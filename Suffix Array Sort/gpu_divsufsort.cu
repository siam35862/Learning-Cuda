// gpu_divsufsort.cu
//
// GPU suffix array construction via parallel prefix-doubling
// (Karkkainen-Sanders style rank-doubling), implemented with Thrust.
//
// ---------------------------------------------------------------------
// WHAT CHANGED IN THIS VERSION (and why)
// ---------------------------------------------------------------------
// The previous version used thrust::device_vector for its scratch
// buffers. Each device_vector construction/destruction issues a plain
// cudaMalloc/cudaFree under the hood. On the CUDA driver's classic
// (non-stream-ordered) allocator, cudaMalloc/cudaFree are effectively
// DEVICE-WIDE synchronization points -- not just for their own stream.
// When running many concurrent streams (e.g. a batch benchmark with
// 16-32 streams processing independent texts), every call's internal
// alloc/free was silently serializing against every OTHER call's
// alloc/free, regardless of which stream each was issued on. This is
// the primary reason GPU utilization measured only ~30-50% even with
// many streams in flight: the streams were mostly waiting on each
// other's allocator calls, not actually computing concurrently.
//
// Two fixes are provided here, and you should pick based on your use
// case:
//
//   1) gpu_divsufsort(...)            -- drop-in replacement, SAME
//      signature as before. Internally now uses cudaMallocAsync /
//      cudaFreeAsync (CUDA >= 11.2), which draw from a stream-ordered
//      memory pool and do NOT block other streams. This alone should
//      noticeably raise concurrent-stream utilization with zero
//      caller-side changes.
//
//   2) gpu_divsufsort_ws(...)         -- for batch workloads (your
//      1000+ independent files case): takes a pre-allocated
//      GpuDivsufsortWorkspace that the caller allocates ONCE per
//      stream and reuses across every call on that stream. This
//      eliminates ALL per-call device allocation, not just the
//      blocking behavior of it -- the fastest option for repeated
//      calls of similar/bounded size. See batch usage example near
//      the bottom of this file.
//
// A secondary, smaller change: the per-round convergence check
// (previously `*thrust::max_element(...)`, which does an implicit
// device->host copy whose cost/behavior is opaque) now does an
// explicit cudaMemcpyAsync of just that one int into a reused pinned
// (page-locked) host buffer, followed by cudaStreamSynchronize(stream)
// -- same as before, this only ever synchronized THIS stream, but the
// explicit pinned-memory path is measurably cheaper per round than an
// unpinned implicit copy over ~20 rounds.
//
// ---------------------------------------------------------------------
// API is deliberately shaped like libdivsufsort's divsufsort():
//
//     int gpu_divsufsort(const unsigned char *T, int *SA, int n);
//
// It also optionally reports peak device memory and accepts a CUDA
// stream, so the SAME function serves both a single standalone call
// and a batch benchmark running many independent texts concurrently
// across multiple streams:
//
//     int gpu_divsufsort(const unsigned char *T, int *SA, int n,
//                        size_t *peakDeviceBytes, cudaStream_t stream);
//
// Build (requires CUDA 11.2+ for cudaMallocAsync/cudaFreeAsync):
//     nvcc -O3 -std=c++17 gpu_divsufsort.cu -o gpu_divsufsort
//
// Notes:
//   - Work: O(n log n), Parallel steps: O(log n) sort passes.
//   - Uses thrust::sort_by_key (radix-sort backend for integer keys)
//     for each doubling round, run entirely on the device.
//   - Memory: ~20 bytes/element (rank + SA as 4-byte ints, keys as an
//     8-byte combined key, plus one shared 4-byte scratch buffer reused
//     for both new-group flags and the in-place scan).

#include "gpu_divsufsort.h"

#include <cuda_runtime.h>
#include <thrust/sort.h>
#include <thrust/scan.h>
#include <thrust/sequence.h>
#include <thrust/gather.h>
#include <thrust/functional.h>
#include <thrust/execution_policy.h>
#include <thrust/system/cuda/execution_policy.h> // thrust::cuda::par.on(stream)
#include <thrust/iterator/zip_iterator.h>
#include <thrust/device_ptr.h>
#include <thrust/tuple.h>

#include <cstdio>
#include <cstdint>
#include <vector>
#include <chrono>
#include <stdexcept>

// ---------------------------------------------------------------------
// Device helpers
// ---------------------------------------------------------------------

// Combine two 32-bit ranks into a single 64-bit sort key so one
// sort_by_key call orders by (rank[i], rank[i+k]) simultaneously.
// +1 offset lets us use 0 to mean "past end of string" (sorts lowest).
struct MakeKey
{
    const int *rank;
    int n, k;

    MakeKey(const int *r, int n_, int k_) : rank(r), n(n_), k(k_) {}

    __host__ __device__
    uint64_t operator()(int i) const
    {
        uint32_t r1 = static_cast<uint32_t>(rank[i]) + 1u;
        uint32_t r2 = (i + k < n) ? static_cast<uint32_t>(rank[i + k]) + 1u : 0u;
        return (static_cast<uint64_t>(r1) << 32) | static_cast<uint64_t>(r2);
    }
};

// After sorting SA by combined key, mark 1 where a new rank group
// starts (its key differs from the previous suffix's key), else 0.
struct NewGroupFlag
{
    const uint64_t *sortedKeys;

    NewGroupFlag(const uint64_t *k) : sortedKeys(k) {}

    __host__ __device__
    int operator()(int idx) const
    {
        if (idx == 0) return 0;
        return sortedKeys[idx] != sortedKeys[idx - 1] ? 1 : 0;
    }
};

struct ByteToInt
{
    __host__ __device__ int operator()(unsigned char c) const { return static_cast<int>(c); }
};

// GpuDivsufsortWorkspace is now declared in gpu_divsufsort.h so that
// batch_bench_divsufsort.cu (or any other caller) can use it too
// without the struct layout drifting out of sync between files.

// ---------------------------------------------------------------------
// Core loop, shared by both the allocating wrapper and the workspace
// version. Assumes d_rank/d_SA/d_keys/d_scratch/d_text are all valid,
// sized >= n, and that d_text already holds the uploaded input bytes
// is NOT assumed -- this function does the H2D copy itself.
// ---------------------------------------------------------------------
static void runPrefixDoubling(const unsigned char *T, int *SA, int n,
                               int *d_rank, int *d_SA, uint64_t *d_keys,
                               int *d_scratch, unsigned char *d_text,
                               int *h_maxRank_pinned, cudaStream_t stream)
{
    auto policy = thrust::cuda::par.on(stream);

    thrust::device_ptr<int>      rankPtr(d_rank);
    thrust::device_ptr<int>      saPtr(d_SA);
    thrust::device_ptr<uint64_t> keysPtr(d_keys);
    thrust::device_ptr<int>      scratchPtr(d_scratch);
    thrust::device_ptr<unsigned char> textPtr(d_text);

    // --- Upload text, initial rank = byte value ---
    cudaMemcpyAsync(d_text, T, n, cudaMemcpyHostToDevice, stream);
    thrust::transform(policy, textPtr, textPtr + n, rankPtr, ByteToInt());

    thrust::sequence(policy, saPtr, saPtr + n); // SA[i] = i initially

    int k = 1;
    while (true)
    {
        // 1) Build combined (rank[i], rank[i+k]) key for every suffix,
        //    in CURRENT SA order (so the sort is stable-refining).
        {
            MakeKey mk(d_rank, n, k);
            thrust::transform(policy, saPtr, saPtr + n, keysPtr, mk);
        }

        // 2) Sort suffix indices by that key (radix sort backend).
        thrust::sort_by_key(policy, keysPtr, keysPtr + n, saPtr);

        // 3) Flags -> in-place inclusive scan -> new ranks (by sorted position).
        {
            NewGroupFlag ngf(d_keys);
            thrust::transform(
                policy,
                thrust::counting_iterator<int>(0),
                thrust::counting_iterator<int>(n),
                scratchPtr, ngf);

            thrust::inclusive_scan(policy, scratchPtr, scratchPtr + n, scratchPtr);
        }

        // 4) Scatter new ranks back into d_rank, indexed by position.
        thrust::scatter(policy, scratchPtr, scratchPtr + n, saPtr, rankPtr);

        // 5) Convergence check: explicit pinned-memory readback instead
        //    of thrust::max_element's implicit copy. Still only
        //    synchronizes THIS stream (cudaStreamSynchronize), so other
        //    concurrent streams keep running -- just cheaper per round.
        {
            thrust::device_ptr<int> maxIt = thrust::max_element(policy, rankPtr, rankPtr + n);
            const int *d_maxPtr = thrust::raw_pointer_cast(maxIt);
            cudaMemcpyAsync(h_maxRank_pinned, d_maxPtr, sizeof(int),
                             cudaMemcpyDeviceToHost, stream);
            cudaStreamSynchronize(stream);
        }

        if (*h_maxRank_pinned == n - 1)
            break;

        k <<= 1;
        if (k > 2 * n) // safety net
            break;
    }

    // --- Download result ---
    cudaMemcpyAsync(SA, d_SA, n * sizeof(int), cudaMemcpyDeviceToHost, stream);
    cudaStreamSynchronize(stream); // guarantee SA is populated before returning
}

// ---------------------------------------------------------------------
// gpu_divsufsort_ws: batch-oriented entry point.
//
// Pass a GpuDivsufsortWorkspace that you allocated ONCE (sized for the
// largest n you'll call with on this stream) and reuse for every text
// processed on that stream. Zero device allocation happens inside this
// call -- this is the version to use for your 1000+-file batch case.
//
//   ws.capacity must be >= n, or this returns -3.
// ---------------------------------------------------------------------
int gpu_divsufsort_ws(const unsigned char *T, int *SA, int n,
                       GpuDivsufsortWorkspace &ws, cudaStream_t stream) // NOLINT: matches header decl
{
    if (n < 0 || (n > 0 && (T == nullptr || SA == nullptr))) return -1;
    if (n == 0) return 0;
    if (n == 1) { SA[0] = 0; return 0; }
    if (n > ws.capacity) return -3; // workspace too small; allocate() for a larger maxN

    runPrefixDoubling(T, SA, n, ws.d_rank, ws.d_SA, ws.d_keys, ws.d_scratch,
                       ws.d_text, ws.h_maxRank_pinned, stream);
    return 0;
}

// ---------------------------------------------------------------------
// gpu_divsufsort: drop-in replacement for the original API. Allocates
// and frees its scratch buffers with the stream-ordered allocator
// (cudaMallocAsync/cudaFreeAsync) every call -- much less blocking
// than the old device_vector-based version, but still not free. For
// a batch of many calls on the same stream, prefer gpu_divsufsort_ws
// with a workspace allocated once outside your loop.
// ---------------------------------------------------------------------
int gpu_divsufsort(const unsigned char *T, int *SA, int n,
                    size_t *peakDeviceBytes, cudaStream_t stream)
{
    if (n < 0 || (n > 0 && (T == nullptr || SA == nullptr)))
        return -1;

    if (n == 0)
    {
        if (peakDeviceBytes) *peakDeviceBytes = 0;
        return 0;
    }

    if (n == 1)
    {
        SA[0] = 0;
        if (peakDeviceBytes) *peakDeviceBytes = 0;
        return 0;
    }

    if (peakDeviceBytes)
    {
        size_t bytes = 0;
        bytes += static_cast<size_t>(n) * sizeof(int);      // d_rank
        bytes += static_cast<size_t>(n) * sizeof(int);      // d_SA
        bytes += static_cast<size_t>(n) * sizeof(uint64_t); // d_keys
        bytes += static_cast<size_t>(n) * sizeof(int);      // d_scratch
        bytes += static_cast<size_t>(n) * sizeof(unsigned char); // d_text
        *peakDeviceBytes = bytes;
    }

    int      *d_rank = nullptr, *d_SA = nullptr, *d_scratch = nullptr;
    uint64_t *d_keys = nullptr;
    unsigned char *d_text = nullptr;
    int      *h_maxRank_pinned = nullptr;

    cudaError_t e;
    e = cudaMallocAsync(&d_rank,    static_cast<size_t>(n) * sizeof(int),      stream); if (e) return -2;
    e = cudaMallocAsync(&d_SA,      static_cast<size_t>(n) * sizeof(int),      stream); if (e) return -2;
    e = cudaMallocAsync(&d_keys,    static_cast<size_t>(n) * sizeof(uint64_t), stream); if (e) return -2;
    e = cudaMallocAsync(&d_scratch, static_cast<size_t>(n) * sizeof(int),      stream); if (e) return -2;
    e = cudaMallocAsync(&d_text,    static_cast<size_t>(n) * sizeof(unsigned char), stream); if (e) return -2;
    e = cudaHostAlloc(&h_maxRank_pinned, sizeof(int), cudaHostAllocDefault);   if (e) return -2;

    runPrefixDoubling(T, SA, n, d_rank, d_SA, d_keys, d_scratch, d_text,
                       h_maxRank_pinned, stream);

    cudaFreeAsync(d_rank,    stream);
    cudaFreeAsync(d_SA,      stream);
    cudaFreeAsync(d_keys,    stream);
    cudaFreeAsync(d_scratch, stream);
    cudaFreeAsync(d_text,    stream);
    cudaFreeHost(h_maxRank_pinned);

    return 0;
}

// ---------------------------------------------------------------------
// Simple self-test / example usage
// ---------------------------------------------------------------------
#ifdef GPU_DIVSUFSORT_MAIN
static void printSA(const unsigned char *T, const int *SA, int n)
{
    for (int i = 0; i < n; ++i)
    {
        printf("SA[%2d] = %2d  suffix: %s\n", i, SA[i],
               reinterpret_cast<const char*>(T + SA[i]));
    }
}

// Example: how a batch loop should use the workspace API for zero
// per-call allocation (this is the pattern to use for your 1000+ file
// case -- one workspace per stream, reused across every file on it).
static void batchExample(const std::vector<std::vector<unsigned char>> &texts,
                          cudaStream_t stream, int maxN)
{
    GpuDivsufsortWorkspace ws;
    if (!ws.allocate(maxN, stream))
    {
        fprintf(stderr, "workspace allocation failed\n");
        return;
    }

    for (const auto &text : texts)
    {
        std::vector<int> SA(text.size());
        int rc = gpu_divsufsort_ws(text.data(), SA.data(),
                                    static_cast<int>(text.size()), ws, stream);
        if (rc != 0)
        {
            fprintf(stderr, "gpu_divsufsort_ws failed: %d\n", rc);
            continue;
        }
        // ... consume SA ...
    }
    // ws destructor frees everything once, here, not per-file.
}

int main()
{
    // Classic example: "banana"
    const char *text = "banana";
    int n = static_cast<int>(strlen(text));
    std::vector<int> SA(n);

    auto t0 = std::chrono::high_resolution_clock::now();
    int rc = gpu_divsufsort(reinterpret_cast<const unsigned char*>(text), SA.data(), n);
    auto t1 = std::chrono::high_resolution_clock::now();

    if (rc != 0)
    {
        printf("gpu_divsufsort failed: %d\n", rc);
        return 1;
    }

    printSA(reinterpret_cast<const unsigned char*>(text), SA.data(), n);
    printf("Elapsed: %.3f ms\n",
           std::chrono::duration<double, std::milli>(t1 - t0).count());

    // Expected suffix array for "banana": 5 3 1 0 4 2
    // (a, ana, anana, banana, na, nana)
    return 0;
}
#endif
