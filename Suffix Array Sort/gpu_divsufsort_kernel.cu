// gpu_divsufsort_kernel.cu
//
// GPU suffix array construction via parallel prefix-doubling,
// implemented with EXPLICIT CUDA KERNELS (no Thrust). Every step that
// the Thrust version delegated to a library algorithm here is its own
// __global__ kernel:
//
//   buildKeysKernel   - replaces thrust::transform (key construction)
//   bitonicStepKernel - replaces thrust::sort_by_key (the sort itself)
//   flagKernel        - replaces thrust::transform (new-group flags)
//   scanStepKernel    - replaces thrust::inclusive_scan (Hillis-Steele,
//                       one kernel launch per stride = one global sync)
//   scatterKernel     - replaces thrust::scatter (rank writeback)
//
// Same external contract as libdivsufsort / the Thrust version:
//
//     int gpu_divsufsort(const unsigned char *T, int *SA, int n);
//
// Build:
//     nvcc -O3 -std=c++17 --extended-lambda gpu_divsufsort_kernel.cu -o gpu_divsufsort_kernel -DGPU_DIVSUFSORT_MAIN
//
// Tradeoff vs the Thrust version:
//   - Sort here is a naive global-memory bitonic sort: O(n log^2 n)
//     compare-exchanges, padded to the next power of two. Thrust's
//     sort_by_key uses a radix sort (O(n) passes) and will be faster
//     at scale. This version trades that speed for full visibility
//     into the actual kernels doing the work.
//   - Memory: SA/keys arrays are padded to the next power of two (m),
//     so worst case (n just over a power of two) nearly doubles the
//     footprint versus the exact-size Thrust arrays.

#include <cuda_runtime.h>

#include <cstdio>
#include <cstdint>
#include <cstring>
#include <vector>
#include <chrono>
#include <algorithm>

#define CUDA_CHECK(call)                                                    \
    do {                                                                    \
        cudaError_t _e = (call);                                            \
        if (_e != cudaSuccess) {                                            \
            fprintf(stderr, "CUDA error %s at %s:%d\n",                     \
                    cudaGetErrorString(_e), __FILE__, __LINE__);             \
            return -2;                                                      \
        }                                                                   \
    } while (0)

static const uint64_t SENTINEL_MAX = ~0ULL; // padded slots sort last

// ---------------------------------------------------------------------
// Kernel 1: build the combined (rank[SA[i]], rank[SA[i]+k]) sort key
// for every slot i in [0, m). Slots >= n are padding and get the max
// sentinel key so they always sort to the end.
// ---------------------------------------------------------------------
__global__ void buildKeysKernel(const int *rank, const int *SA,
                                 uint64_t *keys, int n, int m, int k)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= m) return;

    if (i >= n) { keys[i] = SENTINEL_MAX; return; }

    int pos = SA[i];
    uint32_t r1 = static_cast<uint32_t>(rank[pos]) + 1u;
    uint32_t r2 = (pos + k < n) ? static_cast<uint32_t>(rank[pos + k]) + 1u : 0u;
    keys[i] = (static_cast<uint64_t>(r1) << 32) | static_cast<uint64_t>(r2);
}

// ---------------------------------------------------------------------
// Kernel 2: one compare-exchange step of bitonic sort over (keys, SA)
// pairs, array length m (must be a power of two). Standard formula:
// for stage size `k` and step `j`, each thread handles index i and its
// partner ixj = i ^ j; ascending/descending direction depends on bit
// pattern of i relative to k.
// ---------------------------------------------------------------------
__global__ void bitonicStepKernel(uint64_t *keys, int *SA, int m, int k, int j)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= m) return;

    int ixj = i ^ j;
    if (ixj <= i) return; // only the lower index of each pair acts

    bool ascending = ((i & k) == 0);
    uint64_t ki = keys[i], kixj = keys[ixj];

    bool shouldSwap = ascending ? (ki > kixj) : (ki < kixj);
    if (shouldSwap)
    {
        keys[i] = kixj; keys[ixj] = ki;
        int si = SA[i], sixj = SA[ixj];
        SA[i] = sixj; SA[ixj] = si;
    }
}

// ---------------------------------------------------------------------
// Kernel 3: mark the start of a new equivalence class. flags[0] = 0 by
// convention; flags[i] = 1 if sortedKeys[i] != sortedKeys[i-1].
// Only meaningful (and only launched) over the first n slots, since
// slots >= n are padding that got sorted to the end.
// ---------------------------------------------------------------------
__global__ void flagKernel(const uint64_t *sortedKeys, int *flags, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    flags[i] = (i == 0) ? 0 : (sortedKeys[i] != sortedKeys[i - 1] ? 1 : 0);
}

// ---------------------------------------------------------------------
// Kernel 4: one stride-`d` step of an inclusive Hillis-Steele scan.
// Reads from `in`, writes to `out`; caller ping-pongs the two buffers
// across strides 1, 2, 4, ... doubling until d >= n. Each kernel launch
// is itself the global synchronization point between scan levels.
// ---------------------------------------------------------------------
__global__ void scanStepKernel(const int *in, int *out, int n, int d)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    out[i] = (i >= d) ? in[i] + in[i - d] : in[i];
}

// ---------------------------------------------------------------------
// Kernel 5: scatter new ranks (indexed by sorted order) back into a
// position-indexed rank array: rank[SA[i]] = newRank[i].
// ---------------------------------------------------------------------
__global__ void scatterKernel(const int *newRank, const int *SA, int *rankOut, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    rankOut[SA[i]] = newRank[i];
}

// ---------------------------------------------------------------------
// Kernel 0: initial rank = byte value; SA[i] = i for real slots,
// SA[i] = 0 (unused, key is sentinel so it never wins a compare) for
// padding slots.
// ---------------------------------------------------------------------
__global__ void initKernel(const unsigned char *text, int *rank, int *SA, int n, int m)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= m) return;
    if (i < n) { rank[i] = static_cast<int>(text[i]); SA[i] = i; }
    else       { SA[i] = 0; }
}

static inline int nextPow2(int n)
{
    int m = 1;
    while (m < n) m <<= 1;
    return m;
}

// ---------------------------------------------------------------------
// gpu_divsufsort — kernel-only implementation
// ---------------------------------------------------------------------
int gpu_divsufsort(const unsigned char *T, int *SA_out, int n)
{
    if (n < 0 || (n > 0 && (T == nullptr || SA_out == nullptr)))
        return -1;
    if (n == 0) return 0;
    if (n == 1) { SA_out[0] = 0; return 0; }

    const int m = nextPow2(n); // bitonic sort needs a power-of-two length

    unsigned char *d_text = nullptr;
    int *d_rank = nullptr, *d_SA = nullptr;
    int *d_flags = nullptr, *d_scanA = nullptr, *d_scanB = nullptr;
    uint64_t *d_keys = nullptr;

    CUDA_CHECK(cudaMalloc(&d_text, n * sizeof(unsigned char)));
    CUDA_CHECK(cudaMalloc(&d_rank, n * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_SA,   m * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_keys, m * sizeof(uint64_t)));
    CUDA_CHECK(cudaMalloc(&d_flags,  n * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_scanA,  n * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_scanB,  n * sizeof(int)));

    CUDA_CHECK(cudaMemcpy(d_text, T, n * sizeof(unsigned char), cudaMemcpyHostToDevice));

    const int block = 256;
    auto grid = [&](int count) { return (count + block - 1) / block; };

    initKernel<<<grid(m), block>>>(d_text, d_rank, d_SA, n, m);
    CUDA_CHECK(cudaGetLastError());

    int k = 1;
    while (true)
    {
        // 1) Build keys over the full padded length m.
        buildKeysKernel<<<grid(m), block>>>(d_rank, d_SA, d_keys, n, m, k);
        CUDA_CHECK(cudaGetLastError());

        // 2) Bitonic sort (keys, SA) ascending over length m.
        for (int stage = 2; stage <= m; stage <<= 1)
            for (int step = stage >> 1; step > 0; step >>= 1)
            {
                bitonicStepKernel<<<grid(m), block>>>(d_keys, d_SA, m, stage, step);
                CUDA_CHECK(cudaGetLastError());
            }
        // Padding slots (key = SENTINEL_MAX) are now at the tail; the
        // first n slots of d_keys/d_SA hold the real, sorted suffixes.

        // 3) New-group flags over the first n (real) slots.
        flagKernel<<<grid(n), block>>>(d_keys, d_flags, n);
        CUDA_CHECK(cudaGetLastError());

        // 4) Inclusive Hillis-Steele scan of flags -> new ranks.
        //    Ping-pong d_scanA/d_scanB; each stride doubling is one
        //    kernel launch (= one global barrier).
        CUDA_CHECK(cudaMemcpy(d_scanA, d_flags, n * sizeof(int), cudaMemcpyDeviceToDevice));
        int *curIn = d_scanA, *curOut = d_scanB;
        for (int d = 1; d < n; d <<= 1)
        {
            scanStepKernel<<<grid(n), block>>>(curIn, curOut, n, d);
            CUDA_CHECK(cudaGetLastError());
            std::swap(curIn, curOut);
        }
        // curIn now holds the final inclusive scan (new ranks, 0-based
        // relative order preserved since flags start at 0 for slot 0).

        // 5) Scatter new ranks back into position-indexed rank array.
        scatterKernel<<<grid(n), block>>>(curIn, d_SA, d_rank, n);
        CUDA_CHECK(cudaGetLastError());

        // 6) Termination check: read back the last (max) rank value.
        int maxRank = -1;
        CUDA_CHECK(cudaMemcpy(&maxRank, curIn + (n - 1), sizeof(int), cudaMemcpyDeviceToHost));
        if (maxRank == n - 1) break;

        k <<= 1;
        if (k > 2 * n) break; // safety net
    }

    CUDA_CHECK(cudaMemcpy(SA_out, d_SA, n * sizeof(int), cudaMemcpyDeviceToHost));

    cudaFree(d_text); cudaFree(d_rank); cudaFree(d_SA); cudaFree(d_keys);
    cudaFree(d_flags); cudaFree(d_scanA); cudaFree(d_scanB);

    return 0;
}

// ---------------------------------------------------------------------
// Self-test
// ---------------------------------------------------------------------
#ifdef GPU_DIVSUFSORT_MAIN
static void printSA(const unsigned char *T, const int *SA, int n)
{
    for (int i = 0; i < n; ++i)
        printf("SA[%2d] = %2d  suffix: %s\n", i, SA[i],
               reinterpret_cast<const char*>(T + SA[i]));
}

int main()
{
    const char *text = "banana";
    int n = static_cast<int>(strlen(text));
    std::vector<int> SA(n);

    auto t0 = std::chrono::high_resolution_clock::now();
    int rc = gpu_divsufsort(reinterpret_cast<const unsigned char*>(text), SA.data(), n);
    auto t1 = std::chrono::high_resolution_clock::now();

    if (rc != 0) { printf("gpu_divsufsort failed: %d\n", rc); return 1; }

    printSA(reinterpret_cast<const unsigned char*>(text), SA.data(), n);
    printf("Elapsed: %.3f ms\n",
           std::chrono::duration<double, std::milli>(t1 - t0).count());

    // Expected: 5 3 1 0 4 2  (a, ana, anana, banana, na, nana)
    return 0;
}
#endif
