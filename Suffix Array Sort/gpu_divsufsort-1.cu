// gpu_divsufsort.cu
//
// GPU suffix array construction via parallel prefix-doubling
// (Karkkainen-Sanders style rank-doubling), implemented with Thrust.
//
// API is deliberately shaped like libdivsufsort's divsufsort():
//
//     int gpu_divsufsort(const unsigned char *T, int *SA, int n);
//
// Build:
//     nvcc -O3 -std=c++17 gpu_divsufsort.cu -o gpu_divsufsort
//
// Notes:
//   - Work: O(n log n), Parallel steps: O(log n) sort passes.
//   - Uses thrust::sort_by_key (radix-sort backend for integer keys)
//     for each doubling round, run entirely on the device.
//   - Memory: ~7 int arrays of size n (rank, rank2, SA, SA2, keys hi/lo,
//     flags) — heavier than divsufsort's footprint, budget accordingly.

#include <cuda_runtime.h>
#include <thrust/device_vector.h>
#include <thrust/sort.h>
#include <thrust/scan.h>
#include <thrust/sequence.h>
#include <thrust/gather.h>
#include <thrust/functional.h>
#include <thrust/execution_policy.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/tuple.h>

#include <cstdio>
#include <cstdint>
#include <vector>
#include <chrono>

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

// ---------------------------------------------------------------------
// gpu_divsufsort
//
//   T  - input bytes (host pointer), length n
//   SA - output suffix array (host pointer), must hold n ints
//   n  - length of T
//
// Returns 0 on success, negative on error (mirrors divsufsort's contract).
// ---------------------------------------------------------------------
int gpu_divsufsort(const unsigned char *T, int *SA, int n)
{
    if (n < 0 || (n > 0 && (T == nullptr || SA == nullptr)))
        return -1;

    if (n == 0)
        return 0;

    if (n == 1)
    {
        SA[0] = 0;
        return 0;
    }

    // --- Upload text, initial rank = byte value ---
    thrust::device_vector<int> d_rank(n);
    {
        thrust::device_vector<unsigned char> d_text(T, T + n);
        thrust::transform(
            d_text.begin(), d_text.end(), d_rank.begin(),
            [] __device__ (unsigned char c) { return static_cast<int>(c); });
    }

    thrust::device_vector<int> d_SA(n);
    thrust::sequence(d_SA.begin(), d_SA.end()); // SA[i] = i initially

    thrust::device_vector<uint64_t> d_keys(n);
    thrust::device_vector<int>      d_flags(n);
    thrust::device_vector<int>      d_newRank(n);
    thrust::device_vector<int>      d_scatterRank(n); // rank indexed by position

    int k = 1;
    while (true)
    {
        // 1) Build combined (rank[i], rank[i+k]) key for every suffix,
        //    in CURRENT SA order (so the sort is stable-refining).
        {
            MakeKey mk(thrust::raw_pointer_cast(d_rank.data()), n, k);
            thrust::transform(d_SA.begin(), d_SA.end(), d_keys.begin(), mk);
        }

        // 2) Sort suffix indices by that key. thrust::sort_by_key on
        //    64-bit keys dispatches to a radix sort — this is the
        //    parallel step that replaces divsufsort's induced sorting.
        thrust::sort_by_key(d_keys.begin(), d_keys.end(), d_SA.begin());

        // 3) Assign new ranks: flag[i] = 1 if this suffix starts a new
        //    equivalence class (its key differs from the previous one
        //    in sorted order), else 0. Inclusive scan of flags gives
        //    the new rank (0-based) for the suffix at sorted position i.
        {
            NewGroupFlag ngf(thrust::raw_pointer_cast(d_keys.data()));
            thrust::transform(
                thrust::counting_iterator<int>(0),
                thrust::counting_iterator<int>(n),
                d_flags.begin(), ngf);

            thrust::inclusive_scan(d_flags.begin(), d_flags.end(), d_newRank.begin());
        }

        // 4) Scatter new ranks back into position-indexed rank array:
        //    d_scatterRank[ d_SA[i] ] = d_newRank[i]
        thrust::scatter(d_newRank.begin(), d_newRank.end(), d_SA.begin(), d_scatterRank.begin());
        d_rank.swap(d_scatterRank);

        // 5) Termination: if the last (highest) rank equals n-1, every
        //    suffix is in its own class -> fully sorted.
        int maxRank = d_rank.empty() ? 0
            : *thrust::max_element(d_rank.begin(), d_rank.end());
        if (maxRank == n - 1)
            break;

        // Also stop if k has grown past n (safety net; shouldn't trigger
        // before maxRank converges for valid input).
        k <<= 1;
        if (k > 2 * n)
            break;
    }

    // --- Download result ---
    thrust::copy(d_SA.begin(), d_SA.end(), SA);

    return 0;
}

// ---------------------------------------------------------------------
// Optional: 64-bit-index variant for inputs > ~2^31 suffixes, mirroring
// divsufsort64(). Included as a thin wrapper; swap `int` for `long long`
// throughout the above if you actually need >2GB inputs.
// ---------------------------------------------------------------------
// int gpu_divsufsort64(const unsigned char *T, long long *SA, long long n);
// (left unimplemented here — flag if you need it, the rank/key types
//  just need widening to int64_t / int64_t pairs.)

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
