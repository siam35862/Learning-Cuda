// gpu_divsufsort_cdp.cu
//
// Device-callable suffix array construction: the ENTIRE prefix-doubling
// loop runs inside a single root kernel, which itself launches every
// stage (key-build, bitonic sort steps, flagging, scan, scatter) as
// child kernels via CUDA Dynamic Parallelism (CDP). This means it can
// be invoked from a single thread of ANOTHER kernel, e.g.:
//
//     __global__ void caller(...) {
//         if (threadIdx.x == 0 && blockIdx.x == 0) {
//             divsufsort_root<<<1, 1>>>(d_text, d_SA, n, d_scratch);
//             cudaDeviceSynchronize(); // wait for the child work to finish
//         }
//     }
//
// IMPORTANT CAVEATS (read before wiring this in):
//   1. This is O(n log n) work (prefix doubling), not O(n) like the CPU's
//      SA-IS. It will not beat divsufsort asymptotically at any n large
//      enough for the crossover to matter — see the chat message this
//      code was delivered with for the full explanation.
//   2. CDP child-launch overhead (~a few microseconds per launch, plus
//      device-side scheduling) is added on top of every kernel in the
//      loop. A host-orchestrated version of the exact same kernels will
//      almost always be faster in wall-clock terms. Use this version
//      only when you actually need the "call it from inside a kernel"
//      property, not for raw speed.
//   3. Requires: compute capability >= 3.5, and compiling/linking with
//      relocatable device code + the device runtime library:
//
//        nvcc -O3 -std=c++17 -rdc=true -arch=sm_75 \
//             gpu_divsufsort_cdp.cu -o gpu_divsufsort_cdp -lcudadevrt \
//             -DGPU_DIVSUFSORT_CDP_MAIN
//
//      (swap -arch=sm_75 for whatever matches your GPU; CDP needs it
//      explicit since it changes code generation.)
//   4. Device-side cudaMalloc draws from a separate device-side heap.
//      If you allocate large scratch buffers from device code, raise
//      the heap limit from the HOST before launching:
//
//        cudaDeviceSetLimit(cudaLimitMallocHeapSize, 256ULL * 1024*1024);
//
//      This example instead takes pre-allocated device scratch memory
//      as a parameter (allocated host-side) to avoid device-side
//      cudaMalloc entirely -- simpler and avoids heap-limit surprises.

#include <cuda_runtime.h>
#include <cstdio>
#include <cstdint>
#include <cstring>
#include <vector>
#include <chrono>

static const uint64_t SENTINEL_MAX = ~0ULL;

// ---------------------------------------------------------------------
// Scratch layout the caller must allocate on the device and pass in.
// Sized against m = nextPow2(n).
// ---------------------------------------------------------------------
struct Scratch
{
    int      *rank;     // [n]
    int      *SA;       // [m]
    uint64_t *keys;      // [m]
    int      *flags;     // [n]
    int      *scanA;     // [n]
    int      *scanB;     // [n]
    int      *maxRank;   // [1]  device-resident, avoids host round-trip
};

__host__ __device__ static inline int nextPow2Dev(int n)
{
    int m = 1;
    while (m < n) m <<= 1;
    return m;
}

// ---------------------------------------------------------------------
// Same five kernels as the plain kernel version (build/sort/flag/scan/
// scatter) -- unchanged logic, just reused here as children of the CDP
// root kernel instead of being launched from the host.
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

__global__ void bitonicStepKernel(uint64_t *keys, int *SA, int m, int k, int j)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= m) return;
    int ixj = i ^ j;
    if (ixj <= i) return;
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

__global__ void flagKernel(const uint64_t *sortedKeys, int *flags, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    flags[i] = (i == 0) ? 0 : (sortedKeys[i] != sortedKeys[i - 1] ? 1 : 0);
}

__global__ void scanStepKernel(const int *in, int *out, int n, int d)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    out[i] = (i >= d) ? in[i] + in[i - d] : in[i];
}

__global__ void scatterKernel(const int *newRank, const int *SA, int *rankOut, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    rankOut[SA[i]] = newRank[i];
}

__global__ void initKernel(const unsigned char *text, int *rank, int *SA, int n, int m)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= m) return;
    if (i < n) { rank[i] = static_cast<int>(text[i]); SA[i] = i; }
    else       { SA[i] = 0; }
}

__global__ void copyLastKernel(const int *arr, int idx, int *out)
{
    if (threadIdx.x == 0 && blockIdx.x == 0) *out = arr[idx];
}

// ---------------------------------------------------------------------
// Root kernel: single thread orchestrates the whole prefix-doubling
// loop, issuing every child kernel from the DEVICE via CDP. Launch
// this with <<<1,1>>> from host or from another kernel.
//
//   text : device pointer, n bytes
//   SAOut: device pointer, n ints (final result written here)
//   n    : length
//   s    : pre-allocated scratch (see Scratch struct); caller sizes
//          rank/flags/scanA/scanB to n and SA/keys to m = nextPow2(n)
// ---------------------------------------------------------------------
__global__ void divsufsort_root(const unsigned char *text, int *SAOut, int n, Scratch s)
{
    if (threadIdx.x != 0 || blockIdx.x != 0) return;

    if (n <= 1)
    {
        if (n == 1) SAOut[0] = 0;
        return;
    }

    const int m = nextPow2Dev(n);
    const int block = 256;
    auto gridOf = [](int count, int b) { return (count + b - 1) / b; };

    initKernel<<<gridOf(m, block), block>>>(text, s.rank, s.SA, n, m);
    cudaDeviceSynchronize();

    int k = 1;
    while (true)
    {
        buildKeysKernel<<<gridOf(m, block), block>>>(s.rank, s.SA, s.keys, n, m, k);
        cudaDeviceSynchronize();

        for (int stage = 2; stage <= m; stage <<= 1)
            for (int step = stage >> 1; step > 0; step >>= 1)
            {
                bitonicStepKernel<<<gridOf(m, block), block>>>(s.keys, s.SA, m, stage, step);
                cudaDeviceSynchronize();
            }

        flagKernel<<<gridOf(n, block), block>>>(s.keys, s.flags, n);
        cudaDeviceSynchronize();

        // Copy flags into scanA to seed the ping-pong scan.
        for (int i = 0; i < n; i += n) {} // no-op (kept for clarity of intent)
        cudaMemcpyAsync(s.scanA, s.flags, n * sizeof(int), cudaMemcpyDeviceToDevice);
        cudaDeviceSynchronize();

        int *curIn = s.scanA, *curOut = s.scanB;
        for (int d = 1; d < n; d <<= 1)
        {
            scanStepKernel<<<gridOf(n, block), block>>>(curIn, curOut, n, d);
            cudaDeviceSynchronize();
            int *tmp = curIn; curIn = curOut; curOut = tmp;
        }

        scatterKernel<<<gridOf(n, block), block>>>(curIn, s.SA, s.rank, n);
        cudaDeviceSynchronize();

        copyLastKernel<<<1, 1>>>(curIn, n - 1, s.maxRank);
        cudaDeviceSynchronize();

        if (*s.maxRank == n - 1) break;

        k <<= 1;
        if (k > 2 * n) break;
    }

    // Final SA is in s.SA[0..n) at this point.
    for (int i = 0; i < n; ++i) SAOut[i] = s.SA[i]; // single-thread copy;
    // fine for the root thread since n is typically modest relative to
    // total kernel time here -- swap for a parallel copy kernel if n is
    // large and this loop becomes a measurable fraction of total time.
}

// ---------------------------------------------------------------------
// Host-side convenience wrapper: allocates scratch + device I/O buffers,
// launches the root kernel, copies the result back. Same external
// contract as divsufsort()/gpu_divsufsort() for easy drop-in testing --
// but note this wrapper itself launches from the HOST. The actual
// "callable from device code" entry point is divsufsort_root() above;
// call that directly (with your own pre-allocated device Scratch) from
// inside your own kernel instead of using this wrapper if that's your
// real use case.
// ---------------------------------------------------------------------
int gpu_divsufsort_cdp(const unsigned char *T, int *SA_out, int n)
{
    if (n < 0 || (n > 0 && (T == nullptr || SA_out == nullptr))) return -1;
    if (n == 0) return 0;
    if (n == 1) { SA_out[0] = 0; return 0; }

    const int m = nextPow2Dev(n);

    unsigned char *d_text = nullptr;
    int *d_SAOut = nullptr;
    Scratch s{};

    cudaMalloc(&d_text, n * sizeof(unsigned char));
    cudaMalloc(&d_SAOut, n * sizeof(int));
    cudaMalloc(&s.rank, n * sizeof(int));
    cudaMalloc(&s.SA, m * sizeof(int));
    cudaMalloc(&s.keys, m * sizeof(uint64_t));
    cudaMalloc(&s.flags, n * sizeof(int));
    cudaMalloc(&s.scanA, n * sizeof(int));
    cudaMalloc(&s.scanB, n * sizeof(int));
    cudaMalloc(&s.maxRank, sizeof(int));

    cudaMemcpy(d_text, T, n * sizeof(unsigned char), cudaMemcpyHostToDevice);

    divsufsort_root<<<1, 1>>>(d_text, d_SAOut, n, s);
    cudaError_t err = cudaDeviceSynchronize();
    if (err != cudaSuccess)
    {
        fprintf(stderr, "CDP launch failed: %s\n", cudaGetErrorString(err));
        return -2;
    }

    cudaMemcpy(SA_out, d_SAOut, n * sizeof(int), cudaMemcpyDeviceToHost);

    cudaFree(d_text); cudaFree(d_SAOut);
    cudaFree(s.rank); cudaFree(s.SA); cudaFree(s.keys);
    cudaFree(s.flags); cudaFree(s.scanA); cudaFree(s.scanB); cudaFree(s.maxRank);

    return 0;
}

#ifdef GPU_DIVSUFSORT_CDP_MAIN
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
    int rc = gpu_divsufsort_cdp(reinterpret_cast<const unsigned char*>(text), SA.data(), n);
    auto t1 = std::chrono::high_resolution_clock::now();

    if (rc != 0) { printf("gpu_divsufsort_cdp failed: %d\n", rc); return 1; }

    printSA(reinterpret_cast<const unsigned char*>(text), SA.data(), n);
    printf("Elapsed: %.3f ms\n",
           std::chrono::duration<double, std::milli>(t1 - t0).count());
    // Expected: 5 3 1 0 4 2
    return 0;
}
#endif
