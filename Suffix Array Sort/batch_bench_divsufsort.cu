// batch_bench_divsufsort.cu
//
// Benchmarks suffix-array construction over MANY INDEPENDENT texts:
//   1. CPU, single-threaded, one divsufsort() call after another.
//   2. CPU, multi-threaded, one thread per hardware core pulling from
//      a shared queue (fair "CPU parallel" baseline).
//   3. GPU, using a small pool of CUDA STREAMS so multiple texts'
//      suffix-array construction genuinely overlaps on the device.
//
// ---------------------------------------------------------------------
// WHAT CHANGED IN THIS VERSION
// ---------------------------------------------------------------------
// The GPU path now uses gpu_divsufsort_ws() with a PRE-ALLOCATED
// GpuDivsufsortWorkspace per stream, instead of calling the allocating
// gpu_divsufsort() once per text. Each worker thread allocates its
// stream's workspace ONCE, before pulling any work off the queue, then
// reuses it for every text it processes. This eliminates all
// per-call device allocation traffic on the GPU path -- previously,
// every single one of the 1000 calls did its own cudaMalloc*/
// cudaFree*, which (even with the async allocator) adds up over many
// calls and was measured capping GPU utilization at ~30-50% during
// concurrent-stream runs. With workspace reuse, utilization should be
// substantially higher.
//
// One consequence: this version assumes all texts are the SAME SIZE
// (as the benchmark already does -- numTexts x textBytes). If you
// adapt this for texts of varying sizes, size each workspace for the
// LARGEST text it will ever see, or resize/reallocate per stream
// lazily on first oversize text (see gpu_divsufsort_ws's -3 return
// code, which signals "workspace too small").
//
// Build (compiles together with gpu_divsufsort.cu -- no duplicated
// algorithm code, one source of truth for the GPU implementation):
//   nvcc -O3 -std=c++17 -Xcompiler "/Zc:preprocessor" --extended-lambda ^
//        batch_bench_divsufsort.cu gpu_divsufsort.cu -o batch_bench_divsufsort ^
//        -I"C:\vcpkg\libdivsufsort\include" -I"C:\vcpkg\libdivsufsort\build\include" ^
//        -L"C:\vcpkg\libdivsufsort\build\lib\Release" -ldivsufsort
//
// Usage:
//   batch_bench_divsufsort [numTexts] [textBytes] [numStreams]
//   (defaults: 1000, 1048576, 16)
//
// Notes on the GPU path:
//   - Concurrency comes from having several HOST THREADS, each driving
//     one dedicated stream (and that stream's own workspace) through a
//     shared work queue.
//   - Host-side text/result buffers here are ordinary (pageable)
//     memory, not pinned -- simplest to get right first. Pinned memory
//     (cudaHostAlloc) would let the H2D/D2H copies themselves overlap
//     across streams too; ask if you want that version.
//   - Memory: with S concurrent streams, each workspace holds ~24
//     bytes/byte-of-text (rank+SA+keys+scratch+text, see
//     gpu_divsufsort.h), allocated ONCE per stream now rather than
//     once per call. Total device memory is roughly
//     S * 24 * textBytes -- same order of magnitude as before, just
//     allocated once instead of numTexts/S times.

#include "gpu_divsufsort.h"

#include <divsufsort.h>

#include <cuda_runtime.h>

#include <cstdio>
#include <cstdint>
#include <cstring>
#include <vector>
#include <string>
#include <random>
#include <chrono>
#include <thread>
#include <atomic>
#include <iostream>

// ---------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------
static std::vector<std::vector<unsigned char>> makeTexts(int numTexts, size_t textBytes)
{
    std::vector<std::vector<unsigned char>> texts(numTexts);
    std::mt19937_64 rng(42);
    std::uniform_int_distribution<int> dist('a', 'd'); // small alphabet -> realistic repeats
    for (int t = 0; t < numTexts; ++t)
    {
        texts[t].resize(textBytes);
        for (size_t i = 0; i < textBytes; ++i)
            texts[t][i] = static_cast<unsigned char>(dist(rng));
    }
    return texts;
}

template <typename Fn>
static double timeMs(Fn &&fn)
{
    auto t0 = std::chrono::high_resolution_clock::now();
    fn();
    auto t1 = std::chrono::high_resolution_clock::now();
    return std::chrono::duration<double, std::milli>(t1 - t0).count();
}

// ---------------------------------------------------------------------
// main
// ---------------------------------------------------------------------
int main(int argc, char **argv)
{
    int numTexts   = (argc >= 2) ? std::atoi(argv[1]) : 1000;
    size_t textLen = (argc >= 3) ? std::stoull(argv[2]) : (1u << 20); // 1 MiB
    int numStreams = (argc >= 4) ? std::atoi(argv[3]) : 16;

    std::cout << "Texts: " << numTexts << " x " << textLen << " bytes"
              << "  (" << (numTexts * textLen) / (1024.0 * 1024.0) << " MiB total)\n";
    std::cout << "GPU concurrent streams: " << numStreams << "\n";

    auto texts = makeTexts(numTexts, textLen);

    // Only saCpu is kept as the full ground-truth array (needed at the
    // end to verify GPU output). saCpuMT and saGpu results are NEVER
    // inspected beyond "does it match saCpu" -- so instead of storing
    // all 1000 of each (which was ~12.4 GiB combined on a 15 GiB
    // system and pushed the OS into paging/swap, badly skewing the
    // CPU-MT timing), each worker below uses a small per-thread/
    // per-stream REUSABLE buffer and discards it immediately after
    // use. This cuts host RAM use from ~12.4 GiB to ~4.8 GiB.
    std::vector<std::vector<int>> saCpu(numTexts, std::vector<int>(textLen));

    // --- 1) CPU, single-threaded ---
    double cpuMs = timeMs([&]() {
        for (int t = 0; t < numTexts; ++t)
        {
            int rc = divsufsort(texts[t].data(), saCpu[t].data(), static_cast<int>(textLen));
            if (rc != 0) { std::cerr << "divsufsort failed on text " << t << "\n"; std::exit(1); }
        }
    });

    // --- 2) CPU, multi-threaded (shared queue, one thread per core) ---
    // Each worker reuses its OWN buffer across every text it processes
    // -- result is discarded (only timing matters here), so there is
    // no need to keep 1000 separate result arrays alive at once.
    std::atomic<int> nextIdxMT{0};
    unsigned hwThreads = std::max(1u, std::thread::hardware_concurrency());
    double cpuMtMs = timeMs([&]() {
        std::vector<std::thread> pool;
        for (unsigned w = 0; w < hwThreads; ++w)
        {
            pool.emplace_back([&]() {
                std::vector<int> localSA(textLen); // reused every iteration
                int idx;
                while ((idx = nextIdxMT.fetch_add(1)) < numTexts)
                    divsufsort(texts[idx].data(), localSA.data(), static_cast<int>(textLen));
            });
        }
        for (auto &th : pool) th.join();
    });

    // --- 3) GPU, concurrent streams, ONE WORKSPACE PER STREAM ---
    // Each worker thread owns exactly one stream and one workspace,
    // both created before the thread touches the shared work queue,
    // and both reused for every text that thread pulls. No device
    // allocation happens inside the timed loop below.
    std::vector<cudaStream_t> streams(numStreams);
    for (auto &s : streams) cudaStreamCreate(&s);

    std::vector<GpuDivsufsortWorkspace> workspaces(numStreams);
    bool wsOk = true;
    for (int s = 0; s < numStreams; ++s)
        wsOk &= workspaces[s].allocate(static_cast<int>(textLen), streams[s]);
    if (!wsOk)
    {
        std::cerr << "Workspace allocation failed (out of VRAM for "
                  << numStreams << " streams x " << textLen << " bytes each?)\n";
        return 1;
    }

    // gpu_divsufsort_ws doesn't report peakDeviceBytes directly (the
    // workspace's own allocate() already tells you the size you asked
    // for); compute the same ~24 bytes/element figure here for the
    // printout below, matching what gpu_divsufsort.h's buffers hold.
    size_t gpuPerWorkspaceBytes =
        static_cast<size_t>(textLen) * (sizeof(int) * 3 + sizeof(uint64_t) + sizeof(unsigned char));

    // Each GPU worker also reuses its own host-side result buffer, and
    // verifies against saCpu[idx] IMMEDIATELY after each call instead
    // of stashing all 1000 results for a separate comparison pass
    // afterward -- same reasoning as the CPU-MT change above.
    std::atomic<int> nextIdxGpu{0};
    std::atomic<bool> allMatch{true};
    double gpuMs = timeMs([&]() {
        std::vector<std::thread> pool;
        for (int s = 0; s < numStreams; ++s)
        {
            pool.emplace_back([&, s]() {
                std::vector<int> localSA(textLen); // reused every iteration
                int idx;
                while ((idx = nextIdxGpu.fetch_add(1)) < numTexts)
                {
                    int rc = gpu_divsufsort_ws(texts[idx].data(), localSA.data(),
                                                static_cast<int>(textLen), workspaces[s], streams[s]);
                    if (rc != 0)
                    {
                        std::cerr << "gpu_divsufsort_ws failed on text " << idx << ": rc=" << rc << "\n";
                        allMatch = false;
                        continue;
                    }
                    if (localSA != saCpu[idx])
                    {
                        std::cerr << "Mismatch on text " << idx << "\n";
                        allMatch = false;
                    }
                }
            });
        }
        for (auto &th : pool) th.join();
    });

    for (auto &ws : workspaces) ws.release(); // also runs in destructor, explicit here for clarity
    for (auto &s : streams) cudaStreamDestroy(s);

    double totalMiB = (numTexts * textLen) / (1024.0 * 1024.0);
    std::cout << "-----------------------------------------\n";
    std::cout << "CPU single-threaded : " << cpuMs   << " ms  (" << (totalMiB / (cpuMs / 1000.0))   << " MiB/s)\n";
    std::cout << "CPU multi-threaded  : " << cpuMtMs << " ms  (" << (totalMiB / (cpuMtMs / 1000.0)) << " MiB/s)  [" << hwThreads << " threads]\n";
    std::cout << "GPU " << numStreams << " streams   : " << gpuMs   << " ms  (" << (totalMiB / (gpuMs / 1000.0))   << " MiB/s)\n";
    std::cout << "-----------------------------------------\n";
    std::cout << "Speedup GPU vs CPU(1)  : " << (cpuMs / gpuMs)   << "x\n";
    std::cout << "Speedup GPU vs CPU(MT) : " << (cpuMtMs / gpuMs) << "x\n";
    std::cout << "Per-stream GPU workspace: " << (gpuPerWorkspaceBytes / (1024.0 * 1024.0)) << " MiB"
              << "  (x" << numStreams << " concurrent = ~"
              << (gpuPerWorkspaceBytes * numStreams / (1024.0 * 1024.0)) << " MiB total, allocated ONCE not per-call)\n";
    std::cout << "Result match (GPU vs CPU): " << (allMatch ? "YES" : "NO") << "\n";

    return allMatch ? 0 : 2;
}
