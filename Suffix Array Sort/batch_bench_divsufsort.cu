// batch_bench_divsufsort.cu
//
// Benchmarks suffix-array construction over MANY INDEPENDENT texts:
//   1. CPU, single-threaded, one divsufsort() call after another.
//   2. CPU, multi-threaded, one thread per hardware core pulling from
//      a shared queue (fair "CPU parallel" baseline).
//   3. GPU, using a small pool of CUDA STREAMS so multiple texts'
//      suffix-array construction genuinely overlaps on the device.
//      Uses the SAME gpu_divsufsort() from gpu_divsufsort.cu as the
//      single-call benchmark (bench_divsufsort.cu) -- just passing it
//      a per-thread stream instead of the default.
//
// Default: 1000 texts x 1 MiB each (override via CLI args).
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
//     one dedicated stream through a shared work queue. Each thread's
//     call to gpu_divsufsort() blocks that thread on its own stream's
//     results, but other streams' work keeps running on the GPU in the
//     meantime.
//   - Host-side text/result buffers here are ordinary (pageable)
//     memory, not pinned -- simplest to get right first. Pinned memory
//     (cudaHostAlloc) would let the H2D/D2H copies themselves overlap
//     across streams too; ask if you want that version.
//   - Memory: with S concurrent streams and per-text footprint of
//     ~20 bytes/byte-of-text (see gpu_divsufsort.cu), total device
//     memory is roughly S * (24 * textBytes). Keep S modest relative
//     to your VRAM -- gpu_divsufsort()'s peakDeviceBytes out-param can
//     confirm the exact per-call figure if you want to size this.

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

// Declared in gpu_divsufsort.cu.
int gpu_divsufsort(const unsigned char *T, int *SA, int n,
                    size_t *peakDeviceBytes = nullptr, cudaStream_t stream = 0);

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

    std::vector<std::vector<int>> saCpu(numTexts, std::vector<int>(textLen));
    std::vector<std::vector<int>> saCpuMT(numTexts, std::vector<int>(textLen));
    std::vector<std::vector<int>> saGpu(numTexts, std::vector<int>(textLen));

    // --- 1) CPU, single-threaded ---
    double cpuMs = timeMs([&]() {
        for (int t = 0; t < numTexts; ++t)
        {
            int rc = divsufsort(texts[t].data(), saCpu[t].data(), static_cast<int>(textLen));
            if (rc != 0) { std::cerr << "divsufsort failed on text " << t << "\n"; std::exit(1); }
        }
    });

    // --- 2) CPU, multi-threaded (shared queue, one thread per core) ---
    std::atomic<int> nextIdxMT{0};
    unsigned hwThreads = std::max(1u, std::thread::hardware_concurrency());
    double cpuMtMs = timeMs([&]() {
        std::vector<std::thread> pool;
        for (unsigned w = 0; w < hwThreads; ++w)
        {
            pool.emplace_back([&]() {
                int idx;
                while ((idx = nextIdxMT.fetch_add(1)) < numTexts)
                    divsufsort(texts[idx].data(), saCpuMT[idx].data(), static_cast<int>(textLen));
            });
        }
        for (auto &th : pool) th.join();
    });

    // --- 3) GPU, concurrent streams (shared queue, one thread per stream) ---
    std::vector<cudaStream_t> streams(numStreams);
    for (auto &s : streams) cudaStreamCreate(&s);

    size_t gpuPeakPerCallBytes = 0; // same for every call at a given textLen
    std::atomic<int> nextIdxGpu{0};
    double gpuMs = timeMs([&]() {
        std::vector<std::thread> pool;
        for (int s = 0; s < numStreams; ++s)
        {
            pool.emplace_back([&, s]() {
                int idx;
                while ((idx = nextIdxGpu.fetch_add(1)) < numTexts)
                    gpu_divsufsort(texts[idx].data(), saGpu[idx].data(),
                                   static_cast<int>(textLen), &gpuPeakPerCallBytes, streams[s]);
            });
        }
        for (auto &th : pool) th.join();
    });

    for (auto &s : streams) cudaStreamDestroy(s);

    // --- Verify correctness against the CPU baseline ---
    bool allMatch = true;
    for (int t = 0; t < numTexts && allMatch; ++t)
        if (saCpu[t] != saGpu[t]) { allMatch = false; std::cerr << "Mismatch on text " << t << "\n"; }

    double totalMiB = (numTexts * textLen) / (1024.0 * 1024.0);
    std::cout << "-----------------------------------------\n";
    std::cout << "CPU single-threaded : " << cpuMs   << " ms  (" << (totalMiB / (cpuMs / 1000.0))   << " MiB/s)\n";
    std::cout << "CPU multi-threaded  : " << cpuMtMs << " ms  (" << (totalMiB / (cpuMtMs / 1000.0)) << " MiB/s)  [" << hwThreads << " threads]\n";
    std::cout << "GPU " << numStreams << " streams   : " << gpuMs   << " ms  (" << (totalMiB / (gpuMs / 1000.0))   << " MiB/s)\n";
    std::cout << "-----------------------------------------\n";
    std::cout << "Speedup GPU vs CPU(1)  : " << (cpuMs / gpuMs)   << "x\n";
    std::cout << "Speedup GPU vs CPU(MT) : " << (cpuMtMs / gpuMs) << "x\n";
    std::cout << "Per-call GPU device memory: " << (gpuPeakPerCallBytes / (1024.0 * 1024.0)) << " MiB"
              << "  (x" << numStreams << " concurrent = ~"
              << (gpuPeakPerCallBytes * numStreams / (1024.0 * 1024.0)) << " MiB total)\n";
    std::cout << "Result match (GPU vs CPU): " << (allMatch ? "YES" : "NO") << "\n";

    return allMatch ? 0 : 2;
}
