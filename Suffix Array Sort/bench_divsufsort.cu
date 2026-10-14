// bench_divsufsort.cu
//
// Benchmarks CPU libdivsufsort() against the GPU gpu_divsufsort()
// (prefix-doubling / Thrust) on the same input, and verifies both
// produce an identical suffix array.
//
// Requires:
//   - libdivsufsort installed (headers: divsufsort.h, lib: divsufsort / divsufsort64)
//   - gpu_divsufsort.cu (the GPU implementation) compiled into the same binary
//
// Build (Windows / MSVC, vcpkg-installed divsufsort):
//   nvcc -O3 -std=c++17 -Xcompiler "/Zc:preprocessor" --extended-lambda ^
//        bench_divsufsort.cu gpu_divsufsort.cu -o bench_divsufsort ^
//        -I"C:\vcpkg\installed\x64-windows\include" ^
//        -L"C:\vcpkg\installed\x64-windows\lib" -ldivsufsort
//
// Build (Linux, libdivsufsort installed via apt/build-from-source):
//   nvcc -O3 -std=c++17 --extended-lambda \
//        bench_divsufsort.cu gpu_divsufsort.cu -o bench_divsufsort \
//        -ldivsufsort
//
// Usage:
//   bench_divsufsort [n]            random text of length n (default 1<<20)
//   bench_divsufsort --file path    read text from a file instead

#include <divsufsort.h>   // saidx_t, divsufsort()

#include <cuda_runtime.h> // cudaStream_t

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cstdint>
#include <string>
#include <vector>
#include <random>
#include <chrono>
#include <fstream>
#include <iostream>

#ifdef _WIN32
  #include <windows.h>
  #include <psapi.h>
  #pragma comment(lib, "psapi.lib")
#else
  #include <sys/resource.h>
#endif

// Declared in gpu_divsufsort.cu — same signature/contract as divsufsort(),
// plus an optional peak-device-memory out-param and an optional stream
// (defaults to the legacy stream, so existing calls here are unaffected).
int gpu_divsufsort(const unsigned char *T, int *SA, int n,
                    size_t *peakDeviceBytes = nullptr, cudaStream_t stream = 0);

// ---------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------

static std::vector<unsigned char> makeRandomText(size_t n, int alphabetSize = 4)
{
    std::vector<unsigned char> text(n);
    std::mt19937_64 rng(12345); // fixed seed for reproducible benchmarks
    std::uniform_int_distribution<int> dist('a', 'a' + alphabetSize - 1);
    for (size_t i = 0; i < n; ++i)
        text[i] = static_cast<unsigned char>(dist(rng));
    return text;
}

static std::vector<unsigned char> readFile(const std::string &path)
{
    std::ifstream f(path, std::ios::binary);
    if (!f)
    {
        std::cerr << "Could not open file: " << path << "\n";
        std::exit(1);
    }
    return std::vector<unsigned char>(
        (std::istreambuf_iterator<char>(f)),
        std::istreambuf_iterator<char>());
}

// Compare two suffix arrays element-by-element; report first mismatch.
static bool compareSA(const std::vector<int> &a, const std::vector<int> &b, int n)
{
    for (int i = 0; i < n; ++i)
    {
        if (a[i] != b[i])
        {
            std::cerr << "Mismatch at index " << i
                       << ": cpu=" << a[i] << " gpu=" << b[i] << "\n";
            return false;
        }
    }
    return true;
}

template <typename Fn>
static double timeMs(Fn &&fn)
{
    auto t0 = std::chrono::high_resolution_clock::now();
    fn();
    auto t1 = std::chrono::high_resolution_clock::now();
    return std::chrono::duration<double, std::milli>(t1 - t0).count();
}

// Peak resident/working-set memory for the WHOLE PROCESS so far (a
// high-water mark that only grows) -- not just the last call. Read it
// right after each phase to see how much that phase pushed the peak up.
static size_t peakProcessMemoryBytes()
{
#ifdef _WIN32
    PROCESS_MEMORY_COUNTERS pmc;
    if (GetProcessMemoryInfo(GetCurrentProcess(), &pmc, sizeof(pmc)))
        return static_cast<size_t>(pmc.PeakWorkingSetSize);
    return 0;
#else
    struct rusage ru;
    getrusage(RUSAGE_SELF, &ru);
    return static_cast<size_t>(ru.ru_maxrss) * 1024; // ru_maxrss is KB on Linux
#endif
}

static double toMiB(size_t bytes) { return bytes / (1024.0 * 1024.0); }

// ---------------------------------------------------------------------
// main
// ---------------------------------------------------------------------

int main(int argc, char **argv)
{
    std::vector<unsigned char> text;
    int m;
    std::cin>>m;
    size_t n = 1u << m; // 1M chars default

    if (argc >= 2 && std::string(argv[1]) == "--file")
    {
        if (argc < 3)
        {
            std::cerr << "Usage: " << argv[0] << " --file <path>\n";
            return 1;
        }
        text = readFile(argv[2]);
        n = text.size();
    }
    else if (argc >= 2)
    {
        n = std::stoull(argv[1]);
        text = makeRandomText(n);
    }
    else
    {
        text = makeRandomText(n);
    }

    std::cout << "Input length: " << n << " bytes\n";

    if (n == 0)
    {
        std::cerr << "Empty input, nothing to benchmark.\n";
        return 1;
    }
    if (n > static_cast<size_t>(INT32_MAX))
    {
        std::cerr << "Input too large for 32-bit SA indices; use divsufsort64 / a widened GPU variant.\n";
        return 1;
    }

    int in = static_cast<int>(n);
    std::vector<int> saCpu(in);
    std::vector<int> saGpu(in);

    size_t memBeforeCpu = peakProcessMemoryBytes();

    // --- CPU: libdivsufsort ---
    double cpuMs = timeMs([&]() {
        int rc = divsufsort(text.data(), saCpu.data(), in);
        if (rc != 0)
        {
            std::cerr << "divsufsort() failed with code " << rc << "\n";
            std::exit(1);
        }
    });
    size_t memAfterCpu = peakProcessMemoryBytes();

    // --- GPU: prefix-doubling / Thrust ---
    // Note: this includes host<->device transfer time and all sort
    // rounds, i.e. wall-clock "black box" time, not just kernel time.
    size_t gpuPeakDeviceBytes = 0;
    double gpuMs = timeMs([&]() {
        int rc = gpu_divsufsort(text.data(), saGpu.data(), in, &gpuPeakDeviceBytes);
        if (rc != 0)
        {
            std::cerr << "gpu_divsufsort() failed with code " << rc << "\n";
            std::exit(1);
        }
    });
    size_t memAfterGpu = peakProcessMemoryBytes(); // host-side peak (includes CUDA driver's host allocations)

    bool match = compareSA(saCpu, saGpu, in);

    std::cout << "-----------------------------------------\n";
    std::cout << "CPU (divsufsort)   : " << cpuMs << " ms\n";
    std::cout << "GPU (gpu_divsufsort): " << gpuMs << " ms\n";
    std::cout << "Speedup (cpu/gpu)   : " << (cpuMs / gpuMs) << "x\n";
    std::cout << "Result match        : " << (match ? "YES" : "NO (see mismatch above)") << "\n";
    std::cout << "-----------------------------------------\n";
    std::cout << "Peak process memory (host RAM, whole-process high-water mark):\n";
    std::cout << "  before CPU sort  : " << toMiB(memBeforeCpu) << " MiB\n";
    std::cout << "  after CPU sort   : " << toMiB(memAfterCpu)  << " MiB  (+"
               << toMiB(memAfterCpu - memBeforeCpu) << " MiB)\n";
    std::cout << "  after GPU sort   : " << toMiB(memAfterGpu)  << " MiB  (+"
               << toMiB(memAfterGpu - memAfterCpu) << " MiB, mostly CUDA driver/host staging)\n";
    std::cout << "Peak GPU device memory (gpu_divsufsort's own buffers): "
               << toMiB(gpuPeakDeviceBytes) << " MiB\n";

    return match ? 0 : 2;
}
