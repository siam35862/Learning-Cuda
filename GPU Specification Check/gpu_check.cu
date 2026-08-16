#include <iostream>
#include <cuda_runtime.h>

void checkGPUSpecifications() {
    int deviceCount = 0;
    cudaError_t err = cudaGetDeviceCount(&deviceCount);

    if (err != cudaSuccess) {
        std::cerr << "CUDA Error: " << cudaGetErrorString(err) << std::endl;
        return;
    }

    if (deviceCount == 0) {
        std::cout << "No CUDA-capable GPU found on this system!" << std::endl;
        return;
    }

    std::cout << "==========================================" << std::endl;
    std::cout << "        NVIDIA GPU SPECIFICATIONS         " << std::endl;
    std::cout << "==========================================" << std::endl;

    for (int dev = 0; dev < deviceCount; ++dev) {
        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, dev);

        std::cout << "Device Number: " << dev << std::endl;
        std::cout << "Device Name: " << prop.name << std::endl;
        std::cout << "Compute Capability: " << prop.major << "." << prop.minor << std::endl;
        
        // VRAM Specs
        size_t totalMemGB = prop.totalGlobalMem / (1024 * 1024 * 1024);
        double totalMemMB = prop.totalGlobalMem / (1024.0 * 1024.0);
        std::cout << "Total Global Memory (VRAM): " << totalMemGB << " GB (" << totalMemMB << " MB)" << std::endl;

        // Execution Specs
        std::cout << "Multiprocessors (SM Count): " << prop.multiProcessorCount << std::endl;
        std::cout << "Memory Bus Width: " << prop.memoryBusWidth << " bits" << std::endl;
        
        // CUDA Thread & Block Configuration
        std::cout << "------------------------------------------" << std::endl;
        std::cout << "Max Threads Per Block: " << prop.maxThreadsPerBlock << std::endl;
        std::cout << "Max Threads Per MultiProcessor (SM): " << prop.maxThreadsPerMultiProcessor << std::endl;
        std::cout << "Warp Size: " << prop.warpSize << " threads" << std::endl;
        
        // Memory Limits
        std::cout << "Shared Memory Per Block: " << prop.sharedMemPerBlock / 1024.0 << " KB" << std::endl;
        std::cout << "Shared Memory Per Multiprocessor: " << prop.sharedMemPerMultiprocessor / 1024.0 << " KB" << std::endl;
        std::cout << "Total Constant Memory: " << prop.totalConstMem / 1024.0 << " KB" << std::endl;
        
        // Feature Support Check
        std::cout << "Concurrent Kernels Execution: " << (prop.concurrentKernels ? "Yes" : "No") << std::endl;
        std::cout << "Async Engine Count (Overlap Copy & Exec): " << prop.asyncEngineCount << std::endl;
        std::cout << "==========================================" << std::endl;
    }
}

int main() {
    checkGPUSpecifications();
    return 0;
}