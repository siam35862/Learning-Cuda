#include <iostream>
#include <vector>
#include <thread>
#include <mutex>
#include <chrono>
#include <cmath>
#include <atomic>
#include <iomanip>
#include <algorithm>
#include <cuda_runtime.h>

#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            std::cerr << "CUDA Error: " << cudaGetErrorString(err) \
                      << " at line " << __LINE__ << std::endl; \
            exit(EXIT_FAILURE); \
        } \
    } while (0)

// Lock-free 64-bit Atomic Work Dispatcher (O(1) Memory)
class WorkDispatcher {
private:
    std::atomic<long long> current_index_{0};
    long long total_chunks_;

public:
    WorkDispatcher(long long total) : total_chunks_(total) {}

    // Atomically claim a contiguous block of indices
    long long get_batch(long long max_batch_size, long long& start_idx) {
        start_idx = current_index_.fetch_add(max_batch_size, std::memory_order_relaxed);
        if (start_idx >= total_chunks_) {
            return 0;
        }
        return std::min(max_batch_size, total_chunks_ - start_idx);
    }
};

// GPU Kernel
__global__ void compute_kernel(const float* __restrict__ d_in, float* __restrict__ d_out, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float val = d_in[idx];
        #pragma unroll
        for (int i = 0; i < 100; ++i) {
            val = sinf(val) * cosf(val) + sqrtf(fabsf(val) + 1.0f);
        }
        d_out[idx] = val;
    }
}

// Host CPU computation function
inline float compute_cpu(float val) {
    for (int i = 0; i < 100; ++i) {
        val = std::sin(val) * std::cos(val) + std::sqrt(std::abs(val) + 1.0f);
    }
    return val;
}

// CPU Worker Routine (Streams computation without storing massive arrays in RAM)
void cpu_worker_thread(WorkDispatcher& dispatcher,
                       std::atomic<long long>& cpu_processed_count,
                       size_t cpu_batch_size,
                       std::atomic<long long>& cpu_active_time_us) {
    auto start = std::chrono::high_resolution_clock::now();

    long long start_idx = 0;
    long long count = 0;

    while ((count = dispatcher.get_batch(cpu_batch_size, start_idx)) > 0) {
        for (long long i = 0; i < count; ++i) {
            float val = static_cast<float>(start_idx + i + 1);
            volatile float res = compute_cpu(val); // volatile prevents dead-code elimination
            (void)res;
        }
        cpu_processed_count.fetch_add(count, std::memory_order_relaxed);
    }

    auto end = std::chrono::high_resolution_clock::now();
    long long duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();

    long long current_max = cpu_active_time_us.load();
    while (duration > current_max && !cpu_active_time_us.compare_exchange_weak(current_max, duration));
}

// GPU Orchestrator Routine (Streaming pinned host buffers)
void gpu_worker_thread(WorkDispatcher& dispatcher,
                       std::atomic<long long>& gpu_processed_count,
                       size_t gpu_batch_size,
                       double& gpu_time_ms) {
    auto start = std::chrono::high_resolution_clock::now();

    constexpr int NUM_STREAMS = 2;
    cudaStream_t streams[NUM_STREAMS];
    float *h_in[NUM_STREAMS], *h_out[NUM_STREAMS];
    float *d_in[NUM_STREAMS], *d_out[NUM_STREAMS];
    long long stream_active_count[NUM_STREAMS] = {0};

    for (int i = 0; i < NUM_STREAMS; ++i) {
        CUDA_CHECK(cudaStreamCreate(&streams[i]));
        CUDA_CHECK(cudaMallocHost((void**)&h_in[i], gpu_batch_size * sizeof(float)));
        CUDA_CHECK(cudaMallocHost((void**)&h_out[i], gpu_batch_size * sizeof(float)));
        CUDA_CHECK(cudaMalloc((void**)&d_in[i], gpu_batch_size * sizeof(float)));
        CUDA_CHECK(cudaMalloc((void**)&d_out[i], gpu_batch_size * sizeof(float)));
    }

    int stream_idx = 0;
    long long start_idx = 0;

    while (true) {
        // 1. Synchronize prior operations on this stream
        if (stream_active_count[stream_idx] > 0) {
            CUDA_CHECK(cudaStreamSynchronize(streams[stream_idx]));
            gpu_processed_count.fetch_add(stream_active_count[stream_idx], std::memory_order_relaxed);
            stream_active_count[stream_idx] = 0;
        }

        // 2. Claim next macro-batch
        long long count = dispatcher.get_batch(gpu_batch_size, start_idx);
        if (count <= 0) break;

        // 3. Generate input payload on-the-fly into pinned buffer
        for (long long k = 0; k < count; ++k) {
            h_in[stream_idx][k] = static_cast<float>(start_idx + k + 1);
        }
        stream_active_count[stream_idx] = count;

        // 4. Async transfers & kernel launch
        CUDA_CHECK(cudaMemcpyAsync(d_in[stream_idx], h_in[stream_idx], count * sizeof(float),
                                   cudaMemcpyHostToDevice, streams[stream_idx]));

        int threads_per_block = 256;
        int blocks_per_grid = static_cast<int>((count + threads_per_block - 1) / threads_per_block);
        compute_kernel<<<blocks_per_grid, threads_per_block, 0, streams[stream_idx]>>>(
            d_in[stream_idx], d_out[stream_idx], static_cast<int>(count)
        );

        CUDA_CHECK(cudaMemcpyAsync(h_out[stream_idx], d_out[stream_idx], count * sizeof(float),
                                   cudaMemcpyDeviceToHost, streams[stream_idx]));

        stream_idx = (stream_idx + 1) % NUM_STREAMS;
    }

    // Drain active streams
    for (int i = 0; i < NUM_STREAMS; ++i) {
        if (stream_active_count[i] > 0) {
            CUDA_CHECK(cudaStreamSynchronize(streams[i]));
            gpu_processed_count.fetch_add(stream_active_count[i], std::memory_order_relaxed);
        }
        CUDA_CHECK(cudaFreeHost(h_in[i]));
        CUDA_CHECK(cudaFreeHost(h_out[i]));
        CUDA_CHECK(cudaFree(d_in[i]));
        CUDA_CHECK(cudaFree(d_out[i]));
        CUDA_CHECK(cudaStreamDestroy(streams[i]));
    }

    auto end = std::chrono::high_resolution_clock::now();
    gpu_time_ms = std::chrono::duration<double, std::milli>(end - start).count();
}

int main() {
    std::cout << "Enter total chunks (e.g., 10000000000 for 10^10): ";
    long long TOTAL_CHUNKS = 0;
    std::cin >> TOTAL_CHUNKS;

    constexpr size_t CPU_BATCH_SIZE = 4096;          // Scaled up for high throughput
    constexpr size_t GPU_BATCH_SIZE = 1048576;       // 1M chunks per batch (2^20)

    WorkDispatcher dispatcher(TOTAL_CHUNKS);

    std::atomic<long long> cpu_processed_count{0};
    std::atomic<long long> gpu_processed_count{0};
    std::atomic<long long> cpu_max_time_us{0};
    double gpu_time_ms = 0.0;

    unsigned int num_cpu_threads = std::thread::hardware_concurrency();
    if (num_cpu_threads > 1) num_cpu_threads -= 1;

    std::cout << "\nStarting Dynamic Heterogeneous Execution..." << std::endl;
    std::cout << "Total Work Chunks: " << TOTAL_CHUNKS << std::endl;
    std::cout << "CPU Worker Threads: " << num_cpu_threads << " (Batch Size: " << CPU_BATCH_SIZE << ")" << std::endl;
    std::cout << "GPU Orchestrator Active (Batch Size: " << GPU_BATCH_SIZE << ")\n" << std::endl;

    auto total_start_time = std::chrono::high_resolution_clock::now();

    std::vector<std::thread> workers;
    for (unsigned int i = 0; i < num_cpu_threads; ++i) {
        workers.emplace_back(cpu_worker_thread, std::ref(dispatcher),
                             std::ref(cpu_processed_count), CPU_BATCH_SIZE,
                             std::ref(cpu_max_time_us));
    }

    std::thread gpu_thread(gpu_worker_thread, std::ref(dispatcher),
                           std::ref(gpu_processed_count), GPU_BATCH_SIZE,
                           std::ref(gpu_time_ms));

    for (auto& w : workers) {
        w.join();
    }
    gpu_thread.join();

    auto total_end_time = std::chrono::high_resolution_clock::now();
    double total_time_ms = std::chrono::duration<double, std::milli>(total_end_time - total_start_time).count();
    double cpu_time_ms = static_cast<double>(cpu_max_time_us.load()) / 1000.0;

    double cpu_throughput = (cpu_processed_count.load() > 0 && cpu_time_ms > 0)
                                ? (cpu_processed_count.load() / (cpu_time_ms / 1000.0)) : 0.0;
    double gpu_throughput = (gpu_processed_count.load() > 0 && gpu_time_ms > 0)
                                ? (gpu_processed_count.load() / (gpu_time_ms / 1000.0)) : 0.0;

    double speed_factor = (cpu_throughput > 0) ? (gpu_throughput / cpu_throughput) : 0.0;

    std::cout << std::fixed << std::setprecision(2);
    std::cout << "==========================================" << std::endl;
    std::cout << "            EXECUTION METRICS             " << std::endl;
    std::cout << "==========================================" << std::endl;
    std::cout << "CPU Active Time      : " << cpu_time_ms << " ms" << std::endl;
    std::cout << "GPU Active Time      : " << gpu_time_ms << " ms" << std::endl;
    std::cout << "Total Wall Time      : " << total_time_ms << " ms (" << total_time_ms / 1000.0 << " s)" << std::endl;
    std::cout << "------------------------------------------" << std::endl;
    std::cout << "CPU Chunks Processed : " << cpu_processed_count.load() << std::endl;
    std::cout << "GPU Chunks Processed : " << gpu_processed_count.load() << std::endl;
    std::cout << "CPU Throughput       : " << cpu_throughput << " chunks/sec" << std::endl;
    std::cout << "GPU Throughput       : " << gpu_throughput << " chunks/sec" << std::endl;
    std::cout << "------------------------------------------" << std::endl;
    std::cout << "GPU/CPU Speed Factor : " << speed_factor << "x" << std::endl;
    std::cout << "==========================================" << std::endl;

    return 0;
}