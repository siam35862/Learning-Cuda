// gpu_divsufsort.h
//
// Shared declarations for gpu_divsufsort.cu, included by any .cu file
// that needs to call it (e.g. batch_bench_divsufsort.cu). Keeping this
// in one header means the GpuDivsufsortWorkspace layout can't drift
// out of sync between translation units.

#pragma once

#include <cuda_runtime.h>
#include <cstdint>
#include <cstdio>

// ---------------------------------------------------------------------
// Persistent per-stream workspace, for batch use (gpu_divsufsort_ws).
//
// Allocate ONCE per stream (sized for the largest n you'll ever pass
// on that stream), then pass the SAME workspace into every call on
// that stream. This eliminates all per-call cudaMalloc*/cudaFree*
// traffic -- the fastest option for a batch of many texts.
// ---------------------------------------------------------------------
struct GpuDivsufsortWorkspace
{
    int       *d_rank    = nullptr;
    int       *d_SA      = nullptr;
    uint64_t  *d_keys    = nullptr;
    int       *d_scratch = nullptr;
    unsigned char *d_text = nullptr;
    int       *h_maxRank_pinned = nullptr; // pinned host staging for convergence readback
    int        capacity  = 0;              // max n this workspace supports
    cudaStream_t ownerStream = 0;

    // Allocate for up to `maxN` elements, all work enqueued on `stream`.
    // Safe to call once, well before the batch loop starts.
    bool allocate(int maxN, cudaStream_t stream)
    {
        release(); // idempotent if already allocated
        ownerStream = stream;
        capacity = maxN;

        cudaError_t e;
        e = cudaMallocAsync(&d_rank,    static_cast<size_t>(maxN) * sizeof(int),      stream); if (e) return false;
        e = cudaMallocAsync(&d_SA,      static_cast<size_t>(maxN) * sizeof(int),      stream); if (e) return false;
        e = cudaMallocAsync(&d_keys,    static_cast<size_t>(maxN) * sizeof(uint64_t), stream); if (e) return false;
        e = cudaMallocAsync(&d_scratch, static_cast<size_t>(maxN) * sizeof(int),      stream); if (e) return false;
        e = cudaMallocAsync(&d_text,    static_cast<size_t>(maxN) * sizeof(unsigned char), stream); if (e) return false;
        e = cudaHostAlloc(&h_maxRank_pinned, sizeof(int), cudaHostAllocDefault); if (e) return false;

        return true;
    }

    void release()
    {
        if (d_rank)    { cudaFreeAsync(d_rank,    ownerStream); d_rank = nullptr; }
        if (d_SA)      { cudaFreeAsync(d_SA,      ownerStream); d_SA = nullptr; }
        if (d_keys)    { cudaFreeAsync(d_keys,    ownerStream); d_keys = nullptr; }
        if (d_scratch) { cudaFreeAsync(d_scratch, ownerStream); d_scratch = nullptr; }
        if (d_text)    { cudaFreeAsync(d_text,    ownerStream); d_text = nullptr; }
        if (h_maxRank_pinned) { cudaFreeHost(h_maxRank_pinned); h_maxRank_pinned = nullptr; }
        capacity = 0;
    }

    ~GpuDivsufsortWorkspace() { release(); }
};

// Drop-in replacement for the original gpu_divsufsort(): allocates its
// own scratch via cudaMallocAsync/cudaFreeAsync every call. Fine for a
// single one-off call; for a batch loop, prefer gpu_divsufsort_ws below.
int gpu_divsufsort(const unsigned char *T, int *SA, int n,
                    size_t *peakDeviceBytes = nullptr, cudaStream_t stream = 0);

// Batch-oriented entry point: zero device allocation per call. `ws`
// must already be allocate()'d with capacity >= n, on the same stream
// you pass here. Returns -3 if n > ws.capacity.
int gpu_divsufsort_ws(const unsigned char *T, int *SA, int n,
                       GpuDivsufsortWorkspace &ws, cudaStream_t stream);
