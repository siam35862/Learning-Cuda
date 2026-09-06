#include <iostream>
#include <cuda_runtime.h>

using namespace std;

// Macro to wrap CUDA runtime API calls
#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
    if (code != cudaSuccess) 
    {
        cerr << "GPU Error: " << cudaGetErrorString(code) 
             << " | File: " << file 
             << " | Line: " << line << endl;
        if (abort) exit(code);
    }
}

__global__ void matrixadd(const int *a, const int *b, int *c, int n, int m)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x; // Row index
    int j = blockIdx.y * blockDim.y + threadIdx.y; // Column index

    // Bounds check to prevent illegal memory access
    if (i < n && j < m)
    {
        int idx = i * m + j;
        c[idx] = a[idx] + b[idx];
    }
}

int main()
{
    const int n = 6, m = 5;
    size_t bytes = n * m * sizeof(int);

    int a[n][m], b[n][m], c[n][m];

    for (int i = 0; i < n; i++)
    {
        for (int j = 0; j < m; j++)
        {
            a[i][j] = (i + 1) * (j + 1);
            b[i][j] = (n - i) * (m - j);
        }
    }

    int *da = nullptr, *db = nullptr, *dc = nullptr;

    // 1. Check memory allocation errors
    gpuErrchk(cudaMalloc((void **)&da, bytes));
    gpuErrchk(cudaMalloc((void **)&db, bytes));
    gpuErrchk(cudaMalloc((void **)&dc, bytes));

    // 2. Check Host-to-Device transfer errors
    gpuErrchk(cudaMemcpy(da, a, bytes, cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(db, b, bytes, cudaMemcpyHostToDevice));

    dim3 gridsize(2, 1);
    dim3 blocksize(n, m);

    // Launch kernel
    matrixadd<<<gridsize, blocksize>>>(da, db, dc, n, m);

    // 3. Check for immediate kernel launch configuration errors (invalid grid/block setup)
    gpuErrchk(cudaGetLastError());

    // 4. Synchronize and check for asynchronous execution errors (e.g., out-of-bounds memory access)
    gpuErrchk(cudaDeviceSynchronize());

    // 5. Check Device-to-Host transfer errors
    gpuErrchk(cudaMemcpy(c, dc, bytes, cudaMemcpyDeviceToHost));

    // Print result
    for (int i = 0; i < n; i++)
    {
        for (int j = 0; j < m; j++)
        {
            cout << c[i][j] << " ";
        }
        cout << endl;
    }

    // 6. Check memory cleanup errors
    gpuErrchk(cudaFree(da));
    gpuErrchk(cudaFree(db));
    gpuErrchk(cudaFree(dc));

    return 0;
}