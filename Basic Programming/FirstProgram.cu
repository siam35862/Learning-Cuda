#include <iostream>
#include <cuda_runtime.h>
using namespace std;

__global__ void copyArray(int *a, int *b, int n)
{
    int i = threadIdx.x;
    if (i < n)
    {
        b[i] = a[i];
    }
}
__global__ void swap_(int *a, int *b)
{
    int p;
    p = *a;
    *a = *b;
    *b = p;
}
int main()
{
    int h_a[] = {1, 2, 2, 3, 4, 5};
    int h_b[6];
    int *d_a, *d_b;
    cudaMalloc((void **)&d_a, 6 * sizeof(int));
    cudaMalloc((void **)&d_b, 6 * sizeof(int));

    cudaMemcpy(d_a, h_a, 6 * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, 6 * sizeof(int), cudaMemcpyHostToDevice);

    dim3 gridsize(1), blocksize(6);
    copyArray<<<gridsize, blocksize>>>(d_a, d_b, 6);

    cudaMemcpy(h_b, d_b, 6 * sizeof(int), cudaMemcpyDeviceToHost);

    for (int i = 0; i < 6; i++)
        cout << h_b[i] << " ";
    cout << endl;

    int h_x = 5, h_y = 10;
    int *d_x, *d_y;

    cudaMalloc((void **)&d_x, sizeof(int));
    cudaMalloc((void **)&d_y, sizeof(int));
    cudaMemcpy(d_x, &h_x, sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_y, &h_y, sizeof(int), cudaMemcpyHostToDevice);


    dim3 newblocksize(1);
    swap_<<<gridsize, newblocksize>>>(d_x, d_y);



    cudaMemcpy(&h_x, d_x, sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(&h_y, d_y, sizeof(int), cudaMemcpyDeviceToHost);

    cout << h_x << " " << h_y << endl;
}