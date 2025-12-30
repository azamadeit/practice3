#include <iostream>                 // std::cout
#include <vector>                   // std::vector
#include <algorithm>                // std::sort
#include <cstdlib>                  // rand()
#include <chrono>                   // std::chrono
#include <cuda_runtime.h>           // CUDA Runtime API

// ---------------- GPU MERGE SORT (используем из Task 1) ----------------

__global__ void mergeKernel(int* data, int* temp, int width, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int start = idx * width * 2;

    if (start < n) {
        int mid = min(start + width, n);
        int end = min(start + width * 2, n);
        int i = start, j = mid, k = start;

        while (i < mid && j < end) {
            if (data[i] <= data[j]) temp[k++] = data[i++];
            else temp[k++] = data[j++];
        }
        while (i < mid) temp[k++] = data[i++];
        while (j < end) temp[k++] = data[j++];
    }
}

void gpuMergeSort(std::vector<int>& data) {
    int n = data.size();
    int* d_data;
    int* d_temp;
    size_t bytes = n * sizeof(int);

    cudaMalloc(&d_data, bytes);
    cudaMalloc(&d_temp, bytes);
    cudaMemcpy(d_data, data.data(), bytes, cudaMemcpyHostToDevice);

    int threads = 256;

    for (int width = 1; width < n; width *= 2) {
        int merges = (n + 2 * width - 1) / (2 * width);
        int blocks = (merges + threads - 1) / threads;
        mergeKernel<<<blocks, threads>>>(d_data, d_temp, width, n);
        cudaDeviceSynchronize();
        std::swap(d_data, d_temp);
    }

    cudaMemcpy(data.data(), d_data, bytes, cudaMemcpyDeviceToHost);
    cudaFree(d_data);
    cudaFree(d_temp);
}

// ---------------- BENCHMARK ----------------

void runTest(int n) {
    std::vector<int> dataCPU(n);
    std::vector<int> dataGPU(n);

    for (int i = 0; i < n; i++)
        dataCPU[i] = dataGPU[i] = rand();

    // ---------- CPU ----------
    auto cpuStart = std::chrono::high_resolution_clock::now();
    std::sort(dataCPU.begin(), dataCPU.end());
    auto cpuEnd = std::chrono::high_resolution_clock::now();
    double cpuTime = std::chrono::duration<double, std::milli>(cpuEnd - cpuStart).count();

    // ---------- GPU ----------
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    gpuMergeSort(dataGPU);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float gpuTime;
    cudaEventElapsedTime(&gpuTime, start, stop);

    // ---------- OUTPUT ----------
    std::cout << "Размер массива: " << n << "\n";
    std::cout << "CPU time: " << cpuTime << " ms\n";
    std::cout << "GPU time: " << gpuTime << " ms\n";
    std::cout << "Ускорение: " << cpuTime / gpuTime << "x\n\n";

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
}

int main() {
    runTest(1000);
    runTest(100000);
    runTest(1000000);
    return 0;
}
