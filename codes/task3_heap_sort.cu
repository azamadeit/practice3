#include <iostream>
#include <cuda_runtime.h>
#include <cstdlib>

// Итеративное восстановление свойств кучи
__device__ void heapify(int* data, int n, int i) {
    while (true) {
        int largest = i;
        int l = 2 * i + 1;
        int r = 2 * i + 2;

        if (l < n && data[l] > data[largest]) largest = l;
        if (r < n && data[r] > data[largest]) largest = r;

        if (largest == i) break;

        int temp = data[i];
        data[i] = data[largest];
        data[largest] = temp;

        i = largest;
    }
}

// Построение кучи (корректно, один поток на GPU)
__global__ void buildHeap(int* data, int n) {
    if (blockIdx.x == 0 && threadIdx.x == 0) {
        for (int i = n / 2 - 1; i >= 0; --i) {
            heapify(data, n, i);
        }
    }
}

// Обмен корня с последним элементом
__global__ void swapRoot(int* data, int last) {
    if (blockIdx.x == 0 && threadIdx.x == 0) {
        int temp = data[0];
        data[0] = data[last];
        data[last] = temp;
    }
}

// Восстановление кучи от корня
__global__ void heapifyRoot(int* data, int n) {
    if (blockIdx.x == 0 && threadIdx.x == 0) {
        heapify(data, n, 0);
    }
}

int main() {
    const int n = 1024;
    int* h_data = new int[n];

    for (int i = 0; i < n; i++)
        h_data[i] = rand() % 1000;

    int* d_data;
    cudaMalloc(&d_data, n * sizeof(int));
    cudaMemcpy(d_data, h_data, n * sizeof(int), cudaMemcpyHostToDevice);

    // Построение кучи
    buildHeap<<<1, 1>>>(d_data, n);
    cudaDeviceSynchronize();

    // Heapsort
    for (int last = n - 1; last > 0; --last) {
        swapRoot<<<1, 1>>>(d_data, last);
        cudaDeviceSynchronize();
        heapifyRoot<<<1, 1>>>(d_data, last);
        cudaDeviceSynchronize();
    }

    cudaMemcpy(h_data, d_data, n * sizeof(int), cudaMemcpyDeviceToHost);

    // Вывод результата
    std::cout << "Первые 10 элементов: ";
    for (int i = 0; i < 10; i++)
        std::cout << h_data[i] << " ";
    std::cout << "\n";

    // Проверка
    bool sorted = true;
    for (int i = 1; i < n; i++) {
        if (h_data[i - 1] > h_data[i]) {
            sorted = false;
            break;
        }
    }

    if (sorted)
        std::cout << "Массив отсортирован корректно.\n";
    else
        std::cout << "Ошибка сортировки.\n";

    cudaFree(d_data);
    delete[] h_data;
    return 0;
}
