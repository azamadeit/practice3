#include <iostream>             // std::cout
#include <cuda_runtime.h>       // CUDA Runtime API
#include <algorithm>            // std::swap
#include <cstdlib>              // rand()

// Ядро: один поток выполняет слияние двух отсортированных отрезков длины width
__global__ void mergePass(int* src, int* dst, int width, int n)
{
    int t = blockIdx.x * blockDim.x + threadIdx.x;      // Номер потока
    int base = t * (width << 1);                        // Начальный индекс пары отрезков

    if (base >= n) return;                               // Если пара начинается за массивом — выходим

    int leftBeg  = base;                                 // Начало левого отрезка
    int leftEnd  = min(base + width, n);                 // Конец левого отрезка
    int rightBeg = leftEnd;                              // Начало правого отрезка
    int rightEnd = min(base + (width << 1), n);          // Конец правого отрезка

    int i = leftBeg;                                     // Индекс по левому отрезку
    int j = rightBeg;                                    // Индекс по правому отрезку
    int out = base;                                      // Позиция записи в dst

    while (i < leftEnd && j < rightEnd) {                // Пока есть элементы в обоих отрезках
        int a = src[i];                                  // Текущий элемент слева
        int b = src[j];                                  // Текущий элемент справа
        if (a <= b) {                                    // Выбираем меньший
            dst[out++] = a;                              // Записываем слева
            i++;                                         // Сдвигаем левый указатель
        } else {
            dst[out++] = b;                              // Записываем справа
            j++;                                         // Сдвигаем правый указатель
        }
    }

    while (i < leftEnd) dst[out++] = src[i++];           // Дописываем остаток левого отрезка
    while (j < rightEnd) dst[out++] = src[j++];          // Дописываем остаток правого отрезка
}

// Функция: сортировка слиянием на GPU (итеративные проходы по width)
void gpuMergeSort(int* hostData, int n)
{
    int* devA = nullptr;                                 // Буфер A на GPU
    int* devB = nullptr;                                 // Буфер B на GPU
    size_t bytes = static_cast<size_t>(n) * sizeof(int);  // Размер массива в байтах

    cudaMalloc(&devA, bytes);                             // Выделяем devA
    cudaMalloc(&devB, bytes);                             // Выделяем devB
    cudaMemcpy(devA, hostData, bytes, cudaMemcpyHostToDevice); // Копируем вход CPU -> devA

    const int threads = 256;                              // Потоков в блоке

    for (int width = 1; width < n; width <<= 1) {         // width: 1,2,4,8,...
        int merges = (n + (2 * width) - 1) / (2 * width);  // Число операций слияния
        int blocks = (merges + threads - 1) / threads;     // Число блоков

        mergePass<<<blocks, threads>>>(devA, devB, width, n); // Слияние devA -> devB
        cudaDeviceSynchronize();                           // Ожидаем завершения

        std::swap(devA, devB);                             // Следующий проход читает из нового devA
    }

    cudaMemcpy(hostData, devA, bytes, cudaMemcpyDeviceToHost); // Копируем результат devA -> CPU

    cudaFree(devA);                                       // Освобождаем devA
    cudaFree(devB);                                       // Освобождаем devB
}

int main()
{
    const int n = 1024;                                   // Размер массива
    int* data = new int[n];                               // Массив на CPU

    for (int i = 0; i < n; ++i) data[i] = rand() % 1000;   // Заполнение случайными значениями

    gpuMergeSort(data, n);                                // Запуск сортировки на GPU

    std::cout << "Первые 10 элементов: ";                 // Заголовок вывода
    for (int i = 0; i < 10; ++i) std::cout << data[i] << " "; // Печатаем 10 элементов
    std::cout << "\n";                                    // Перевод строки

    delete[] data;                                        // Освобождаем массив на CPU
    return 0;                                             // Код завершения
}
