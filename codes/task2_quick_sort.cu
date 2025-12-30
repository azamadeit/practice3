#include <iostream>                   // Подключение потока ввода-вывода
#include <cuda_runtime.h>             // Подключение CUDA Runtime API
#include <device_launch_parameters.h> // Доступ к blockIdx, threadIdx
#include <algorithm>                  // Для std::swap
#include <cstdlib>                    // Для rand()

__device__ void quickSortRange(int* a, int l, int r) // Быстрая сортировка диапазона на устройстве
{
    if (l >= r) return;                               // Выход при пустом или единичном диапазоне

    int pivot = a[(l + r) >> 1];                      // Выбор опорного элемента
    int i = l;                                        // Левый индекс
    int j = r;                                        // Правый индекс

    while (i <= j) {                                  // Пока индексы не пересеклись
        while (a[i] < pivot) i++;                     // Поиск элемента >= pivot слева
        while (a[j] > pivot) j--;                     // Поиск элемента <= pivot справа
        if (i <= j) {                                 // Если элементы найдены
            int t = a[i];                             // Сохраняем левый элемент
            a[i] = a[j];                              // Перемещаем правый элемент влево
            a[j] = t;                                 // Сохраняем левый элемент справа
            i++;                                      // Сдвигаем левый индекс
            j--;                                      // Сдвигаем правый индекс
        }
    }

    if (l < j) quickSortRange(a, l, j);               // Рекурсивная сортировка левой части
    if (i < r) quickSortRange(a, i, r);               // Рекурсивная сортировка правой части
}

__global__ void quickSortChunks(int* data, int n, int chunkSize) // Ядро сортировки сегментов
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;  // Глобальный индекс потока
    int left = tid * chunkSize;                        // Начало сегмента
    int right = left + chunkSize - 1;                 // Конец сегмента

    if (left >= n) return;                             // Проверка выхода за границы массива
    if (right >= n) right = n - 1;                    // Коррекция правой границы сегмента

    quickSortRange(data, left, right);                // Сортировка сегмента
}

__global__ void mergePass(int* src, int* dst, int width, int n) // Ядро слияния сегментов
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;  // Глобальный индекс потока
    int base = tid * (width << 1);                    // Начальный индекс пары сегментов

    if (base >= n) return;                            // Проверка выхода за пределы массива

    int l1 = base;                                    // Начало первого сегмента
    int r1 = min(base + width, n);                    // Конец первого сегмента
    int l2 = r1;                                      // Начало второго сегмента
    int r2 = min(base + (width << 1), n);             // Конец второго сегмента

    int i = l1;                                       // Индекс первого сегмента
    int j = l2;                                       // Индекс второго сегмента
    int k = base;                                     // Индекс записи результата

    while (i < r1 && j < r2) {                        // Пока есть элементы в обоих сегментах
        if (src[i] <= src[j]) dst[k++] = src[i++];    // Записываем меньший элемент слева
        else dst[k++] = src[j++];                     // Записываем меньший элемент справа
    }
    while (i < r1) dst[k++] = src[i++];               // Дописываем остаток первого сегмента
    while (j < r2) dst[k++] = src[j++];               // Дописываем остаток второго сегмента
}

int main()                                            // Точка входа программы
{
    const int n = 10000;                              // Размер массива
    const int chunkSize = 500;                        // Размер сегмента для одного потока
    size_t bytes = static_cast<size_t>(n) * sizeof(int); // Размер массива в байтах

    int* h_data = new int[n];                         // Выделение массива на CPU

    for (int i = 0; i < n; ++i)                       // Заполнение массива
        h_data[i] = rand() % 10000;                   // Случайные значения

    int* d_data = nullptr;                            // Указатель на данные на GPU
    int* d_tmp  = nullptr;                            // Указатель на временный буфер GPU

    cudaMalloc(&d_data, bytes);                       // Выделение памяти под данные на GPU
    cudaMalloc(&d_tmp, bytes);                        // Выделение временного буфера на GPU
    cudaMemcpy(d_data, h_data, bytes, cudaMemcpyHostToDevice); // Копирование CPU -> GPU

    int threads = 256;                                // Количество потоков в блоке
    int chunks  = (n + chunkSize - 1) / chunkSize;    // Количество сегментов
    int blocks  = (chunks + threads - 1) / threads;   // Количество блоков

    quickSortChunks<<<blocks, threads>>>(d_data, n, chunkSize); // Сортировка сегментов
    cudaDeviceSynchronize();                          // Ожидание завершения ядра

    int* src = d_data;                                // Текущий источник данных
    int* dst = d_tmp;                                 // Текущий приёмник данных

    for (int width = chunkSize; width < n; width <<= 1) { // Итеративное слияние сегментов
        int merges = (n + (2 * width) - 1) / (2 * width); // Количество операций слияния
        int mBlocks = (merges + threads - 1) / threads;   // Количество блоков

        mergePass<<<mBlocks, threads>>>(src, dst, width, n); // Запуск ядра слияния
        cudaDeviceSynchronize();                      // Ожидание завершения ядра

        std::swap(src, dst);                          // Смена буферов местами
    }

    cudaMemcpy(h_data, src, bytes, cudaMemcpyDeviceToHost); // Копирование GPU -> CPU

    std::cout << "Первые 10 элементов: ";             // Вывод первых элементов
    for (int i = 0; i < 10; ++i)                       // Цикл вывода
        std::cout << h_data[i] << " ";                // Печать элемента
    std::cout << "\n";                                // Перевод строки

    bool sorted = true;                               // Флаг корректности сортировки
    for (int i = 1; i < n; ++i) {                     // Проверка упорядоченности
        if (h_data[i - 1] > h_data[i]) {              // Нарушение порядка
            sorted = false;                           // Помечаем ошибку
            break;                                    // Прерываем проверку
        }
    }

    if (sorted)                                       // Если массив отсортирован
        std::cout << "Массив отсортирован корректно.\n"; // Сообщение об успехе
    else                                              // Иначе
        std::cout << "Ошибка сортировки.\n";          // Сообщение об ошибке

    cudaFree(d_data);                                 // Освобождение памяти GPU
    cudaFree(d_tmp);                                  // Освобождение временного буфера GPU
    delete[] h_data;                                  // Освобождение памяти CPU
    return 0;                                         // Завершение программы
}

