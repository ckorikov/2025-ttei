#pragma once

#include <algorithm>
#include <iostream>
#include <stdexcept>
#include <vector>

// Оператор сложения матриц
std::vector<float> operator+(const std::vector<float> &A,
                             const std::vector<float> &B)
{
    if (A.size() != B.size())
    {
        throw std::invalid_argument(
            "Matrix dimensions must agree for addition");
    }

    std::vector<float> result(A.size());
    for (size_t i = 0; i < A.size(); ++i)
    {
        result[i] = A[i] + B[i];
    }

    return result;
}

// Оператор вычитания матриц
std::vector<float> operator-(const std::vector<float> &A,
                             const std::vector<float> &B)
{
    if (A.size() != B.size())
    {
        throw std::invalid_argument(
            "Matrix dimensions must agree for subtraction");
    }

    std::vector<float> result(A.size());
    for (size_t i = 0; i < A.size(); ++i)
    {
        result[i] = A[i] - B[i];
    }

    return result;
}

class MatrixMultiplier
{
  public:
    // Классический метод умножения матриц
    std::vector<float> classic_multiply(const std::vector<float> &A,
                                        const std::vector<float> &B,
                                        size_t rowsA, size_t colsA,
                                        size_t colsB)
    {
        if (A.size() != rowsA * colsA || B.size() != colsA * colsB)
        {
            throw std::invalid_argument(
                "Invalid matrix dimensions for multiplication");
        }

        std::vector<float> result(rowsA * colsB, 0.0f);

        for (unsigned int i = 0; i < rowsA; ++i)
        {
            for (unsigned int j = 0; j < colsB; ++j)
            {
                for (unsigned int k = 0; k < colsA; ++k)
                {
                    result[i * colsB + j] +=
                        A[i * colsA + k] * B[k * colsB + j];
                }
            }
        }

        return result;
    }

    // Метод Штрассена (работает только для квадратных матриц, размеры кратны 2)
    std::vector<float> strassen_multiply(const std::vector<float> &A,
                                         const std::vector<float> &B, int n)
    {
        if (n == 2)
        {
            return strassen_multiply_2x2(A, B);
        }

        int mid = n / 2;
        int size = mid * mid;

        std::vector<float> A11(size), A12(size), A21(size), A22(size);
        std::vector<float> B11(size), B12(size), B21(size), B22(size);

        // Разделение матриц
        for (unsigned int i = 0; i < mid; ++i)
        {
            for (unsigned int j = 0; j < mid; ++j)
            {
                int index = i * mid + j;
                A11[index] = A[i * n + j];
                A12[index] = A[i * n + j + mid];
                A21[index] = A[(i + mid) * n + j];
                A22[index] = A[(i + mid) * n + j + mid];

                B11[index] = B[i * n + j];
                B12[index] = B[i * n + j + mid];
                B21[index] = B[(i + mid) * n + j];
                B22[index] = B[(i + mid) * n + j + mid];
            }
        }

        auto M1 = strassen_multiply(A11, B12 - B22, mid);
        auto M2 = strassen_multiply(A11 + A12, B22, mid);
        auto M3 = strassen_multiply(A21 + A22, B11, mid);
        auto M4 = strassen_multiply(A22, B21 - B11, mid);
        auto M5 = strassen_multiply(A11 + A22, B11 + B22, mid);
        auto M6 = strassen_multiply(A12 - A22, B21 + B22, mid);
        auto M7 = strassen_multiply(A11 - A21, B11 + B12, mid);

        std::vector<float> result(n * n, 0);

        // Сборка результата
        for (size_t i = 0; i < mid; ++i)
        {
            for (size_t j = 0; j < mid; ++j)
            {
                int index = i * mid + j;
                result[i * n + j] =
                    M5[index] + M4[index] - M2[index] + M6[index];
                result[i * n + j + mid] = M1[index] + M2[index];
                result[(i + mid) * n + j] = M3[index] + M4[index];
                result[(i + mid) * n + j + mid] =
                    M5[index] + M1[index] - M3[index] - M7[index];
            }
        }

        return result;
    }

    // Метод Винограда
    std::vector<float> winograd_multiply(const std::vector<float> &A,
                                         const std::vector<float> &B, int rowsA,
                                         int colsA, int colsB)
    {
        if (A.size() != rowsA * colsA || B.size() != colsA * colsB)
        {
            throw std::invalid_argument(
                "Invalid matrix dimensions for multiplication");
        }

        std::vector<float> rowFactor(rowsA, 0.0f);
        std::vector<float> colFactor(colsB, 0.0f);
        std::vector<float> result(rowsA * colsB, 0.0f);

        // Предобработка: расчет rowFactor и colFactor
        for (size_t i = 0; i < rowsA; ++i)
        {
            for (size_t k = 0; k < colsA / 2; ++k)
            {
                rowFactor[i] += A[i * colsA + 2 * k] * A[i * colsA + 2 * k + 1];
            }
        }

        for (size_t j = 0; j < colsB; ++j)
        {
            for (size_t k = 0; k < colsA / 2; ++k)
            {
                colFactor[j] +=
                    B[2 * k * colsB + j] * B[(2 * k + 1) * colsB + j];
            }
        }

        // Основной цикл умножения
        for (size_t i = 0; i < rowsA; ++i)
        {
            for (size_t j = 0; j < colsB; ++j)
            {
                result[i * colsB + j] = -rowFactor[i] - colFactor[j];
                for (size_t k = 0; k < colsA / 2; ++k)
                {
                    result[i * colsB + j] +=
                        (A[i * colsA + 2 * k] + B[(2 * k + 1) * colsB + j]) *
                        (A[i * colsA + 2 * k + 1] + B[2 * k * colsB + j]);
                }
            }
        }

        // Для нечетных матриц
        if (colsA % 2)
        {
            for (size_t i = 0; i < rowsA; ++i)
            {
                for (size_t j = 0; j < colsB; ++j)
                {
                    result[i * colsB + j] +=
                        A[i * colsA + colsA - 1] * B[(colsA - 1) * colsB + j];
                }
            }
        }

        return result;
    }

    // Метод автоматического выбора подходящего алгоритма
    std::vector<float> auto_multiply(const std::vector<float> &A,
                                     const std::vector<float> &B, int rowsA,
                                     int colsA, int colsB)
    {
        if (A.size() != rowsA * colsA || B.size() != colsA * colsB)
        {
            throw std::invalid_argument(
                "Invalid matrix dimensions for multiplication");
        }

        // Если матрицы квадратные и кратны 2, используем Штрассена
        if (rowsA == colsA && colsA == colsB && (rowsA & (rowsA - 1)) == 0)
        {
            return strassen_multiply(A, B, rowsA);
        }

        // Если четные размеры, используем Винограда
        if (colsA % 2 == 0)
        {
            return winograd_multiply(A, B, rowsA, colsA, colsB);
        }

        // По умолчанию классический метод
        return classic_multiply(A, B, rowsA, colsA, colsB);
    }

  private:
    // Вспомогательный метод для умножения 2x2 матриц методом Штрассена
    std::vector<float> strassen_multiply_2x2(const std::vector<float> &A,
                                             const std::vector<float> &B)
    {
        float a = A[0], b = A[1], c = A[2], d = A[3];
        float e = B[0], f = B[1], g = B[2], h = B[3];

        float p1 = a * (f - h);
        float p2 = (a + b) * h;
        float p3 = (c + d) * e;
        float p4 = d * (g - e);
        float p5 = (a + d) * (e + h);
        float p6 = (b - d) * (g + h);
        float p7 = (a - c) * (e + f);

        return {p5 + p4 - p2 + p6, p1 + p2, p3 + p4, p1 + p5 - p3 - p7};
    }
};
