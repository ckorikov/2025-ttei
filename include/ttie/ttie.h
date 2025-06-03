#ifndef TTIE_H
#define TTIE_H

#include <cassert>
#include <algorithm>
#include <iostream>
#include <random>
#include <stdexcept>
#include <sstream>
#include <string>
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


namespace ttie
{
template <typename T>
static std::string vector_to_string(const std::vector<T> &vec, size_t limit = 5)
{
    std::stringstream ss;
    ss << "[";
    size_t preview_size = std::min(limit, vec.size());
    for (size_t i = 0; i < preview_size; ++i)
    {
        ss << vec[i];
        if (i < preview_size - 1)
            ss << ", ";
    }
    if (vec.size() > preview_size)
        ss << ", ...";
    ss << "]";
    return ss.str();
}

struct Tensor
{
    std::vector<size_t> shape;

    std::vector<float> data;
    std::vector<float> grad;

    bool validate_shape() const
    {
        if (shape.empty())
        {
            return false;
        }

        for (size_t dim : shape)
        {
            if (dim == 0)
            {
                return false;
            }
        }

        return true;
    }
    size_t size() const
    {
        if (!validate_shape())
        {
            throw std::invalid_argument("Invalid tensor shape");
        }
        size_t total = 1;
        for (size_t dim : shape)
        {
            total *= dim;
        }
        return total;
    }

    void resize() { data.resize(size()); }

    void resize_grad() { grad.resize(size()); }

    void zero_grad() { std::fill(grad.begin(), grad.end(), 0.0f); }

    friend std::ostream &operator<<(std::ostream &os, const Tensor &t)
    {
        os << "Tensor@" << &t;

        if (t.shape.empty())
        {
            os << "(not initialized)";
            return os;
        }

        os << "(shape=" << vector_to_string(t.shape);

        if (!t.data.empty())
        {
            os << ", data=" << vector_to_string(t.data);
        }
        else
        {
            os << ", data=[no data]";
        }

        if (!t.grad.empty())
        {
            os << ", grad=" << vector_to_string(t.grad);
        }

        os << ")";
        return os;
    }
};

struct Layer
{
    virtual void forward(const Tensor &input, Tensor &output) = 0;
    virtual void backward(const Tensor &grad_output, Tensor &grad_input) = 0;
    virtual std::string to_string() const = 0;
    virtual std::vector<Tensor *> parameters() = 0;
    virtual ~Layer() {}
};

struct Linear : Layer
{
    Tensor weight;
    Tensor bias;

    Linear(size_t in_features, size_t out_features)
    {
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_real_distribution<float> dis(-0.1f, 0.1f);

        weight.shape = {in_features, out_features};
        weight.resize();
        for (size_t i = 0; i < in_features * out_features; ++i)
        {
            weight.data[i] = dis(gen);
        }

        bias.shape = {out_features};
        bias.resize();
        for (size_t i = 0; i < out_features; ++i)
        {
            bias.data[i] = dis(gen);
        }
    }

    std::vector<Tensor *> parameters() override { return {&weight, &bias}; }

    void forward(const Tensor &input, Tensor &output) override
    {
        size_t in_features = weight.shape[0];
        size_t out_features = weight.shape[1];
        output.shape = {input.shape[0], out_features};
        output.resize();

        for (size_t i = 0; i < input.shape[0]; ++i)
        {
            for (size_t j = 0; j < out_features; ++j)
            {
                output.data[i * out_features + j] = bias.data[j];
                for (size_t k = 0; k < in_features; ++k)
                {
                    output.data[i * out_features + j] +=
                        input.data[i * in_features + k] *
                        weight.data[k * out_features + j];
                }
            }
        }
    }

    void forward_with_strassen(const Tensor &input, Tensor &output)
    {
        size_t in_features = weight.shape[0];
        size_t out_features = weight.shape[1];
        output.shape = {input.shape[0], out_features};
        output.resize();

        // Create matrix multiplier instance
        MatrixMultiplier matmul;

        // Check if we can use Strassen's method (square matrices with size
        // power of 2)
        bool can_use_strassen = (input.shape[0] == in_features) &&
                                (in_features == out_features) &&
                                ((input.shape[0] & (input.shape[0] - 1)) == 0);

        for (size_t i = 0; i < input.shape[0]; ++i)
        {
            // Get current input sample (row vector)
            std::vector<float> input_row(in_features);
            for (size_t k = 0; k < in_features; ++k)
            {
                input_row[k] = input.data[i * in_features + k];
            }

            // Multiply using appropriate method
            std::vector<float> multiplied;

            // For Strassen, we need to reshape the input to square matrix
            std::vector<float> input_square(input_row.begin(), input_row.end());
            input_square.resize(in_features * in_features,
                                0.0f); // Pad if needed

            multiplied = matmul.strassen_multiply(input_square, weight.data,
                                                  in_features);

            // Extract the first row of the result (since we multiplied by a row
            // vector)
            for (size_t j = 0; j < out_features; ++j)
            {
                output.data[i * out_features + j] =
                    multiplied[j] + bias.data[j];
            }
        }
    }

    void backward(const Tensor &output, Tensor &input) override
    {
        size_t in_features = weight.shape[0];
        size_t out_features = weight.shape[1];
        size_t batch_size = output.shape[0];

        input.resize_grad();
        weight.resize_grad();
        bias.resize_grad();

        for (size_t i = 0; i < batch_size; ++i)
        {
            for (size_t j = 0; j < in_features; ++j)
            {
                input.grad[i * in_features + j] = 0;
                for (size_t k = 0; k < out_features; ++k)
                {
                    input.grad[i * in_features + j] +=
                        output.grad[i * out_features + k] *
                        weight.data[j * out_features + k];
                    weight.grad[j * out_features + k] +=
                        output.grad[i * out_features + k] *
                        input.data[i * in_features + j];
                }
            }
        }

        for (size_t i = 0; i < batch_size; ++i)
        {
            for (size_t k = 0; k < out_features; ++k)
            {
                bias.grad[k] += output.grad[i * out_features + k];
            }
        }
    }

    std::string to_string() const override
    {
        std::stringstream ss;
        ss << "Linear(in_features=" << weight.shape[0]
           << ", out_features=" << weight.shape[1] << ")";
        return ss.str();
    }
};

struct ReLU : Layer
{
    std::vector<Tensor *> parameters() override { return {}; }

    void forward(const Tensor &input, Tensor &output) override
    {
        output.shape = input.shape;
        output.resize();
        for (size_t i = 0; i < input.data.size(); ++i)
        {
            output.data[i] = std::max(0.0f, input.data[i]);
        }
    }

    void backward(const Tensor &output, Tensor &input) override
    {
        input.resize_grad();
        for (size_t i = 0; i < output.data.size(); ++i)
        {
            input.grad[i] = (output.data[i] > 0) ? output.grad[i] : 0;
        }
    }

    std::string to_string() const override { return "ReLU()"; }
};

struct Sigmoid : Layer
{
    std::vector<Tensor *> parameters() override { return {}; }

    void forward(const Tensor &input, Tensor &output) override
    {
        output.shape = input.shape;
        output.resize();
        for (size_t i = 0; i < input.data.size(); ++i)
        {
            output.data[i] = 1.0f / (1.0f + std::exp(-input.data[i]));
        }
    }

    void backward(const Tensor &output, Tensor &input) override
    {
        input.resize_grad();
        for (size_t i = 0; i < output.data.size(); ++i)
        {
            float s = output.data[i];
            input.grad[i] = output.grad[i] * s *
                            (1 - s); // Производная Sigmoid: σ(x)*(1-σ(x))
        }
    }

    std::string to_string() const override { return "Sigmoid()"; }
};

struct Tanh : Layer
{
    std::vector<Tensor *> parameters() override { return {}; }

    void forward(const Tensor &input, Tensor &output) override
    {
        output.shape = input.shape;
        output.resize();
        for (size_t i = 0; i < input.data.size(); ++i)
        {
            output.data[i] = std::tanh(input.data[i]);
        }
    }

    void backward(const Tensor &output, Tensor &input) override
    {
        input.resize_grad();
        for (size_t i = 0; i < output.data.size(); ++i)
        {
            float t = output.data[i];
            input.grad[i] =
                output.grad[i] * (1 - t * t); // Производная Tanh: 1 - tanh²(x)
        }
    }

    std::string to_string() const override { return "Tanh()"; }
};

struct Model
{
    std::vector<Layer *> layers;
    std::vector<Tensor> activations;

    void add_layer(Layer *layer) { layers.push_back(layer); }

    void forward(const Tensor &input, Tensor &output)
    {
        activations.resize(layers.size() - 1);

        const Tensor *current = &input;
        for (size_t i = 0; i < layers.size(); ++i)
        {
            Tensor *next = (i == layers.size() - 1) ? &output : &activations[i];
            layers[i]->forward(*current, *next);
            current = next;
        }
    }

    void backward(const Tensor &output, Tensor &input)
    {
        if (activations.size() != layers.size() - 1)
        {
            throw std::runtime_error(
                "Forward pass must be called before backward pass");
        }

        const Tensor *current = &output;
        for (int i = layers.size() - 1; i >= 0; --i)
        {
            Tensor *prev = (i > 0) ? &activations[i - 1] : &input;
            layers[i]->backward(*current, *prev);
            current = prev;
        }
    }

    std::vector<Tensor *> parameters()
    {
        std::vector<Tensor *> params;
        for (Layer *layer : layers)
        {
            auto layer_params = layer->parameters();
            params.insert(params.end(), layer_params.begin(),
                          layer_params.end());
        }
        return params;
    }

    std::string to_string() const
    {
        std::stringstream ss;
        for (Layer *layer : layers)
        {
            ss << layer->to_string() << "\n";
        }
        return ss.str();
    }

    ~Model()
    {
        for (Layer *layer : layers)
        {
            delete layer;
        }
    }
};

Tensor mse_loss(const Tensor &pred, const Tensor &target)
{
    if (pred.data.size() != target.data.size())
    {
        throw std::invalid_argument(
            "Prediction and target tensors must have same size");
    }

    Tensor loss;
    loss.shape = {1};
    loss.resize();
    loss.data[0] = 0.0f;

    for (size_t i = 0; i < pred.data.size(); ++i)
    {
        float diff = pred.data[i] - target.data[i];
        loss.data[0] += diff * diff;
    }
    loss.data[0] /= pred.data.size();
    return loss;
}

} // namespace ttie

#endif TTIE_H