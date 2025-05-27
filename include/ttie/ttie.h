#ifndef TTIE_H
#define TTIE_H

#include <cassert>
#include <iostream>
#include <random>
#include <sstream>
#include <string>
#include <vector>
#include <cmath>
#include <numeric>
#include <algorithm>

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

    Tensor transpose(size_t dim1, size_t dim2) const
    {
        if (dim1 >= shape.size() || dim2 >= shape.size())
        {
            throw std::invalid_argument("Invalid dimension index");
        }

        if (dim1 == dim2)
        {
            return *this;
        }

        // Создаем новую перестановку осей
        std::vector<size_t> new_order(shape.size());
        std::iota(new_order.begin(), new_order.end(), 0); // заполняем 0, 1, 2, ...

        // Меняем местами выбранные оси
        std::swap(new_order[dim1], new_order[dim2]);

        // Новые размерности после транспонирования
        std::vector<size_t> new_shape(shape.size());
        for (size_t i = 0; i < shape.size(); ++i)
        {
            new_shape[i] = shape[new_order[i]];
        }

        // Вычисляем шаги для исходных размерностей
        std::vector<size_t> strides(shape.size());
        strides.back() = 1;
        for (int i = shape.size() - 2; i >= 0; --i)
        {
            strides[i] = strides[i + 1] * shape[i + 1];
        }

        // Вычисляем шаги для новых размерностей
        std::vector<size_t> new_strides(new_shape.size());
        new_strides.back() = 1;
        for (int i = new_shape.size() - 2; i >= 0; --i)
        {
            new_strides[i] = new_strides[i + 1] * new_shape[i + 1];
        }

        // Создаем результирующий тензор
        Tensor result;
        result.shape = new_shape;
        result.resize();

        // Вспомогательный вектор для индексации
        std::vector<size_t> indices(shape.size(), 0);

        // Перебираем все элементы и переставляем их
        for (size_t flat_idx = 0; flat_idx < result.size(); ++flat_idx)
        {
            // Вычисляем многомерный индекс в исходном порядке
            size_t remaining = flat_idx;
            for (size_t i = 0; i < shape.size(); ++i)
            {
                indices[i] = remaining / strides[i];
                remaining %= strides[i];
            }

            // Вычисляем новый плоский индекс
            size_t new_flat_idx = 0;
            for (size_t i = 0; i < new_order.size(); ++i)
            {
                new_flat_idx += indices[new_order[i]] * new_strides[i];
            }

            result.data[new_flat_idx] = data[flat_idx];
        }
        
        return result;
    }

    Tensor view(const std::vector<size_t>& new_shape) const
    {
        if (!validate_shape())
        {
            throw std::invalid_argument("Invalid tensor shape");
        }

        // Проверяем, что новая форма совместима по размеру
        size_t new_size = 1;
        for (size_t dim : new_shape)
        {
            new_size *= dim;
        }

        if (new_size != size())
        {
            throw std::invalid_argument("Total size of new shape must match original tensor size");
        }

        // Создаем новый тензор с новой формой
        Tensor result;
        result.shape = new_shape;
        result.data = data;
        result.grad = grad;

        return result;
    }

    Tensor copy() const
    {
        Tensor result;
        result.shape = shape;
        result.data = data;
        result.grad = grad;
        return result;
    }


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

inline Tensor matmul(const Tensor &a, const Tensor &b)
{
    if (a.shape.size() < 2 || b.shape.size() < 2)
    {
        throw std::invalid_argument("Tensors must have at least 2 dimensions");
    }

    // Проверяем совместимость последних двух осей
    size_t a_cols = a.shape.back();
    size_t b_rows = b.shape[b.shape.size()-2];
    size_t b_cols = b.shape.back();

    if (a_cols != b_rows)
    {
        throw std::invalid_argument("Incompatible matrix dimensions for multiplication");
    }

    // Проверяем, что все остальные размерности совпадают
    if (a.shape.size() != b.shape.size())
    {
        throw std::invalid_argument("Tensors must have same number of dimensions");
    }

    for (size_t i = 0; i < a.shape.size()-2; ++i)
    {
        if (a.shape[i] != b.shape[i])
        {
            throw std::invalid_argument("Batch dimensions must match");
        }
    }

    // Создаем форму результата
    std::vector<size_t> result_shape = a.shape;
    result_shape.back() = b_cols; // Заменяем последнюю размерность
    result_shape[result_shape.size()-2] = a.shape[a.shape.size()-2]; // Сохраняем предпоследнюю

    Tensor result;
    result.shape = result_shape;
    result.resize();

    // Вспомогательные переменные для индексации
    size_t a_row_stride = a_cols;
    size_t a_batch_stride = a.shape[a.shape.size()-2] * a_row_stride;
    
    size_t b_row_stride = b_cols;
    size_t b_batch_stride = b_rows * b_row_stride;
    
    size_t result_row_stride = b_cols;
    size_t result_batch_stride = a.shape[a.shape.size()-2] * result_row_stride;

    // Общее количество "матриц" для перемножения
    size_t total_matrices = 1;
    for (size_t i = 0; i < a.shape.size()-2; ++i)
    {
        total_matrices *= a.shape[i];
    }

    // Выполняем матричное умножение для каждого батча/головы
    for (size_t matrix = 0; matrix < total_matrices; ++matrix)
    {
        size_t a_offset = matrix * a_batch_stride;
        size_t b_offset = matrix * b_batch_stride;
        size_t result_offset = matrix * result_batch_stride;

        for (size_t i = 0; i < a.shape[a.shape.size()-2]; ++i)
        {
            for (size_t j = 0; j < b_cols; ++j)
            {
                float sum = 0.0f;
                for (size_t k = 0; k < a_cols; ++k)
                {
                    size_t a_idx = a_offset + i * a_row_stride + k;
                    size_t b_idx = b_offset + k * b_row_stride + j;
                    sum += a.data[a_idx] * b.data[b_idx];
                }
                size_t result_idx = result_offset + i * result_row_stride + j;
                result.data[result_idx] = sum;
            }
        }
    }

    return result;
}

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

static Tensor softmax_for_mha(const Tensor &attention_score)
{
    // Применяем softmax по последнему измерению
    Tensor attention;
    attention.shape = attention_score.shape;
    attention.resize();

    const size_t last_dim = attention.shape.back();
    const size_t num_elements = attention.data.size();

    for (size_t i = 0; i < num_elements; i += last_dim)
    {
        // Находим максимум для численной стабильности
        float max_val = attention_score.data[i];
        for (size_t j = 1; j < last_dim; ++j)
        {
            max_val = std::max(max_val, attention_score.data[i + j]);
        }

        // Вычисляем экспоненты и их сумму
        float sum_exp = 0.0f;
        for (size_t j = 0; j < last_dim; ++j)
        {
            attention.data[i + j] = std::exp(attention_score.data[i + j] - max_val);
            sum_exp += attention.data[i + j];
        }

        // Нормализуем для получения вероятностей
        for (size_t j = 0; j < last_dim; ++j)
        {
            attention.data[i + j] /= sum_exp;
        }
    }
    return attention;
}

struct ScaledDotProductAttention
{
    Tensor saved_q, saved_k, saved_v;
    Tensor attn_scores, attention;
    float scale = 0.0f;

    void forward(const Tensor &q, const Tensor &k, const Tensor &v, Tensor &values)
    {
        if (q.shape.size() != k.shape.size() || q.shape.size() != v.shape.size())
        {
            throw std::runtime_error("All input tensors must have the same number of dimensions");
        }
        saved_q = q.copy();
        saved_k = k.copy();
        saved_v = v.copy();

        const size_t d_k = k.shape.back();

        attn_scores.shape = q.shape;
        attn_scores.shape.back() = k.shape[k.shape.size() - 2];
        attn_scores.resize();

        attn_scores = matmul(q, k.transpose(k.shape.size() - 2, k.shape.size() - 1));

        scale = 1.0f / std::sqrt(static_cast<float>(d_k));
        for (size_t i = 0; i < attn_scores.data.size(); ++i)
        {
            attn_scores.data[i] *= scale;
        }

        attention = softmax_for_mha(attn_scores);

        values = matmul(attention, v);
    }

    void backward(const Tensor &grad_output, Tensor &dq, Tensor &dk, Tensor &dv)
    {

        // Инициализация градиентов
        dq.resize_grad();
        dk.resize_grad();
        dv.resize_grad();

        // Вычисление градиента для dv (dV = Attention^T * grad_output)
        Tensor attention_T = attention.transpose(attention.shape.size() - 2, attention.shape.size() - 1);
        attention_T.resize_grad();
        dv = matmul(attention_T, grad_output);
        dv.resize_grad();
        for (size_t i = 0; i < dv.grad.size(); ++i)
        {
            dv.grad[i] += grad_output.grad[i]; // Накопление градиентов
        }

        // Вычисление градиента для attention (dAttention = grad_output * V^T)
        Tensor v_T = saved_v.transpose(saved_v.shape.size() - 2, saved_v.shape.size() - 1);
        v_T.resize_grad();
        Tensor dAttention;
        dAttention = matmul(grad_output, v_T);
        dAttention.resize_grad();
        for (size_t i = 0; i < dAttention.grad.size(); ++i)
        {
            dAttention.grad[i] = grad_output.grad[i]; // Передаем градиенты
        }

        // Вычисление градиента для scores (dScores = dSoftmax(attention) * dAttention)
        Tensor dScores;
        dScores.shape = attn_scores.shape;
        dScores.resize();
        dScores.resize_grad();
        const size_t BHT = attention.data.size() / attention.shape.back();
        const size_t T_k = attention.shape.back();
        for (size_t i = 0; i < BHT; ++i)
        {
            float dot = 0.0f;
            for (size_t j = 0; j < T_k; ++j)
            {
                dot += attention.data[i * T_k + j] * dAttention.grad[i * T_k + j];
            }
            for (size_t j = 0; j < T_k; ++j)
            {
                dScores.grad[i * T_k + j] = attention.data[i * T_k + j] * (dAttention.grad[i * T_k + j] - dot);
            }
        }
        for (float &val : dScores.grad)
        {
            val *= scale; // Масштабирование
        }

        // Вычисление dq (dq = dScores * K)
        dq = matmul(dScores, saved_k);
        dq.resize_grad();
        for (size_t i = 0; i < dq.grad.size(); ++i)
        {
            dq.grad[i] += dScores.grad[i]; // Накопление градиентов
        }

        // Вычисление dk (dk = dScores^T * Q)
        Tensor dScores_T = dScores.transpose(dScores.shape.size() - 2, dScores.shape.size() - 1);
        dScores_T.resize_grad();
        for (size_t i = 0; i < dScores_T.grad.size(); ++i)
        {
            dScores_T.grad[i] = dScores.grad[i];
        }

        dk = matmul(dScores_T, saved_q);
        dk.resize_grad();
        for (size_t i = 0; i < dk.grad.size(); ++i)
        {
            dk.grad[i] += dScores_T.grad[i]; // Накопление градиентов
        }
    }
};

struct MultiHeadAttention
{
    ScaledDotProductAttention attention;
    size_t head_dim = 0ULL;
    size_t d_model = 0ULL;
    size_t num_heads = 0ULL;
    Linear w_q;
    Linear w_k;
    Linear w_v;
    Linear w_concat;
    Tensor q;
    Tensor k;
    Tensor v;
    Tensor w_concat_in;

    MultiHeadAttention(size_t d_model, size_t num_heads) : w_q(d_model, d_model),
    w_k(d_model, d_model), w_v(d_model, d_model), 
    w_concat(d_model, d_model), head_dim(d_model / num_heads), d_model(d_model), num_heads(num_heads)
    {
        if (d_model % num_heads != 0)
        {
            throw std::runtime_error("Embedding dimension must be 0 modulo number of heads");
        }
    }

    void split(Tensor &x)
    {
        const size_t batch_size = x.shape[0];
        const size_t seq_len = x.shape[1];

        x = x.view({batch_size, seq_len, num_heads, head_dim});
        x = x.transpose(1, 2);
    }

    void concat(Tensor &x)
    {
        const size_t batch_size = x.shape[0];
        const size_t seq_len = x.shape[2];
        x = x.transpose(1, 2);

        x = x.view({batch_size, seq_len, d_model});

    }

    void forward(const Tensor &q_in, const Tensor &k_in, const Tensor &v_in, Tensor &out)
    {
        q = q_in.copy();
        k = k_in.copy();
        v = v_in.copy();

        const size_t batch = q.shape[0];
        const size_t seq_len = q.shape[1];
        const size_t d_model = q.shape[2];

        q = q.view({batch * seq_len, d_model});
        k = k.view({batch * seq_len, d_model});
        v = v.view({batch * seq_len, d_model});

        w_q.forward(q, q);
        w_k.forward(k, k);
        w_v.forward(v, v);

        q = q.view({batch, seq_len, d_model});
        k = k.view({batch, seq_len, d_model});
        v = v.view({batch, seq_len, d_model});

        split(q);
        split(k);
        split(v);

        attention.forward(q, k, v, out);

        concat(out);

        out = out.view({batch * seq_len, d_model});
        w_concat.forward(out, out);
        out = out.view({batch, seq_len, d_model});
        w_concat_in = out.copy();
    }

    void backward(const Tensor &grad_output, Tensor &dq, Tensor &dk, Tensor &dv)
    {

        // Инициализация градиентов выходов
        dq.resize_grad();
        dk.resize_grad();
        dv.resize_grad();

        // Изменение формы grad_output на (batch * seq_len, d_model) для w_concat.backward
        const size_t batch = grad_output.shape[0];
        const size_t seq_len = grad_output.shape[1];
        Tensor grad_out_reshaped = grad_output.view({batch * seq_len, d_model});
        grad_out_reshaped.resize_grad();
        for (size_t i = 0; i < grad_out_reshaped.grad.size(); ++i)
        {
            grad_out_reshaped.grad[i] = grad_output.grad[i];
        }

        // Обратный проход через w_concat
        Tensor grad_concat_in;
        grad_concat_in.shape = {batch * seq_len, d_model};
        grad_concat_in.data = w_concat_in.data;
        w_concat.backward(grad_out_reshaped, grad_concat_in);

        // Возврат к форме (batch, seq_len, d_model)
        Tensor grad_concat_out = grad_concat_in.view({batch, seq_len, d_model});
        grad_concat_out.resize_grad();
        for (size_t i = 0; i < grad_concat_out.grad.size(); ++i)
        {
            grad_concat_out.grad[i] = grad_concat_in.grad[i];
        }

        // Обратный проход через concat (транспонирование: (batch, seq_len, num_heads, head_dim) -> (batch, num_heads, seq_len, head_dim))
        Tensor grad_concat_transposed = grad_concat_out.view({batch, seq_len, num_heads, head_dim});
        grad_concat_transposed = grad_concat_transposed.transpose(1, 2);
        grad_concat_transposed.resize_grad();
        for (size_t i = 0; i < grad_concat_transposed.grad.size(); ++i)
        {
            grad_concat_transposed.grad[i] = grad_concat_out.grad[i];
        }

        // Обратный проход через ScaledDotProductAttention
        Tensor grad_q, grad_k, grad_v;
        grad_q.shape = {batch, num_heads, seq_len, head_dim};
        grad_k.shape = {batch, num_heads, seq_len, head_dim};
        grad_v.shape = {batch, num_heads, seq_len, head_dim};
        attention.backward(grad_concat_transposed, grad_q, grad_k, grad_v);

        // Обратный проход через split (транспонирование: (batch, num_heads, seq_len, head_dim) -> (batch, seq_len, num_heads, head_dim))
        Tensor grad_q_reshaped = grad_q.transpose(1, 2).view({batch, seq_len, d_model});
        Tensor grad_k_reshaped = grad_k.transpose(1, 2).view({batch, seq_len, d_model});
        Tensor grad_v_reshaped = grad_v.transpose(1, 2).view({batch, seq_len, d_model});
        grad_q_reshaped.resize_grad();
        grad_k_reshaped.resize_grad();
        grad_v_reshaped.resize_grad();
        for (size_t i = 0; i < grad_q_reshaped.grad.size(); ++i)
        {
            grad_q_reshaped.grad[i] = grad_q.grad[i];
            grad_k_reshaped.grad[i] = grad_k.grad[i];
            grad_v_reshaped.grad[i] = grad_v.grad[i];
        }

        // Обратный проход через w_q, w_k, w_v
        Tensor grad_q_in, grad_k_in, grad_v_in;
        grad_q_in.shape = {batch * seq_len, d_model};
        grad_k_in.shape = {batch * seq_len, d_model};
        grad_v_in.shape = {batch * seq_len, d_model};
        grad_q_in.data = q.data;
        grad_k_in.data = k.data;
        grad_v_in.data = v.data;

        w_q.backward(grad_q_reshaped, grad_q_in);
        w_k.backward(grad_k_reshaped, grad_k_in);
        w_v.backward(grad_v_reshaped, grad_v_in);

        // Накопление градиентов в dq, dk, dv
        Tensor grad_q_final = grad_q_in.view({batch, seq_len, d_model});
        Tensor grad_k_final = grad_k_in.view({batch, seq_len, d_model});
        Tensor grad_v_final = grad_v_in.view({batch, seq_len, d_model});
        grad_q_final.resize_grad();
        grad_k_final.resize_grad();
        grad_v_final.resize_grad();
        for (size_t i = 0; i < grad_q_final.grad.size(); ++i)
        {
            grad_q_final.grad[i] = grad_q_in.grad[i];
            grad_k_final.grad[i] = grad_k_in.grad[i];
            grad_v_final.grad[i] = grad_v_in.grad[i];
        }

        for (size_t i = 0; i < dq.grad.size(); ++i)
        {
            dq.grad[i] += grad_q_final.grad[i];
        }
        for (size_t i = 0; i < dk.grad.size(); ++i)
        {
            dk.grad[i] += grad_k_final.grad[i];
        }
        for (size_t i = 0; i < dv.grad.size(); ++i)
        {
            dv.grad[i] += grad_v_final.grad[i];
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

class BatchNorm1d : public Layer {
private:
    size_t num_features;
    float eps;
    float momentum;
    bool affine;
    bool track_running_stats;

    Tensor gamma; 
    Tensor beta; 
    Tensor running_mean;
    Tensor running_var;

    std::vector<float> input_data; // сохраняем входные данные для backward
    bool first_update = true;

public:
    BatchNorm1d(size_t num_features,
                float eps = 1e-5,
                float momentum = 0.1,
                bool affine = true,
                bool track_running_stats = true)
        : num_features(num_features),
            eps(eps),
            momentum(momentum),
            affine(affine),
            track_running_stats(track_running_stats) {

        if (affine) {
            gamma.shape = {num_features};
            gamma.resize();
            gamma.resize_grad();
            beta.shape = {num_features};
            beta.resize();
            beta.resize_grad();
            std::random_device rd;
            std::mt19937 gen(rd());
            std::uniform_real_distribution<float> dis(0.9f, 1.1f);
            for (size_t i = 0; i < num_features; ++i) {
                gamma.data[i] = dis(gen);
                beta.data[i] = 0.0f;
            }
        }

        if (track_running_stats) {
            running_mean.shape = {num_features};
            running_mean.resize();
            running_var.shape = {num_features};
            running_var.resize();
        }
    }

    void forward(const Tensor &input, Tensor &output) override {
        if (input.shape.size() != 2 || input.shape[1] != num_features) {
            throw std::invalid_argument("Входной тензор должен быть [batch_size, num_features]");
        }

        size_t batch_size = input.shape[0];
        output.shape = {batch_size, num_features};
        output.resize();

        input_data = input.data;

        for (size_t f = 0; f < num_features; ++f) {
            float mean = 0.0f, var = 0.0f;

            for (size_t b = 0; b < batch_size; ++b) {
                mean += input.data[b * num_features + f];
            }
            mean /= batch_size;

            for (size_t b = 0; b < batch_size; ++b) {
                float diff = input.data[b * num_features + f] - mean;
                var += diff * diff;
            }
            var /= batch_size;

            if (track_running_stats) {
                if (first_update) {
                    running_mean.data[f] = mean;
                    running_var.data[f] = var;
                } else {
                    running_mean.data[f] = (1 - momentum) * running_mean.data[f] + momentum * mean;
                    running_var.data[f] = (1 - momentum) * running_var.data[f] + momentum * var;
                }
            }

            float inv_std = 1.0f / std::sqrt(var + eps);

            for (size_t b = 0; b < batch_size; ++b) {
                float x_hat = (input.data[b * num_features + f] - mean) * inv_std;
                output.data[b * num_features + f] = affine ? gamma.data[f] * x_hat + beta.data[f] : x_hat;
            }
        }

        first_update = false;
    }

    void backward(const Tensor &grad_output, Tensor &grad_input) override {
        if (grad_output.shape.size() != 2 || grad_output.shape[1] != num_features) {
            throw std::invalid_argument("grad_output должен быть [batch_size, num_features]");
        }
        if (grad_output.grad.empty()) {
            throw std::runtime_error("grad_output.grad пустой");
        }
        if (input_data.empty()) {
            throw std::runtime_error("input_data пустой. Сначала вызовите forward()");
        }
        if (input_data.size() != grad_output.shape[0] * num_features) {
            throw std::runtime_error("Размер input_data не соответствует ожидаемому");
        }
    
        size_t batch_size = grad_output.shape[0];
        grad_input.shape = {batch_size, num_features};
        grad_input.resize();
        grad_input.resize_grad();
    
        const size_t count = batch_size; // Перемещено и помечено как const
        for (size_t f = 0; f < num_features; ++f) {
            float mean = 0.0f, var = 0.0f;
    
            for (size_t b = 0; b < batch_size; ++b) {
                mean += input_data[b * num_features + f];
            }
            mean /= count;
    
            for (size_t b = 0; b < batch_size; ++b) {
                float diff = input_data[b * num_features + f] - mean;
                var += diff * diff;
            }
            var /= count;
    
            float inv_std = 1.0f / std::sqrt(var + eps);
    
            std::vector<float> x_hat(count);
            for (size_t b = 0; b < batch_size; ++b) {
                x_hat[b] = (input_data[b * num_features + f] - mean) * inv_std;
            }
    
            float sum_dy = 0.0f, sum_dy_x_hat = 0.0f;
            for (size_t b = 0; b < batch_size; ++b) {
                float dy = grad_output.grad[b * num_features + f];
                sum_dy += dy;
                sum_dy_x_hat += dy * x_hat[b];
            }
    
            float factor = static_cast<float>(count);
            float gamma_val = affine ? gamma.data[f] : 1.0f;
    
            for (size_t b = 0; b < batch_size; ++b) {
                float dy = grad_output.grad[b * num_features + f];
                float dx_hat = dy;
                float term1 = dx_hat * inv_std * gamma_val;
                float term2 = inv_std * sum_dy / factor * gamma_val;
                float term3 = inv_std * x_hat[b] * sum_dy_x_hat / factor * gamma_val;
    
                grad_input.grad[b * num_features + f] = term1 - term2 - term3;
    
                if (affine) {
                    beta.grad[f] += dy;
                    gamma.grad[f] += dy * x_hat[b];
                }
            }
        }
    }

    std::string to_string() const override {
        std::stringstream ss;
        ss << "BatchNorm1d(" << num_features << ")";
        return ss.str();
    }

    std::vector<Tensor *> parameters() override {
        if (affine) {
            return {&gamma, &beta};
        }
        return {};
    }
};

class BatchNorm2d : public Layer {
private:
    size_t num_features;
    float eps;
    float momentum;
    bool affine;
    bool track_running_stats;

    Tensor gamma;
    Tensor beta;
    Tensor running_mean;
    Tensor running_var;

    std::vector<float> input_data;
    bool first_update = true;

public:
    BatchNorm2d(size_t num_features,
                float eps = 1e-5,
                float momentum = 0.1,
                bool affine = true,
                bool track_running_stats = true)
        : num_features(num_features),
            eps(eps),
            momentum(momentum),
            affine(affine),
            track_running_stats(track_running_stats) {

        if (affine) {
            gamma.shape = {num_features};
            gamma.resize();
            gamma.resize_grad();
            beta.shape = {num_features};
            beta.resize();
            beta.resize_grad();
            std::random_device rd;
            std::mt19937 gen(rd());
            std::uniform_real_distribution<float> dis(0.9f, 1.1f);
            for (size_t i = 0; i < num_features; ++i) {
                gamma.data[i] = dis(gen);
                beta.data[i] = 0.0f;
            }
        }

        if (track_running_stats) {
            running_mean.shape = {num_features};
            running_mean.resize();
            running_var.shape = {num_features};
            running_var.resize();
        }
    }

    void forward(const Tensor &input, Tensor &output) override {
        if (input.shape.size() != 4) {
            throw std::invalid_argument("Входной тензор должен быть [N, C, H, W]");
        }
        if (input.shape[1] != num_features) {
            throw std::invalid_argument("Количество каналов должно соответствовать num_features");
        }
        if (input.data.size() != input.size()) {
            throw std::runtime_error("Размер входных данных не соответствует shape");
        }
    
        size_t N = input.shape[0];
        size_t C = input.shape[1];
        size_t H = input.shape[2];
        size_t W = input.shape[3];
    
        output.shape = {N, C, H, W};
        output.resize();
    
        input_data = input.data;
    
        for (size_t c = 0; c < C; ++c) {
            float mean = 0.0f, var = 0.0f;
            size_t count = N * H * W;
    
            // Вычисление среднего
            for (size_t n = 0; n < N; ++n) {
                for (size_t h = 0; h < H; ++h) {
                    for (size_t w = 0; w < W; ++w) {
                        size_t idx = n * C * H * W + c * H * W + h * W + w;
                        mean += input.data[idx];
                    }
                }
            }
            mean /= count;
    
            // Вычисление дисперсии
            for (size_t n = 0; n < N; ++n) {
                for (size_t h = 0; h < H; ++h) {
                    for (size_t w = 0; w < W; ++w) {
                        size_t idx = n * C * H * W + c * H * W + h * W + w;
                        float diff = input.data[idx] - mean;
                        var += diff * diff;
                    }
                }
            }
            var /= count;
    
            // Обновление running_mean и running_var
            if (track_running_stats) {
                if (first_update) {
                    running_mean.data[c] = mean;
                    running_var.data[c] = var;
                } else {
                    running_mean.data[c] = (1 - momentum) * running_mean.data[c] + momentum * mean;
                    running_var.data[c] = (1 - momentum) * running_var.data[c] + momentum * var;
                }
            }
    
            float inv_std = 1.0f / std::sqrt(var + eps);
    
            // Вычисление выхода с учетом affine за пределами вложенных циклов
            if (affine) {
                for (size_t n = 0; n < N; ++n) {
                    for (size_t h = 0; h < H; ++h) {
                        for (size_t w = 0; w < W; ++w) {
                            size_t idx = n * C * H * W + c * H * W + h * W + w;
                            float x_hat = (input.data[idx] - mean) * inv_std;
                            output.data[idx] = gamma.data[c] * x_hat + beta.data[c];
                        }
                    }
                }
            } else {
                for (size_t n = 0; n < N; ++n) {
                    for (size_t h = 0; h < H; ++h) {
                        for (size_t w = 0; w < W; ++w) {
                            size_t idx = n * C * H * W + c * H * W + h * W + w;
                            float x_hat = (input.data[idx] - mean) * inv_std;
                            output.data[idx] = x_hat;
                        }
                    }
                }
            }
        }
    
        first_update = false;
    }

    void backward(const Tensor &grad_output, Tensor &grad_input) override {
        if (input_data.empty()) {
            throw std::runtime_error("input_data пустой. Сначала вызовите forward()");
        }
        if (grad_output.shape.size() != 4 || grad_output.shape[1] != num_features) {
            throw std::invalid_argument("grad_output должен быть [N, C, H, W]");
        }
        if (grad_output.grad.empty()) {
            throw std::runtime_error("grad_output.grad пустой");
        }
        if (input_data.size() != grad_output.shape[0] * num_features * grad_output.shape[2] * grad_output.shape[3]) {
            throw std::runtime_error("Размер input_data не соответствует ожидаемому");
        }
        if (affine && (gamma.grad.size() != num_features || beta.grad.size() != num_features)) {
            throw std::runtime_error("Градиенты gamma или beta имеют неправильный размер");
        }
    
        size_t N = grad_output.shape[0];
        size_t C = grad_output.shape[1];
        size_t H = grad_output.shape[2];
        size_t W = grad_output.shape[3];
    
        grad_input.shape = {N, C, H, W};
        grad_input.resize();
        grad_input.resize_grad();
    
        const size_t count = N * H * W; 
        for (size_t c = 0; c < C; ++c) {
            float mean = 0.0f, var = 0.0f;
    
            for (size_t n = 0; n < N; ++n) {
                for (size_t h = 0; h < H; ++h) {
                    for (size_t w = 0; w < W; ++w) {
                        size_t idx = n * C * H * W + c * H * W + h * W + w;
                        mean += input_data[idx];
                    }
                }
            }
            mean /= count;
    
            for (size_t n = 0; n < N; ++n) {
                for (size_t h = 0; h < H; ++h) {
                    for (size_t w = 0; w < W; ++w) {
                        size_t idx = n * C * H * W + c * H * W + h * W + w;
                        float diff = input_data[idx] - mean;
                        var += diff * diff;
                    }
                }
            }
            var /= count;
    
            float inv_std = 1.0f / std::sqrt(var + eps);
            float gamma_val = affine ? gamma.data[c] : 1.0f;
    
            float sum_dy = 0.0f, sum_dy_x_hat = 0.0f;
            std::vector<float> x_hat(count);
            size_t pos = 0;
            for (size_t n = 0; n < N; ++n) {
                for (size_t h = 0; h < H; ++h) {
                    for (size_t w = 0; w < W; ++w) {
                        size_t idx = n * C * H * W + c * H * W + h * W + w;
                        x_hat[pos] = (input_data[idx] - mean) * inv_std;
                        sum_dy += grad_output.grad[idx];
                        sum_dy_x_hat += grad_output.grad[idx] * x_hat[pos];
                        pos++;
                    }
                }
            }
    
            pos = 0;
            for (size_t n = 0; n < N; ++n) {
                for (size_t h = 0; h < H; ++h) {
                    for (size_t w = 0; w < W; ++w) {
                        size_t idx = n * C * H * W + c * H * W + h * W + w;
                        float dy = grad_output.grad[idx];
                        float dx_hat = dy;
                        float term1 = dx_hat * inv_std * gamma_val;
                        float term2 = inv_std * sum_dy / count * gamma_val;
                        float term3 = inv_std * x_hat[pos] * sum_dy_x_hat / count * gamma_val;
                        grad_input.grad[idx] = term1 - term2 - term3;
    
                        if (affine) {
                            beta.grad[c] += dy;
                            gamma.grad[c] += dy * x_hat[pos];
                        }
                        pos++;
                    }
                }
            }
        }
    }

    std::string to_string() const override {
        std::stringstream ss;
        ss << "BatchNorm2d(" << num_features << ")";
        return ss.str();
    }

    std::vector<Tensor *> parameters() override {
        if (affine) {
            return {&gamma, &beta};
        }
        return {};
    }
};

class BatchNorm3d : public Layer {
private:
    size_t num_features;
    float eps;
    float momentum;
    bool affine;
    bool track_running_stats;

    Tensor gamma;
    Tensor beta;
    Tensor running_mean;
    Tensor running_var;

    std::vector<float> input_data;
    bool first_update = true;

public:
    BatchNorm3d(size_t num_features,
                float eps = 1e-5,
                float momentum = 0.1,
                bool affine = true,
                bool track_running_stats = true)
        : num_features(num_features),
            eps(eps),
            momentum(momentum),
            affine(affine),
            track_running_stats(track_running_stats) {

        if (affine) {
            gamma.shape = {num_features};
            gamma.resize();
            gamma.resize_grad();
            beta.shape = {num_features};
            beta.resize();
            beta.resize_grad();
            std::random_device rd;
            std::mt19937 gen(rd());
            std::uniform_real_distribution<float> dis(0.9f, 1.1f);
            for (size_t i = 0; i < num_features; ++i) {
                gamma.data[i] = dis(gen);
                beta.data[i] = 0.0f;
            }
        }

        if (track_running_stats) {
            running_mean.shape = {num_features};
            running_mean.resize();
            running_var.shape = {num_features};
            running_var.resize();
        }
    }

    void forward(const Tensor &input, Tensor &output) override {
        if (input.shape.size() != 5) {
            throw std::invalid_argument("Входной тензор должен быть [N, C, D, H, W]");
        }
        if (input.shape[1] != num_features) {
            throw std::invalid_argument("Количество каналов должно соответствовать num_features");
        }
        if (input.data.size() != input.size()) {
            throw std::runtime_error("Размер входных данных не соответствует shape");
        }
    
        size_t N = input.shape[0];
        size_t C = input.shape[1];
        size_t D = input.shape[2];
        size_t H = input.shape[3];
        size_t W = input.shape[4];
    
        output.shape = {N, C, D, H, W};
        output.resize();
    
        input_data = input.data;
    
        for (size_t c = 0; c < C; ++c) {
            float mean = 0.0f, var = 0.0f;
            size_t count = N * D * H * W;
    
            // Вычисление среднего
            for (size_t n = 0; n < N; ++n) {
                for (size_t d = 0; d < D; ++d) {
                    for (size_t h = 0; h < H; ++h) {
                        for (size_t w = 0; w < W; ++w) {
                            size_t idx = n * C * D * H * W + c * D * H * W + d * H * W + h * W + w;
                            mean += input.data[idx];
                        }
                    }
                }
            }
            mean /= count;
    
            // Вычисление дисперсии
            for (size_t n = 0; n < N; ++n) {
                for (size_t d = 0; d < D; ++d) {
                    for (size_t h = 0; h < H; ++h) {
                        for (size_t w = 0; w < W; ++w) {
                            size_t idx = n * C * D * H * W + c * D * H * W + d * H * W + h * W + w;
                            float diff = input.data[idx] - mean;
                            var += diff * diff;
                        }
                    }
                }
            }
            var /= count;
    
            // Обновление running_mean и running_var
            if (track_running_stats) {
                if (first_update) {
                    running_mean.data[c] = mean;
                    running_var.data[c] = var;
                } else {
                    running_mean.data[c] = (1 - momentum) * running_mean.data[c] + momentum * mean;
                    running_var.data[c] = (1 - momentum) * running_var.data[c] + momentum * var;
                }
            }
    
            float inv_std = 1.0f / std::sqrt(var + eps);
    
            // Вычисление выхода с учетом affine за пределами вложенных циклов
            if (affine) {
                for (size_t n = 0; n < N; ++n) {
                    for (size_t d = 0; d < D; ++d) {
                        for (size_t h = 0; h < H; ++h) {
                            for (size_t w = 0; w < W; ++w) {
                                size_t idx = n * C * D * H * W + c * D * H * W + d * H * W + h * W + w;
                                float x_hat = (input.data[idx] - mean) * inv_std;
                                output.data[idx] = gamma.data[c] * x_hat + beta.data[c];
                            }
                        }
                    }
                }
            } else {
                for (size_t n = 0; n < N; ++n) {
                    for (size_t d = 0; d < D; ++d) {
                        for (size_t h = 0; h < H; ++h) {
                            for (size_t w = 0; w < W; ++w) {
                                size_t idx = n * C * D * H * W + c * D * H * W + d * H * W + h * W + w;
                                float x_hat = (input.data[idx] - mean) * inv_std;
                                output.data[idx] = x_hat;
                            }
                        }
                    }
                }
            }
        }
    
        first_update = false;
    }

    void backward(const Tensor &grad_output, Tensor &grad_input) override {
        if (input_data.empty()) {
            throw std::runtime_error("input_data пустой. Сначала вызовите forward()");
        }
        if (grad_output.shape.size() != 5 || grad_output.shape[1] != num_features) {
            throw std::invalid_argument("grad_output должен быть [N, C, D, H, W]");
        }
        if (grad_output.grad.empty()) {
            throw std::runtime_error("grad_output.grad пустой");
        }
        if (input_data.size() != grad_output.shape[0] * num_features * grad_output.shape[2] * grad_output.shape[3] * grad_output.shape[4]) {
            throw std::runtime_error("Размер input_data не соответствует ожидаемому");
        }
        if (affine && (gamma.grad.size() != num_features || beta.grad.size() != num_features)) {
            throw std::runtime_error("Градиенты gamma или beta имеют неправильный размер");
        }
    
        size_t N = grad_output.shape[0];
        size_t C = grad_output.shape[1];
        size_t D = grad_output.shape[2];
        size_t H = grad_output.shape[3];
        size_t W = grad_output.shape[4];
    
        grad_input.shape = {N, C, D, H, W};
        grad_input.resize();
        grad_input.resize_grad();
    
        const size_t count = N * D * H * W; // Перемещено и помечено как const
        for (size_t c = 0; c < C; ++c) {
            float mean = 0.0f, var = 0.0f;
    
            for (size_t n = 0; n < N; ++n) {
                for (size_t d = 0; d < D; ++d) {
                    for (size_t h = 0; h < H; ++h) {
                        for (size_t w = 0; w < W; ++w) {
                            size_t idx = n * C * D * H * W + c * D * H * W + d * H * W + h * W + w;
                            mean += input_data[idx];
                        }
                    }
                }
            }
            mean /= count;
    
            for (size_t n = 0; n < N; ++n) {
                for (size_t d = 0; d < D; ++d) {
                    for (size_t h = 0; h < H; ++h) {
                        for (size_t w = 0; w < W; ++w) {
                            size_t idx = n * C * D * H * W + c * D * H * W + d * H * W + h * W + w;
                            float diff = input_data[idx] - mean;
                            var += diff * diff;
                        }
                    }
                }
            }
            var /= count;
    
            float inv_std = 1.0f / std::sqrt(var + eps);
            float gamma_val = affine ? gamma.data[c] : 1.0f;
    
            float sum_dy = 0.0f, sum_dy_x_hat = 0.0f;
            std::vector<float> x_hat(count);
            size_t pos = 0;
            for (size_t n = 0; n < N; ++n) {
                for (size_t d = 0; d < D; ++d) {
                    for (size_t h = 0; h < H; ++h) {
                        for (size_t w = 0; w < W; ++w) {
                            size_t idx = n * C * D * H * W + c * D * H * W + d * H * W + h * W + w;
                            x_hat[pos] = (input_data[idx] - mean) * inv_std;
                            sum_dy += grad_output.grad[idx];
                            sum_dy_x_hat += grad_output.grad[idx] * x_hat[pos];
                            pos++;
                        }
                    }
                }
            }
    
            pos = 0;
            for (size_t n = 0; n < N; ++n) {
                for (size_t d = 0; d < D; ++d) {
                    for (size_t h = 0; h < H; ++h) {
                        for (size_t w = 0; w < W; ++w) {
                            size_t idx = n * C * D * H * W + c * D * H * W + d * H * W + h * W + w;
                            float dy = grad_output.grad[idx];
                            float dx_hat = dy;
                            float term1 = dx_hat * inv_std * gamma_val;
                            float term2 = inv_std * sum_dy / count * gamma_val;
                            float term3 = inv_std * x_hat[pos] * sum_dy_x_hat / count * gamma_val;
                            grad_input.grad[idx] = term1 - term2 - term3;
    
                            if (affine) {
                                beta.grad[c] += dy;
                                gamma.grad[c] += dy * x_hat[pos];
                            }
                            pos++;
                        }
                    }
                }
            }
        }
    }

    std::string to_string() const override {
        std::stringstream ss;
        ss << "BatchNorm3d(" << num_features << ")";
        return ss.str();
    }

    std::vector<Tensor *> parameters() override {
        if (affine) {
            return {&gamma, &beta};
        }
        return {};
    }
};

} // namespace ttie

#endif // TTIE_H