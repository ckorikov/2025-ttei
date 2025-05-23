#ifndef TTIE_H
#define TTIE_H

#include <cassert>
#include <iostream>
#include <random>
#include <sstream>
#include <string>
#include <vector>

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

        for (size_t f = 0; f < num_features; ++f) {
            float mean = 0.0f, var = 0.0f;

            for (size_t b = 0; b < batch_size; ++b) {
                mean += input_data[b * num_features + f];
            }
            mean /= batch_size;

            for (size_t b = 0; b < batch_size; ++b) {
                float diff = input_data[b * num_features + f] - mean;
                var += diff * diff;
            }
            var /= batch_size;

            float inv_std = 1.0f / std::sqrt(var + eps);

            std::vector<float> x_hat(batch_size);
            for (size_t b = 0; b < batch_size; ++b) {
                x_hat[b] = (input_data[b * num_features + f] - mean) * inv_std;
            }

            float sum_dy = 0.0f, sum_dy_x_hat = 0.0f;
            for (size_t b = 0; b < batch_size; ++b) {
                float dy = grad_output.grad[b * num_features + f];
                sum_dy += dy;
                sum_dy_x_hat += dy * x_hat[b];
            }

            float factor = static_cast<float>(batch_size);
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

            for (size_t n = 0; n < N; ++n) {
                for (size_t h = 0; h < H; ++h) {
                    for (size_t w = 0; w < W; ++w) {
                        size_t idx = n * C * H * W + c * H * W + h * W + w;
                        mean += input.data[idx];
                    }
                }
            }
            mean /= count;

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
            for (size_t n = 0; n < N; ++n) {
                for (size_t h = 0; h < H; ++h) {
                    for (size_t w = 0; w < W; ++w) {
                        size_t idx = n * C * H * W + c * H * W + h * W + w;
                        float x_hat = (input.data[idx] - mean) * inv_std;
                        output.data[idx] = affine ? gamma.data[c] * x_hat + beta.data[c] : x_hat;
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

        if (affine) {
            std::fill(gamma.grad.begin(), gamma.grad.end(), 0.0f);
            std::fill(beta.grad.begin(), beta.grad.end(), 0.0f);
        }

        for (size_t c = 0; c < C; ++c) {
            float mean = 0.0f, var = 0.0f;
            size_t count = N * H * W;

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
            for (size_t n = 0; n < N; ++n) {
                for (size_t d = 0; d < D; ++d) {
                    for (size_t h = 0; h < H; ++h) {
                        for (size_t w = 0; w < W; ++w) {
                            size_t idx = n * C * D * H * W + c * D * H * W + d * H * W + h * W + w;
                            float x_hat = (input.data[idx] - mean) * inv_std;
                            output.data[idx] = affine ? gamma.data[c] * x_hat + beta.data[c] : x_hat;
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

        if (affine) {
            std::fill(gamma.grad.begin(), gamma.grad.end(), 0.0f);
            std::fill(beta.grad.begin(), beta.grad.end(), 0.0f);
        }

        for (size_t c = 0; c < C; ++c) {
            float mean = 0.0f, var = 0.0f;
            size_t count = N * D * H * W;

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