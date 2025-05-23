#ifndef TTIE_H
#define TTIE_H

#pragma once

#include <cassert>
#include <iostream>
#include <random>
#include <sstream>
#include <string>
#include <vector>

#include "MatrixMultiplier.h"

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


    void forward_with_strassen(const Tensor& input, Tensor& output) {
        size_t in_features = weight.shape[0];
        size_t out_features = weight.shape[1];
        output.shape = {input.shape[0], out_features};
        output.resize();

        // Create matrix multiplier instance
        MatrixMultiplier matmul;

        // Check if we can use Strassen's method (square matrices with size power of 2)
        bool can_use_strassen = (input.shape[0] == in_features) && 
                            (in_features == out_features) &&
                            ((input.shape[0] & (input.shape[0] - 1)) == 0);

        for (size_t i = 0; i < input.shape[0]; ++i) {
            // Get current input sample (row vector)
            std::vector<float> input_row(in_features);
            for (size_t k = 0; k < in_features; ++k) {
                input_row[k] = input.data[i * in_features + k];
            }

            // Multiply using appropriate method
            std::vector<float> multiplied;
            
            // For Strassen, we need to reshape the input to square matrix
            std::vector<float> input_square(input_row.begin(), input_row.end());
            input_square.resize(in_features * in_features, 0.0f);  // Pad if needed
            
            multiplied = matmul.strassen_multiply(input_square, weight.data, in_features);
            
            // Extract the first row of the result (since we multiplied by a row vector)
            for (size_t j = 0; j < out_features; ++j) {
                output.data[i * out_features + j] = multiplied[j] + bias.data[j];
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

#endif // TTIE_H