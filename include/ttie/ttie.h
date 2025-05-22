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

struct LSTM : public Model
{
    size_t input_dim;
    size_t hidden_dim;
    size_t num_layers;

    LSTM(size_t D_in, size_t H, size_t L = 1)
        : input_dim(D_in), hidden_dim(H), num_layers(L)
    {
        for (size_t layer = 0; layer < num_layers; ++layer)
        {
            size_t in_dim = (layer == 0 ? input_dim : hidden_dim);
            add_layer(new Linear(in_dim, hidden_dim));
            add_layer(new Sigmoid()); // ii
            add_layer(new Linear(in_dim, hidden_dim));
            add_layer(new Sigmoid()); // if
            add_layer(new Linear(in_dim, hidden_dim));
            add_layer(new Tanh()); // ig
            add_layer(new Linear(in_dim, hidden_dim));
            add_layer(new Sigmoid()); // io

            add_layer(new Linear(hidden_dim, hidden_dim));
            add_layer(new Sigmoid()); // hi
            add_layer(new Linear(hidden_dim, hidden_dim));
            add_layer(new Sigmoid()); // hf
            add_layer(new Linear(hidden_dim, hidden_dim));
            add_layer(new Tanh()); // hg
            add_layer(new Linear(hidden_dim, hidden_dim));
            add_layer(new Sigmoid()); // ho
        }
    }

    std::string to_string() const
    {
        std::stringstream ss;
        ss << "LSTM(D_in=" << input_dim << ", H=" << hidden_dim
           << ", layers=" << num_layers << ")";
        return ss.str();
    }

    void forward(const std::vector<Tensor> &inputs,
                 std::vector<Tensor> &outputs) const
    {
        const Tensor &input_seq = inputs[0];
        Tensor &output_seq = outputs[0];
        const size_t T = input_seq.shape[0];
        const size_t D = input_seq.shape[1];

        std::vector<Tensor> h(num_layers), c(num_layers);
        for (size_t l = 0; l < num_layers; ++l)
        {
            h[l].shape = {1, hidden_dim};
            h[l].resize();
            std::fill(h[l].data.begin(), h[l].data.end(), 0.0f);
            c[l].shape = {1, hidden_dim};
            c[l].resize();
            std::fill(c[l].data.begin(), c[l].data.end(), 0.0f);
        }

        output_seq.shape = {T, hidden_dim};
        output_seq.resize();

        Tanh tanh_fn;

        Tensor x_t;
        x_t.shape = {1, D};
        x_t.resize();
        Tensor temp1, temp2, i_t, f_t, g_t, o_t;
        temp1.shape = temp2.shape = i_t.shape = f_t.shape = g_t.shape =
            o_t.shape = {hidden_dim};
        temp1.resize();
        temp2.resize();
        i_t.resize();
        f_t.resize();
        g_t.resize();
        o_t.resize();

        for (size_t t = 0; t < T; ++t)
        {
            std::copy(input_seq.data.begin() + t * D,
                      input_seq.data.begin() + (t + 1) * D, x_t.data.begin());

            for (size_t l = 0; l < num_layers; ++l)
            {
                size_t base = l * 16;

                // i
                assert(base + 0 < layers.size());
                layers[base + 0]->forward(x_t, temp1);
                layers[base + 8]->forward(h[l], temp2);
                for (size_t i = 0; i < hidden_dim; ++i)
                    i_t.data[i] = temp1.data[i] + temp2.data[i];
                layers[base + 1]->forward(i_t, i_t);

                // f
                layers[base + 2]->forward(x_t, temp1);
                layers[base + 10]->forward(h[l], temp2);
                for (size_t i = 0; i < hidden_dim; ++i)
                    f_t.data[i] = temp1.data[i] + temp2.data[i];
                layers[base + 3]->forward(f_t, f_t);

                // g
                layers[base + 4]->forward(x_t, temp1);
                layers[base + 12]->forward(h[l], temp2);
                for (size_t i = 0; i < hidden_dim; ++i)
                    g_t.data[i] = temp1.data[i] + temp2.data[i];
                layers[base + 5]->forward(g_t, g_t);

                // o
                layers[base + 6]->forward(x_t, temp1);
                layers[base + 14]->forward(h[l], temp2);
                for (size_t i = 0; i < hidden_dim; ++i)
                    o_t.data[i] = temp1.data[i] + temp2.data[i];
                layers[base + 7]->forward(o_t, o_t);

                // c, h
                for (size_t i = 0; i < hidden_dim; ++i)
                {
                    c[l].data[i] =
                        f_t.data[i] * c[l].data[i] + i_t.data[i] * g_t.data[i];
                }

                tanh_fn.forward(c[l], temp1);
                for (size_t i = 0; i < hidden_dim; ++i)
                {
                    h[l].data[i] = o_t.data[i] * temp1.data[i];
                }

                x_t = h[l];
            }

            std::copy(h.back().data.begin(), h.back().data.end(),
                      output_seq.data.begin() + t * hidden_dim);
        }
    }

    void backward(const std::vector<Tensor> &outputs,
                  std::vector<Tensor> &inputs) const
    {

        const Tensor &output_seq = outputs[0];
        Tensor &input_seq = inputs[0];
        throw std::runtime_error("LSTM::backward() not implemented yet — "
                                 "requires forward pass state caching.");
    }
};

} // namespace ttie

#endif // TTIE_H