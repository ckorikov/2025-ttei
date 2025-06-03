#include "ttie/ttie.h"
#include <cmath>
#include <gtest/gtest.h>
#include <random>

using namespace ttie;

TEST(TensorTest, UninitializedTensor)
{
    Tensor x;
    EXPECT_TRUE(x.shape.empty());
}

TEST(TensorTest, ValidateShape)
{
    Tensor t;
    t.shape = {2, 3};
    EXPECT_TRUE(t.validate_shape());

    t.shape = {0, 3};
    EXPECT_FALSE(t.validate_shape());

    t.shape = {};
    EXPECT_FALSE(t.validate_shape());

    EXPECT_THROW(t.resize(), std::exception);
}

TEST(TensorTest, InitializeAndSetData)
{
    Tensor t;
    t.shape = {2, 3};
    t.resize();

    EXPECT_EQ(t.data.size(), 6);
}

TEST(TensorTest, GradientOperations)
{
    Tensor t;
    t.shape = {2, 3};
    t.resize();
    t.resize_grad();

    EXPECT_EQ(t.grad.size(), 6);

    for (size_t i = 0; i < t.grad.size(); ++i)
    {
        t.grad[i] = static_cast<float>(i);
    }

    t.zero_grad();

    for (size_t i = 0; i < t.grad.size(); ++i)
    {
        EXPECT_FLOAT_EQ(t.grad[i], 0.0f);
    }
}

TEST(LayerTest, ReLU)
{
    ReLU relu;
    EXPECT_EQ(relu.parameters().size(), 0);

    Tensor input;
    input.shape = {2, 3};
    input.resize();
    for (size_t i = 0; i < input.data.size(); ++i)
    {
        input.data[i] = static_cast<float>(i) - 2.0f;
    }

    Tensor output;
    relu.forward(input, output);

    for (size_t i = 0; i < output.data.size(); ++i)
    {
        EXPECT_FLOAT_EQ(output.data[i], std::max(0.0f, input.data[i]));
    }

    output.resize_grad();
    for (size_t i = 0; i < output.grad.size(); ++i)
    {
        output.grad[i] = 1.0f;
    }

    relu.backward(output, input);

    for (size_t i = 0; i < input.grad.size(); ++i)
    {
        EXPECT_FLOAT_EQ(input.grad[i], input.data[i] > 0 ? 1.0f : 0.0f);
    }
}

TEST(LayerTest, Sigmoid)
{
    Sigmoid sigmoid;
    Tensor input;
    input.shape = {2};
    input.data = {0.0f, 1.0f};
    input.resize();

    Tensor output;
    sigmoid.forward(input, output);

    EXPECT_NEAR(output.data[0], 0.5f, 1e-5f);        // σ(0) = 0.5
    EXPECT_NEAR(output.data[1], 0.73105858f, 1e-5f); // σ(1) ≈ 0.731

    output.grad = {1.0f, 1.0f};
    sigmoid.backward(output, input);

    EXPECT_NEAR(input.grad[0], 0.25f, 1e-5f);       // σ'(0) = 0.25
    EXPECT_NEAR(input.grad[1], 0.19661193f, 1e-5f); // σ'(1) ≈ 0.197
}

TEST(LayerTest, Tanh)
{
    Tanh tanh;
    Tensor input;
    input.shape = {2};
    input.data = {0.0f, 1.0f};
    input.resize();

    Tensor output;
    tanh.forward(input, output);

    EXPECT_NEAR(output.data[0], 0.0f, 1e-5f);        // tanh(0) = 0
    EXPECT_NEAR(output.data[1], 0.76159416f, 1e-5f); // tanh(1) ≈ 0.761

    output.grad = {1.0f, 1.0f};
    tanh.backward(output, input);

    EXPECT_NEAR(input.grad[0], 1.0f, 1e-5f);        // tanh'(0) = 1
    EXPECT_NEAR(input.grad[1], 0.41997434f, 1e-5f); // tanh'(1) ≈ 0.420
}

TEST(Softmax, SoftmaxBasic)
{
    Softmax softmax;
    Tensor input;
    input.shape = {3};
    input.data = {1.0f, 2.0f, 3.0f};
    input.resize();

    Tensor output;
    softmax.forward(input, output);

    float sum = output.data[0] + output.data[1] + output.data[2];
    EXPECT_NEAR(sum, 1.0f, 1e-5f); // Сумма вероятностей должна быть равна 1

    EXPECT_GT(output.data[2], output.data[1]);
    EXPECT_GT(output.data[1], output.data[0]);

    output.grad = {1.0f, 0.0f, 0.0f};
    softmax.backward(output, input);

    float grad_sum = input.grad[0] + input.grad[1] + input.grad[2];
    EXPECT_NEAR(grad_sum, 0.0f,
                1e-5f); // сумма градиентов Softmax должна быть 0
}

TEST(Softmax, SoftmaxEqualInputs)
{
    Softmax softmax;
    Tensor input;
    input.shape = {2};
    input.data = {0.0f, 0.0f};
    input.resize();

    Tensor output;
    softmax.forward(input, output);

    EXPECT_NEAR(output.data[0], 0.5f, 1e-5f);
    EXPECT_NEAR(output.data[1], 0.5f, 1e-5f);

    output.grad = {1.0f, -1.0f};
    softmax.backward(output, input);

    EXPECT_NEAR(input.grad[0], 0.5f, 1e-5f);
    EXPECT_NEAR(input.grad[1], -0.5f, 1e-5f);
}

TEST(Softmax, SoftmaxSingleLargeInput)
{
    Softmax softmax;
    Tensor input;
    input.shape = {3};
    input.data = {-1000.0f, 0.0f, 1000.0f}; // Большая разница
    input.resize();

    Tensor output;
    softmax.forward(input, output);

    EXPECT_NEAR(output.data[2], 1.0f,
                1e-5f); // максимальный элемент должен приблизиться к 1
    EXPECT_NEAR(output.data[0], 0.0f, 1e-5f);
    EXPECT_NEAR(output.data[1], 0.0f, 1e-5f);

    output.grad = {0.0f, 0.0f, 1.0f};
    softmax.backward(output, input);

    EXPECT_NEAR(input.grad[2], 0.0f, 1e-5f);
    EXPECT_NEAR(input.grad[0], 0.0f, 1e-5f);
    EXPECT_NEAR(input.grad[1], 0.0f, 1e-5f);
}

TEST(Softmax, StableSoftmaxLSEExtremeValues)
{
    StableSoftmaxLSE softmax;

    Tensor input;
    input.shape = {5};
    input.data = {1000.0f, -1000.0f, 0.0f, 500.0f, -500.0f}; // крайние значения
    input.resize();

    Tensor output;
    softmax.forward(input, output);

    float sum = std::accumulate(output.data.begin(), output.data.end(), 0.0f);
    EXPECT_NEAR(sum, 1.0f, 1e-5f);

    for (float val : output.data)
    {
        EXPECT_TRUE(std::isfinite(val));
        EXPECT_GE(val, 0.0f);
        EXPECT_LE(val, 1.0f);
    }

    // Самая высокая вероятность у первого элемента (1000)
    EXPECT_NEAR(output.data[0], 1.0f, 1e-5f);
    EXPECT_NEAR(output.data[1], 0.0f, 1e-5f);
    EXPECT_NEAR(output.data[2], 0.0f, 1e-5f);
    EXPECT_NEAR(output.data[3], 0.0f, 1e-5f);
    EXPECT_NEAR(output.data[4], 0.0f, 1e-5f);

    output.grad = {1.0f, -1.0f, 0.0f, 0.0f, 0.0f};
    softmax.backward(output, input);

    for (float grad_val : input.grad)
    {
        EXPECT_TRUE(std::isfinite(grad_val));
    }
}

TEST(MaxPool1d, ForwardBasic)
{
    MaxPool1d pool(2, 2);

    Tensor input;
    input.shape = {6};
    input.data = {1, 3, 2, 4, 5, 0};
    input.resize();

    Tensor output;
    pool.forward(input, output);

    EXPECT_EQ(output.shape[0], 3);
    EXPECT_FLOAT_EQ(output.data[0], 3);
    EXPECT_FLOAT_EQ(output.data[1], 4);
    EXPECT_FLOAT_EQ(output.data[2], 5);
}

TEST(MaxPool1d, BackwardBasic)
{
    MaxPool1d pool(3, 1);

    Tensor input;
    input.shape = {5};
    input.data = {1, 5, 2, 3, 4};
    input.resize();

    Tensor output;
    pool.forward(input, output);
    output.grad = {1, 2, 3};

    pool.backward(output, input);

    EXPECT_FLOAT_EQ(input.grad[0], 0);
    EXPECT_FLOAT_EQ(input.grad[1], 3);
    EXPECT_FLOAT_EQ(input.grad[2], 0);
    EXPECT_FLOAT_EQ(input.grad[3], 0);
    EXPECT_FLOAT_EQ(input.grad[4], 3);
}

TEST(QuantizedLinear, ForwardBasic)
{
    QuantizedLinear layer(3, 2);
    layer.init_weights({{0.1f, -0.2f, 0.3f}, {0.4f, 0.5f, -0.6f}},
                       {0.01f, -0.02f});

    Tensor input;
    input.shape = {3};
    input.data = {1.0f, 0.5f, -1.5f};
    input.resize();

    Tensor output;
    layer.forward(input, output);

    ASSERT_EQ(output.shape.size(), 1);
    ASSERT_EQ(output.shape[0], 2);
    EXPECT_TRUE(std::isfinite(output.data[0]));
    EXPECT_TRUE(std::isfinite(output.data[1]));
}

TEST(QuantizedLinear, BackwardBasic)
{
    QuantizedLinear layer(2, 2);
    layer.init_weights({{0.5f, -0.1f}, {-0.3f, 0.7f}}, {0.02f, -0.03f});

    Tensor input;
    input.shape = {2};
    input.data = {0.2f, -0.4f};
    input.resize();

    Tensor output;
    layer.forward(input, output);
    output.grad = {1.0f, -1.0f};
    layer.backward(output, input);

    ASSERT_EQ(input.grad.size(), 2);
    EXPECT_TRUE(std::isfinite(input.grad[0]));
    EXPECT_TRUE(std::isfinite(input.grad[1]));
}
TEST(QuantizedLinear, LargeIntegrationTest)
{
    // Структура: Input -> QuantizedLinear(4->6) -> MaxPool1d(2,2) ->
    // QuantizedLinear(3->2)

    QuantizedLinear layer1(4, 6);
    layer1.init_weights({{0.1f, -0.2f, 0.3f, -0.4f},
                         {0.5f, 0.6f, -0.7f, 0.8f},
                         {-0.9f, 1.0f, -1.1f, 1.2f},
                         {1.3f, -1.4f, 1.5f, -1.6f},
                         {-1.7f, 1.8f, -1.9f, 2.0f},
                         {2.1f, -2.2f, 2.3f, -2.4f}},
                        {0.05f, -0.05f, 0.05f, -0.05f, 0.05f, -0.05f});

    MaxPool1d pool(2, 2);

    QuantizedLinear layer2(3, 2);
    layer2.init_weights({{0.2f, -0.3f, 0.4f}, {-0.5f, 0.6f, -0.7f}},
                        {0.01f, -0.02f});

    Tensor input;
    input.shape = {4};
    input.data = {1.0f, -1.0f, 2.0f, -2.0f};
    input.resize();

    Tensor output1, output2, final_output;

    layer1.forward(input, output1);
    pool.forward(output1, output2);
    layer2.forward(output2, final_output);

    ASSERT_EQ(final_output.shape[0], 2);
    EXPECT_TRUE(std::isfinite(final_output.data[0]));
    EXPECT_TRUE(std::isfinite(final_output.data[1]));

    // Проверка backward pass
    final_output.grad = {1.0f, -1.0f};
    Tensor grad_pool, grad_input;

    layer2.backward(final_output, output2);
    pool.backward(output2, output1);
    layer1.backward(output1, input);

    ASSERT_EQ(input.grad.size(), 4);
    for (float grad : input.grad)
        EXPECT_TRUE(std::isfinite(grad));
}
TEST(LayerTest, Linear)
{
    Linear linear(3, 2);

    auto params = linear.parameters();
    EXPECT_EQ(params.size(), 2);

    EXPECT_EQ(linear.weight.shape.size(), 2);
    EXPECT_EQ(linear.weight.shape[0], 3);
    EXPECT_EQ(linear.weight.shape[1], 2);
    EXPECT_EQ(linear.bias.shape.size(), 1);
    EXPECT_EQ(linear.bias.shape[0], 2);

    Tensor input;
    input.shape = {2, 3};
    input.resize();
    for (size_t i = 0; i < input.data.size(); ++i)
    {
        input.data[i] = static_cast<float>(i) / input.data.size();
    }

    Tensor output;
    linear.forward(input, output);

    EXPECT_EQ(output.shape.size(), 2);
    EXPECT_EQ(output.shape[0], 2);
    EXPECT_EQ(output.shape[1], 2);

    output.resize_grad();
    for (size_t i = 0; i < output.grad.size(); ++i)
    {
        output.grad[i] = 1.0f;
    }

    input.resize_grad();
    input.zero_grad();

    linear.backward(output, input);

    bool has_nonzero = false;
    for (size_t i = 0; i < input.grad.size(); ++i)
    {
        if (input.grad[i] != 0.0f)
        {
            has_nonzero = true;
            break;
        }
    }
    EXPECT_TRUE(has_nonzero);

    has_nonzero = false;
    for (size_t i = 0; i < linear.weight.grad.size(); ++i)
    {
        if (linear.weight.grad[i] != 0.0f)
        {
            has_nonzero = true;
            break;
        }
    }
    EXPECT_TRUE(has_nonzero);

    has_nonzero = false;
    for (size_t i = 0; i < linear.bias.grad.size(); ++i)
    {
        if (linear.bias.grad[i] != 0.0f)
        {
            has_nonzero = true;
            break;
        }
    }
    EXPECT_TRUE(has_nonzero);
}

TEST(LayerTest, LinearVSTorch)
{
    /* PyTorch reference forward and backward with predefined weights and biases
    import torch

    input = torch.tensor([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]], requires_grad=True)
    weight = torch.tensor([[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]],
    requires_grad=True) bias = torch.tensor([0.1, 0.2], requires_grad=True)

    output = torch.addmm(bias, input, weight)
    output.backward(torch.ones_like(output))

    # Print results
    print("PyTorch Output:", output)
    print("PyTorch Output Grad:", torch.ones_like(output))
    print("PyTorch Input Grad:", input.grad)
    print("PyTorch Weight Grad:", weight.grad)
    print("PyTorch Bias Grad:", bias.grad)

    # Output:
    PyTorch Output: tensor([[0.3200, 0.4800],
        [0.5900, 0.8400]], grad_fn=<AddmmBackward0>)
    PyTorch Output Grad: tensor([[1., 1.],
            [1., 1.]])
    PyTorch Input Grad: tensor([[0.3000, 0.7000, 1.1000],
            [0.3000, 0.7000, 1.1000]])
    PyTorch Weight Grad: tensor([[0.5000, 0.5000],
            [0.7000, 0.7000],
            [0.9000, 0.9000]])
    PyTorch Bias Grad: tensor([2., 2.])
    */

    Linear linear(3, 2);
    linear.weight.data = {0.1f, 0.2f, 0.3f, 0.4f, 0.5f, 0.6f};
    linear.bias.data = {0.1f, 0.2f};

    Tensor input;
    input.shape = {2, 3};
    input.resize();
    input.data = {0.1f, 0.2f, 0.3f, 0.4f, 0.5f, 0.6f};
    input.resize_grad();

    // Forward pass
    Tensor output;
    linear.forward(input, output);

    // Check output values match PyTorch
    ASSERT_EQ(output.shape.size(), 2);
    ASSERT_EQ(output.shape[0], 2);
    ASSERT_EQ(output.shape[1], 2);
    EXPECT_NEAR(output.data[0], 0.32f, 1e-5f);
    EXPECT_NEAR(output.data[1], 0.48f, 1e-5f);
    EXPECT_NEAR(output.data[2], 0.59f, 1e-5f);
    EXPECT_NEAR(output.data[3], 0.84f, 1e-5f);

    // Backward pass
    output.resize_grad();
    output.grad = {1.0f, 1.0f, 1.0f, 1.0f};

    linear.backward(output, input);

    // Check input gradients match PyTorch
    ASSERT_EQ(input.grad.size(), 6);
    EXPECT_NEAR(input.grad[0], 0.3f, 1e-5f);
    EXPECT_NEAR(input.grad[1], 0.7f, 1e-5f);
    EXPECT_NEAR(input.grad[2], 1.1f, 1e-5f);
    EXPECT_NEAR(input.grad[3], 0.3f, 1e-5f);
    EXPECT_NEAR(input.grad[4], 0.7f, 1e-5f);
    EXPECT_NEAR(input.grad[5], 1.1f, 1e-5f);

    // Check weight gradients match PyTorch
    ASSERT_EQ(linear.weight.grad.size(), 6);
    EXPECT_NEAR(linear.weight.grad[0], 0.5f, 1e-5f);
    EXPECT_NEAR(linear.weight.grad[1], 0.5f, 1e-5f);
    EXPECT_NEAR(linear.weight.grad[2], 0.7f, 1e-5f);
    EXPECT_NEAR(linear.weight.grad[3], 0.7f, 1e-5f);
    EXPECT_NEAR(linear.weight.grad[4], 0.9f, 1e-5f);
    EXPECT_NEAR(linear.weight.grad[5], 0.9f, 1e-5f);

    // Check bias gradients match PyTorch
    ASSERT_EQ(linear.bias.grad.size(), 2);
    EXPECT_NEAR(linear.bias.grad[0], 2.0f, 1e-5f);
    EXPECT_NEAR(linear.bias.grad[1], 2.0f, 1e-5f);
}

TEST(ModelTest, ForwardAndBackwardVSTorch)
{
    /* Pytorch reference
    import torch

    input = torch.tensor([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]], requires_grad=True)
    weight = torch.tensor([[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]],
    requires_grad=True) bias = torch.tensor([0.1, 0.2], requires_grad=True)

    # First addmm operation
    output1 = torch.addmm(bias, input, weight)
    relu_output = torch.relu(output1)

    # Define weights and bias for the second addmm (2->1)
    weight2 = torch.tensor([[0.7], [0.8]], requires_grad=True)
    bias2 = torch.tensor([0.3], requires_grad=True)

    output = torch.addmm(bias2, relu_output, weight2)
    output.backward(torch.ones_like(output))

    # Print results
    print("PyTorch First Output:", output1)
    print("PyTorch ReLU Output:", relu_output)
    print("PyTorch Final Output:", output)
    print("PyTorch Output Grad:", torch.ones_like(output))
    print("PyTorch Input Grad:", input.grad)
    print("PyTorch Weight Grad:", weight.grad)
    print("PyTorch Bias Grad:", bias.grad)
    print("PyTorch Weight2 Grad:", weight2.grad)
    print("PyTorch Bias2 Grad:", bias2.grad)

    Output:
    PyTorch First Output: tensor([[0.3200, 0.4800],
        [0.5900, 0.8400]], grad_fn=<AddmmBackward0>)
    PyTorch ReLU Output: tensor([[0.3200, 0.4800],
            [0.5900, 0.8400]], grad_fn=<ReluBackward0>)
    PyTorch Final Output: tensor([[0.9080],
            [1.3850]], grad_fn=<AddmmBackward0>)
    PyTorch Output Grad: tensor([[1.],
            [1.]])
    PyTorch Input Grad: tensor([[0.2300, 0.5300, 0.8300],
            [0.2300, 0.5300, 0.8300]])
    PyTorch Weight Grad: tensor([[0.3500, 0.4000],
            [0.4900, 0.5600],
            [0.6300, 0.7200]])
    PyTorch Bias Grad: tensor([1.4000, 1.6000])
    PyTorch Weight2 Grad: tensor([[0.9100],
            [1.3200]])
    PyTorch Bias2 Grad: tensor([2.])
    */

    Model model;
    model.add_layer(new Linear(3, 2));
    model.add_layer(new ReLU());
    model.add_layer(new Linear(2, 1));

    // Initialize with the same values as in PyTorch reference
    Linear *layer1 = static_cast<Linear *>(model.layers[0]);
    layer1->weight.data = {0.1f, 0.2f, 0.3f, 0.4f, 0.5f, 0.6f};
    layer1->bias.data = {0.1f, 0.2f};

    Linear *layer2 = static_cast<Linear *>(model.layers[2]);
    layer2->weight.data = {0.7f, 0.8f};
    layer2->bias.data = {0.3f};

    // Prepare input
    Tensor input;
    input.shape = {2, 3};
    input.resize();
    input.data = {0.1f, 0.2f, 0.3f, 0.4f, 0.5f, 0.6f};
    input.resize_grad();

    // Forward pass
    Tensor output;
    model.forward(input, output);

    // Verify final output shape and values
    ASSERT_EQ(output.shape.size(), 2);
    ASSERT_EQ(output.shape[0], 2);
    ASSERT_EQ(output.shape[1], 1);
    EXPECT_NEAR(output.data[0], 0.9080f, 1e-4f);
    EXPECT_NEAR(output.data[1], 1.3850f, 1e-4f);

    // Backward pass
    output.resize_grad();
    output.grad = {1.0f, 1.0f};

    model.backward(output, input);

    // Check input gradients
    ASSERT_EQ(input.grad.size(), 6);
    EXPECT_NEAR(input.grad[0], 0.2300f, 1e-4f);
    EXPECT_NEAR(input.grad[1], 0.5300f, 1e-4f);
    EXPECT_NEAR(input.grad[2], 0.8300f, 1e-4f);
    EXPECT_NEAR(input.grad[3], 0.2300f, 1e-4f);
    EXPECT_NEAR(input.grad[4], 0.5300f, 1e-4f);
    EXPECT_NEAR(input.grad[5], 0.8300f, 1e-4f);

    // Check layer1 weight gradients
    ASSERT_EQ(layer1->weight.grad.size(), 6);
    EXPECT_NEAR(layer1->weight.grad[0], 0.3500f, 1e-4f);
    EXPECT_NEAR(layer1->weight.grad[1], 0.4000f, 1e-4f);
    EXPECT_NEAR(layer1->weight.grad[2], 0.4900f, 1e-4f);
    EXPECT_NEAR(layer1->weight.grad[3], 0.5600f, 1e-4f);
    EXPECT_NEAR(layer1->weight.grad[4], 0.6300f, 1e-4f);
    EXPECT_NEAR(layer1->weight.grad[5], 0.7200f, 1e-4f);

    // Check layer1 bias gradients
    ASSERT_EQ(layer1->bias.grad.size(), 2);
    EXPECT_NEAR(layer1->bias.grad[0], 1.4000f, 1e-4f);
    EXPECT_NEAR(layer1->bias.grad[1], 1.6000f, 1e-4f);

    // Check layer2 weight gradients
    ASSERT_EQ(layer2->weight.grad.size(), 2);
    EXPECT_NEAR(layer2->weight.grad[0], 0.9100f, 1e-4f);
    EXPECT_NEAR(layer2->weight.grad[1], 1.3200f, 1e-4f);

    // Check layer2 bias gradients
    ASSERT_EQ(layer2->bias.grad.size(), 1);
    EXPECT_NEAR(layer2->bias.grad[0], 2.0000f, 1e-4f);
}

TEST(LSTMTest, Initialization) // Validation of LSTM
{
    LSTM lstm(3, 4); // input_size = 3, hidden_size = 4

    auto params = lstm.parameters();
    EXPECT_EQ(params.size(), 16); // 2 x (4 weight matrices + 4 bias vectors)

    for (Tensor *t : params)
    {
        EXPECT_TRUE(t->validate_shape());
    }
}

TEST(LSTMTest, ForwardShapeAndSanity) // Shape of tensors and sanity of outputs
{
    // batch_size = 1 as default
    const size_t seq_len = 1;
    const size_t input_size = 3;
    const size_t hidden_size = 4;

    LSTM lstm(input_size, hidden_size);

    Tensor input_seq;
    input_seq.shape = {seq_len, input_size};
    input_seq.resize();

    for (size_t i = 0; i < input_seq.data.size(); ++i)
    {
        input_seq.data[i] = static_cast<float>(i) / 10.0f;
    }

    Tensor output_seq;
    output_seq.shape = {seq_len, hidden_size};
    output_seq.resize();

    std::vector<Tensor> inputs = {input_seq};
    std::vector<Tensor> outputs = {output_seq};

    lstm.forward(input_seq, output_seq);

    EXPECT_EQ(outputs[0].shape[0], seq_len);
    EXPECT_EQ(outputs[0].shape[1], hidden_size);

    for (float v : outputs[0].data)
    {
        EXPECT_TRUE(std::isfinite(v));
    }
}

TEST(LSTMTest,
     ForwardAndBackwardVSTorch) // LSTM forward and backward propagation tests
{
    /*
    # 1 Layer
    import torch
    import torch.nn as nn

    seq_len = 2
    input_dim = 2
    hidden_dim = 2
    batch_size = 1
    num_layers = 1

    x = torch.tensor([
        [0.5, 1.0],
        [1.0, 1.5]
    ], dtype=torch.float32).unsqueeze(1)
    x.requires_grad_()

    lstm = nn.LSTM(input_dim, hidden_dim, num_layers=num_layers,
    batch_first=False)

    weight_ih = torch.tensor([
        [0.5, 0.6],
        [0.7, 0.8],
        [0.9, 1.0],
        [1.1, 1.2],
        [0.5, 0.6],
        [0.7, 0.8],
        [0.9, 1.0],
        [1.1, 1.2]
    ])

    weight_hh = torch.tensor([
        [0.2, 0.3],
        [0.3, 0.4],
        [0.4, 0.5],
        [0.5, 0.6],
        [0.2, 0.3],
        [0.3, 0.4],
        [0.4, 0.5],
        [0.5, 0.6]
    ])

    bias_ih = torch.ones(8) * 0.5
    bias_hh = torch.ones(8) * 0.5

    with torch.no_grad():
        lstm.weight_ih_l0.copy_(weight_ih)
        lstm.weight_hh_l0.copy_(weight_hh)
        lstm.bias_ih_l0.copy_(bias_ih)
        lstm.bias_hh_l0.copy_(bias_hh)

    x.requires_grad_()

    output, (hn, cn) = lstm(x)

    grad_output = torch.zeros_like(output)
    grad_output[1, 0] = torch.tensor([1.0, 1.0])  # Только по h₁
    grad_output[0, 0] = torch.tensor([1.0, 1.0])
    output.backward(grad_output)

    print("Input (x):")
    print(x.squeeze(1))

    print("\n--- LSTM Parameters ---")
    print("weight_ih_l0:")
    print(lstm.weight_ih_l0.data)

    print("\nweight_hh_l0:")
    print(lstm.weight_hh_l0.data)

    print("\nbias_ih_l0:")
    print(lstm.bias_ih_l0.data)

    print("\nbias_hh_l0:")
    print(lstm.bias_hh_l0.data)

    print("\nInitial h_0:")
    print(torch.zeros(1, batch_size, hidden_dim))  # explicitly showing default

    print("\nInitial c_0:")
    print(torch.zeros(1, batch_size, hidden_dim))  # explicitly showing default

    print("\n--- LSTM Outputs ---")
    print("Output:")
    print(output.squeeze(1))

    print("\nHidden state hn:")
    print(hn.squeeze(0))

    print("\nCell state cn:")
    print(cn.squeeze(0))

    print("\nWeight gradients:")
    print("weight_ih_l0.grad:")
    print(lstm.weight_ih_l0.grad)
    print("\nweight_hh_l0.grad:")
    print(lstm.weight_hh_l0.grad)

    print("\nBias gradients:")
    print("bias_ih_l0.grad:")
    print(lstm.bias_ih_l0.grad)
    print("\nbias_hh_l0.grad:")
    print(lstm.bias_hh_l0.grad)

    print("\nGradient of input x:")
    print(x.grad.squeeze(1))

    Input (x):
    tensor([[0.5000, 1.0000],
            [1.0000, 1.5000]], grad_fn=<SqueezeBackward1>)

    --- LSTM Parameters ---
    weight_ih_l0:
    tensor([[0.5000, 0.6000],
            [0.7000, 0.8000],
            [0.9000, 1.0000],
            [1.1000, 1.2000],
            [0.5000, 0.6000],
            [0.7000, 0.8000],
            [0.9000, 1.0000],
            [1.1000, 1.2000]])

    weight_hh_l0:
    tensor([[0.2000, 0.3000],
            [0.3000, 0.4000],
            [0.4000, 0.5000],
            [0.5000, 0.6000],
            [0.2000, 0.3000],
            [0.3000, 0.4000],
            [0.4000, 0.5000],
            [0.5000, 0.6000]])

    bias_ih_l0:
    tensor([0.5000, 0.5000, 0.5000, 0.5000, 0.5000, 0.5000, 0.5000, 0.5000])

    bias_hh_l0:
    tensor([0.5000, 0.5000, 0.5000, 0.5000, 0.5000, 0.5000, 0.5000, 0.5000])

    Initial h_0:
    tensor([[[0., 0.]]])

    Initial c_0:
    tensor([[[0., 0.]]])

    --- LSTM Outputs ---
    Output:
    tensor([[0.6227, 0.6600],
            [0.9227, 0.9401]], grad_fn=<SqueezeBackward1>)

    Hidden state hn:
    tensor([[0.9227, 0.9401]], grad_fn=<SqueezeBackward1>)

    Cell state cn:
    tensor([[1.7376, 1.8268]], grad_fn=<SqueezeBackward1>)

    Weight gradients:
    weight_ih_l0.grad:
    tensor([[0.0412, 0.0790],
            [0.0297, 0.0577],
            [0.0017, 0.0025],
            [0.0008, 0.0012],
            [0.0270, 0.0532],
            [0.0142, 0.0282],
            [0.0420, 0.0756],
            [0.0295, 0.0544]])

    weight_hh_l0.grad:
    tensor([[0.0041, 0.0043],
            [0.0020, 0.0021],
            [0.0011, 0.0011],
            [0.0005, 0.0005],
            [0.0011, 0.0012],
            [0.0003, 0.0003],
            [0.0105, 0.0112],
            [0.0058, 0.0061]])

    Bias gradients:
    bias_ih_l0.grad:
    tensor([0.0758, 0.0561, 0.0017, 0.0008, 0.0522, 0.0280, 0.0672, 0.0497])

    bias_hh_l0.grad:
    tensor([0.0758, 0.0561, 0.0017, 0.0008, 0.0522, 0.0280, 0.0672, 0.0497])

    Gradient of input x:
    tensor([[0.2058, 0.2349],
            [0.0347, 0.0387]])
    */

    const size_t seq_len = 2;
    const size_t input_size = 2;
    const size_t hidden_size = 2;
    const size_t num_layers = 1;

    LSTM lstm(input_size, hidden_size, num_layers); // batch_size = 1 as default

    // input and output filling
    Tensor input;
    input.shape = {seq_len, input_size};
    input.resize();

    input.data[0] = 0.5f;
    input.data[1] = 1.0f;
    input.data[2] = 1.0f;
    input.data[3] = 1.5f;

    Tensor output;
    output.shape = {seq_len, hidden_size};
    output.resize();

    // Layers initialization
    Linear *layer_ii = static_cast<Linear *>(lstm.layers[0]);
    layer_ii->weight.data = {0.5f, 0.7f, 0.6f, 0.8f};
    layer_ii->bias.data = {0.5f, 0.5f};

    Linear *layer_if = static_cast<Linear *>(lstm.layers[2]);
    layer_if->weight.data = {0.9f, 1.1f, 1.0f, 1.2f};
    layer_if->bias.data = {0.5f, 0.5f};

    Linear *layer_ig = static_cast<Linear *>(lstm.layers[4]);
    layer_ig->weight.data = {0.5f, 0.7f, 0.6f, 0.8f};
    layer_ig->bias.data = {0.5f, 0.5f};

    Linear *layer_io = static_cast<Linear *>(lstm.layers[6]);
    layer_io->weight.data = {0.9f, 1.1f, 1.0f, 1.2f};
    layer_io->bias.data = {0.5f, 0.5f};

    Linear *layer_hi = static_cast<Linear *>(lstm.layers[8]);
    layer_hi->weight.data = {0.2f, 0.3f, 0.3f, 0.4f};
    layer_hi->bias.data = {0.5f, 0.5f};

    Linear *layer_hf = static_cast<Linear *>(lstm.layers[10]);
    layer_hf->weight.data = {0.4f, 0.5f, 0.5f, 0.6f};
    layer_hf->bias.data = {0.5f, 0.5f};

    Linear *layer_hg = static_cast<Linear *>(lstm.layers[12]);
    layer_hg->weight.data = {0.2f, 0.3f, 0.3f, 0.4f};
    layer_hg->bias.data = {0.5f, 0.5f};

    Linear *layer_ho = static_cast<Linear *>(lstm.layers[14]);
    layer_ho->weight.data = {0.4f, 0.5f, 0.5f, 0.6f};
    layer_ho->bias.data = {0.5f, 0.5f};

    // forward checking
    lstm.forward(input, output);
    std::vector<float> torch_output = {0.6227f, 0.6600f, 0.9227f, 0.9401f};

    ASSERT_EQ(output.data.size(), torch_output.size());
    for (size_t i = 0; i < torch_output.size(); ++i)
    {
        EXPECT_NEAR(output.data[i], torch_output[i], 1e-4f);
    }

    // Gradients filling
    output.resize_grad();
    output.grad = {1.0f, 1.0f, 1.0f, 1.0f};
    input.resize_grad();
    input.zero_grad();

    // backward checking
    lstm.backward(output, input);
    std::vector<float> torch_input_grad = {0.2058f, 0.2349f, 0.0347f, 0.0387f};

    // Validation of gradients for input and all layers
    for (size_t i = 0; i < torch_input_grad.size(); ++i)
    {
        EXPECT_NEAR(input.grad[i], torch_input_grad[i], 1e-4f);
    }

    std::vector<float> torch_grad_weight_ii = {0.0412f, 0.0297f, 0.0790f,
                                               0.0577f};
    std::vector<float> torch_grad_weight_if = {0.0017f, 0.0008f, 0.0025f,
                                               0.0012f};
    std::vector<float> torch_grad_weight_ig = {0.0270f, 0.0142f, 0.0532f,
                                               0.0282f};
    std::vector<float> torch_grad_weight_io = {0.0420f, 0.0295f, 0.0756f,
                                               0.0544f};

    std::vector<float> torch_grad_weight_hi = {0.0041f, 0.0020f, 0.0043f,
                                               0.0021f};
    std::vector<float> torch_grad_weight_hf = {0.0011f, 0.0005f, 0.0011f,
                                               0.0005f};
    std::vector<float> torch_grad_weight_hg = {0.0011f, 0.0003f, 0.0012f,
                                               0.0003f};
    std::vector<float> torch_grad_weight_ho = {0.0105f, 0.0058f, 0.0112f,
                                               0.0061f};

    ASSERT_EQ(layer_ii->weight.grad.size(), torch_grad_weight_ii.size());
    ASSERT_EQ(layer_if->weight.grad.size(), torch_grad_weight_if.size());
    ASSERT_EQ(layer_ig->weight.grad.size(), torch_grad_weight_ig.size());
    ASSERT_EQ(layer_io->weight.grad.size(), torch_grad_weight_io.size());

    ASSERT_EQ(layer_hi->weight.grad.size(), torch_grad_weight_hi.size());
    ASSERT_EQ(layer_hf->weight.grad.size(), torch_grad_weight_hf.size());
    ASSERT_EQ(layer_hg->weight.grad.size(), torch_grad_weight_hg.size());
    ASSERT_EQ(layer_ho->weight.grad.size(), torch_grad_weight_ho.size());

    for (size_t i = 0; i < torch_grad_weight_ii.size(); ++i)
    {
        EXPECT_NEAR(layer_ii->weight.grad[i], torch_grad_weight_ii[i], 1e-4f);
        EXPECT_NEAR(layer_if->weight.grad[i], torch_grad_weight_if[i], 1e-4f);
        EXPECT_NEAR(layer_ig->weight.grad[i], torch_grad_weight_ig[i], 1e-4f);
        EXPECT_NEAR(layer_io->weight.grad[i], torch_grad_weight_io[i], 1e-4f);

        EXPECT_NEAR(layer_hi->weight.grad[i], torch_grad_weight_hi[i], 1e-4f);
        EXPECT_NEAR(layer_hf->weight.grad[i], torch_grad_weight_hf[i], 1e-4f);
        EXPECT_NEAR(layer_hg->weight.grad[i], torch_grad_weight_hg[i], 1e-4f);
        EXPECT_NEAR(layer_ho->weight.grad[i], torch_grad_weight_ho[i], 1e-4f);
    }

    std::vector<float> torch_grad_bias_ii = {0.0758f, 0.0561f};
    std::vector<float> torch_grad_bias_if = {0.0017f, 0.0008f};
    std::vector<float> torch_grad_bias_ig = {0.0522f, 0.0280f};
    std::vector<float> torch_grad_bias_io = {0.0672f, 0.0497f};

    std::vector<float> torch_grad_bias_hi = {0.0758f, 0.0561f};
    std::vector<float> torch_grad_bias_hf = {0.0017f, 0.0008f};
    std::vector<float> torch_grad_bias_hg = {0.0522f, 0.0280f};
    std::vector<float> torch_grad_bias_ho = {0.0672f, 0.0497f};

    ASSERT_EQ(layer_ii->bias.grad.size(), torch_grad_bias_ii.size());
    for (size_t i = 0; i < torch_grad_bias_ii.size(); ++i)
    {
        EXPECT_NEAR(layer_ii->bias.grad[i], torch_grad_bias_ii[i], 1e-4f);
        EXPECT_NEAR(layer_if->bias.grad[i], torch_grad_bias_if[i], 1e-4f);
        EXPECT_NEAR(layer_ig->bias.grad[i], torch_grad_bias_ig[i], 1e-4f);
        EXPECT_NEAR(layer_io->bias.grad[i], torch_grad_bias_io[i], 1e-4f);

        EXPECT_NEAR(layer_hi->bias.grad[i], torch_grad_bias_hi[i], 1e-4f);
        EXPECT_NEAR(layer_hf->bias.grad[i], torch_grad_bias_hf[i], 1e-4f);
        EXPECT_NEAR(layer_hg->bias.grad[i], torch_grad_bias_hg[i], 1e-4f);
        EXPECT_NEAR(layer_ho->bias.grad[i], torch_grad_bias_ho[i], 1e-4f);
    }
}

TEST(TensorTransposeTest, BasicTransposition)
{
    Tensor t;
    t.shape = {2, 3};
    t.data = {1, 2, 3, 4, 5, 6};

    Tensor result = t.transpose(0, 1);

    EXPECT_EQ(result.shape, std::vector<size_t>({3, 2}));
    EXPECT_EQ(result.data, std::vector<float>({1, 4, 2, 5, 3, 6}));
}

TEST(TensorTransposeTest, IdentityTransposition)
{
    Tensor t;
    t.shape = {2, 3, 4};
    t.resize();
    std::iota(t.data.begin(), t.data.end(), 1);

    Tensor result = t.transpose(1, 1);

    EXPECT_EQ(result.shape, t.shape);
    EXPECT_EQ(result.data, t.data);
}

TEST(TensorTransposeTest, HigherDimTransposition)
{
    /*PyTorch refernce
    import torch

    tensor = torch.arange(1, 25).reshape(2, 3, 4)

    transposed_tensor = tensor.transpose(0, 2)

    print(transposed_tensor.flatten()[0])
    print(transposed_tensor.flatten()[1])
    print(transposed_tensor.flatten()[4])
    print(transposed_tensor.flatten()[23])
    Output:
    tensor(1)
    tensor(13)
    tensor(9)
    tensor(24)
*/

    Tensor t;
    t.shape = {2, 3, 4};
    t.resize();
    std::iota(t.data.begin(), t.data.end(), 1);

    Tensor result = t.transpose(0, 2);

    EXPECT_EQ(result.shape, std::vector<size_t>({4, 3, 2}));

    EXPECT_FLOAT_EQ(result.data[0], 1);
    EXPECT_FLOAT_EQ(result.data[1], 13);
    EXPECT_FLOAT_EQ(result.data[4], 9);
    EXPECT_FLOAT_EQ(result.data[23], 24);
}

TEST(TensorTransposeTest, InvalidDimensions)
{
    Tensor t;
    t.shape = {2, 3};

    EXPECT_THROW(t.transpose(0, 5), std::invalid_argument);
    EXPECT_THROW(t.transpose(5, 0), std::invalid_argument);
    EXPECT_THROW(t.transpose(5, 5), std::invalid_argument);
}

TEST(TensorTransposeTest, EmptyTensor)
{
    Tensor t;

    EXPECT_THROW(t.transpose(0, 1), std::invalid_argument);
}

TEST(TensorTransposeTest, DataIntegrity)
{
    Tensor t;
    t.shape = {3, 3};
    t.data = {1, 2, 3, 4, 5, 6, 7, 8, 9};

    Tensor t1 = t.transpose(0, 1);
    Tensor t2 = t1.transpose(0, 1);

    EXPECT_EQ(t2.shape, t.shape);
    EXPECT_EQ(t2.data, t.data);
}

TEST(TensorTransposeTest, NoChangeTensor)
{
    Tensor t;
    t.shape = {2, 3};
    t.data = {1, 2, 3, 4, 5, 6};

    Tensor t1 = t.transpose(0, 1);
    std::vector<size_t> test = {2, 3};
    EXPECT_EQ(test, t.shape);
}

TEST(TensorMatMulTest, SimpleMatrixMultiplication)
{
    Tensor a;
    a.shape = {2, 3};
    a.data = {1, 2, 3, 4, 5, 6};

    Tensor b;
    b.shape = {3, 2};
    b.data = {7, 8, 9, 10, 11, 12};

    Tensor expected;
    expected.shape = {2, 2};
    expected.data = {58, 64, 139, 154};

    Tensor result;
    result = matmul(a, b);
    EXPECT_EQ(result.shape, expected.shape);
    for (size_t i = 0; i < expected.data.size(); ++i)
    {
        EXPECT_FLOAT_EQ(result.data[i], expected.data[i]);
    }
}

TEST(TensorMatMulTest, BatchMatrixMultiplication)
{
    /*PyTorch refernce
    import torch

    tensor1 = torch.arange(1, 13).reshape(2, 2, 3)
    tensor2 = torch.arange(1, 13).reshape(2, 3, 2)

    print(tensor1 @ tensor2)

    Output:
    tensor([[[ 22,  28],
            [ 49,  64]],

            [[220, 244],
            [301, 334]]])
*/
    Tensor a;
    a.shape = {2, 2, 3};
    a.data = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12};

    Tensor b;
    b.shape = {2, 3, 2};
    b.data = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12};

    Tensor expected;
    expected.shape = {2, 2, 2};
    expected.data = {22, 28, 49, 64, 220, 244, 301, 334};

    Tensor result;
    result = matmul(a, b);
    ASSERT_EQ(result.shape, expected.shape);
    for (size_t i = 0; i < expected.data.size(); ++i)
    {
        EXPECT_FLOAT_EQ(result.data[i], expected.data[i]);
    }
}

TEST(TensorMatMulTest, VectorMatrixMultiplication)
{
    Tensor a;
    a.shape = {1, 3};
    a.data = {1, 2, 3};

    Tensor b;
    b.shape = {3, 2};
    b.data = {4, 5, 6, 7, 8, 9};

    Tensor expected;
    expected.shape = {1, 2};
    expected.data = {40, 46};

    Tensor result;
    result = matmul(a, b);
    EXPECT_EQ(result.shape, expected.shape);
    EXPECT_FLOAT_EQ(result.data[0], expected.data[0]);
    EXPECT_FLOAT_EQ(result.data[1], expected.data[1]);
}

TEST(TensorMatMulTest, DimensionMismatchError)
{
    Tensor a;
    a.shape = {2, 3};
    Tensor b;
    b.shape = {4, 5};

    EXPECT_THROW(matmul(a, b), std::invalid_argument);
}

TEST(TensorMatMulTest, NotEnoughDimensionsError)
{
    Tensor a;
    a.shape = {3};
    Tensor b;
    b.shape = {3};

    EXPECT_THROW(matmul(a, b), std::invalid_argument);
}

TEST(TensorMatMulTest, BatchSizeMismatchError)
{
    Tensor a;
    a.shape = {2, 2, 3};
    Tensor b;
    b.shape = {3, 3, 3};

    EXPECT_THROW(matmul(a, b), std::invalid_argument);
}

TEST(TensorMatMulTest, FloatingPointPrecision)
{
    Tensor a;
    a.shape = {2, 2};
    a.data = {0.1f, 0.2f, 0.3f, 0.4f};

    Tensor b;
    b.shape = {2, 2};
    b.data = {0.5f, 0.6f, 0.7f, 0.8f};

    Tensor result = matmul(a, b);

    EXPECT_NEAR(result.data[0], 0.19f, 1e-6f);
    EXPECT_NEAR(result.data[1], 0.22f, 1e-6f);
    EXPECT_NEAR(result.data[2], 0.43f, 1e-6f);
    EXPECT_NEAR(result.data[3], 0.5f, 1e-6f);
}

TEST(TensorViewTest, BasicReshape)
{
    Tensor t;
    t.shape = {2, 3};
    t.data = {1, 2, 3, 4, 5, 6};

    Tensor result = t.view({3, 2});

    EXPECT_EQ(result.shape, std::vector<size_t>({3, 2}));

    EXPECT_EQ(result.data, t.data);
}

TEST(TensorViewTest, FlattenToVector)
{
    Tensor t;
    t.shape = {2, 2, 2};
    t.resize();
    std::iota(t.data.begin(), t.data.end(), 1);

    Tensor flat = t.view({8});

    EXPECT_EQ(flat.shape, std::vector<size_t>({8}));
    EXPECT_EQ(flat.data.size(), 8);
    EXPECT_EQ(flat.data, t.data);
}

TEST(TensorViewTest, AddDimensions)
{
    Tensor t;
    t.shape = {6};
    t.data = {1, 2, 3, 4, 5, 6};

    Tensor result = t.view({1, 3, 2});

    EXPECT_EQ(result.shape, std::vector<size_t>({1, 3, 2}));
    EXPECT_EQ(result.data, t.data);
}

TEST(TensorViewTest, InvalidTotalSize)
{
    Tensor t;
    t.shape = {4, 5};
    t.resize();

    EXPECT_THROW(t.view({3, 7}), std::invalid_argument);
}

TEST(TensorViewTest, ZeroDimension)
{
    Tensor t;
    t.shape = {4, 5};
    t.resize();

    EXPECT_THROW(t.view({0, 20}), std::invalid_argument);
}

TEST(ScaledDotProductAttentionTest, BasicForward)
{
    ScaledDotProductAttention attn;
    Tensor q, k, v, out;

    q.shape = {1, 1, 1};
    q.data = {1.0f};
    k.shape = {1, 1, 1};
    k.data = {1.0f};
    v.shape = {1, 1, 1};
    v.data = {1.0f};

    attn.forward(q, k, v, out);

    EXPECT_EQ(out.shape, v.shape);
    EXPECT_NEAR(out.data[0], 1.0f, 1e-6f);
}

TEST(ScaledDotProductAttentionTest, SmallMatrix)
{
    /*PyTorch refernce
    import torch.nn as nn
    import torch
    import math

    class ScaledDotProductAttention(nn.Module):
        def __init__(self, dropout=0.1):
            super(ScaledDotProductAttention, self).__init__()
            self.dropout = nn.Dropout(dropout)

        def forward(self, q, k, v, mask=None):
            d_k = q.size(-1)
            attn_logits = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(d_k)

            if mask is not None:
                attn_logits = attn_logits.masked_fill(mask==0, -1e9)

            attention = torch.softmax(attn_logits, dim=-1)
            # attention = self.dropout(attention)
            values = torch.matmul(attention, v)

            return values, attention


    q = torch.tensor([1, 0, 1, 0, 1, 0], dtype=torch.float).reshape(1, 2, 3)
    k = torch.tensor([1, 1, 0, 0, 1, 1], dtype=torch.float).reshape(1, 2, 3)
    v = torch.tensor([1, 2, 3, 4, 5, 6], dtype=torch.float).reshape(1, 2, 3)

    s = ScaledDotProductAttention()

    out, _ = s.forward(q, k, v)
    print(out)

    Output:
    tensor([[[2.5000, 3.5000, 4.5000],
            [2.5000, 3.5000, 4.5000]]])
*/
    ScaledDotProductAttention attn;
    Tensor q, k, v, out;

    q.shape = {1, 2, 3};
    q.data = {1, 0, 1, 0, 1, 0};
    k.shape = {1, 2, 3};
    k.data = {1, 1, 0, 0, 1, 1};
    v.shape = {1, 2, 3};
    v.data = {1, 2, 3, 4, 5, 6};

    attn.forward(q, k, v, out);

    EXPECT_EQ(out.shape, std::vector<size_t>({1, 2, 3}));

    EXPECT_NEAR(out.data[0], 2.5f, 1e-6f);
    EXPECT_NEAR(out.data[2], 4.5f, 1e-6f);
    EXPECT_NEAR(out.data[4], 3.5f, 1e-6f);
}

TEST(ScaledDotProductAttentionTest, BatchProcessing)
{
    ScaledDotProductAttention attn;
    Tensor q, k, v, out;

    q.shape = {2, 2, 3};
    q.data = {1, 0, 1, 0, 1, 0, 1, 1, 0, 0, 0, 1};
    k.shape = {2, 2, 3};
    k.data = {1, 1, 0, 0, 1, 1, 1, 0, 1, 0, 1, 0};
    v.shape = {2, 2, 3};
    v.data = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12};

    attn.forward(q, k, v, out);

    EXPECT_EQ(out.shape, std::vector<size_t>({2, 2, 3}));

    EXPECT_EQ(out.data.size(), 12);
}

TEST(ScaledDotProductAttentionTest, DimensionMismatch)
{
    ScaledDotProductAttention attn;
    Tensor q, k, v, out;

    q.shape = {1, 2, 3};
    q.resize();
    k.shape = {1, 4, 3};
    k.resize();
    v.shape = {1, 2, 3};
    v.resize();

    EXPECT_THROW(attn.forward(q, k, v, out), std::invalid_argument);
}

TEST(ScaledDotProductAttentionTest, DifferentBatchSizes)
{
    ScaledDotProductAttention attn;
    Tensor q, k, v, out;

    q.shape = {2, 2, 3};
    q.resize();
    k.shape = {3, 2, 3};
    k.resize();
    v.shape = {2, 2, 3};
    v.resize();

    EXPECT_THROW(attn.forward(q, k, v, out), std::invalid_argument);
}

TEST(ScaledDotProductAttentionTest, LargeValues)
{
    ScaledDotProductAttention attn;
    Tensor q, k, v, out;

    q.shape = {1, 2, 2};
    q.data = {1000, 1000, 1000, 1000};
    k.shape = {1, 2, 2};
    k.data = {1000, 1000, 1000, 1000};
    v.shape = {1, 2, 2};
    v.data = {1, 2, 3, 4};

    EXPECT_NO_THROW(attn.forward(q, k, v, out));

    for (float val : out.data)
    {
        EXPECT_FALSE(std::isnan(val));
        EXPECT_FALSE(std::isinf(val));
    }
}

TEST(ScaledDotProductAttentionTest, ScaleFactor)
{
    ScaledDotProductAttention attn;
    Tensor q, k, v, out;

    q.shape = {1, 1, 10};
    k.shape = {1, 1, 10};
    v.shape = {1, 1, 10};
    q.data.resize(10, 1.0f);
    k.data.resize(10, 1.0f);
    v.data.resize(10, 1.0f);

    attn.forward(q, k, v, out);

    EXPECT_NEAR(out.data[0], 1.0f, 0.5f);
}

TEST(MultiHeadAttentionTest, BasicForwardPass)
{
    const size_t d_model = 8;
    const size_t num_heads = 2;
    MultiHeadAttention mha(d_model, num_heads);

    Tensor q;
    q.shape = {2, 3, d_model};
    Tensor k;
    k.shape = {2, 3, d_model};
    Tensor v;
    v.shape = {2, 3, d_model};

    q.resize();
    std::iota(q.data.begin(), q.data.end(), 0.1f);
    k.resize();
    std::iota(k.data.begin(), k.data.end(), 0.2f);
    v.resize();
    std::iota(v.data.begin(), v.data.end(), 0.3f);

    Tensor out;
    EXPECT_NO_THROW(mha.forward(q, k, v, out));

    EXPECT_EQ(out.shape, std::vector<size_t>({2, 3, d_model}));

    // Проверяем, что выходные значения в разумных пределах
    for (float val : out.data)
    {
        EXPECT_FALSE(std::isnan(val));
        EXPECT_FALSE(std::isinf(val));
    }
}

TEST(MultiHeadAttentionTest, OutputShapeConsistency)
{
    const size_t d_model = 12;
    const size_t num_heads = 3;
    MultiHeadAttention mha(d_model, num_heads);

    Tensor q1;
    q1.shape = {1, 5, d_model};
    q1.resize();
    Tensor q2;
    q2.shape = {3, 1, d_model};
    q2.resize();

    Tensor k = q1.copy(), v = q1.copy(), out;

    mha.forward(q1, k, v, out);
    EXPECT_EQ(out.shape, std::vector<size_t>({1, 5, d_model}));

    mha.forward(q2, q2, q2, out);
    EXPECT_EQ(out.shape, std::vector<size_t>({3, 1, d_model}));
}

TEST(MultiHeadAttentionTest, InvalidInputDimensions)
{
    const size_t d_model = 8;
    const size_t num_heads = 2;
    MultiHeadAttention mha(d_model, num_heads);

    Tensor q;
    q.shape = {2, 3, d_model};
    q.resize();
    Tensor k;
    k.shape = {2, 4, d_model};
    k.resize();
    Tensor v;
    v.shape = {2, 3, d_model};
    v.resize();
    Tensor out;

    EXPECT_THROW(mha.forward(q, k, v, out), std::invalid_argument);
}

TEST(MultiHeadAttentionTest, InvalidHeadConfiguration)
{
    EXPECT_THROW(MultiHeadAttention(7, 2), std::runtime_error);
}

TEST(MultiHeadAttentionTest, SplitAndConcatOperations)
{
    const size_t d_model = 8;
    const size_t num_heads = 2;
    MultiHeadAttention mha(d_model, num_heads);

    Tensor t;
    t.shape = {1, 4, d_model};
    t.resize();
    std::iota(t.data.begin(), t.data.end(), 1.0f);

    // spit
    Tensor original = t.copy();
    mha.split(t);

    EXPECT_EQ(t.shape,
              std::vector<size_t>({1, num_heads, 4, d_model / num_heads}));

    // concat
    mha.concat(t);
    EXPECT_EQ(t.shape, original.shape);

    // Проверяем, что данные не изменились
    for (size_t i = 0; i < original.data.size(); ++i)
    {
        EXPECT_FLOAT_EQ(t.data[i], original.data[i]);
    }
}

TEST(MultiHeadAttentionTest, BasicBackwardPass)
{
    const size_t d_model = 8;
    const size_t num_heads = 2;
    const size_t batch_size = 2;
    const size_t seq_len = 3;

    MultiHeadAttention mha(d_model, num_heads);

    // Инициализация входных тензоров
    Tensor q, k, v;
    q.shape = {batch_size, seq_len, d_model};
    q.resize();
    k.shape = {batch_size, seq_len, d_model};
    k.resize();
    v.shape = {batch_size, seq_len, d_model};
    v.resize();

    std::iota(q.data.begin(), q.data.end(), 0.1f);
    std::iota(k.data.begin(), k.data.end(), 0.2f);
    std::iota(v.data.begin(), v.data.end(), 0.3f);

    // Прямой проход
    Tensor out;
    mha.forward(q, k, v, out);

    // Подготовка градиентов
    out.grad.resize(out.data.size(), 1.0f); // Устанавливаем градиент в 1

    Tensor dq, dk, dv;
    dq.shape = q.shape;
    dq.resize_grad();
    dk.shape = k.shape;
    dk.resize_grad();
    dv.shape = v.shape;
    dv.resize_grad();

    // Обратный проход
    EXPECT_NO_THROW(mha.backward(out, dq, dk, dv));

    // Проверка градиентов
    EXPECT_EQ(dq.grad.size(), q.data.size());
    EXPECT_EQ(dk.grad.size(), k.data.size());
    EXPECT_EQ(dv.grad.size(), v.data.size());

    // Проверка, что градиенты не нулевые
    EXPECT_FALSE(std::all_of(dq.grad.begin(), dq.grad.end(),
                             [](float g) { return g == 0.0f; }));
    EXPECT_FALSE(std::all_of(dk.grad.begin(), dk.grad.end(),
                             [](float g) { return g == 0.0f; }));
    EXPECT_FALSE(std::all_of(dv.grad.begin(), dv.grad.end(),
                             [](float g) { return g == 0.0f; }));
}

TEST(MultiHeadAttentionTest, WeightGradients)
{
    const size_t d_model = 12;
    const size_t num_heads = 3;

    MultiHeadAttention mha(d_model, num_heads);

    Tensor q;
    q.shape = {1, 5, d_model};
    q.resize();
    Tensor k = q.copy(), v = q.copy(), out;

    mha.forward(q, k, v, out);

    // Сохраняем исходные веса
    auto w_q_before = mha.w_q.weight.data;
    auto w_k_before = mha.w_k.weight.data;
    auto w_v_before = mha.w_v.weight.data;
    auto w_concat_before = mha.w_concat.weight.data;

    // Обратный проход
    out.grad.resize(out.data.size(), 1.0f);
    Tensor dq, dk, dv;
    dq.shape = q.shape;
    dq.resize_grad();
    dk.shape = k.shape;
    dk.resize_grad();
    dv.shape = v.shape;
    dv.resize_grad();

    mha.backward(out, dq, dk, dv);

    // Проверка, что веса не изменились
    EXPECT_EQ(mha.w_q.weight.data, w_q_before);
    EXPECT_EQ(mha.w_k.weight.data, w_k_before);
    EXPECT_EQ(mha.w_v.weight.data, w_v_before);
    EXPECT_EQ(mha.w_concat.weight.data, w_concat_before);

    // Проверка, что градиенты весов изменились
    EXPECT_FALSE(mha.w_q.weight.grad.empty());
    EXPECT_FALSE(mha.w_k.weight.grad.empty());
    EXPECT_FALSE(mha.w_v.weight.grad.empty());
    EXPECT_FALSE(mha.w_concat.weight.grad.empty());
}

TEST(MultiHeadAttentionTest, MultipleBackwardCalls)
{
    const size_t d_model = 16;
    const size_t num_heads = 4;

    MultiHeadAttention mha(d_model, num_heads);

    Tensor q;
    q.shape = {1, 4, d_model};
    q.resize();
    Tensor k = q.copy(), v = q.copy(), out;

    std::iota(q.data.begin(), q.data.end(), 0.1f);

    mha.forward(q, k, v, out);

    // Первый обратный проход
    out.grad.resize(out.data.size(), 1.0f);
    Tensor dq1, dk1, dv1;
    dq1.shape = q.shape;
    dq1.resize_grad();
    dk1.shape = k.shape;
    dk1.resize_grad();
    dv1.shape = v.shape;
    dv1.resize_grad();

    mha.backward(out, dq1, dk1, dv1);

    // Второй обратный проход с другими градиентами
    for (size_t i = 0; i < out.grad.size(); ++i)
    {
        out.grad[i] = (i % 2 + 1) * 0.5f;
    }

    Tensor dq2, dk2, dv2;
    dq2.shape = q.shape;
    dq2.resize_grad();
    dk2.shape = k.shape;
    dk2.resize_grad();
    dv2.shape = v.shape;
    dv2.resize_grad();

    mha.backward(out, dq2, dk2, dv2);

    // Проверка, что градиенты разные при разных входных градиентах
    EXPECT_NE(dq1.grad, dq2.grad);
    EXPECT_NE(dk1.grad, dk2.grad);
    EXPECT_NE(dv1.grad, dv2.grad);
}

#include "ttie/ttie.h"
#include <cmath>
#include <gtest/gtest.h>
#include <random>

// Вспомогательная функция для вычисления MSE Loss
float compute_mse_loss(const Tensor &output, const Tensor &target)
{
    if (output.data.size() != target.data.size())
    {
        throw std::invalid_argument("Output and target sizes do not match");
    }
    float sum_sq_diff = 0.0f;
    for (size_t i = 0; i < output.data.size(); ++i)
    {
        float diff = output.data[i] - target.data[i];
        sum_sq_diff += diff * diff;
    }
    return sum_sq_diff / output.data.size();
}

// Вспомогательная функция для обновления параметров (градиентный спуск)
void update_parameters(const std::vector<Tensor *> &params, float learning_rate)
{
    for (Tensor *param : params)
    {
        if (!param->grad.empty())
        {
            for (size_t i = 0; i < param->data.size(); ++i)
            {
                param->data[i] -= learning_rate * param->grad[i];
            }
        }
    }
}

// Тесты для BatchNorm1d
TEST(BatchNorm1dTest, Initialization)
{
    BatchNorm1d bn(64, 1e-5, 0.1, true, true);
    EXPECT_EQ(bn.to_string(), "BatchNorm1d(64)");
    EXPECT_EQ(bn.parameters().size(), 2); // gamma и beta
    EXPECT_EQ(bn.parameters()[0]->shape, std::vector<size_t>{64}); // gamma
    EXPECT_EQ(bn.parameters()[1]->shape, std::vector<size_t>{64}); // beta
    EXPECT_FALSE(bn.parameters()[0]->data.empty());
    EXPECT_FALSE(bn.parameters()[1]->data.empty());
}

TEST(BatchNorm1dTest, ForwardValidInput)
{
    BatchNorm1d bn(2, 1e-5, 0.1, true, true);
    Tensor input;
    input.shape = {4, 2}; // [batch_size, num_features]
    input.resize();
    input.data = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f};
    Tensor output;
    bn.forward(input, output);
    EXPECT_EQ(output.shape, (std::vector<size_t>{4, 2}));

    // Проверяем параметры gamma и beta
    std::cout << "BatchNorm1d gamma: ";
    for (float g : bn.parameters()[0]->data)
        std::cout << g << " ";
    std::cout << "\nBatchNorm1d beta: ";
    for (float b : bn.parameters()[1]->data)
        std::cout << b << " ";
    std::cout << "\n";

    // Проверяем нормализацию по батчу для каждого канала
    size_t batch_size = input.shape[0];
    size_t num_features = input.shape[1];
    for (size_t f = 0; f < num_features; ++f)
    {
        // Вычисляем среднее и дисперсию по батчу для канала f
        float mean = 0.0f;
        for (size_t b = 0; b < batch_size; ++b)
        {
            mean += input.data[b * num_features + f];
        }
        mean /= batch_size;

        float var = 0.0f;
        for (size_t b = 0; b < batch_size; ++b)
        {
            float diff = input.data[b * num_features + f] - mean;
            var += diff * diff;
        }
        var /= batch_size;

        float inv_std = 1.0f / std::sqrt(var + 1e-5f);
        float gamma = bn.parameters()[0]->data[f];
        float beta = bn.parameters()[1]->data[f];

        // Проверяем, что выход соответствует нормализации
        for (size_t b = 0; b < batch_size; ++b)
        {
            float x = input.data[b * num_features + f];
            float x_hat = (x - mean) * inv_std;
            float expected_output = gamma * x_hat + beta;
            float actual_output = output.data[b * num_features + f];
            EXPECT_NEAR(actual_output, expected_output, 1e-5f);
        }
    }
}

TEST(BatchNorm1dTest, ForwardInvalidInputShape)
{
    BatchNorm1d bn(2);
    Tensor input;
    input.shape = {4, 3}; // неправильное число каналов
    input.resize();
    Tensor output;
    EXPECT_THROW(bn.forward(input, output), std::invalid_argument);
}

TEST(BatchNorm1dTest, BackwardValidInput)
{
    BatchNorm1d bn(2, 1e-5, 0.1, true, true);
    Tensor input;
    input.shape = {4, 2};
    input.resize();
    input.data = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f};
    Tensor output;
    bn.forward(input, output);
    Tensor grad_output;
    grad_output.shape = {4, 2};
    grad_output.resize();
    grad_output.resize_grad();
    grad_output.grad = {1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f};
    Tensor grad_input;
    bn.backward(grad_output, grad_input);
    EXPECT_EQ(grad_input.shape, (std::vector<size_t>{4, 2}));
    EXPECT_FALSE(grad_input.grad.empty());
    EXPECT_FALSE(bn.parameters()[0]->grad.empty());
    EXPECT_FALSE(bn.parameters()[1]->grad.empty());
}

TEST(BatchNorm1dTest, BackwardWithoutForward)
{
    BatchNorm1d bn(2);
    Tensor grad_output;
    grad_output.shape = {4, 2};
    grad_output.resize();
    grad_output.resize_grad();
    grad_output.grad = {1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f};
    Tensor grad_input;
    EXPECT_THROW(bn.backward(grad_output, grad_input), std::runtime_error);
}

TEST(BatchNorm1dTest, BackwardGradientDescent)
{
    BatchNorm1d bn(2, 1e-5, 0.1, true, true);
    Tensor input;
    input.shape = {4, 2};
    input.resize();
    input.data = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f};
    input.resize_grad();

    // Целевой тензор (target)
    Tensor target;
    target.shape = {4, 2};
    target.resize();
    target.data = {0.5f, 1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f};

    float learning_rate = 0.01f;
    int num_iterations = 10;
    float initial_loss = 0.0f;

    // Выполняем градиентный спуск
    for (int iter = 0; iter < num_iterations; ++iter)
    {
        Tensor output;
        bn.forward(input, output);
        float loss = compute_mse_loss(output, target);
        if (iter == 0)
        {
            initial_loss = loss;
        }

        Tensor grad_output;
        grad_output.shape = output.shape;
        grad_output.resize();
        grad_output.resize_grad();
        for (size_t i = 0; i < output.data.size(); ++i)
        {
            grad_output.grad[i] =
                2.0f * (output.data[i] - target.data[i]) / output.data.size();
        }

        Tensor grad_input;
        bn.backward(grad_output, grad_input);

        // Обновляем параметры
        update_parameters(bn.parameters(), learning_rate);

        // Очищаем градиенты
        for (Tensor *param : bn.parameters())
        {
            std::fill(param->grad.begin(), param->grad.end(), 0.0f);
        }
    }

    // Проверяем, что лосс уменьшился
    Tensor final_output;
    bn.forward(input, final_output);
    float final_loss = compute_mse_loss(final_output, target);
    EXPECT_LT(final_loss, initial_loss);
}

// Тесты для BatchNorm2d
TEST(BatchNorm2dTest, Initialization)
{
    BatchNorm2d bn(32, 1e-5, 0.1, true, true);
    EXPECT_EQ(bn.to_string(), "BatchNorm2d(32)");
    EXPECT_EQ(bn.parameters().size(), 2); // gamma и beta
    EXPECT_EQ(bn.parameters()[0]->shape, std::vector<size_t>{32}); // gamma
    EXPECT_EQ(bn.parameters()[1]->shape, std::vector<size_t>{32}); // beta
    EXPECT_FALSE(bn.parameters()[0]->data.empty());
    EXPECT_FALSE(bn.parameters()[1]->data.empty());
}

TEST(BatchNorm2dTest, ForwardValidInput)
{
    BatchNorm2d bn(2, 1e-5, 0.1, true, true);
    Tensor input;
    input.shape = {2, 2, 3, 3}; // [N, C, H, W]
    input.resize();
    for (size_t i = 0; i < input.size(); ++i)
    {
        input.data[i] = static_cast<float>(i % 5 + 1);
    }
    Tensor output;
    bn.forward(input, output);
    EXPECT_EQ(output.shape, (std::vector<size_t>{2, 2, 3, 3}));

    // Проверяем параметры gamma и beta
    std::cout << "BatchNorm2d gamma: ";
    for (float g : bn.parameters()[0]->data)
        std::cout << g << " ";
    std::cout << "\nBatchNorm2d beta: ";
    for (float b : bn.parameters()[1]->data)
        std::cout << b << " ";
    std::cout << "\n";

    // Проверяем нормализацию по батчу для каждого канала
    size_t N = input.shape[0];
    size_t C = input.shape[1];
    size_t H = input.shape[2];
    size_t W = input.shape[3];
    size_t count = N * H * W;

    for (size_t c = 0; c < C; ++c)
    {
        float mean = 0.0f;
        for (size_t n = 0; n < N; ++n)
        {
            for (size_t h = 0; h < H; ++h)
            {
                for (size_t w = 0; w < W; ++w)
                {
                    size_t idx = n * C * H * W + c * H * W + h * W + w;
                    mean += input.data[idx];
                }
            }
        }
        mean /= count;

        float var = 0.0f;
        for (size_t n = 0; n < N; ++n)
        {
            for (size_t h = 0; h < H; ++h)
            {
                for (size_t w = 0; w < W; ++w)
                {
                    size_t idx = n * C * H * W + c * H * W + h * W + w;
                    float diff = input.data[idx] - mean;
                    var += diff * diff;
                }
            }
        }
        var /= count;

        float inv_std = 1.0f / std::sqrt(var + 1e-5f);
        float gamma = bn.parameters()[0]->data[c];
        float beta = bn.parameters()[1]->data[c];

        for (size_t n = 0; n < N; ++n)
        {
            for (size_t h = 0; h < H; ++h)
            {
                for (size_t w = 0; w < W; ++w)
                {
                    size_t idx = n * C * H * W + c * H * W + h * W + w;
                    float x = input.data[idx];
                    float x_hat = (x - mean) * inv_std;
                    float expected_output = gamma * x_hat + beta;
                    float actual_output = output.data[idx];
                    EXPECT_NEAR(actual_output, expected_output, 1e-5f);
                }
            }
        }
    }
}

TEST(BatchNorm2dTest, ForwardInvalidInputShape)
{
    BatchNorm2d bn(2);
    Tensor input;
    input.shape = {2, 3, 3, 3}; // неправильное число каналов
    input.resize();
    Tensor output;
    EXPECT_THROW(bn.forward(input, output), std::invalid_argument);
}

TEST(BatchNorm2dTest, BackwardValidInput)
{
    BatchNorm2d bn(2, 1e-5, 0.1, true, true);
    Tensor input;
    input.shape = {2, 2, 3, 3};
    input.resize();
    input.data = std::vector<float>(input.size(), 1.0f);
    Tensor output;
    bn.forward(input, output);
    Tensor grad_output;
    grad_output.shape = {2, 2, 3, 3};
    grad_output.resize();
    grad_output.resize_grad();
    grad_output.grad = std::vector<float>(grad_output.size(), 1.0f);
    Tensor grad_input;
    bn.backward(grad_output, grad_input);
    EXPECT_EQ(grad_input.shape, (std::vector<size_t>{2, 2, 3, 3}));
    EXPECT_FALSE(grad_input.grad.empty());
    EXPECT_FALSE(bn.parameters()[0]->grad.empty());
    EXPECT_FALSE(bn.parameters()[1]->grad.empty());
}

TEST(BatchNorm2dTest, BackwardInvalidGradOutput)
{
    BatchNorm2d bn(2);
    Tensor input;
    input.shape = {2, 2, 3, 3};
    input.resize();
    input.data = std::vector<float>(input.size(), 1.0f);
    Tensor output;
    bn.forward(input, output);
    Tensor grad_output;
    grad_output.shape = {2, 3, 3, 3}; // неправильное число каналов
    grad_output.resize();
    grad_output.resize_grad();
    Tensor grad_input;
    EXPECT_THROW(bn.backward(grad_output, grad_input), std::invalid_argument);
}

TEST(BatchNorm2dTest, BackwardGradientDescent)
{
    BatchNorm2d bn(2, 1e-5, 0.1, true, true);
    Tensor input;
    input.shape = {2, 2, 3, 3};
    input.resize();
    input.data = std::vector<float>(input.size(), 1.0f);
    input.resize_grad();

    // Целевой тензор
    Tensor target;
    target.shape = {2, 2, 3, 3};
    target.resize();
    target.data = std::vector<float>(target.size(), 0.5f);

    float learning_rate = 0.01f;
    int num_iterations = 10;
    float initial_loss = 0.0f;

    for (int iter = 0; iter < num_iterations; ++iter)
    {
        Tensor output;
        bn.forward(input, output);
        float loss = compute_mse_loss(output, target);
        if (iter == 0)
        {
            initial_loss = loss;
        }

        Tensor grad_output;
        grad_output.shape = output.shape;
        grad_output.resize();
        grad_output.resize_grad();
        for (size_t i = 0; i < output.data.size(); ++i)
        {
            grad_output.grad[i] =
                2.0f * (output.data[i] - target.data[i]) / output.data.size();
        }

        Tensor grad_input;
        bn.backward(grad_output, grad_input);

        update_parameters(bn.parameters(), learning_rate);

        for (Tensor *param : bn.parameters())
        {
            std::fill(param->grad.begin(), param->grad.end(), 0.0f);
        }
    }

    Tensor final_output;
    bn.forward(input, final_output);
    float final_loss = compute_mse_loss(final_output, target);
    EXPECT_LT(final_loss, initial_loss);
}

// Тесты для BatchNorm3d
TEST(BatchNorm3dTest, Initialization)
{
    BatchNorm3d bn(16, 1e-5, 0.1, true, true);
    EXPECT_EQ(bn.to_string(), "BatchNorm3d(16)");
    EXPECT_EQ(bn.parameters().size(), 2); // gamma и beta
    EXPECT_EQ(bn.parameters()[0]->shape, std::vector<size_t>{16}); // gamma
    EXPECT_EQ(bn.parameters()[1]->shape, std::vector<size_t>{16}); // beta
    EXPECT_FALSE(bn.parameters()[0]->data.empty());
    EXPECT_FALSE(bn.parameters()[1]->data.empty());
}

TEST(BatchNorm3dTest, ForwardValidInput)
{
    BatchNorm3d bn(2, 1e-5, 0.1, true, true);
    Tensor input;
    input.shape = {2, 2, 3, 3, 3}; // [N, C, D, H, W]
    input.resize();
    for (size_t i = 0; i < input.size(); ++i)
    {
        input.data[i] = static_cast<float>(i % 5 + 1);
    }
    Tensor output;
    bn.forward(input, output);
    EXPECT_EQ(output.shape, (std::vector<size_t>{2, 2, 3, 3, 3}));

    // Проверяем параметры gamma и beta
    std::cout << "BatchNorm3d gamma: ";
    for (float g : bn.parameters()[0]->data)
        std::cout << g << " ";
    std::cout << "\nBatchNorm3d beta: ";
    for (float b : bn.parameters()[1]->data)
        std::cout << b << " ";
    std::cout << "\n";

    // Проверяем нормализацию по батчу для каждого канала
    size_t N = input.shape[0];
    size_t C = input.shape[1];
    size_t D = input.shape[2];
    size_t H = input.shape[3];
    size_t W = input.shape[4];
    size_t count = N * D * H * W;

    for (size_t c = 0; c < C; ++c)
    {
        float mean = 0.0f;
        for (size_t n = 0; n < N; ++n)
        {
            for (size_t d = 0; d < D; ++d)
            {
                for (size_t h = 0; h < H; ++h)
                {
                    for (size_t w = 0; w < W; ++w)
                    {
                        size_t idx = n * C * D * H * W + c * D * H * W +
                                     d * H * W + h * W + w;
                        mean += input.data[idx];
                    }
                }
            }
        }
        mean /= count;

        float var = 0.0f;
        for (size_t n = 0; n < N; ++n)
        {
            for (size_t d = 0; d < D; ++d)
            {
                for (size_t h = 0; h < H; ++h)
                {
                    for (size_t w = 0; w < W; ++w)
                    {
                        size_t idx = n * C * D * H * W + c * D * H * W +
                                     d * H * W + h * W + w;
                        float diff = input.data[idx] - mean;
                        var += diff * diff;
                    }
                }
            }
        }
        var /= count;

        float inv_std = 1.0f / std::sqrt(var + 1e-5f);
        float gamma = bn.parameters()[0]->data[c];
        float beta = bn.parameters()[1]->data[c];

        for (size_t n = 0; n < N; ++n)
        {
            for (size_t d = 0; d < D; ++d)
            {
                for (size_t h = 0; h < H; ++h)
                {
                    for (size_t w = 0; w < W; ++w)
                    {
                        size_t idx = n * C * D * H * W + c * D * H * W +
                                     d * H * W + h * W + w;
                        float x = input.data[idx];
                        float x_hat = (x - mean) * inv_std;
                        float expected_output = gamma * x_hat + beta;
                        float actual_output = output.data[idx];
                        EXPECT_NEAR(actual_output, expected_output, 1e-5f);
                    }
                }
            }
        }
    }
}

TEST(BatchNorm3dTest, ForwardInvalidInputShape)
{
    BatchNorm3d bn(2);
    Tensor input;
    input.shape = {2, 3, 3, 3, 3}; // неправильное число каналов
    input.resize();
    Tensor output;
    EXPECT_THROW(bn.forward(input, output), std::invalid_argument);
}

TEST(BatchNorm3dTest, BackwardValidInput)
{
    BatchNorm3d bn(2, 1e-5, 0.1, true, true);
    Tensor input;
    input.shape = {2, 2, 3, 3, 3};
    input.resize();
    input.data = std::vector<float>(input.size(), 1.0f);
    Tensor output;
    bn.forward(input, output);
    Tensor grad_output;
    grad_output.shape = {2, 2, 3, 3, 3};
    grad_output.resize();
    grad_output.resize_grad();
    grad_output.grad = std::vector<float>(grad_output.size(), 1.0f);
    Tensor grad_input;
    bn.backward(grad_output, grad_input);
    EXPECT_EQ(grad_input.shape, (std::vector<size_t>{2, 2, 3, 3, 3}));
    EXPECT_FALSE(grad_input.grad.empty());
    EXPECT_FALSE(bn.parameters()[0]->grad.empty());
    EXPECT_FALSE(bn.parameters()[1]->grad.empty());
}

TEST(BatchNorm3dTest, BackwardWithoutForward)
{
    BatchNorm3d bn(2);
    Tensor grad_output;
    grad_output.shape = {2, 2, 3, 3, 3};
    grad_output.resize();
    grad_output.resize_grad();
    grad_output.grad = std::vector<float>(grad_output.size(), 1.0f);
    Tensor grad_input;
    EXPECT_THROW(bn.backward(grad_output, grad_input), std::runtime_error);
}

TEST(BatchNorm3dTest, BackwardGradientDescent)
{
    BatchNorm3d bn(2, 1e-5, 0.1, true, true);
    Tensor input;
    input.shape = {2, 2, 3, 3, 3};
    input.resize();
    input.data = std::vector<float>(input.size(), 1.0f);
    input.resize_grad();

    // Целевой тензор
    Tensor target;
    target.shape = {2, 2, 3, 3, 3};
    target.resize();
    target.data = std::vector<float>(target.size(), 0.5f);

    float learning_rate = 0.01f;
    int num_iterations = 10;
    float initial_loss = 0.0f;

    for (int iter = 0; iter < num_iterations; ++iter)
    {
        Tensor output;
        bn.forward(input, output);
        float loss = compute_mse_loss(output, target);
        if (iter == 0)
        {
            initial_loss = loss;
        }

        Tensor grad_output;
        grad_output.shape = output.shape;
        grad_output.resize();
        grad_output.resize_grad();
        for (size_t i = 0; i < output.data.size(); ++i)
        {
            grad_output.grad[i] =
                2.0f * (output.data[i] - target.data[i]) / output.data.size();
        }

        Tensor grad_input;
        bn.backward(grad_output, grad_input);

        update_parameters(bn.parameters(), learning_rate);

        for (Tensor *param : bn.parameters())
        {
            std::fill(param->grad.begin(), param->grad.end(), 0.0f);
        }
    }

    Tensor final_output;
    bn.forward(input, final_output);
    float final_loss = compute_mse_loss(final_output, target);
    EXPECT_LT(final_loss, initial_loss);
}

int main(int argc, char **argv)
{
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}