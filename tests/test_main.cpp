#include "ttie/ttie.h"
#include <gtest/gtest.h>

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

TEST(LSTMTest, Initialization)
{
    LSTM lstm(3, 4); // input_size = 3, hidden_size = 4

    auto params = lstm.parameters();
    EXPECT_EQ(params.size(), 16); // 2 x (4 weight matrices + 4 bias vectors)

    for (Tensor *t : params)
    {
        EXPECT_TRUE(t->validate_shape());
    }
}

TEST(LSTMTest, ForwardShapeAndSanity)
{
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

    lstm.forward(inputs, outputs);

    EXPECT_EQ(outputs[0].shape[0], seq_len);
    EXPECT_EQ(outputs[0].shape[1], hidden_size);

    for (float v : outputs[0].data)
    {
        EXPECT_TRUE(std::isfinite(v));
    }
}

TEST(LSTMTest, ForwardVSTorch)
{
    /* Pytorch reference
    # 1 Layer
    import torch
    import torch.nn as nn

    seq_len = 2
    input_dim = 2
    hidden_dim = 2
    batch_size = 1
    num_layers = 1

    x = torch.tensor([
        [0, 0.1],
        [0.2, 0.3]
    ], dtype=torch.float32).unsqueeze(1)  # [seq_len, batch, input_dim]

    lstm = nn.LSTM(input_dim, hidden_dim, num_layers=num_layers,
    batch_first=False)

    weight_ih = torch.tensor([
        [0.1, 0.2],   # W_ii
        [0.3, 0.4],
        [0.5, 0.6],   # W_if
        [0.7, 0.8],
        [0.9, 1.0],   # W_ig
        [1.1, 1.2],
        [1.3, 1.4],   # W_io
        [1.5, 1.6]
    ], dtype=torch.float32)

    weight_hh = torch.tensor([
        [2.1, 2.2],   # W_fi
        [2.3, 2.4],
        [2.5, 2.6],   # W_ff
        [2.7, 2.8],
        [2.9, 3.0],   # W_fg
        [3.1, 3.2],
        [3.3, 3.4],   # W_fo
        [3.5, 3.6]
    ], dtype=torch.float32)

    bias_ih = torch.tensor([4.1, 4.2, 4.3, 4.4, 4.5, 4.6, 4.7, 4.8],
    dtype=torch.float32) bias_hh =
    torch.tensor([5.1, 5.2, 5.3, 5.4, 5.5, 5.6, 5.7, 5.8], dtype=torch.float32)

    with torch.no_grad():
        lstm.weight_ih_l0.copy_(weight_ih)
        lstm.weight_hh_l0.copy_(weight_hh)
        lstm.bias_ih_l0.copy_(bias_ih)
        lstm.bias_hh_l0.copy_(bias_hh)

    # LSTM (1 layer)
    #for param in lstm.parameters():
    #    nn.init.uniform_(param, a=-0.1, b=0.1)  # ensure deterministic
    initialization

    # Initial states (h_0, c_0): default is zeros
    output, (hn, cn) = lstm(x)

    grad_output = torch.ones_like(output)
    # Backward pass
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
    */

    /*
    Input (x):
    tensor([[1., 2.],
            [3., 4.]])

    --- LSTM Parameters ---
    weight_ih_l0:
    tensor([[0.1000, 0.2000],
            [0.3000, 0.4000],
            [0.5000, 0.6000],
            [0.7000, 0.8000],
            [0.9000, 1.0000],
            [1.1000, 1.2000],
            [1.3000, 1.4000],
            [1.5000, 1.6000]])

    weight_hh_l0:
    tensor([[2.1000, 2.2000],
            [2.3000, 2.4000],
            [2.5000, 2.6000],
            [2.7000, 2.8000],
            [2.9000, 3.0000],
            [3.1000, 3.2000],
            [3.3000, 3.4000],
            [3.5000, 3.6000]])

    bias_ih_l0:
    tensor([4.1000, 4.2000, 4.3000, 4.4000, 4.5000, 4.6000, 4.7000, 4.8000])

    bias_hh_l0:
    tensor([5.1000, 5.2000, 5.3000, 5.4000, 5.5000, 5.6000, 5.7000, 5.8000])

    Initial h_0:
    tensor([[[0., 0.]]])

    Initial c_0:
    tensor([[[0., 0.]]])

    --- LSTM Outputs ---
    Output:
    tensor([[0.7616, 0.7616],
            [0.9640, 0.9640]], grad_fn=<SqueezeBackward1>)

    Hidden state hn:
    tensor([[0.9640, 0.9640]], grad_fn=<SqueezeBackward1>)

    Cell state cn:
    tensor([[1.9999, 2.0000]], grad_fn=<SqueezeBackward1>)
    */

    const size_t seq_len = 2;
    const size_t input_size = 2;
    const size_t hidden_size = 2;
    const size_t num_layers = 1;

    LSTM lstm(input_size, hidden_size, num_layers);

    Tensor input;
    input.shape = {seq_len, input_size};
    input.resize();

    for (size_t i = 0; i < input.data.size(); ++i)
    {
        input.data[i] = static_cast<float>(i) / 10.0f;
    }

    Tensor output;
    output.shape = {seq_len, hidden_size};
    output.resize();

    std::vector<Tensor> inputs = {input};
    std::vector<Tensor> outputs = {output};

    Linear *layer_ii = static_cast<Linear *>(lstm.layers[0]);
    layer_ii->weight.data = {0.1, 0.2, 0.3, 0.4};
    layer_ii->bias.data = {4.1, 4.2};

    Linear *layer_if = static_cast<Linear *>(lstm.layers[2]);
    layer_if->weight.data = {0.5, 0.6, 0.7, 0.8};
    layer_if->bias.data = {4.3, 4.4};

    Linear *layer_ig = static_cast<Linear *>(lstm.layers[4]);
    layer_ig->weight.data = {0.9, 1.0, 1.1, 1.2};
    layer_ig->bias.data = {4.5, 4.6};

    Linear *layer_io = static_cast<Linear *>(lstm.layers[6]);
    layer_io->weight.data = {1.3, 1.4, 1.5, 1.6};
    layer_io->bias.data = {4.7, 4.8};

    Linear *layer_hi = static_cast<Linear *>(lstm.layers[8]);
    layer_hi->weight.data = {2.1, 2.2, 2.3, 2.4};
    layer_hi->bias.data = {5.1, 5.2};

    Linear *layer_hf = static_cast<Linear *>(lstm.layers[10]);
    layer_hf->weight.data = {2.5, 2.6, 2.7, 2.8};
    layer_hf->bias.data = {5.3, 5.4};

    Linear *layer_hg = static_cast<Linear *>(lstm.layers[12]);
    layer_hg->weight.data = {2.9, 3.0, 3.1, 3.2};
    layer_hg->bias.data = {5.5, 5.6};

    Linear *layer_ho = static_cast<Linear *>(lstm.layers[14]);
    layer_ho->weight.data = {3.3, 3.4, 3.5, 3.6};
    layer_ho->bias.data = {5.7, 5.8};

    // Second layer (TODO)

    /*
    Linear* layer_ii_2 = static_cast<Linear*>(lstm.layers[16 + 0]);
    layer_ii_2->weight.data = {-0.1, -0.2, -0.3, -0.4};
    layer_ii_2->bias.data = {-4.1, -4.2};

    Linear* layer_if_2 = static_cast<Linear*>(lstm.layers[16 + 2]);
    layer_if_2->weight.data = {-0.5, -0.6, -0.7, -0.8};
    layer_if_2->bias.data = {-4.3, -4.4};

    Linear* layer_ig_2 = static_cast<Linear*>(lstm.layers[16 + 4]);
    layer_ig_2->weight.data = {-0.9, -1.0, -1.1, -1.2};
    layer_ig_2->bias.data = {-4.5, -4.6};

    Linear* layer_io_2 = static_cast<Linear*>(lstm.layers[16 + 6]);
    layer_io_2->weight.data = {-1.3, -1.4, -1.5, -1.6};
    layer_io_2->bias.data = {-4.7, -4.8};

    Linear* layer_hi_2 = static_cast<Linear*>(lstm.layers[16 + 8]);
    layer_hi_2->weight.data = {-2.1, -2.2, -2.3, -2.4};
    layer_hi_2->bias.data = {-5.1, -5.2};

    Linear* layer_hf_2 = static_cast<Linear*>(lstm.layers[16 + 10]);
    layer_hf_2->weight.data = {-2.5, -2.6, -2.7, -2.8};
    layer_hf_2->bias.data = {-5.3, -5.4};

    Linear* layer_hg_2 = static_cast<Linear*>(lstm.layers[16 + 12]);
    layer_hg_2->weight.data = {-2.9, -3.0, -3.1, -3.2};
    layer_hg_2->bias.data = {-5.5, -5.6};

    Linear* layer_ho_2 = static_cast<Linear*>(lstm.layers[16 + 14]);
    layer_ho_2->weight.data = {-3.3, -3.4, -3.5, -3.6};
    layer_ho_2->bias.data = {-5.7, -5.8};

    */
    lstm.forward(inputs, outputs);
    std::vector<float> torch_output = {0.7615f, 0.7615f, 0.9640f, 0.9640f};

    ASSERT_EQ(outputs[0].data.size(), torch_output.size());
    for (size_t i = 0; i < torch_output.size(); ++i)
    {
        EXPECT_NEAR(outputs[0].data[i], torch_output[i], 1e-4f);
    }
}

int main(int argc, char **argv)
{
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}