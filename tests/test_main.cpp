#include <gtest/gtest.h>
#include "ttie/ttie.h"

using namespace ttie;

TEST(TensorTest, UninitializedTensor) {
    Tensor x;
    EXPECT_TRUE(x.empty());
}

TEST(TensorTest, BasicOps) {
    Tensor a(std::vector<float>({1, 2, 3, 4}));
    Tensor b(std::vector<float>({1, 2, 3, 4}));
    a.reshape({1, 2, 2});
    b.reshape({2, 1, 2});

    EXPECT_THROW(a + b, std::exception);
}

TEST(TensorTest, InitializeAndSetData) {
    Tensor t;
    t.reshape({2, 3});

    EXPECT_EQ(t.size(), 6);
}

TEST(AutoGradTest, AddOp_) {
    Tensor a({1.0f, 1.0f, 1.0f, 1.0f}, true);
    Tensor b({1.0f, 1.0f, 1.0f, 1.0f}, true);

    Tensor c = a + b;

    c.backward();

    auto a_grad = a.get_grad();
    auto b_grad = b.get_grad();

    EXPECT_EQ(a_grad[0], 1);
    EXPECT_EQ(a_grad[1], 1);
    EXPECT_EQ(a_grad[2], 1);
    EXPECT_EQ(a_grad[3], 1);
    EXPECT_EQ(b_grad[0], 1);
    EXPECT_EQ(b_grad[1], 1);
    EXPECT_EQ(b_grad[2], 1);
    EXPECT_EQ(b_grad[3], 1);
}

TEST(AutoGradTest, MulOp_) {
    Tensor a({1.0f, 2.0f, 3.0f, 4.0f}, true);
    Tensor b({5.0f, 6.0f, 7.0f, 8.0f}, true);

    Tensor c = a * b;

    c.backward();

    auto a_grad = a.get_grad();
    auto b_grad = b.get_grad();

    EXPECT_NEAR(a_grad[0], 5.0, 1e-6);
    EXPECT_NEAR(a_grad[1], 6.0, 1e-6);
    EXPECT_NEAR(a_grad[2], 7.0, 1e-6);
    EXPECT_NEAR(a_grad[3], 8.0, 1e-6);
    EXPECT_NEAR(b_grad[0], 1.0, 1e-6);
    EXPECT_NEAR(b_grad[1], 2.0, 1e-6);
    EXPECT_NEAR(b_grad[2], 3.0, 1e-6);
    EXPECT_NEAR(b_grad[3], 4.0, 1e-6);
}

TEST(AutoGradTest, SubOp_) {
    Tensor a({1.0f, 1.0f, 1.0f, 1.0f}, true);
    Tensor b({1.0f, 1.0f, 1.0f, 1.0f}, true);

    Tensor c = a - b;

    c.backward();

    auto a_grad = a.get_grad();
    auto b_grad = b.get_grad();

    EXPECT_NEAR(a_grad[0], 1.0, 1e-6);
    EXPECT_NEAR(a_grad[1], 1.0, 1e-6);
    EXPECT_NEAR(a_grad[2], 1.0, 1e-6);
    EXPECT_NEAR(a_grad[3], 1.0, 1e-6);
    EXPECT_NEAR(b_grad[0], -1.0, 1e-6);
    EXPECT_NEAR(b_grad[1], -1.0, 1e-6);
    EXPECT_NEAR(b_grad[2], -1.0, 1e-6);
    EXPECT_NEAR(b_grad[3], -1.0, 1e-6);
}

TEST(AutoGradTest, UnaryMinusOp_) {
    Tensor a({2.0f, 1.0f, 2.0f, 1.0f}, true);

    Tensor c = -a;

    c.backward();

    auto a_grad = a.get_grad();

    EXPECT_NEAR(a_grad[0], -1.0, 1e-6);
    EXPECT_NEAR(a_grad[1], -1.0, 1e-6);
    EXPECT_NEAR(a_grad[2], -1.0, 1e-6);
    EXPECT_NEAR(a_grad[3], -1.0, 1e-6);
}

TEST(AutoGradTest, MaxOp_) {
    Tensor a({2.0f, 1.0f, 2.0f, 1.0f}, true);
    Tensor b({1.0f, 2.0f, 1.0f, 2.0f}, true);

    Tensor c = Tensor::max(a, b);

    c.backward();

    auto a_grad = a.get_grad();
    auto b_grad = b.get_grad();

    EXPECT_NEAR(a_grad[0], 1.0, 1e-6);
    EXPECT_NEAR(a_grad[1], 0.0, 1e-6);
    EXPECT_NEAR(a_grad[2], 1.0, 1e-6);
    EXPECT_NEAR(a_grad[3], 0.0, 1e-6);
    EXPECT_NEAR(b_grad[0], 0.0, 1e-6);
    EXPECT_NEAR(b_grad[1], 1.0, 1e-6);
    EXPECT_NEAR(b_grad[2], 0.0, 1e-6);
    EXPECT_NEAR(b_grad[3], 1.0, 1e-6);
}

TEST(AutoGradTest, SqrtOp_) {
    Tensor a({2.0f, 1.0f, 2.0f, 1.0f}, true);

    Tensor c = Tensor::sqrt(a);

    c.backward();

    auto a_grad = a.get_grad();

    EXPECT_NEAR(a_grad[0], 0.35355, 1e-5);
    EXPECT_NEAR(a_grad[1], 0.5, 1e-5);
    EXPECT_NEAR(a_grad[2], 0.35355, 1e-5);
    EXPECT_NEAR(a_grad[3], 0.5, 1e-5);
}

TEST(AutoGradTest, LogOp_) {
    Tensor a({2.0f, 1.0f, 2.0f, 1.0f}, true);

    Tensor c = Tensor::log(a);

    c.backward();

    auto a_grad = a.get_grad();

    EXPECT_NEAR(a_grad[0], 0.5, 1e-5);
    EXPECT_NEAR(a_grad[1], 1.0, 1e-5);
    EXPECT_NEAR(a_grad[2], 0.5, 1e-5);
    EXPECT_NEAR(a_grad[3], 1.0, 1e-5);
}

TEST(AutoGradTest, DivOp_) {
    Tensor a({1.0f, 1.0f, 1.0f, 1.0f}, true);
    Tensor b({2.0f, 2.0f, 2.0f, 2.0f}, true);

    Tensor c = a / b;

    c.backward();

    auto a_grad = a.get_grad();
    auto b_grad = b.get_grad();

    EXPECT_NEAR(a_grad[0], 0.5, 1e-6);
    EXPECT_NEAR(a_grad[1], 0.5, 1e-6);
    EXPECT_NEAR(a_grad[2], 0.5, 1e-6);
    EXPECT_NEAR(a_grad[3], 0.5, 1e-6);
    EXPECT_NEAR(b_grad[0], -0.25, 1e-6);
    EXPECT_NEAR(b_grad[1], -0.25, 1e-6);
    EXPECT_NEAR(b_grad[2], -0.25, 1e-6);
    EXPECT_NEAR(b_grad[3], -0.25, 1e-6);
}

TEST(AutoGradTest, MatmulOp) {
    Tensor a({1.5f, 1.0f, 1.5f}, true);
    Tensor b({1.0f, 1.5f, 1.0f}, true);

    a.reshape({1, 3});
    b.reshape({3, 1});

    Tensor c = Tensor::matmul(a, b);

    c.backward();

    auto a_grad = a.get_grad();
    auto b_grad = b.get_grad();

    EXPECT_NEAR(a_grad[0], 1.0f, 1e-6);
    EXPECT_NEAR(a_grad[1], 1.5f, 1e-6);
    EXPECT_NEAR(a_grad[2], 1.0f, 1e-6);
    EXPECT_NEAR(b_grad[0], 1.5f, 1e-6);
    EXPECT_NEAR(b_grad[1], 1.0f, 1e-6);
    EXPECT_NEAR(b_grad[2], 1.5f, 1e-6);
}

TEST(AutoGradTest, BroadcastOp_) {
    Tensor a({1.5f, 1.0f, 1.5f}, true);

    a.reshape({3});

    Tensor c = a.broadcast_to({2, 3});

    c.backward();

    auto a_grad = a.get_grad();

    EXPECT_NEAR(a_grad[0], 2.0f, 1e-6);
    EXPECT_NEAR(a_grad[1], 2.0f, 1e-6);
    EXPECT_NEAR(a_grad[2], 2.0f, 1e-6);
}

TEST(LayerTest, ReLU) {
    ReLU relu;
    EXPECT_EQ(relu.parameters().size(), 0);

    Tensor input;
    input.reshape({2, 3});
    for (size_t i = 0; i < input.size(); ++i) {
        input[i] = static_cast<float>(i) - 2.0f;
    }

    Tensor output;
    relu.forward(input, output);

    for (size_t i = 0; i < output.size(); ++i) {
        EXPECT_FLOAT_EQ(output[i], std::max(0.0f, input[i]));
    }

    output.backward();

    auto grad_ = output.get_grad();

    auto input_grad = input.get_grad();
    for(size_t i = 0; i < input.size(); ++i) {
        EXPECT_FLOAT_EQ(input_grad[i], input[i] >= 0 ? 1.0f : 0.0f);
    }
}

TEST(LayerTest, Sigmoid) {
    Sigmoid sigmoid;
    Tensor input({0.0f, 1.0f}, true);

    Tensor output;
    sigmoid.forward(input, output);

    EXPECT_NEAR(output[0], 0.5f, 1e-5f);        // σ(0) = 0.5
    EXPECT_NEAR(output[1], 0.73105858f, 1e-5f); // σ(1) ≈ 0.731

    output.backward();

    auto grad = input.get_grad();

    EXPECT_NEAR(grad[0], 0.25f, 1e-5f);        // σ'(0) * 1 = 0.25
    EXPECT_NEAR(grad[1], 0.19661193f, 1e-5f);  // σ'(1) * 1 ≈ 0.197
}

TEST(LayerTest, Tanh) {
    Tanh tanh;
    Tensor input({0.0f, 1.0f}, true);
    input.reshape({2});

    Tensor output;
    tanh.forward(input, output);

    EXPECT_NEAR(output[0], 0.0f, 1e-5f);        // tanh(0) = 0
    EXPECT_NEAR(output[1], 0.76159416f, 1e-5f); // tanh(1) ≈ 0.761

    output.backward();

    auto grad = input.get_grad();

    EXPECT_NEAR(grad[0], 1.0f, 1e-5f);        // tanh'(0) = 1
    EXPECT_NEAR(grad[1], 0.41997434f, 1e-5f); // tanh'(1) ≈ 0.420
}


TEST(LayerTest, Linear) {
    Linear linear(3, 2);

    auto params = linear.parameters();
    EXPECT_EQ(params.size(), 2);

    EXPECT_EQ(linear.weight.shape().size(), 2);
    EXPECT_EQ(linear.weight.shape()[0], 3);
    EXPECT_EQ(linear.weight.shape()[1], 2);
    EXPECT_EQ(linear.bias.shape().size(), 1);
    EXPECT_EQ(linear.bias.shape()[0], 2);

    Tensor input(true);
    input.reshape({2, 3});
    for (size_t i = 0; i < input.size(); ++i) {
        input[i] = static_cast<float>(i) / input.size();
    }

    Tensor output(true);
    linear.forward(input, output);

    auto shape_ = output.shape();

    EXPECT_EQ(shape_.size(), 2);
    EXPECT_EQ(shape_[0], 2);
    EXPECT_EQ(shape_[1], 2); // w.shape = [3, 2], input.shape = [2, 3] => (input @ weight).shape = [2, 2]

    for (size_t i = 0; i < output.size(); ++i) {
        output[i] = 1.0f;
    }

    input.zero_grad();

    output.backward();

    bool has_nonzero = false;
    auto inp_grad = input.get_grad();
    for (size_t i = 0; i < input.size(); ++i) {
        if(std::abs(inp_grad[i]) > 1e-5) {
            has_nonzero = true;
            break;
        }
    }
    EXPECT_TRUE(has_nonzero);

    has_nonzero = false;
    auto w_grad = linear.weight.get_grad();
    for (size_t i = 0; i < linear.weight.size(); ++i) {
        if (std::fabs(w_grad[i]) > 1e-8) {
            has_nonzero = true;
            break;
        }
    }
    EXPECT_TRUE(has_nonzero);

    has_nonzero = false;
    auto bias_grad = linear.bias.get_grad();
    for (size_t i = 0; i < bias_grad.size(); ++i) {
        if (std::fabs(bias_grad[i]) > 1e-8) {
            has_nonzero = true;
            break;
        }
    }
    EXPECT_TRUE(has_nonzero);
}

TEST(LayerTest, LinearVSTorch) {
    /* PyTorch reference forward and backward with predefined weights and biases
    import torch

    input = torch.tensor([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]], requires_grad=True)
    weight = torch.tensor([[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]], requires_grad=True)
    bias = torch.tensor([0.1, 0.2], requires_grad=True)

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
    linear.weight.init({0.1f, 0.2f, 0.3f, 0.4f, 0.5f, 0.6f});
    linear.weight.reshape({3, 2});
    linear.bias.init({0.1f, 0.2f});

    Tensor input({0.1f, 0.2f, 0.3f, 0.4f, 0.5f, 0.6f});
    input.reshape({2, 3});


    // Forward pass
    Tensor output;
    linear.forward(input, output);

    // Check output values match PyTorch
    ASSERT_EQ(output.shape().size(), 2);
    ASSERT_EQ(output.shape()[0], 2);
    ASSERT_EQ(output.shape()[1], 2);
    EXPECT_NEAR(output[0], 0.32f, 1e-5f);
    EXPECT_NEAR(output[1], 0.48f, 1e-5f);
    EXPECT_NEAR(output[2], 0.59f, 1e-5f);
    EXPECT_NEAR(output[3], 0.84f, 1e-5f);

    // Backward pass
    // output.grad({1.0f, 1.0f, 1.0f, 1.0f}); // implemented internally

    // linear.backward(output, input);
    output.backward();

    // Check input gradients match PyTorch
    auto input_grad = input.get_grad();
    ASSERT_EQ(input_grad.size(), 6);
    EXPECT_NEAR(input_grad[0], 0.3f, 1e-5f);
    EXPECT_NEAR(input_grad[1], 0.7f, 1e-5f);
    EXPECT_NEAR(input_grad[2], 1.1f, 1e-5f);
    EXPECT_NEAR(input_grad[3], 0.3f, 1e-5f);
    EXPECT_NEAR(input_grad[4], 0.7f, 1e-5f);
    EXPECT_NEAR(input_grad[5], 1.1f, 1e-5f);

    // Check weight gradients match PyTorch
    auto w_grad = linear.weight.get_grad();
    ASSERT_EQ(w_grad.size(), 6);
    EXPECT_NEAR(w_grad[0], 0.5f, 1e-5f);
    EXPECT_NEAR(w_grad[1], 0.5f, 1e-5f);
    EXPECT_NEAR(w_grad[2], 0.7f, 1e-5f);
    EXPECT_NEAR(w_grad[3], 0.7f, 1e-5f);
    EXPECT_NEAR(w_grad[4], 0.9f, 1e-5f);
    EXPECT_NEAR(w_grad[5], 0.9f, 1e-5f);

    // Check bias gradients match PyTorch
    auto bias_grad = linear.bias.get_grad();
    ASSERT_EQ(bias_grad.size(), 2);
    EXPECT_NEAR(bias_grad[0], 2.0f, 1e-5f);
    EXPECT_NEAR(bias_grad[1], 2.0f, 1e-5f);
}

TEST(ModelTest, ForwardAndBackwardVSTorch) {
    /* Pytorch reference
    import torch

    input = torch.tensor([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]], requires_grad=True)
    weight = torch.tensor([[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]], requires_grad=True)
    bias = torch.tensor([0.1, 0.2], requires_grad=True)

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

    Linear* layer1 = static_cast<Linear*>(model.layers[0]);
    layer1->weight.init({0.1f, 0.2f, 0.3f, 0.4f, 0.5f, 0.6f});
    layer1->weight.reshape({3, 2});
    layer1->bias.init({0.1f, 0.2f});

    Linear* layer2 = static_cast<Linear*>(model.layers[2]);
    layer2->weight.init({0.7f, 0.8f});
    layer2->weight.reshape({2, 1});
    layer2->bias.init({0.3f});

    Tensor input({0.1f, 0.2f, 0.3f, 0.4f, 0.5f, 0.6f}, true);
    input.reshape({2, 3});

    Tensor output = model.forward(input);

    output.backward();

    auto params = model.parameters();

    // Verify final output shape and values
    ASSERT_EQ(output.shape().size(), 2);
    ASSERT_EQ(output.shape()[0], 2);
    ASSERT_EQ(output.shape()[1], 1);
    EXPECT_NEAR(output[0], 0.9080f, 1e-4f);
    EXPECT_NEAR(output[1], 1.3850f, 1e-4f);

    // Check input gradients
    auto input_grad = input.get_grad();
    ASSERT_EQ(input_grad.size(), 6);
    EXPECT_NEAR(input_grad[0], 0.2300f, 1e-4f);
    EXPECT_NEAR(input_grad[1], 0.5300f, 1e-4f);
    EXPECT_NEAR(input_grad[2], 0.8300f, 1e-4f);
    EXPECT_NEAR(input_grad[3], 0.2300f, 1e-4f);
    EXPECT_NEAR(input_grad[4], 0.5300f, 1e-4f);
    EXPECT_NEAR(input_grad[5], 0.8300f, 1e-4f);

    // Check layer1 weight gradients
    auto l1_w_grad = layer1->weight.get_grad();
    ASSERT_EQ(l1_w_grad.size(), 6);
    EXPECT_NEAR(l1_w_grad[0], 0.3500f, 1e-4f);
    EXPECT_NEAR(l1_w_grad[1], 0.4000f, 1e-4f);
    EXPECT_NEAR(l1_w_grad[2], 0.4900f, 1e-4f);
    EXPECT_NEAR(l1_w_grad[3], 0.5600f, 1e-4f);
    EXPECT_NEAR(l1_w_grad[4], 0.6300f, 1e-4f);
    EXPECT_NEAR(l1_w_grad[5], 0.7200f, 1e-4f);

    // Check layer1 bias gradients
    auto l1_bias_grad = layer1->bias.get_grad();
    ASSERT_EQ(l1_bias_grad.size(), 2);
    EXPECT_NEAR(l1_bias_grad[0], 1.4000f, 1e-4f);
    EXPECT_NEAR(l1_bias_grad[1], 1.6000f, 1e-4f);

    // Check layer2 weight gradients
    auto l2_w_grad = layer2->weight.get_grad();
    ASSERT_EQ(l2_w_grad.size(), 2);
    EXPECT_NEAR(l2_w_grad[0], 0.9100f, 1e-4f);
    EXPECT_NEAR(l2_w_grad[1], 1.3200f, 1e-4f);

    // Check layer2 bias gradients
    auto l2_bias_grad = layer2->bias.get_grad();
    ASSERT_EQ(l2_bias_grad.size(), 1);
    EXPECT_NEAR(l2_bias_grad[0], 2.0000f, 1e-4f);
}

int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
