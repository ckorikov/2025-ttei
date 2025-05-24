#ifndef TTIE_H
#define TTIE_H

#include <cassert>
#include <iostream>
#include <random>
#include <sstream>
#include <string>
#include <vector>

namespace ttie {

    struct TensorImpl;
    class Tensor;
    struct GradFn {
        std::vector<std::shared_ptr<TensorImpl>> inputs;

        GradFn(const std::initializer_list<std::shared_ptr<TensorImpl>> lst_) : inputs(lst_) {}
        virtual void backward(TensorImpl& output) = 0;
        virtual std::string typestr() const noexcept = 0;
        virtual ~GradFn() {}
    };

    struct TensorImpl final {
        std::vector<size_t> shape_;
        std::vector<float> data;
        std::vector<float> grad;
        std::vector<size_t> strides;
        std::shared_ptr<GradFn> grad_fn;
        size_t ndims;
        bool requires_grad = false;

        TensorImpl() : strides({1}), ndims(1) {}

        explicit TensorImpl(std::initializer_list<float> lst_) : data(lst_) {
            shape_ = {lst_.size()};
            grad.resize(lst_.size());
            ndims = 1;
            strides.resize(ndims);
            strides.back() = 1;
        }

        explicit TensorImpl(std::initializer_list<int> shape_, bool requires_grad_=false) : ndims(shape_.size()), strides(shape_.begin(),
        shape_.end()), shape_(shape_.begin(), shape_.end()), requires_grad(requires_grad_) {
            int total_size = std::accumulate(shape_.begin(), shape_.end(), 1, std::multiplies<size_t>());
            data.resize(total_size);
            for(int i = 0; i < shape_.size() - 1; ++i) {
                strides[i] = std::accumulate(std::next(shape_.begin(), i + 1), shape_.end(), 1, std::multiplies<size_t>());
            }
            strides.back() = 1;
        }

        explicit TensorImpl(std::initializer_list<size_t> shape_, bool requires_grad_=false) : ndims(shape_.size()), strides(shape_.begin(),
        shape_.end()), shape_(shape_.begin(), shape_.end()), requires_grad(requires_grad_) {
            int total_size = std::accumulate(shape_.begin(), shape_.end(), 1, std::multiplies<size_t>());
            data.resize(total_size);
            for(int i = 0; i < shape_.size() - 1; ++i) {
                strides[i] = std::accumulate(std::next(shape_.begin(), i + 1), shape_.end(), 1, std::multiplies<size_t>());
            }
            strides.back() = 1;
        }


        explicit TensorImpl(std::vector<size_t> shape_, bool requires_grad_=false) : ndims(shape_.size()), strides(shape_.begin(),
        shape_.end()), shape_(shape_.begin(), shape_.end()), requires_grad(requires_grad_) {
            int total_size = std::accumulate(shape_.begin(), shape_.end(), 1, std::multiplies<size_t>());
            data.resize(total_size);
            for(int i = 0; i < shape_.size() - 1; ++i) {
                strides[i] = std::accumulate(std::next(shape_.begin(), i + 1), shape_.end(), 1, std::multiplies<size_t>());
            }
            strides.back() = 1;
        }

        void backward() {
            if(grad_fn) {
                if(grad.empty()) {
                    grad.resize(data.size());
                    std::fill(grad.begin(), grad.end(), 1.0f);
                }
                grad_fn->backward(*this);
            }
        }

        void accumulate_grad(const std::vector<float>& grad_) {
            if(grad.empty()) {
                grad.resize(data.size());
                std::copy(grad_.begin(), grad_.end(), grad.begin());
            } else {
                std::transform(grad.begin(), grad.end(), grad_.begin(), grad.begin(), std::plus<float>());
            }
        }
    };


    struct SigmoidOp : public GradFn {
        SigmoidOp(std::shared_ptr<TensorImpl> input) : GradFn({input}) {}

        void backward(TensorImpl &output) override {
            const std::vector<float> &out_data = output.data;
            const std::vector<float> &grad_out = output.grad;
            std::vector<float> grad_input(out_data.size());

            for (size_t i = 0; i < out_data.size(); ++i) {
                float s = out_data[i];
                grad_input[i] = grad_out[i] * s * (1 - s);
            }

            inputs[0]->accumulate_grad(grad_input);
            inputs[0]->backward();
        }

        std::string typestr() const noexcept override { return "SigmoidOp"; }
    };


    struct TanhOp : public GradFn {
        TanhOp(std::shared_ptr<TensorImpl> input) : GradFn({input}) {}

        void backward(TensorImpl &output) override {
            const std::vector<float> &out_data = output.data;
            const std::vector<float> &grad_out = output.grad;
            std::vector<float> grad_input(out_data.size());

            for (size_t i = 0; i < out_data.size(); ++i) {
                float t = out_data[i];
                grad_input[i] = grad_out[i] * (1 - t * t);
            }

            inputs[0]->accumulate_grad(grad_input);
            inputs[0]->backward();
        }

        std::string typestr() const noexcept override { return "TanhOp"; }
    };

    struct AddOp final : public GradFn {
        AddOp(std::shared_ptr<TensorImpl> a, std::shared_ptr<TensorImpl> b) : GradFn({a, b}) {}

        void backward(TensorImpl& output) override { // c = a + b, ∂c/∂a = 1, ∂c/∂b = 1;
            inputs[0]->accumulate_grad(output.grad);
            inputs[1]->accumulate_grad(output.grad);

            inputs[0]->backward();
            inputs[1]->backward();
        }
        std::string typestr() const noexcept override { return "AddOp"; }
    };

    struct MulOp final : public GradFn {
        MulOp(std::shared_ptr<TensorImpl> a, std::shared_ptr<TensorImpl> b) : GradFn({a, b}) {}

        void backward(TensorImpl& output) override { // c = a * b, ∂c/∂a = b, ∂c/∂b = a
            std::vector<float> final_grad_a = inputs[1]->data;
            std::vector<float> final_grad_b = inputs[0]->data;

            std::transform(final_grad_a.begin(), final_grad_a.end(), output.grad.begin(), final_grad_a.begin(), std::multiplies<float>());
            std::transform(final_grad_b.begin(), final_grad_b.end(), output.grad.begin(), final_grad_b.begin(), std::multiplies<float>());

            inputs[0]->accumulate_grad(final_grad_a);
            inputs[1]->accumulate_grad(final_grad_b);

            inputs[0]->backward();
            inputs[1]->backward();
        }
        std::string typestr() const noexcept override { return "MulOp"; }
    };

    struct SubOp final : public GradFn {
        SubOp(std::shared_ptr<TensorImpl> a, std::shared_ptr<TensorImpl> b) : GradFn({a, b}) {}

        void backward(TensorImpl& output) override { // c = a - b, ∂c/∂a = 1, ∂c/∂b = -1
            std::vector<float> final_grad_a(output.grad.size(), 1.0f);
            std::vector<float> final_grad_b(output.grad.size(), -1.0f);

            std::transform(final_grad_a.begin(), final_grad_a.end(), output.grad.begin(), final_grad_a.begin(), std::multiplies<float>());
            std::transform(final_grad_b.begin(), final_grad_b.end(), output.grad.begin(), final_grad_b.begin(), std::multiplies<float>());

            inputs[0]->accumulate_grad(final_grad_a);
            inputs[1]->accumulate_grad(final_grad_b);

            inputs[0]->backward();
            inputs[1]->backward();
        }
        std::string typestr() const noexcept override { return "SubOp"; }
    };

    struct MaxOp final : public GradFn {
        MaxOp(std::shared_ptr<TensorImpl> a, std::shared_ptr<TensorImpl> b) : GradFn({a, b}) {}
        MaxOp(std::shared_ptr<TensorImpl> a) : GradFn({a}) {}

        void backward(TensorImpl& output) override {
            /*
             *
             *  c = max(a, b)
             *  c = {a, a >= b; b, a <= b}
             *  ∂c/∂a = {1, a >= b; 0, a <= b}
             *  ∂c/∂b = {0, a >= b; 1, a <= b}
             *
            */
            std::vector<float> final_grad_a(output.grad.size());
            std::vector<float> final_grad_b(output.grad.size());

            std::transform(
                inputs[0]->data.begin(),
                inputs[0]->data.end(),
                inputs[1]->data.begin(),
                final_grad_a.begin(),
                // [](float a_, float b_){ if (std::abs(a_ - b_) < 1e-8) { return 0.5f; } else { return a_ > b_ ? 1.0f : 0.0f; } }
                [](float a_, float b_){ return a_ > b_ ? 1.0f : 0.0f; }
            );

            std::transform(
                inputs[0]->data.begin(),
                inputs[0]->data.end(),
                inputs[1]->data.begin(),
                final_grad_b.begin(),
                // [](float a_, float b_){ if (std::abs(a_ - b_) < 1e-8) { return 0.5f; } else { return a_ > b_ ? 1.0f : 0.0f; } }
                [](float a_, float b_){ return a_ > b_ ? 0.0f : 1.0f; }
            );

            std::transform(
                final_grad_a.begin(),
                final_grad_a.end(),
                output.grad.begin(),
                final_grad_a.begin(),
                std::multiplies<float>()
            );
            std::transform(
                final_grad_b.begin(),
                final_grad_b.end(),
                output.grad.begin(),
                final_grad_b.begin(),
                std::multiplies<float>()
            );

            inputs[0]->accumulate_grad(final_grad_a);
            inputs[1]->accumulate_grad(final_grad_b);

            inputs[0]->backward();
            inputs[1]->backward();
        }
        std::string typestr() const noexcept override { return "MaxOp"; }
    };

    struct UnaryMinusOp final : public GradFn {
        UnaryMinusOp(std::shared_ptr<TensorImpl> a) : GradFn({a}) {}

        void backward(TensorImpl& output) override { // c = -a, ∂c/∂a = -1
            std::vector<float> final_grad(output.grad.size(), -1.0f);

            std::transform(final_grad.begin(), final_grad.end(), output.grad.begin(), final_grad.begin(), std::multiplies<float>());

            inputs[0]->accumulate_grad(final_grad);
            inputs[0]->backward();
        }
        std::string typestr() const noexcept override { return "UnaryMinusOp"; }
    };

    struct SqrtOp final : public GradFn {
        SqrtOp(std::shared_ptr<TensorImpl> a) : GradFn({a}) {}

        void backward(TensorImpl& output) override { // c = sqrt(a), ∂c/∂a = 1 / (2 * sqrt(a) + 1e-8)
            std::vector<float> final_grad(output.data.size());
            std::transform(inputs[0]->data.begin(), inputs[0]->data.end(), final_grad.begin(), [](float t){ return 1.0f / (2.0 * std::sqrt(t) + 1e-8); });

            std::transform(final_grad.begin(), final_grad.end(), output.grad.begin(), final_grad.begin(), std::multiplies<float>());

            inputs[0]->accumulate_grad(final_grad);

            inputs[0]->backward();
        }
        std::string typestr() const noexcept override { return "SqrtOp"; }
    };

    struct LogOp final : public GradFn {
        LogOp(std::shared_ptr<TensorImpl> a) : GradFn({a}) {}

        void backward(TensorImpl& output) override { // c = log(a), ∂c/∂a = 1 / ( a + 1e-8)
            std::vector<float> final_grad(output.data.size());
            std::transform(inputs[0]->data.begin(), inputs[0]->data.end(), final_grad.begin(), [](float t){ return 1.0f / (t + 1e-8); });

            std::transform(final_grad.begin(), final_grad.end(), output.grad.begin(), final_grad.begin(), std::multiplies<float>());

            inputs[0]->accumulate_grad(final_grad);

            inputs[0]->backward();
        }
        std::string typestr() const noexcept override { return "LogOp"; }
    };

    struct DivOp final : public GradFn {
        DivOp(std::shared_ptr<TensorImpl> a, std::shared_ptr<TensorImpl> b) : GradFn({a, b}) {}

        void backward(TensorImpl& output) override { // c = a / b, ∂c/∂b = -a / (b^2 + 1e-8), ∂c/∂a = 1 / b
            std::vector<float> final_grad_a(output.data.size());
            std::vector<float> final_grad_b(output.data.size());

            std::transform(inputs[0]->data.begin(), inputs[0]->data.end(), inputs[1]->data.begin(), final_grad_a.begin(),
                [](float a, float b){ return 1.0f / (b + 1e-8); });
            std::transform(inputs[0]->data.begin(), inputs[0]->data.end(), inputs[1]->data.begin(), final_grad_b.begin(),
                [](float a, float b){ return -a / ( b * b + 1e-8); });

            std::transform(final_grad_a.begin(), final_grad_a.end(), output.grad.begin(), final_grad_a.begin(), std::multiplies<float>());
            std::transform(final_grad_b.begin(), final_grad_b.end(), output.grad.begin(), final_grad_b.begin(), std::multiplies<float>());

            inputs[0]->accumulate_grad(final_grad_a);
            inputs[1]->accumulate_grad(final_grad_b);

            inputs[0]->backward();
            inputs[1]->backward();
        }
        std::string typestr() const noexcept override { return "DivOp"; }
    };

    struct SumOp final : public GradFn {
        size_t axis;

        SumOp(std::shared_ptr<TensorImpl> input, size_t axis_)
            : GradFn({input}), axis(axis_) {}

        void backward(TensorImpl& output) override;

        std::string typestr() const noexcept override { return "SumOp"; }
    };


    struct MatrixMatrixProdOp final : public GradFn {
        MatrixMatrixProdOp(const std::shared_ptr<TensorImpl>& a, const std::shared_ptr<TensorImpl>& b) : GradFn({a, b}) {}

        void backward(TensorImpl& output) override;
        std::string typestr() const noexcept override { return "MatrixMatrixProdOp"; }
    };

    struct MatrixVectorProdOp final : public GradFn {
        MatrixVectorProdOp(const std::shared_ptr<TensorImpl>& a, const std::shared_ptr<TensorImpl>& b) : GradFn({a, b}) {}

        void backward(TensorImpl& output) override;
        std::string typestr() const noexcept override { return "MatrixVectorProdOp"; }
    };

struct BroadcastOp : public GradFn {
    std::vector<size_t> input_shape;
    std::vector<size_t> target_shape;

    // BroadcastOp(const std::vector<size_t>& input_shape_, const std::vector<size_t>& target_shape_)
    BroadcastOp(std::shared_ptr<TensorImpl> a, const std::vector<size_t>& input_shape_, const std::vector<size_t>& target_shape_)
        : GradFn({a}), input_shape(input_shape_), target_shape(target_shape_) {}

    void backward(TensorImpl& output) override;

    std::string typestr() const noexcept override { return "BroadcastOp"; }

};


    template <typename T>
    static std::string vector_to_string(const std::vector<T> &vec, size_t limit = 5) {
        std::stringstream ss;
        ss << "[";
        size_t preview_size = std::min(limit, vec.size());
        for (size_t i = 0; i < preview_size; ++i) {
            ss << vec[i];
            if (i < preview_size - 1) {
                ss << ", ";
            }
        }
        if (vec.size() > preview_size) {
            ss << ", ...";
        }
        ss << "]";
        return ss.str();
    }

    class Tensor final {
        private:
            std::shared_ptr<TensorImpl> impl_;

        public:
            Tensor(bool requires_grad_=false) : impl_(new TensorImpl) { impl_->requires_grad = requires_grad_; }

            Tensor(const std::vector<float>& data_, bool requires_grad_=false) : impl_(new TensorImpl) {
                impl_->data = data_;
                impl_->ndims = 1;
                impl_->strides.resize(impl_->ndims);
                std::fill(impl_->strides.begin(), impl_->strides.end(), 1);
                impl_->shape_ = {data_.size()};
                impl_->requires_grad = requires_grad_;
            }

            Tensor(const std::vector<float>& data_,  std::vector<size_t> shape, bool requires_grad_=false) : impl_(new TensorImpl) {
                impl_->data = data_;
                impl_->ndims = shape.size();
                impl_->strides.resize(impl_->ndims);
                std::fill(impl_->strides.begin(), impl_->strides.end(), 1);
                impl_->shape_ = shape;
                impl_->requires_grad = requires_grad_;
                for(int i = 0; i < shape.size() - 1; ++i) {
                    impl_->strides[i] = std::accumulate(std::next(shape.begin(), i + 1), shape.end(), 1, std::multiplies<size_t>());
                }
                impl_->strides.back() = 1;
            }

            Tensor(std::initializer_list<float> data_, bool requires_grad_=false) : impl_(new TensorImpl) {
                impl_->data.assign(data_);
                impl_->ndims = 1;
                impl_->strides.resize(impl_->ndims);
                impl_->strides[0] = 1;
                impl_->shape_ = {data_.size()};
                impl_->requires_grad = requires_grad_;
            }

            Tensor(std::initializer_list<int> shape_, bool requires_grad_=false) : impl_(new TensorImpl(shape_, requires_grad_)) {}

            Tensor(std::initializer_list<size_t> shape_, bool requires_grad_=false) : impl_(new TensorImpl(shape_, requires_grad_)) {}

            Tensor(std::vector<size_t> shape_, bool requires_grad_=false) : impl_(new TensorImpl(shape_, requires_grad_)) {}

            Tensor(const TensorImpl& impl) {
                impl_ = std::make_shared<TensorImpl>(impl);
            }

            Tensor(const std::shared_ptr<TensorImpl>& impl, bool deep_copy=false) {
                if(deep_copy) {
                    impl_ = std::make_shared<TensorImpl>(*impl);
                } else {
                    impl_ = impl;
                }
            }

            void init(std::initializer_list<float> data_) {
                if(!empty() && data_.size() != size()) {
                    throw std::invalid_argument("Can't initialize tensor of size " + std::to_string(size()) + " with data which size is " + std::to_string(data_.size()));
                }
                impl_->data = data_;
            }

            Tensor& operator=(std::initializer_list<float> list) {
                Tensor tmp(list);
                std::swap(*this, tmp);
                return *this;
            }


            void set_grad(std::shared_ptr<GradFn> node) {
                impl_->grad_fn = node;
            }

        private:

            Tensor& inplace_add(const Tensor& b) {
                for(size_t i = 0; i < shape()[0]; ++i) {
                    for(size_t j = 0; j < shape()[1]; ++j) {
                        this->operator[]({i, j}) = this->operator[]({i, j}) + b[{i, j}];
                    }
                }

                return *this;
            }

            Tensor& inplace_mul(const Tensor& rhs) {
                for(size_t i = 0; i < shape()[0]; ++i) {
                    for(size_t j = 0; j < shape()[1]; ++j) {
                        this->operator[]({i, j}) = this->operator[]({i, j}) * rhs[{i, j}];
                    }
                }
                return *this;
            }

            Tensor& inplace_sub(const Tensor& rhs) {
                for(size_t i = 0; i < shape()[0]; ++i) {
                    for(size_t j = 0; j < shape()[1]; ++j) {
                        this->operator[]({i, j}) = this->operator[]({i, j}) - rhs[{i, j}];
                    }
                }
                return *this;
            }

            Tensor& inplace_div(const Tensor& rhs) {
                for(size_t i = 0; i < shape()[0]; ++i) {
                    for(size_t j = 0; j < shape()[1]; ++j) {
                        this->operator[]({i, j}) = this->operator[]({i, j}) / rhs[{i, j}];
                    }
                }
                return *this;
            }

        public:

            bool empty() const noexcept { return impl_->data.empty(); }

            bool requires_grad() const noexcept { return impl_->requires_grad; }
            void requires_grad(bool new_state_) noexcept { impl_->requires_grad = new_state_; }

            void backward() const { impl_->backward(); }

            void reshape(std::vector<size_t> new_shape) {
                size_t new_size = std::accumulate(new_shape.begin(), new_shape.end(), 1, std::multiplies<size_t>());
                if(empty()) {
                    impl_->data.resize(new_size);
                    impl_->shape_ = new_shape;
                    impl_->strides.resize(new_shape.size());
                    impl_->ndims = new_shape.size();
                }
                if(size() != new_size) {
                    throw std::invalid_argument("Invalid shape for tensor of size " + std::to_string(size()));
                }

                impl_->shape_ = new_shape;
                impl_->strides.resize(new_shape.size());
                impl_->strides.back() = 1;
                impl_->ndims = new_shape.size();

                for(int i = 0; i < impl_->shape_.size() - 1; ++i) {
                    impl_->strides[i] = std::accumulate(std::next(impl_->shape_.begin(), i + 1), impl_->shape_.end(), 1, std::multiplies<size_t>());
                }
            }

            const size_t dim(int i) const noexcept { return impl_->shape_[i]; }

            const std::vector<size_t>& shape() const noexcept { return impl_->shape_; }

            size_t ndims() const noexcept { return impl_->ndims; }

            float& operator[](int i) { return impl_->data[i]; }
            const float& operator[](int i) const { return impl_->data[i]; }

            float& operator[](std::initializer_list<size_t> idx) {
                return impl_->data[compute_index(idx)];
            }
            const float& operator[](std::initializer_list<size_t> idx) const {
                return impl_->data[compute_index(idx)];
            }

            float& operator[](std::initializer_list<int> idx) {
                return impl_->data[compute_index(idx)];
            }
            const float& operator[](std::initializer_list<int> idx) const {
                return impl_->data[compute_index(idx)];
            }

            Tensor& operator+=(const Tensor& rhs) {
                if(impl_->requires_grad || rhs.impl_->requires_grad) {
                    throw std::runtime_error("A variable requires grad is used in a in-place += operation.");
                }
                if(size() != rhs.size() || !std::equal(impl_->shape_.begin(), impl_->shape_.end(), rhs.impl_->shape_.begin())) {
                    throw std::invalid_argument("Can't apply operator+=() for tensors of shape " + vector_to_string(impl_->shape_) + " and " + vector_to_string(rhs.impl_->shape_));
                }

                std::cout << "impl_->requires_grad || rhs.impl_->requires_grad = " << impl_->requires_grad || rhs.impl_->requires_grad;

                return inplace_add(rhs);
            }

            Tensor& operator-=(const Tensor& rhs) {
                if(impl_->requires_grad || rhs.impl_->requires_grad) {
                    throw std::runtime_error("A variable requires grad is used in a in-place -= operation.");
                }
                if(size() != rhs.size() || !std::equal(impl_->shape_.begin(), impl_->shape_.end(), rhs.impl_->shape_.begin())) {
                    throw std::invalid_argument("Can't apply operator-=() for tensors of shape " + std::to_string(size()) + " and " + std::to_string(rhs.size()));
                }

                return inplace_sub(rhs);
            }

            Tensor& operator*=(const Tensor& rhs) {
                if(impl_->requires_grad || rhs.impl_->requires_grad) {
                    throw std::runtime_error("A variable requires grad is used in a in-place *= operation.");
                }
                if(size() != rhs.size() || !std::equal(impl_->shape_.begin(), impl_->shape_.end(), rhs.impl_->shape_.begin())) {
                    throw std::invalid_argument("Can't apply operator*=() for tensors of shape " + std::to_string(size()) + " and " + std::to_string(rhs.size()));
                }

                return inplace_mul(rhs);
            }

            Tensor& operator/=(const Tensor& rhs) {
                if(impl_->requires_grad || rhs.impl_->requires_grad) {
                    throw std::runtime_error("A variable requires grad is used in a in-place /= operation.");
                }
                if(size() != rhs.size() || !std::equal(impl_->shape_.begin(), impl_->shape_.end(), rhs.impl_->shape_.begin())) {
                    throw std::invalid_argument("Can't apply operator/=() for tensors of shape " + std::to_string(size()) + " and " + std::to_string(rhs.size()));
                }

                return inplace_div(rhs);
            }

        private:
            size_t compute_index(const std::initializer_list<size_t>& idx) const {
                if(std::equal(impl_->shape_.begin(), impl_->shape_.end(), idx.begin(), std::less<float>())) {
                    throw std::invalid_argument("Invalid indexes for tensor of shape" + vector_to_string(impl_->shape_));
                }

                return std::inner_product(impl_->strides.begin(), impl_->strides.end(), idx.begin(), 0);
            }

            int compute_index(const std::initializer_list<int>& idx) const {
                if(std::equal(impl_->shape_.begin(), impl_->shape_.end(), idx.begin(), std::less<float>())) {
                    throw std::invalid_argument("Invalid indexes for tensor of shape" + vector_to_string(impl_->shape_));
                }

                return std::inner_product(impl_->strides.begin(), impl_->strides.end(), idx.begin(), 0);
            }

        public:
            size_t size() const {
                if(empty()) { return 0; }

                #if 0
                    return impl_->data.size();
                #endif
                return std::accumulate(impl_->shape_.begin(), impl_->shape_.end(), 1, std::multiplies<size_t>());
            }

            void zero_grad() {
                std::fill(impl_->grad.begin(), impl_->grad.end(), 0.0f);
            }

            void fill(float arg) { std::fill(impl_->data.begin(), impl_->data.end(), arg); }

            const std::vector<float>& get_data() const noexcept { return impl_->data; }

            const std::vector<float>& get_grad() const noexcept { return impl_->grad; }

            Tensor operator+(const Tensor& rhs) {
                if(size() != rhs.size() || !std::equal(impl_->shape_.begin(), impl_->shape_.end(), rhs.impl_->shape_.begin())) {
                    throw std::invalid_argument("Can't apply operator+() for tensors of shape (" + std::to_string(impl_->shape_[0]) + ", " + \
                        std::to_string(impl_->shape_[1]) + ") and (" + std::to_string(rhs.shape()[0]) + ", " + std::to_string(rhs.shape()[1]) + ").");
                }
                Tensor tmp;
                tmp.impl_ = std::make_shared<TensorImpl>(*impl_);
                tmp.inplace_add(rhs);
                tmp.set_grad(std::make_shared<AddOp>(impl_, rhs.impl_));
                return tmp;
            }

            Tensor operator-(const Tensor& rhs) {
                if(size() != rhs.size() || !std::equal(impl_->shape_.begin(), impl_->shape_.end(), rhs.impl_->shape_.begin())) {
                    throw std::invalid_argument("Can't apply operator-() for tensors of shape " + std::to_string(size()) + " and " + std::to_string(rhs.size()));
                }
                Tensor tmp;
                tmp.impl_ = std::make_shared<TensorImpl>(*impl_);
                tmp.inplace_sub(rhs);
                tmp.set_grad(std::make_shared<SubOp>(impl_, rhs.impl_));
                return tmp;
            }

            Tensor operator-() {
                Tensor tmp;
                tmp.impl_ = std::make_shared<TensorImpl>(*impl_);
                std::transform(tmp.impl_->data.begin(), tmp.impl_->data.end(), tmp.impl_->data.begin(), [](float t){ return -t; });
                tmp.set_grad(std::make_shared<UnaryMinusOp>(impl_));
                return tmp;
            }

            Tensor operator*(const Tensor& rhs) {
                if(size() != rhs.size() || !std::equal(impl_->shape_.begin(), impl_->shape_.end(), rhs.impl_->shape_.begin())) {
                    throw std::invalid_argument("Can't apply operator*() for tensors of shape " + std::to_string(size()) + " and " + std::to_string(rhs.size()));
                }
                Tensor tmp;
                tmp.impl_ = std::make_shared<TensorImpl>(*impl_);
                tmp.inplace_mul(rhs);
                tmp.set_grad(std::make_shared<MulOp>(impl_, rhs.impl_));
                return tmp;
            }

            Tensor operator/(const Tensor& rhs) {
                if(size() != rhs.size() || !std::equal(impl_->shape_.begin(), impl_->shape_.end(), rhs.impl_->shape_.begin())) {
                    throw std::invalid_argument("Can't apply operator/() for tensors of shape " + std::to_string(size()) + " and " + std::to_string(rhs.size()));
                }
                Tensor tmp;
                tmp.impl_ = std::make_shared<TensorImpl>(*impl_);
                tmp.inplace_div(rhs);
                tmp.set_grad(std::make_shared<DivOp>(impl_, rhs.impl_));
                return tmp;
            }

            static Tensor sqrt(const Tensor& arg) {
                Tensor tmp;
                tmp.impl_ = std::make_shared<TensorImpl>(*arg.impl_);
                std::transform(tmp.impl_->data.begin(), tmp.impl_->data.end(), tmp.impl_->data.begin(), std::sqrtf);
                tmp.set_grad(std::make_shared<SqrtOp>(arg.impl_));
                return tmp;
            }

            static Tensor log(const Tensor& arg) {
                Tensor tmp;
                tmp.impl_ = std::make_shared<TensorImpl>(*arg.impl_);
                std::transform(tmp.impl_->data.begin(), tmp.impl_->data.end(), tmp.impl_->data.begin(), std::logf);
                tmp.set_grad(std::make_shared<LogOp>(arg.impl_));
                return tmp;
            }

            static Tensor max(float a, const Tensor& b) {
                Tensor tmp;
                tmp.reshape_as(b);
                tmp.fill(a);
                return max(tmp, b);
            }

            static Tensor matmul(const Tensor& a, const Tensor& b) {
                if(a.ndims() == 1 && b.ndims() == 1) {
                    return dot_product(a, b);
                } else if(a.ndims() == 2 && b.ndims() == 2) {
                    return matrix_matrix_product(a, b);
                } else if(a.ndims() == 1 && b.ndims() == 2) {
                    return matrix_vector_product(b, a);
                } else if(a.ndims() == 2 && b.ndims() == 1) {
                    return matrix_vector_product(a, b);
                } else {
                    throw std::invalid_argument("Invalid tensors shape for matmul.");
                }
            }

            static Tensor dot_product(const Tensor& a, const Tensor& b) {
                if(a.shape()[0] != b.shape()[0]) {
                    throw std::invalid_argument("Invalid shapes for dot product: " + std::to_string(a.shape()[0]) + \
                        " and " + std::to_string(b.shape()[0])
                    );
                }
                Tensor c({a.shape()[0]});
                std::transform(
                    a.impl_->data.begin(), a.impl_->data.end(), b.impl_->data.begin(),
                    c.impl_->data.begin(),
                    std::multiplies<float>()
                );
                return c;
            }

            static Tensor sigmoid(const Tensor& input) {
                Tensor output(input.shape());
                for (size_t i = 0; i < input.size(); ++i) {
                    output[i] = 1.0f / (1.0f + std::exp(-input[i]));
                }

                if (input.requires_grad()) {
                    output.set_grad(std::make_shared<SigmoidOp>(input.impl_));
                }

                return output;
            }

            static Tensor tanh(const Tensor& input) {
                Tensor output(input.shape());
                for (size_t i = 0; i < input.size(); ++i) {
                    output[i] = std::tanh(input[i]);
                }

                if (input.requires_grad()) {
                    output.set_grad(std::make_shared<TanhOp>(input.impl_));
                }

                return output;
            }

            Tensor transpose() {
                if(impl_->ndims != 2) {
                    throw std::invalid_argument("Invalid tensor shape for transpose: (" + std::to_string(shape()[0]) + ", " + std::to_string(shape()[1]) + ").");
                }

                Tensor tmp = *this;

                std::swap(tmp.impl_->shape_[0], tmp.impl_->shape_[1]);
                std::swap(tmp.impl_->strides[0], tmp.impl_->strides[1]);

                return tmp;
            }

            static Tensor matrix_matrix_product(const Tensor& a, const Tensor& b) {
                int M = a.shape()[0], N = a.shape()[1], K = b.shape()[1];
                if(N != b.shape()[0]) {
                    throw std::invalid_argument("Invalid shapes for matmul: (" + std::to_string(M) + ", " + \
                        std::to_string(N) + ") and (" + std::to_string(b.shape()[0]) + ", " + std::to_string(K) + ")."
                    );
                }
                Tensor c({M, K});
                c.impl_->requires_grad = a.requires_grad() || b.requires_grad();
                std::vector<float> fixed_col(N);

                for(size_t i = 0; i < K; ++i) {
                    for(size_t q = 0; q < N; ++q) {
                        fixed_col[q] = b[{q, i}];
                    }
                    for(size_t j = 0; j < M; ++j) {
                        float acc = 0.0f;
                        for (size_t k = 0; k < N; ++k) {
                            acc += a[{j, k}] * fixed_col[k];
                        }
                        c[{j, i}] = acc;
                    }
                }
                c.set_grad(std::make_shared<MatrixMatrixProdOp>(a.impl_, b.impl_));
                return c;
            }

            Tensor broadcast_to(const std::vector<size_t>& target_shape) const {
                if(shape() == target_shape) {
                    return *this;
                }
                if(ndims() > 1) {
                    throw std::invalid_argument("Can't broadcast tensor with ndims > 1");
                }
                size_t new_sz = std::reduce(target_shape.begin(), target_shape.end(), 1, std::multiplies<size_t>());
                if(new_sz < size()) {
                    throw std::invalid_argument("Can't broadcast tensor from size " + std::to_string(size()) + " to size " + std::to_string(new_sz));
                }

                Tensor out(impl_, requires_grad());
                out.impl_->shape_ = target_shape;
                out.impl_->strides = {0, 1};

                if(requires_grad()) {
                    auto grad_fn = std::make_shared<BroadcastOp>(impl_, shape(), target_shape);
                    grad_fn->inputs = {impl_};
                    out.impl_->grad_fn = grad_fn;
                }

                return out;
            }

            static Tensor matrix_vector_product(const Tensor& a, const Tensor& b) {
                if(a.shape()[1] != b.shape()[0]) {
                    throw std::invalid_argument("Invalid arguments shape for matrix_vector_product: (" + std::to_string(a.shape()[0]) + \
                    ", " + std::to_string(a.shape()[1]) + ") and (" + std::to_string(b.shape()[0]) + ")."
                    );
                }
                int M = a.shape()[0], N = a.shape()[1];
                Tensor c({N});

                for(int i = 0; i < M; ++i) {
                    c[i] = std::transform_reduce(
                        std::next(a.impl_->data.begin(), i * M),
                        std::next(a.impl_->data.begin(), (i + 1) * M),
                        b.impl_->data.begin(),
                        0.0f
                    );
                }
                c.set_grad(std::make_shared<MatrixVectorProdOp>(a.impl_, b.impl_));
                return c;
            }


            static Tensor max(const Tensor& a, const Tensor& b) {
                Tensor tmp;
                tmp.impl_ = std::make_shared<TensorImpl>(*a.impl_);
                std::transform(
                    tmp.impl_->data.begin(),
                    tmp.impl_->data.end(),
                    b.impl_->data.begin(),
                    tmp.impl_->data.begin(),
                    [](float a_, float b_){ return std::max(a_, b_); }
                );
                tmp.set_grad(std::make_shared<MaxOp>(a.impl_, b.impl_));
                return tmp;
            }

            void reshape_as(const Tensor& arg) { reshape(arg.shape()); }

            Tensor sum(size_t axis) const {
                if(ndims() > 2) {
                    throw std::invalid_argument("Can't apply sum() for tensor with ndims > 2");
                }
                if(axis > size()) {
                    throw std::invalid_argument("Can't find axis " + std::to_string(axis) + " in tensor");
                }
                if(shape()[axis] == 1 || shape()[axis] == 0) {
                    return *this;
                }

                Tensor out;
                size_t m = shape()[0], n = shape()[1];

                if(axis == 0) {
                    out = Tensor({n});
                    for(size_t i = 0; i < n; ++i) {
                        for(size_t j = 0; j < m; ++j) {
                            out[i] = out[i] + this->operator[]({j, i});
                        }
                    }
                } else {
                    out = Tensor({m});
                    for(size_t i = 0; i < m; ++i) {
                        for(size_t j = 0; j < n; ++j) {
                            out[i] = out[i] + this->operator[]({j, i});
                        }
                    }
                }

                if (requires_grad()) {
                    auto grad_fn = std::make_shared<SumOp>(impl_, axis);
                    grad_fn->inputs = {impl_};
                    out.impl_->grad_fn = grad_fn;
                    out.impl_->requires_grad = true;
                }

                return out;
            }


            friend std::ostream &operator<<(std::ostream &os, const Tensor &t) {
                os << "Tensor@" << &t;

                if(t.empty()) {
                    os << "(not initialized)";
                    return os;
                }

                os << "(shape=" << vector_to_string(t.shape());

                if(!t.empty()) {
                    os << ", data=" << vector_to_string(t.get_data());
                } else {
                    os << ", data=[no data]";
                }

                if(!t.empty()) {
                    os << ", grad=" << vector_to_string(t.get_grad());
                }

                if(t.impl_->grad_fn){
                    os << ", grad_fn=" << t.impl_->grad_fn->typestr();
                } else {
                    os << ", grad_fn=nullptr";
                }

                os << ")";
                return os;
            }
    };

    void MatrixMatrixProdOp::backward(TensorImpl& output) { // C = A @ B.
            Tensor grad_a(inputs[0]->data.size());
            Tensor grad_b(inputs[1]->data.size());

            grad_a = Tensor::matmul(Tensor(output.grad, output.shape_, true), Tensor(inputs[1], true).transpose());
            grad_b = Tensor::matmul(Tensor(inputs[0], true).transpose(), Tensor(output.grad, output.shape_, true));

            std::vector<float> final_grad_a = grad_a.get_data();
            std::vector<float> final_grad_b = grad_b.get_data();

            inputs[0]->accumulate_grad(final_grad_a);
            inputs[1]->accumulate_grad(final_grad_b);

            inputs[0]->backward();
            inputs[1]->backward();
        }

    void MatrixVectorProdOp::backward(TensorImpl& output) { // C = A @ b.
            Tensor grad_a(inputs[0]->data.size());
            Tensor grad_b(inputs[1]->data.size());

            grad_a = Tensor::matmul(Tensor(output), Tensor(inputs[1], true).transpose());
            grad_b = Tensor::matmul(Tensor(inputs[0], true).transpose(), Tensor(output));

            std::vector<float> final_grad_a = grad_a.get_data();
            std::vector<float> final_grad_b = grad_b.get_data();

            inputs[0]->accumulate_grad(final_grad_a);
            inputs[1]->accumulate_grad(final_grad_b);

            inputs[0]->backward();
            inputs[1]->backward();
        }

        Tensor reduce_sum_to_shape(const Tensor& input, const std::vector<size_t>& target_shape) {
            Tensor output = input;
            for (size_t i = 0; i < input.shape().size(); ++i) {
                if (i >= target_shape.size() || input.shape()[i] != target_shape[i]) {
                    output = output.sum(i);
                }
            }
            return output;
        }

        void BroadcastOp::backward(TensorImpl& output) {
            Tensor out_grad(output);
            for(size_t i = 0; i < output.data.size(); ++i) {
                out_grad[i] = output.grad[i];
            }
            Tensor reduced = reduce_sum_to_shape(out_grad, input_shape);

            inputs[0]->accumulate_grad(reduced.get_data());

            inputs[0]->backward();
        }

        void SumOp::backward(TensorImpl& output)  {
            Tensor grad_out(output.grad);  // ∂L/∂(sum(x))

            std::vector<size_t> input_shape = inputs[0]->shape_;
            std::vector<size_t> target_shape = input_shape;
            target_shape[axis] = 1;

            grad_out.reshape(target_shape);
            grad_out = grad_out.broadcast_to(input_shape);

            inputs[0]->accumulate_grad(grad_out.get_grad());

            inputs[0]->backward();
        }


    struct Layer {
        virtual void forward(const Tensor &input, Tensor &output) = 0;
        virtual std::string to_string() const = 0;
        virtual std::vector<Tensor *> parameters() = 0;
        virtual ~Layer() {}
    };

    struct Linear : Layer {
        Tensor weight;
        Tensor bias;

        Linear(size_t in_features, size_t out_features) : weight({in_features, out_features}, true), bias({out_features}, true) {
            std::random_device rd;
            std::mt19937 gen(rd());
            std::uniform_real_distribution<float> dis(-0.1f, 0.1f);

            for(size_t i = 0; i < in_features * out_features; ++i) {
                weight[i] = dis(gen);
            }

            bias.reshape({out_features});

            for(size_t i = 0; i < out_features; ++i) {
                bias[i] = dis(gen);
            }
        }

        std::vector<Tensor *> parameters() override { return {&weight, &bias}; }

        void forward(const Tensor &input, Tensor &output) override {
            size_t in_features = weight.shape()[0];
            size_t out_features = weight.shape()[1];
            output.reshape({input.dim(0), out_features});

            if(input.ndims() == 2) {
                output = Tensor::matmul(input, weight) + bias.broadcast_to({input.shape()[0], out_features});
            } else {
                output = Tensor::matmul(input, weight) + bias;
            }
        }

        std::string to_string() const override {
            std::stringstream ss;
            ss << "Linear(in_features=" << weight.dim(0) << ", out_features=" << weight.dim(1) << ")";
            return ss.str();
        }
    };


    struct Sigmoid final : public Layer {
        std::vector<Tensor *> parameters() override { return {}; }

        void forward(const Tensor &input, Tensor &output) override {
            output = Tensor::sigmoid(input);
        }

        std::string to_string() const override { return "Sigmoid()"; }
    };


    struct Tanh : Layer {
        std::vector<Tensor *> parameters() override { return {}; }

        void forward(const Tensor &input, Tensor &output) override {
            output = Tensor::tanh(input);
        }

        std::string to_string() const override { return "Tanh()"; }
    };


    struct ReLU : Layer {
        std::vector<Tensor *> parameters() override { return {}; }

        void forward(const Tensor &input, Tensor &output) override {
            output.reshape_as(input);
            output = Tensor::max(0.0f, input);
        }

        std::string to_string() const override { return "ReLU()"; }
    };

    struct Model {
        std::vector<Layer*> layers;

        void add_layer(Layer* layer) {
            layers.push_back(layer);
        }

        Tensor forward(const Tensor& input) {
            Tensor current = input;
            for (Layer* layer : layers) {
                Tensor next;
                layer->forward(current, next);
                current = next;
            }
            return current;
        }

        void backward(const Tensor& output) {
            output.backward();
        }

        std::vector<Tensor*> parameters() {
            std::vector<Tensor*> params;
            for (Layer* layer : layers) {
                auto layer_params = layer->parameters();
                params.insert(params.end(), layer_params.begin(), layer_params.end());
            }
            return params;
        }

        std::string to_string() const {
            std::stringstream ss;
            for (Layer* layer : layers) {
                ss << layer->to_string() << "\n";
            }
            return ss.str();
        }

        ~Model() {
            for (Layer* layer : layers) {
                delete layer;
            }
        }
    };

    Tensor mse_loss(const Tensor &pred, const Tensor &target) {
        if(pred.size() != target.size()) {
            throw std::invalid_argument("Prediction and target tensors must have same size");
        }

        Tensor loss;
        loss.reshape({1});
        loss[0] = 0.0f;

        for (size_t i = 0; i < pred.size(); ++i) {
            float diff = pred[i] - target[i];
            loss[0] += diff * diff;
        }
        loss[0] /= pred.size();
        return loss;
    }

}

#endif // TTIE_H
