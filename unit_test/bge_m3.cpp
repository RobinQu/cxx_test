//
// Created by RobinQu on 2024/5/22.
//
#include <ggml.h>
#include <vector>
#include <iostream>

enum ActFunc {
    GELU,
    SILU,
    Tanh,
    RELU,
    RELU2
};

ggml_tensor *inplace_act(ggml_context *ctx, ActFunc act, ggml_tensor *input)
{
    switch (act)
    {
        case ActFunc::GELU:
            return ggml_gelu_inplace(ctx, input);
        case ActFunc::SILU:
            return ggml_silu_inplace(ctx, input);
        case ActFunc::Tanh:
            return ggml_tanh_inplace(ctx, input);
        case ActFunc::RELU:
            return ggml_relu_inplace(ctx, input);
        case ActFunc::RELU2:
        {
            ggml_tensor *output = ggml_relu_inplace(ctx, input);
            output = ggml_sqr_inplace(ctx, output);
            return output;
        }
        default:
            std::cerr << "not implemented act function: " << act;
            return nullptr;
    }
}


struct ForwardContext {
    ggml_context* g_ctx = nullptr;
    ggml_cgraph* g_cgraph = nullptr;
    ggml_scratch g_scratch {};
    virtual ~ForwardContext() {
        ggml_free(g_ctx);
    }
};

struct InitContext {
    ggml_context* g_ctx;
    ggml_type dtype;
};

class Block {
public:
    explicit Block(ggml_prec precision = GGML_PREC_DEFAULT, int id = 0) : precision_(precision), id_(id) {}
    virtual ~Block() = default;
    virtual ggml_tensor* Forward(ForwardContext* ctx, ggml_tensor* input) {
        throw std::runtime_error("Not implemented");
    }
    virtual ggml_tensor* Forward(ForwardContext* ctx, ggml_tensor* input, int n_past) {
        throw std::runtime_error("Not implemented");
    };
protected:
    ggml_prec precision_;
    int id_;
};

class Linear: public Block {
public:
    Linear(ggml_tensor *weight, ggml_tensor *bias) : weight_(weight), bias_(bias) {}

    Linear(InitContext* init_context ,int in_features, int out_features, bool use_bias):
        weight_(ggml_new_tensor_2d(init_context->g_ctx, init_context->dtype, in_features, out_features)),
        bias_(use_bias ? ggml_new_tensor_1d(init_context->g_ctx, GGML_TYPE_F32, out_features) : nullptr)
    {}

    ggml_tensor *Forward(ForwardContext *ctx, ggml_tensor *input) override {
        ggml_tensor *output = ggml_mul_mat(ctx->g_ctx, weight_, input); // [seqlen, out_features]
        ggml_mul_mat_set_prec(output, precision_);
        if (bias_)
        {
            output = ggml_add_inplace(ctx->g_ctx, output, bias_);
        }
        return output;
    }
private:
    ggml_tensor *weight_;
    ggml_tensor *bias_;
};

class LayerNorm: public Block {
public:
    LayerNorm(ggml_tensor *weight, ggml_tensor *bias) : weight_(weight), bias_(bias), eps_(1e-5f) {}
    LayerNorm(InitContext* init_context, int normalized_shape, bool use_bias = true):
        weight_(ggml_new_tensor_1d(init_context->g_ctx, GGML_TYPE_F32, normalized_shape)),
        bias_(use_bias ? ggml_new_tensor_1d(init_context->g_ctx, GGML_TYPE_F32, normalized_shape) : nullptr),
        eps_(1e-5)
    {}

    ggml_tensor *Forward(ForwardContext *ctx, ggml_tensor *input) override {
        // input: [seqlen, normalized_shape]
        ggml_tensor *output = ggml_norm_inplace(ctx->g_ctx, input, eps_);
        output = ggml_mul_inplace(ctx->g_ctx, output, weight_);
        if (bias_)
            output = ggml_add_inplace(ctx->g_ctx, output, bias_);
        return output;
    }

private:
    ggml_tensor* weight_;
    ggml_tensor* bias_;
    float eps_;
};

class RobertaEmbedding {
    ggml_tensor* world_weight_;
    ggml_tensor* position_Weight_;
    ggml_tensor* indices_;
    LayerNorm ln_;
    int pad_index_;
};

class RobertaSelfAttention {

};

class RobertaMLP {
    Linear dense_;
    LayerNorm norm_;
};


class RobertaBlock {
    RobertaSelfAttention attention_;
    LayerNorm post_attention_layer_norm__;
    RobertaMLP mlp_;
    LayerNorm output_layer_norm_;
};


class RobertaClassificationHead {
    Linear dense_;
    ActFunc activation_;
    Linear out_proj_;
};

class XLMRoberta {
    RobertaEmbedding embedding_;
    std::vector<RobertaBlock> layers_;
    RobertaClassificationHead final_;
};

struct GenerationConfig {};

class BGEM3RerankerModel {
public:
    float GetScore(const GenerationConfig& config,  const std::vector<int>& input_tokens) {

    }
private:
    XLMRoberta roberta_;
};

class bge_m3 {
public:
    void load() {

    }
};

int main() {

}