//
// Created by RobinQu on 2024/5/22.
//
#include <ggml.h>
#include <vector>
#include <iostream>
#include <cmath>
#include <memory>
#include <cstring>

enum ActFunc {
    GELU,
    SILU,
    Tanh,
    RELU,
    RELU2
};

static ggml_tensor *inplace_act(ggml_context *ctx, ActFunc act, ggml_tensor *input)
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
    virtual void set_id(int id)
    {
        id_ = id;
    }

protected:
    ggml_prec precision_;
    int id_;
};

class Linear: public Block {
public:
    Linear(ggml_tensor *weight = nullptr, ggml_tensor *bias = nullptr) : weight(weight), bias(bias) {}

    Linear(InitContext* init_context ,int in_features, int out_features, bool use_bias = true):
        Linear(init_context, in_features, out_features, nullptr, use_bias)
    {}

    Linear(InitContext* init_context ,int in_features, int out_features, ggml_tensor *weight, bool use_bias = true):
            weight(weight ? weight : ggml_new_tensor_2d(init_context->g_ctx, init_context->dtype, in_features, out_features)),
            bias(use_bias ? ggml_new_tensor_1d(init_context->g_ctx, GGML_TYPE_F32, out_features) : nullptr)
    {}

    [[nodiscard]] int in_features() const { return (int)weight->ne[0]; }
    [[nodiscard]] int out_features() const { return (int)weight->ne[1]; }

    ggml_tensor *Forward(ForwardContext *ctx, ggml_tensor *input) override {
        ggml_tensor *output = ggml_mul_mat(ctx->g_ctx, weight, input); // [seqlen, out_features]
        ggml_mul_mat_set_prec(output, precision_);
        if (bias)
        {
            output = ggml_add_inplace(ctx->g_ctx, output, bias);
        }
        return output;
    }

    ggml_tensor *weight;
    ggml_tensor *bias;

};

class LayerNorm: public Block {
public:
    explicit LayerNorm(ggml_tensor *weight = nullptr, ggml_tensor *bias = nullptr) : weight(weight), bias(bias), eps_(1e-5f) {}
    LayerNorm(InitContext* init_context, int normalized_shape, bool use_bias = true):
            weight(ggml_new_tensor_1d(init_context->g_ctx, GGML_TYPE_F32, normalized_shape)),
            bias(use_bias ? ggml_new_tensor_1d(init_context->g_ctx, GGML_TYPE_F32, normalized_shape) : nullptr),
            eps_(1e-5)
    {}

    ggml_tensor *Forward(ForwardContext *ctx, ggml_tensor *input) override {
        // input: [seqlen, normalized_shape]
        ggml_tensor *output = ggml_norm_inplace(ctx->g_ctx, input, eps_);
        output = ggml_mul_inplace(ctx->g_ctx, output, weight);
        if (bias)
            output = ggml_add_inplace(ctx->g_ctx, output, bias);
        return output;
    }


    ggml_tensor* weight;
    ggml_tensor* bias;
private:
    float eps_;
};

struct ShiftPending {
    int shift = 0;
    int total = 0;
    void clear() { shift = 0; }
};

class RobertaEmbedding: public Block {
public:
    RobertaEmbedding(): word_weight(nullptr), position_weight(nullptr), indices(nullptr), pad_index(0) {}
    RobertaEmbedding(InitContext* init_context, int num_embeddings, int embedding_dim, int pos_max):
            word_weight(ggml_new_tensor_2d(init_context->g_ctx, init_context->dtype, embedding_dim, num_embeddings)),
            position_weight(ggml_new_tensor_2d(init_context->g_ctx, init_context->dtype, embedding_dim, pos_max)),
            indices(ggml_new_tensor_1d(init_context->g_ctx, GGML_TYPE_I32, pos_max)),
            ln(init_context, embedding_dim),
            pad_index(2) {}

    ggml_tensor *Forward(ForwardContext *ctx, ggml_tensor *input, int n_past) override {
        int qlen = (int)input->ne[0];
        ggml_tensor *idx = ggml_view_1d(ctx->g_ctx, indices, qlen, (n_past + pad_index) * ggml_element_size(indices));

        ggml_tensor *output1 = ggml_get_rows(ctx->g_ctx, word_weight, input);
        ggml_tensor *output2 = ggml_get_rows(ctx->g_ctx, position_weight, idx);

        ggml_tensor *output = ggml_add_inplace(ctx->g_ctx, output1, output2);

        output = ln.Forward(ctx, output);
        return output;
    }

    ggml_tensor* word_weight;
    ggml_tensor* position_weight;
    ggml_tensor* indices;
    LayerNorm ln;
    int pad_index;
};

class CoreAttention: public Block {
public:
    CoreAttention(InitContext *ctx, int num_attention_heads, int num_kv_heads, int max_length, ggml_type cache_type,
                  int k_cache_ele_num, int v_cache_ele_num)
            : num_attention_heads(num_attention_heads),
              num_kv_heads(num_kv_heads),
              k_cache(k_cache_ele_num > 0 ? ggml_new_tensor_1d(ctx->g_ctx, cache_type, k_cache_ele_num)
                                          : nullptr),
              v_cache(v_cache_ele_num > 0 ? ggml_new_tensor_1d(ctx->g_ctx, cache_type, v_cache_ele_num)
                                          : nullptr),
              pos(ggml_new_tensor_1d(ctx->g_ctx, GGML_TYPE_I32, max_length)),
              max_length(max_length),
              shift_pending_(),
              attn_scaling_(true),
              causal_(true)
    {
        if (k_cache_ele_num > 0)
        {
            k_cache->data = new char[ggml_nbytes(k_cache)]();
            ggml_set_name(k_cache, "k_cache");
        }
        if (v_cache_ele_num > 0)
        {
            v_cache->data = new char[ggml_nbytes(v_cache)]();
            ggml_set_name(v_cache, "v_cache");
        }
        pos->data = new char[ggml_nbytes(pos)]();
    }

    void shift_cache(int shift, int total) {
        shift_pending_ = ShiftPending(shift, total);
    }
    const int num_attention_heads;
    const int num_kv_heads;
    ggml_tensor *k_cache;
    ggml_tensor* v_cache;
    ggml_tensor *pos;
    const int max_length;

protected:
    ShiftPending shift_pending_;
    bool attn_scaling_;
    bool causal_;

    virtual ggml_tensor *apply_pos_embedding_kq(ForwardContext *ctx, ggml_tensor *kq, int hidden_size, int qlen, ggml_tensor *past) const {
        return kq;
    }

    // k: [heads, qlen, head_size]
    // q: [heads, qlen, head_size]
    // v: [heads, head_size, klen]
    virtual ggml_tensor *calc_attn_scores(
            ForwardContext *ctx,
            int hidden_size,
            const int n_past,
            const int qlen,
            ggml_tensor *key_layer,
            ggml_tensor *query_layer,
            ggml_tensor *value_layer) {
        const int head_size = hidden_size / num_attention_heads;

        // note auto-broadcasting in ggml_mul_mat for `repeat > 1`
        ggml_tensor *attn_scores = ggml_mul_mat(ctx->g_ctx, key_layer, query_layer); // [heads, qlen, klen]

        ggml_mul_mat_set_prec(attn_scores, precision_);

        if (attn_scaling_)
            attn_scores = ggml_scale_inplace(ctx->g_ctx, attn_scores, 1.f / sqrtf((float)head_size));

        attn_scores = apply_pos_embedding_kq(ctx, attn_scores, hidden_size, qlen, pos);

        // attn_masked = mask_past(attn_scores)
        struct ggml_tensor * attn_masked = causal_ ? ggml_diag_mask_inf_inplace(ctx->g_ctx, attn_scores, n_past)
                                                   : attn_scores;

        // attn_probs = soft_max(attn_masked)
        struct ggml_tensor * attn_probs = ggml_soft_max_inplace(ctx->g_ctx, attn_masked);

        ggml_tensor *context_layer = ggml_mul_mat(ctx->g_ctx, value_layer, attn_probs); // [heads, qlen, head_size]
        context_layer = ggml_reshape_2d(
                ctx->g_ctx,
                ggml_cont(ctx->g_ctx, ggml_permute(ctx->g_ctx, context_layer, 0, 2, 1, 3)),
                hidden_size, qlen);

        return context_layer;
    }
};

static void fill_pos_vector(ggml_tensor *pos, int n_past, int qlen)
{
    int *p = (int *)pos->data;
    for (int i = 0; i < qlen; i++)
        p[i] = n_past + i;
    pos->ne[0] = qlen;
}


class BaseAttention: public CoreAttention {
public:
    BaseAttention(InitContext *ctx, int hidden_size, int num_attention_heads, int num_kv_heads, int head_dim, int max_length,
                  bool qkv_bias, bool o_bias,
                  ggml_type cache_type, int cache_length)
            : CoreAttention(ctx, num_attention_heads, num_kv_heads, max_length, cache_type,
                            head_dim * num_kv_heads * cache_length,
                            head_dim * num_kv_heads * cache_length),
              q_proj(ctx, hidden_size, head_dim * num_attention_heads, nullptr, qkv_bias),
              k_proj(ctx, hidden_size, head_dim * num_kv_heads, nullptr, qkv_bias),
              v_proj(ctx, hidden_size, head_dim * num_kv_heads, nullptr, qkv_bias),
              o_proj(ctx, head_dim * num_attention_heads, hidden_size, nullptr, o_bias),
              cache_length(cache_length)
    {
    }

    BaseAttention(InitContext *ctx, int hidden_size, int num_attention_heads, int num_kv_heads, int max_length,
                  bool qkv_bias, bool o_bias,
                  ggml_type cache_type, int cache_length)
            : BaseAttention(ctx, hidden_size, num_attention_heads, num_kv_heads, hidden_size / num_attention_heads, max_length, qkv_bias, o_bias,
                            cache_type, cache_length)
    {}

    BaseAttention(InitContext *ctx, int hidden_size, int num_attention_heads, int num_kv_heads, int max_length, bool qkv_bias, bool o_bias)
            : BaseAttention(ctx, hidden_size, num_attention_heads, num_kv_heads, hidden_size / num_attention_heads, max_length, qkv_bias, o_bias,
                            GGML_TYPE_F16, max_length)
    {}

    BaseAttention(InitContext *ctx, int hidden_size, int num_attention_heads, int num_kv_heads, int head_dim, int max_length,
                  bool qkv_bias, bool o_bias)
            : BaseAttention(ctx, hidden_size, num_attention_heads, num_kv_heads, head_dim, max_length, qkv_bias, o_bias,
                            GGML_TYPE_F16, max_length)
    {}


protected:
    // input & output: [qlen, heads, head_size]
    virtual ggml_tensor *apply_pos_embedding_k(ForwardContext *ctx, ggml_tensor *k, int hidden_size, int qlen, ggml_tensor * past) const { return k; }
    virtual ggml_tensor *apply_pos_embedding_q(ForwardContext *ctx, ggml_tensor *q, int hidden_size, int qlen, ggml_tensor * past) const { return q; }

//
    virtual void before_forward(ForwardContext *ctx, const int kv_hidden_size, const int n_past, const int qlen) {
        fill_pos_vector(pos, n_past, qlen);

        // shift cache
        if (shift_pending_.shift > 0)
        {
            int remain = shift_pending_.total - shift_pending_.shift;
            if (remain > 0)
            {
                struct ggml_tensor * k_cache_remain = ggml_view_1d(ctx->g_ctx, k_cache, remain * kv_hidden_size,
                                                                   ggml_element_size(k_cache) * kv_hidden_size * shift_pending_.shift);
                struct ggml_tensor * k_cache_1d = ggml_view_1d(ctx->g_ctx, k_cache, remain * kv_hidden_size,
                                                               0);

                struct ggml_tensor * v_cache_remain = ggml_view_2d(ctx->g_ctx, v_cache, remain, kv_hidden_size,
                                                                   cache_length * ggml_element_size(v_cache),
                                                                   shift_pending_.shift * ggml_element_size(v_cache));
                struct ggml_tensor * v_cache_2d =     ggml_view_2d(ctx->g_ctx, v_cache, remain, kv_hidden_size,
                                                                   cache_length * ggml_element_size(v_cache),
                                                                   0);

                ggml_build_forward_expand(ctx->g_cgraph, ggml_cpy(ctx->g_ctx, k_cache_remain, k_cache_1d));
                ggml_build_forward_expand(ctx->g_cgraph, ggml_cpy(ctx->g_ctx, v_cache_remain, v_cache_2d));
            }
            shift_pending_.clear();
        }
    }


    virtual void save_to_cache(ForwardContext *ctx, const int kv_hidden_size, const int n_past, const int qlen, ggml_tensor *k, ggml_tensor *v) = 0;
    virtual ggml_tensor *get_k_from_cache(ForwardContext *ctx, const int hidden_size, const int n_past, const int qlen) = 0;
    virtual ggml_tensor *get_v_from_cache(ForwardContext *ctx, const int hidden_size, const int n_past, const int qlen) = 0;

    ggml_tensor *cross_attention(ForwardContext *ctx, const int hidden_size, const int n_past, const int qlen,
                                         ggml_tensor *q, ggml_tensor *k, ggml_tensor *v) {
        const int head_size = hidden_size / num_attention_heads;
        const int repeat = num_attention_heads / num_kv_heads;
        const int kv_hidden_size = hidden_size / repeat;

        // [qlen, heads, head_size]
        ggml_tensor * key_layer = ggml_reshape_3d(ctx->g_ctx, k, head_size, num_kv_heads, qlen);
        key_layer = apply_pos_embedding_k(ctx, key_layer, hidden_size, qlen, pos);

        // [qlen, heads, head_size]
        ggml_tensor * query_layer = ggml_reshape_3d(ctx->g_ctx, q, head_size, num_attention_heads, qlen);
        query_layer = apply_pos_embedding_q(ctx, query_layer, hidden_size, qlen, pos);

        if (!attn_scaling_)
            query_layer = ggml_scale(ctx->g_ctx, query_layer, 1.f / sqrtf((float)head_size));

        // store key and value to memory
        save_to_cache(ctx, kv_hidden_size, n_past, qlen, key_layer, v);

        query_layer = ggml_permute(ctx->g_ctx, query_layer, 0, 2, 1, 3);                     // [heads, qlen, head_size]

        key_layer = get_k_from_cache(ctx, hidden_size, n_past, qlen);

        ggml_tensor * value_layer = get_v_from_cache(ctx, hidden_size, n_past, qlen);

        ggml_tensor *attn_scores = calc_attn_scores(ctx, hidden_size, n_past, qlen, key_layer, query_layer, value_layer);
        return attn_scores;
    }

    int cache_length;

public:
    Linear q_proj, k_proj, v_proj;
    Linear o_proj;
};

class BaseCachelessAttention : public BaseAttention
{
public:
    BaseCachelessAttention() = delete;

    BaseCachelessAttention(InitContext *ctx, int hidden_size, int num_attention_heads, int num_kv_heads, int max_length, bool qkv_bias, bool o_bias)
            : BaseCachelessAttention(ctx, hidden_size, num_attention_heads, num_kv_heads, hidden_size / num_attention_heads, max_length, qkv_bias, o_bias)
    {}

    BaseCachelessAttention(InitContext *ctx, int hidden_size, int num_attention_heads, int num_kv_heads, int head_dim, int max_length, bool qkv_bias, bool o_bias)
            : BaseAttention(ctx, hidden_size, num_attention_heads, num_kv_heads, head_dim, max_length, qkv_bias, o_bias, GGML_TYPE_F16, 0),
              raw_k(nullptr),
              raw_v(nullptr)
    {}

protected:
    void save_to_cache(ForwardContext *ctx, const int kv_hidden_size, const int n_past, const int qlen, ggml_tensor *k, ggml_tensor *v) override {
        raw_k = k;
        raw_v = v;
    }

    // output: [heads, qlen, head_size]
    ggml_tensor *get_k_from_cache(ForwardContext *ctx, const int hidden_size, const int n_past, const int qlen) override {
        ggml_tensor *r = ggml_permute(ctx->g_ctx, raw_k, 0, 2, 1, 3);
        return r;
    }

    // output: [heads, head_size, klen]
    ggml_tensor *get_v_from_cache(ForwardContext *ctx, const int hidden_size, const int n_past, const int qlen) override {
        const int head_size = hidden_size / num_attention_heads;

        // [qlen, hidden_size] -> [heads, head_size, qlen]
        ggml_tensor *r = ggml_reshape_3d(ctx->g_ctx, raw_v, head_size, num_kv_heads, qlen);  // -> [qlen, heads, head_size]
        r = ggml_permute(ctx->g_ctx, r, 1, 2, 0, 3);   // [heads, head_size, qlen]
        r = ggml_cont(ctx->g_ctx, r);
        return r;
    }
private:
    ggml_tensor *raw_k;
    ggml_tensor *raw_v;
};


enum RoPEMode
{
    Interleaved = 0,        // IQIQ......IQ
    Original = 2,           // II...IQQ...Q
    GLM = 4,
};


template <class BaseAttn = BaseCachelessAttention> class BaseSelfAttention : public BaseAttn
{
public:
    BaseSelfAttention(InitContext *ctx, int hidden_size, int num_attention_heads, int max_length, bool qkv_bias, bool o_bias)
            : BaseSelfAttention(ctx, hidden_size, num_attention_heads, num_attention_heads, max_length, qkv_bias, o_bias)
    {
    }

    BaseSelfAttention(InitContext *ctx, int hidden_size, int num_attention_heads, int num_kv_heads, int max_length, bool qkv_bias, bool o_bias)
            : BaseSelfAttention(ctx, hidden_size, num_attention_heads, num_kv_heads, hidden_size / num_attention_heads,  max_length, qkv_bias, o_bias)
    {
    }

    BaseSelfAttention(InitContext *ctx, int hidden_size, int num_attention_heads, int num_kv_heads, int head_dim, int max_length, bool qkv_bias, bool o_bias)
            : BaseAttn(ctx, hidden_size, num_attention_heads, num_kv_heads, head_dim, max_length, qkv_bias, o_bias),
              freq_base(10000.0f),
              freq_scale(1.0f),
              ext_factor(0.0f),
              attn_factor(1.0f),
              beta_fast(0.0f),
              beta_slow(0.0f),
              rope_dim(head_dim),
              rope_mode(RoPEMode::Interleaved),
              last_attn_scores(nullptr)
    {
    }

    using Block::Forward;
    ggml_tensor *Forward(ForwardContext *ctx, ggml_tensor *hidden_states, int n_past) override
    {
        const int hidden_size = BaseAttn::o_proj.in_features();
        const int qlen = (int)hidden_states->ne[1];
        const int repeat = BaseAttn::num_attention_heads / BaseAttn::num_kv_heads;
        const int kv_hidden_size = hidden_size / repeat;

        BaseAttn::before_forward(ctx, kv_hidden_size, n_past, qlen);

        ggml_tensor *tmpq = BaseAttn::q_proj.Forward(ctx, hidden_states);
        ggml_tensor *tmpk = BaseAttn::k_proj.Forward(ctx, hidden_states);
        ggml_tensor *tmpv = BaseAttn::v_proj.Forward(ctx, hidden_states);

        ggml_mul_mat_set_prec(tmpk, BaseAttn::precision_);
        ggml_mul_mat_set_prec(tmpq, BaseAttn::precision_);
        ggml_mul_mat_set_prec(tmpv, BaseAttn::precision_);

        last_attn_scores = BaseAttn::cross_attention(ctx, hidden_size, n_past, qlen, tmpq, tmpk, tmpv);

        ggml_tensor *attn_output = BaseAttn::o_proj.Forward(ctx, last_attn_scores);
        return attn_output;
    }

    ggml_tensor *get_last_attn_scores(void)
    {
        return last_attn_scores;
    }

public:
    // rope param
    float freq_base;
    float freq_scale;
    float ext_factor;
    float attn_factor;
    float beta_fast;
    float beta_slow;
    int   rope_dim;
    RoPEMode rope_mode;

protected:
    // input & output: [qlen, heads, head_size]
    ggml_tensor *apply_pos_embedding_k(ForwardContext *ctx, ggml_tensor *k, int hidden_size, int qlen, ggml_tensor * past) const override
    {
        return ggml_rope_custom_inplace(ctx->g_ctx, k, past, rope_dim, rope_mode, 0, 0,
                                        freq_base, freq_scale, ext_factor, attn_factor, beta_fast, beta_slow);    // [qlen, heads, head_size]
    }
    ggml_tensor *apply_pos_embedding_q(ForwardContext *ctx, ggml_tensor *q, int hidden_size, int qlen, ggml_tensor * past) const override
    {
        return ggml_rope_custom_inplace(ctx->g_ctx, q, past, rope_dim, rope_mode, 0, 0,
                                        freq_base, freq_scale, ext_factor, attn_factor, beta_fast, beta_slow);    // [qlen, heads, head_size];
    }

private:
    ggml_tensor *last_attn_scores;
};

class RobertaSelfAttention: public BaseSelfAttention<BaseCachelessAttention> {
public:
    RobertaSelfAttention(InitContext *ctx, int hidden_size, int num_attention_heads, int max_length)
            : RobertaSelfAttention(ctx, hidden_size, num_attention_heads, num_attention_heads, max_length) {}

    RobertaSelfAttention(InitContext *ctx, int hidden_size, int num_attention_heads, int num_kv_heads, int max_length)
            : RobertaSelfAttention(ctx, hidden_size, num_attention_heads, num_kv_heads, max_length, true, true) {}

    RobertaSelfAttention(InitContext *ctx, int hidden_size, int num_attention_heads, int num_kv_heads, int max_length, bool qkv_bias, bool o_bias)
            : BaseSelfAttention(ctx, hidden_size, num_attention_heads, num_kv_heads, max_length, qkv_bias, o_bias)
    {
        causal_ = false;
    }

protected:
    // input & output: [qlen, heads, head_size]
    ggml_tensor *apply_pos_embedding_k(ForwardContext *ctx, ggml_tensor *k, int hidden_size, int qlen, ggml_tensor * past) const override
    {
        return k;
    }
    ggml_tensor *apply_pos_embedding_q(ForwardContext *ctx, ggml_tensor *q, int hidden_size, int qlen, ggml_tensor * past) const override
    {
        return q;
    }



};

class RobertaOutput: public Block {
public:

    RobertaOutput(InitContext *ctx, int hidden_size, bool use_bias = true): RobertaOutput(ctx, hidden_size, hidden_size, use_bias) {}

    RobertaOutput(InitContext *init_context, int hidden_size, int intermediate_size, bool use_bias = true):
            dense(init_context, intermediate_size, hidden_size, nullptr, use_bias),
            norm(init_context, hidden_size) {}

    ggml_tensor *Forward(ForwardContext *ctx, ggml_tensor *hidden_states, ggml_tensor *attention_output) {
        ggml_tensor *r = dense.Forward(ctx, hidden_states);
        r = ggml_add_inplace(ctx->g_ctx, r, attention_output);
        r = norm.Forward(ctx, r);
        return r;
    }

    Linear dense;
    LayerNorm norm;

};

class RobertaMLP: public Block {
public:
    RobertaMLP(InitContext* init_context, int hidden_size, int intermediate_size):
            RobertaMLP(init_context, hidden_size, intermediate_size, ActFunc::GELU, true) {}

    RobertaMLP(InitContext* init_context, int hidden_size, int intermediate_size, ActFunc act, bool bias):
            intermediate(init_context, hidden_size, intermediate_size, nullptr, bias),
            output(init_context, hidden_size, intermediate_size, bias),
            act(act) {}


    ggml_tensor *Forward(ForwardContext *ctx, ggml_tensor *hidden_states) override {
        ggml_tensor *temp = intermediate.Forward(ctx, hidden_states);
        temp = inplace_act(ctx->g_ctx, act, temp);
        temp = output.Forward(ctx, temp, hidden_states);
        return temp;
    }

    Linear intermediate;
    RobertaOutput output;
    ActFunc act;
};


class RobertaBlock: public Block {
public:
    RobertaBlock(InitContext *ctx, int hidden_size, int num_attention_heads, int intermediate_size, int num_kv_heads, int max_length)
            : RobertaBlock(ctx, hidden_size, num_attention_heads, intermediate_size, num_kv_heads, max_length, true, true)
    {}

    RobertaBlock(InitContext* init_context, int hidden_size, int num_attention_heads, int intermediate_size, int num_kv_heads, int max_length, bool qkv_bias, bool o_bias):
            attention(init_context, hidden_size, num_attention_heads, num_kv_heads, max_length, qkv_bias, o_bias),
            post_attention_layer_norm(init_context, hidden_size),
            mlp(init_context, hidden_size, intermediate_size),
            output_layer_norm(init_context, hidden_size)
    {}

    ggml_tensor *Forward(ForwardContext *ctx, ggml_tensor *hidden_states, int n_past) override {
        ggml_tensor *attn_outputs = attention.Forward(ctx, hidden_states, n_past);

        // see XLMRobertaSelfOutput
        ggml_tensor *sum = ggml_add(ctx->g_ctx, hidden_states, attn_outputs);
        ggml_tensor *attention_output = post_attention_layer_norm.Forward(ctx, sum);

        ggml_tensor *r = mlp.Forward(ctx, attention_output);
        return r;
    }


    RobertaSelfAttention attention;
    LayerNorm post_attention_layer_norm;
    RobertaMLP mlp;
    LayerNorm output_layer_norm;

};

static void ggml_compute_forward_sigmoid_f32(struct ggml_tensor * dst , const struct ggml_tensor * src0, int ith, int nth, void * userdata) {
    GGML_ASSERT(ggml_are_same_shape(src0, dst));

    GGML_ASSERT(dst->type == GGML_TYPE_F32);

    GGML_TENSOR_UNARY_OP_LOCALS

    // TODO: make this tunable
    float eps = 1e-6f;

    GGML_ASSERT(eps > 0.0f);

    // TODO: optimize
    for (int64_t i03 = 0; i03 < ne03; i03++) {
        for (int64_t i02 = 0; i02 < ne02; i02++) {
            for (int64_t i01 = ith; i01 < ne01; i01 += nth) {

                for (int64_t i00 = 0; i00 < ne00; i00++) {
                    const float * x = (float *) ((char *) src0->data + i00*nb00 + i01*nb01 + i02*nb02 + i03*nb03);
                    float * y = (float *) ((char *) dst->data + i00*nb0 + i01*nb1 + i02*nb2 + i03*nb3);
                    *y = 1 / (1 + expf(- *x));
                }
            }
        }
    }
}

static void ggml_compute_forward_sigmoid(struct ggml_tensor * dst , const struct ggml_tensor * src, int ith, int nth, void * userdata) {
    switch (src->type) {
        case GGML_TYPE_F32:
        {
            ggml_compute_forward_sigmoid_f32(dst, src, ith, nth, userdata);
        } break;
        default:
        {
            GGML_ASSERT(false);
        } break;
    }
}


class RobertaClassificationHead: public Block {
public:
    RobertaClassificationHead()=default;
    RobertaClassificationHead(InitContext *init_context, int hidden_size):
            dense(init_context, hidden_size, hidden_size, nullptr),
            activation(ActFunc::Tanh),
            out_proj(init_context, hidden_size, 1, nullptr)
    {}


    ggml_tensor *Forward(ForwardContext *ctx, ggml_tensor *hidden_states) override {
        int hidden_size = (int)hidden_states->ne[0];

        // We "pool" the model by simply taking the hidden state corresponding to the first token.
        ggml_tensor *first_token_tensor = ggml_view_2d(ctx->g_ctx, hidden_states, hidden_size, 1,
                                                       hidden_size * ggml_element_size(hidden_states), 0);
        ggml_tensor *output = dense.Forward(ctx, first_token_tensor);
        output = inplace_act(ctx->g_ctx, activation, output);
        output = out_proj.Forward(ctx, output);
        output = ggml_map_custom1(ctx->g_ctx, output, ggml_compute_forward_sigmoid, 1, nullptr);
        return output;
    }

    Linear dense;
    ActFunc activation = ActFunc::Tanh;
    Linear out_proj;
};


struct Config
{
    // common attributes
    ggml_type dtype;
    int vocab_size;
    int hidden_size;
    int num_attention_heads;
    int num_hidden_layers;
    int intermediate_size;
    // for sequence generation
    int max_length;
    // for tokenizer
    int bos_token_id;
    int eos_token_id;
    int pad_token_id;
    int sep_token_id;
};


class XLMRoberta: public Block {
public:
    XLMRoberta(InitContext* init_context, const Config& config):
            config_(config),
            word_embeddings(init_context, config.vocab_size, config.hidden_size, config.max_length),
            layers(),
            final(init_context, config.hidden_size)
    {
        layers.reserve(config.num_hidden_layers);
        for(int layer_id=0; layer_id<config.num_hidden_layers; ++layer_id) {
            layers.emplace_back(
                    init_context,
                     config.hidden_size,
                     config.num_attention_heads,
                     config.intermediate_size,
                     config.num_attention_heads,
                     config.max_length
            );
            layers[layer_id].set_id(layer_id);
        }
    }

    ggml_tensor *Forward(ForwardContext *ctx, ggml_tensor *input_ids, int n_past) override {
        ggml_tensor *hidden_states = word_embeddings.Forward(ctx, input_ids, n_past);
        for (auto &layer : layers) {
            ggml_set_scratch(ctx->g_ctx, ctx->g_scratch);
            hidden_states = layer.Forward(ctx, hidden_states, n_past);
        }
        return final_steps(ctx, input_ids, hidden_states);
    }

    RobertaEmbedding word_embeddings;
    std::vector<RobertaBlock> layers;
    RobertaClassificationHead final;

private:
    ggml_tensor *final_steps(ForwardContext *ctx, ggml_tensor *input_ids, ggml_tensor *hidden_states)
    {
        ggml_set_scratch(ctx->g_ctx, {.offs = 0, .size = 0, .data = nullptr});
        ggml_tensor *transformer_outputs = final.Forward(ctx, hidden_states);
        return transformer_outputs;
    }


    Config config_;

};

struct GenerationConfig {
    int max_length;
    int max_context_length;
    bool do_sample;
    int top_k;
    float top_p;
    float temperature;
    int num_threads;
    float presence_penalty;
    float tfs_z;
    std::string sampling;
};


class ModelLoader {
public:
    ModelLoader(const std::string& model_file_path) {

    }

    void read_tensor(const std::string &name, ggml_tensor *tensor) {

    }
};

class BaseModel {
public:
    virtual ~BaseModel()=default;
    virtual void Load(ModelLoader& loader) = 0;
};

class BGEM3RerankerModel: public BaseModel {
public:

    BGEM3RerankerModel(Config config, size_t mem_size, size_t scratch_size):
            GRAPH_SIZE(GGML_DEFAULT_GRAPH_SIZE),
            batch_input(true),
            logit_scale(-1.0f),
            config_(config),
            mem_size_(mem_size),
            mem_buffer_(new char[mem_size]),
            scratch_size_(scratch_size),
            scratch_buffer_(new char[scratch_size]),
            w_ctx_(
                ggml_init({.mem_size = ((9 + config.num_hidden_layers * 19) * (GGML_TENSOR_SIZE + GGML_OBJECT_SIZE)), .mem_buffer = nullptr, .no_alloc = true}),
                config.dtype
        ),
            transformer_(&w_ctx_, config_)
    {
        for (int i = 0; i < config.num_hidden_layers; i++)
            layer_ids.push_back(i);
    }

    float GetScore(const GenerationConfig& config,  const std::vector<int>& input_ids) {
        ggml_tensor *lm = run_model(input_ids, config, 0);
        // "lm->type must be GGML_TYPE_F32"
        GGML_ASSERT(lm->type == GGML_TYPE_F32);
        // "ouput must be scaler"
        GGML_ASSERT((lm->ne[0] == 1) && (ggml_n_dims(lm) <= 1));

        return *(float *)lm->data;
    }

    void Load(ModelLoader &loader) override {
        loader.read_tensor("embeddings.word_embeddings.weight",         transformer_.word_embeddings.word_weight);
        loader.read_tensor("embeddings.position_embeddings.weight",     transformer_.word_embeddings.position_weight);
        loader.read_tensor("embeddings.LayerNorm.weight",               transformer_.word_embeddings.ln.weight);
        loader.read_tensor("embeddings.LayerNorm.bias",                 transformer_.word_embeddings.ln.bias);

        for (int i = 0; i < config_.num_hidden_layers; i++)
        {
            std::string layer_prefix = "encoder.layer." + std::to_string(layer_ids[i]) + '.';
            loader.read_tensor(layer_prefix + "attention.self.query.weight",    transformer_.layers[i].attention.q_proj.weight);
            loader.read_tensor(layer_prefix + "attention.self.query.bias",      transformer_.layers[i].attention.q_proj.bias);
            loader.read_tensor(layer_prefix + "attention.self.key.weight",      transformer_.layers[i].attention.k_proj.weight);
            loader.read_tensor(layer_prefix + "attention.self.key.bias",        transformer_.layers[i].attention.k_proj.bias);
            loader.read_tensor(layer_prefix + "attention.self.value.weight",    transformer_.layers[i].attention.v_proj.weight);
            loader.read_tensor(layer_prefix + "attention.self.value.bias",      transformer_.layers[i].attention.v_proj.bias);
            loader.read_tensor(layer_prefix + "attention.output.dense.weight",  transformer_.layers[i].attention.o_proj.weight);
            loader.read_tensor(layer_prefix + "attention.output.dense.bias",    transformer_.layers[i].attention.o_proj.bias);
            loader.read_tensor(layer_prefix + "attention.output.LayerNorm.weight",    transformer_.layers[i].post_attention_layer_norm.weight);
            loader.read_tensor(layer_prefix + "attention.output.LayerNorm.bias",      transformer_.layers[i].post_attention_layer_norm.bias);

            loader.read_tensor(layer_prefix + "intermediate.dense.weight",  transformer_.layers[i].mlp.intermediate.weight);
            loader.read_tensor(layer_prefix + "intermediate.dense.bias",    transformer_.layers[i].mlp.intermediate.bias);
            loader.read_tensor(layer_prefix + "output.dense.weight",        transformer_.layers[i].mlp.output.dense.weight);
            loader.read_tensor(layer_prefix + "output.dense.bias",          transformer_.layers[i].mlp.output.dense.bias);

            loader.read_tensor(layer_prefix + "output.LayerNorm.weight",    transformer_.layers[i].mlp.output.norm.weight);
            loader.read_tensor(layer_prefix + "output.LayerNorm.bias",      transformer_.layers[i].mlp.output.norm.bias);
        }

        loader.read_tensor("classifier.dense.weight",       transformer_.final.dense.weight);
        loader.read_tensor("classifier.dense.bias",         transformer_.final.dense.bias);
        loader.read_tensor("classifier.out_proj.weight",    transformer_.final.out_proj.weight);
        loader.read_tensor("classifier.out_proj.bias",      transformer_.final.out_proj.bias);

        GGML_ASSERT(ggml_used_mem(w_ctx_.g_ctx) == ggml_get_mem_size(w_ctx_.g_ctx));
    }

protected:

    virtual ggml_tensor *run_model(const std::vector<int> &input_ids,
                                   const GenerationConfig &gen_config,
                                   int past)
    {
        ForwardContext ctx;
        ctx.g_ctx = ggml_init({.mem_size = mem_size_, .mem_buffer = mem_buffer_.get(), .no_alloc = false});
        ctx.g_scratch = {.offs = 0, .size = scratch_size_, .data = scratch_buffer_.get()};

        int n_threads = input_ids.size() >= 32 && ggml_cpu_has_blas() && !ggml_cpu_has_gpublas() ? 1 : gen_config.num_threads;
        ctx.g_cgraph = ggml_new_graph_custom(ctx.g_ctx, GRAPH_SIZE, false);

        ggml_tensor *input_ids_tensor = ggml_new_tensor_1d(ctx.g_ctx, GGML_TYPE_I32, input_ids.size());
        std::memcpy(input_ids_tensor->data, input_ids.data(), ggml_nbytes(input_ids_tensor));

        ggml_tensor *r = transformer_.Forward(&ctx, input_ids_tensor, past);

        if (logit_scale > 0)
            r = ggml_scale_inplace(ctx.g_ctx, r, logit_scale);

        ggml_build_forward_expand(ctx.g_cgraph, r);
        ggml_graph_compute_with_ctx(ctx.g_ctx, ctx.g_cgraph, n_threads);

#ifdef GGML_PERF
        ggml_graph_print(&ctx.gf);
#endif
        return r;
    }

    XLMRoberta transformer_;
    size_t GRAPH_SIZE;
    bool batch_input;
    float logit_scale;
    std::vector<int> layer_ids;

private:
    Config config_;
    size_t mem_size_;
    std::unique_ptr<char[]> mem_buffer_; // BLAS buffer
    size_t scratch_size_;
    std::unique_ptr<char[]> scratch_buffer_; // intermediate tensor buffer
    InitContext w_ctx_; // weight context
};


int main() {

}