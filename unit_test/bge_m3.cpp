//
// Created by RobinQu on 2024/5/22.
//
#include <ggml.h>
#include <vector>

class Activation {

};

class Linear {
    ggml_tensor *weight__;
    ggml_tensor *bias__;
};

class LayerNorm {
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

class RebertaMLP {
    Linear dense_;
    LayerNorm norm_;
};


class RobertaBlock {
    RobertaSelfAttention attention_;
    LayerNorm post_attention_layer_norm__;
    RebertaMLP mlp_;
    LayerNorm output_layer_norm_;
};


class RobertaClassificationHead {
    Linear dense_;
    Activation activation_;
    Linear out_proj_;
};

class XLMRoberta {
    RobertaEmbedding embedding_;
    std::vector<RobertaBlock> layers_;
    RobertaClassificationHead final_;
};

class BGEM3Model {
    XLMRoberta roberta_;
};

class bge_m3 {
public:
    void load() {

    }
};

int main() {

}