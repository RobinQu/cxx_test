//
// Created by RobinQu on 2024/5/22.
//
#include "reranker/bge_m3.hpp"
#include "reranker/config.hpp"
#include "reranker/tokenizer.hpp"

int main() {
    static constexpr size_t MEM_SIZE = 812ull * 1024 * 1024;
    static constexpr size_t SCRATCH_SIZE = 544ull * 1024 * 1024;
    ModelLoader loader {"bge_m3_reranker.bin"};

    // read headers
    loader.seek(0, SEEK_SET);
    std::string magic = loader.read_string(4);
    GGML_ASSERT(magic == "ggml");
    const auto model_type = loader.read_basic<int>();
    GGML_ASSERT(model_type == 0x10000103);
    const auto version = loader.read_basic<int>();

    // read config
    auto config = loader.read_basic<Config>();

    // load tokenizer
    auto *tokenizer = new tokenizer::Tokenizer(config);
    auto proto_size = tokenizer->load(loader.data + loader.tell(), config.vocab_size);
    loader.seek(proto_size, SEEK_CUR);

    if (0 == loader.offset_tensors)
    {
        loader.offset_tensors = loader.tell();
        loader.load_all_tensors();
    }

    BGEM3RerankerModel model  {config, MEM_SIZE, SCRATCH_SIZE};

}