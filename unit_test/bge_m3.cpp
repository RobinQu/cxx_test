//
// Created by RobinQu on 2024/5/22.
//
#include "reranker/bge_m3.hpp"
#include "reranker/config.hpp"
#include "reranker/tokenizer.hpp"


static float get_rank_score(tokenizer::BaseTokenizer* tokenizer, BGEM3RerankerModel* model, const std::string &q, const std::string& a) {
    GenerationConfig generationConfig {.num_threads = 1};
    std::vector<int> ids;
    tokenizer->encode_qa(q, a, ids);
    return model->GetScore(generationConfig, ids);
}

int main() {
    static constexpr size_t MEM_SIZE = 812ull * 1024 * 1024;
    static constexpr size_t SCRATCH_SIZE = 544ull * 1024 * 1024;

    std::string model_path = "/IdeaProjects/baai-bge-m3-guff/bge-reranker-v2-m3.bin";
    ModelLoader loader {model_path};

    // read headers
    loader.seek(0, SEEK_SET);
    std::string magic = loader.read_string(4);
    GGML_ASSERT(magic == "ggml");
    const auto model_type = loader.read_basic<int>();
    GGML_ASSERT(model_type == 0x10000103); // BGE-M3-reanker
    const auto version = loader.read_basic<int>();

    // read config
    if (0 == loader.offset_config)
        loader.offset_config = loader.tell();
    else
        loader.seek(loader.offset_config, SEEK_SET);

    // load config
    auto config = loader.read_basic<Config>();

    // load tokenizer
    loader.offset_tokenizer = loader.tell();
    loader.seek(loader.offset_tokenizer, SEEK_SET);
    auto *tokenizer = new tokenizer::UnigramTokenizer(config);
    auto proto_size = tokenizer->load(loader.data + loader.tell(), config.vocab_size);
    loader.seek(proto_size, SEEK_CUR);

    // load tensors
    if (0 == loader.offset_tensors)
    {
        loader.offset_tensors = loader.tell();
        loader.load_all_tensors();
    }
    loader.seek(loader.offset_tensors, SEEK_SET);

    // test tokenizer
    std::vector<int> result1;
    tokenizer->encode("hello", result1);
    for(const auto&id :result1) {
        std::cout << id << ",";
    }
    std::cout << std::endl;

    // load model
    BGEM3RerankerModel model {config, MEM_SIZE, SCRATCH_SIZE};
    model.Load(loader);

    std::cout << "score=" << get_rank_score(tokenizer, &model, "hello", "welcome") << std::endl;
    std::cout << "score=" << get_rank_score(tokenizer, &model, "hello", "你好") << std::endl;
    std::cout << "score=" << get_rank_score(tokenizer, &model, "hello", "bye") << std::endl;
    std::cout << "score=" << get_rank_score(tokenizer, &model, "hello", "farewell") << std::endl;
}