//
// Created by RobinQu on 2024/3/20.
//

#ifndef MAPPINGCHAIN_HPP
#define MAPPINGCHAIN_HPP
#include <unordered_map>

#include "HashMapContext.hpp"
#include "IChainable.hpp"
#include "IRunnable.hpp"


template<typename MapInput, typename MapOutput>
class MappingChain: public IRunnable<MapInput, MapOutput> {
    std::unordered_map<std::string, ChainablePtr> chainables_;
public:
    explicit MappingChain(std::unordered_map<std::string, ChainablePtr> chainables)
        : chainables_(std::move(chainables)) {
    }

    MapOutput Invoke(const MapInput& input) override {
        auto input_context = std::make_shared<HashMapContext>();
        input.DumpContext(input_context);
        auto output_context = std::make_shared<HashMapContext>();
        for (const auto& [k,v]: chainables_) {
            v->Invoke(input_context->Clone());
            output_context->MapContext(k, input_context);
        }
        return MapOutput::Create(output_context);
    }
};

#endif //MAPPINGCHAIN_HPP
