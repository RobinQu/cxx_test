//
// Created by RobinQu on 2024/3/20.
//

#ifndef SEQUENCECHAIN_HPP
#define SEQUENCECHAIN_HPP

#include <vector>

#include "HashMapContext.hpp"
#include "IChainable.hpp"
#include "IRunnable.hpp"

// struct Foo {
//
// };
//
// struct Bar {
//
// };

template<typename Input, typename Output>
class SequenceChain: public IRunnable<Input, Output> {
    std::vector<ChainablePtr> chains_;

public:
    explicit SequenceChain(std::vector<ChainablePtr> chains)
        : chains_(std::move(chains)) {
    }

    Output Invoke(const Input& input) override {
        auto context = std::make_shared<HashMapContext>();
        input.DumpContext(context);
        for (const auto& chainable: chains_) {
            chainable->Invoke(context);
        }
        return Output::Create(context);
    }
};


#endif //SEQUENCECHAIN_HPP
