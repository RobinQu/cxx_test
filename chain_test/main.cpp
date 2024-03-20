//
// Created by RobinQu on 2024/3/20.
//
#include <assert.h>
#include <iostream>

#include "AlphaJob.hpp"
#include "BetaJob.hpp"
#include "HashMapContext.hpp"
#include "IChainContext.hpp"
#include "SequenceChain.hpp"

struct Foo {
    std::string input;
    std::string alpha;
    std::string beta;

    void DumpContext(const ContextPtr& context) const {
        // ContextPtr context = std::make_shared<HashMapContext>();
        context->SetString("input", input);
        // return context;
    }

    static Foo Create(const ContextPtr& context) {
        return  {
            .alpha = context->GetString("alpha_output", ""),
            .beta = context->GetString("beta_output", "")
        };
    }
};


int main()
{
    ChainablePtr alpha_job = std::make_shared<AlphaJob>();
    ChainablePtr beta_job = std::make_shared<BetaJob>();
    SequenceChain<Foo, Foo> sequence_chain({alpha_job, beta_job});

    Foo foo = sequence_chain.Invoke({.input = "hello"});

    assert(foo.alpha == "hello");
    assert(foo.beta == "hellohello");

    return 0;
}
