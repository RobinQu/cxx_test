//
// Created by RobinQu on 2024/3/20.
//

#ifndef ALPHAJOB_HPP
#define ALPHAJOB_HPP
#include "IChainable.hpp"

class AlphaJob: public IChainable<> {
public:
    void Invoke(const std::shared_ptr<IChainContext>& context) override {
        context->SetString("alpha_output", context->GetString("input", ""));
    }
};


#endif //ALPHAJOB_HPP
