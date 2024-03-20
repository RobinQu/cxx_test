//
// Created by RobinQu on 2024/3/20.
//

#ifndef BETAJOB_HPP
#define BETAJOB_HPP
#include "IChainable.hpp"

class BetaJob: public IChainable<> {
public:
    void Invoke(const std::shared_ptr<IChainContext>& context) override {
        auto alpha = context->GetString("alpha_output", "");
        context->SetString("beta_output", alpha + alpha);
    }
};


#endif //BETAJOB_HPP
