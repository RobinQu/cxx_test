//
// Created by RobinQu on 2024/3/20.
//

#ifndef SEQUENCECHAIN_HPP
#define SEQUENCECHAIN_HPP

#include <vector>

#include "IStepFunction.hpp"
#include "IRunnable.hpp"
#include "JSONObjectFuncContext.hpp"


class SequenceSteps final: public IStepFunction<> {
    std::vector<StepFunctionPtr> steps{};
public:
    explicit SequenceSteps(std::vector<StepFunctionPtr> steps)
        : steps(std::move(steps)) {
    }

    void Invoke(std::shared_ptr<IFunctionContext<nlohmann::basic_json<>>>& context) override {
        for(const auto& step: steps) {
            step->Invoke(context);
        }
    }
};


#endif //SEQUENCECHAIN_HPP
