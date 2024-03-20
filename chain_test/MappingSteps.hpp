//
// Created by RobinQu on 2024/3/20.
//

#ifndef MAPPINGCHAIN_HPP
#define MAPPINGCHAIN_HPP
#include <unordered_map>

#include "IStepFunction.hpp"
#include "IRunnable.hpp"


class MappingSteps final: public IStepFunction<> {
    std::unordered_map<std::string, StepFunctionPtr> steps_{};
public:
    explicit MappingSteps(std::unordered_map<std::string, StepFunctionPtr> chainables)
        : steps_(std::move(chainables)) {
    }

    void Invoke(std::shared_ptr<IFunctionContext<nlohmann::basic_json<>>>& context) override {
        auto output_context = std::make_shared<JSONObjectFuncContext>();
        for (const auto& [k,v]: steps_) {
            auto child_context = context->Clone();
            v->Invoke(child_context);
            output_context->MapPayload(k, child_context->MutablePayload());
        }
        context = std::move(output_context);
    }
};

#endif //MAPPINGCHAIN_HPP
