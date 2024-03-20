//
// Created by RobinQu on 2024/3/20.
//

#ifndef CHAINBUILDER_HPP
#define CHAINBUILDER_HPP
#include <nlohmann/json_fwd.hpp>

#include "IFunctionContext.hpp"
#include "IRunnable.hpp"
#include "IStepFunction.hpp"
#include "MappingSteps.hpp"
#include "SequenceSteps.hpp"
#include "RunnableChain.hpp"


template<
    typename Input,
    typename Output,
    typename ContextPayload = nlohmann::json,
    typename InputConverter = RunnablePtr<Input, ContextPtr<ContextPayload>>,
    typename OutputConverter = RunnablePtr<ContextPtr<ContextPayload>, Output>
>
class ChainBuilder final {
    InputConverter input_converter_;
    OutputConverter output_converter_;
    std::vector<StepFunctionPtr> steps_{};

    std::unordered_map<std::string, StepFunctionPtr> current_map_;

public:

    static auto Create() {
        return std::make_shared<ChainBuilder>();
    }

    auto WithInputConverter(const InputConverter& input_converter) {
        input_converter_ = input_converter;
        return this;
    }

    auto WithOutputConvter(const OutputConverter& output_converter) {
        output_converter_ = output_converter;
        return this;
    }

    auto WithMappingSteps(const std::unordered_map<std::string, StepFunctionPtr>& mapped) {
        steps_.push_back(std::make_shared<MappingSteps>(mapped));
        return this;
    }

    auto WithMappingStep(const std::string& name, const StepFunctionPtr& step) {
        current_map_[name] = step;
        return this;
    }

    auto FinishMappingSteps() {
        steps_.push_back(std::make_shared<MappingSteps>(current_map_));
        current_map_.clear();
        return this;
    }

    auto WithStep(const StepFunctionPtr& step) {
        steps_.push_back(step);
        return this;
    }

    ChainPtr<Input,Output> Build() {
        if (!current_map_.empty()) {
            FinishMappingSteps();
        }
        assert(!steps_.empty());
        return std::make_shared<RunnableChain<Input, Output>>(
            input_converter_,
            output_converter_,
            steps_.size() == 1 ? steps_[0]: std::make_shared<SequenceSteps>(steps_)
        );
    }

};


#endif //CHAINBUILDER_HPP
