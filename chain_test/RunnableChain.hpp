//
// Created by RobinQu on 2024/3/20.
//

#ifndef RUNNABLECHAIN_HPP
#define RUNNABLECHAIN_HPP
#include "IFunctionContext.hpp"
#include "IRunnable.hpp"
#include "IStepFunction.hpp"

template<
    typename Input,
    typename Output,
    typename ContextPayload = nlohmann::json,
    typename InputConverter = RunnablePtr<Input, ContextPtr<ContextPayload>>,
    typename OutputConverter = RunnablePtr<ContextPtr<ContextPayload>, Output>
>
class RunnableChain final: public IRunnable<Input, Output> {
    InputConverter input_converter_;
    OutputConverter output_converter_;
    StepFunctionPtr step_function_{};
public:
    RunnableChain(InputConverter input_converter, OutputConverter output_converter, StepFunctionPtr step_function)
        : input_converter_(std::move(input_converter)),
          output_converter_(std::move(output_converter)),
          step_function_(std::move(step_function)) {
    }

    Output Invoke(const Input& input) override {
        auto ctx = input_converter_->Invoke(input);
        step_function_->Invoke(ctx);
        return output_converter_->Invoke(ctx);
    }
};

#endif //RUNNABLECHAIN_HPP
