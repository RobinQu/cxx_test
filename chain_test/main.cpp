//
// Created by RobinQu on 2024/3/20.
//
#include <assert.h>
#include <iostream>

#include "ChainBuilder.hpp"
#include "IFunctionContext.hpp"
#include "MappingSteps.hpp"
#include "RunnableChain.hpp"
#include "SequenceSteps.hpp"


class StepA: public IStepFunction<> {
public:
    void Invoke(std::shared_ptr<IFunctionContext<nlohmann::basic_json<>>>& context) override {
        context->MutablePayload()["stepA_output"] = "hello " + context->MutablePayload()["input"].get<std::string>();
    }
};

class StepB: public IStepFunction<> {
public:
    void Invoke(std::shared_ptr<IFunctionContext<nlohmann::basic_json<>>>& context) override {
        const auto input = context->MutablePayload().at("stepA_output").get<std::string>();
        context->MutablePayload()["output"] = (input + input + input);
    }
};

class StepC: public IStepFunction<> {
public:
    void Invoke(std::shared_ptr<IFunctionContext<nlohmann::basic_json<>>>& context) override {
        const auto input = context->MutablePayload()["input"].get<std::string>();
        auto output = input + input + input;
        context->MutablePayload()["stepC_output"] = output;
        context->MutablePayload()["output"] = output;
    }
};

class StringInputConverter: public IRunnable<std::string, JSONObjectContextPtr> {
public:
    std::shared_ptr<IFunctionContext<nlohmann::basic_json<>>> Invoke(const std::string& input) override {
        auto ctx = std::make_shared<JSONObjectFuncContext>();
        ctx->MutablePayload()["input"] = input;
        return ctx;
    }
};

class StringOutputConvter: public IRunnable<JSONObjectContextPtr, std::string> {
public:
    std::string Invoke(const std::shared_ptr<IFunctionContext<nlohmann::basic_json<>>>& input) override {
        auto a_plus_b = input->MutablePayload().at(nlohmann::json::json_pointer("/a_plus_b/output")).get<std::string>();
        auto c = input->MutablePayload().at(nlohmann::json::json_pointer("/c/output")).get<std::string>();
        return "a_plus_b=" + a_plus_b + ",c=" + c;
    }
};


StepFunctionPtr operator|(const StepFunctionPtr& first, const StepFunctionPtr& second) {
    return std::make_shared<SequenceSteps>(std::vector {first, second});
}


int main()
{
    StepFunctionPtr step_a = std::make_shared<StepA>();
    StepFunctionPtr step_b = std::make_shared<StepB>();
    StepFunctionPtr step_c = std::make_shared<StepC>();
    StepFunctionPtr sequence_steps = std::make_shared<SequenceSteps>( std::vector {step_a, step_b});
    StepFunctionPtr mapping_steps = std::make_shared<MappingSteps>(std::unordered_map<std::string, StepFunctionPtr> {
        {"a_plus_b", sequence_steps},
        {"c", step_c}
    });
    ContextPtr<nlohmann::json> context = std::make_shared<JSONObjectFuncContext>( nlohmann::json{ {"input", "hello"}});
    mapping_steps->Invoke(context);

    std::cout << context->MutablePayload().dump() << std::endl;
    auto input_conv = std::make_shared<StringInputConverter>();
    auto output_conv = std::make_shared<StringOutputConvter>();
    auto chain = std::make_shared<RunnableChain<std::string, std::string>>(
        input_conv,
        output_conv,
        mapping_steps
        );
    auto result = chain->Invoke("nice boy");
    std::cout << result << std::endl;
    auto chain_builder = ChainBuilder<std::string, std::string>::Create();
    chain_builder->WithMappingStep("a_plus_b", step_a | step_b);
    chain_builder->WithMappingStep("c", step_c);
    auto chain2 = chain_builder->Build();
    auto result2 = chain->Invoke("nice boy");
    std::cout << result2 << std::endl;
    assert(result == result2);


    return 0;
}
