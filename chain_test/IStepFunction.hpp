//
// Created by RobinQu on 2024/3/20.
//

#ifndef CHAINABLE_HPP
#define CHAINABLE_HPP
#include <memory>


#include "JSONObjectFuncContext.hpp"

template<typename Context = ContextPtr<nlohmann::json>>
class IStepFunction {
public:
    IStepFunction()=default;
    IStepFunction(IStepFunction&&)=delete;
    IStepFunction(const IStepFunction&)=delete;
    virtual ~IStepFunction() = default;

    virtual void Invoke(Context& context) = 0;
};

using StepFunctionPtr = std::shared_ptr<IStepFunction<>>;


#endif //CHAINABLE_HPP
