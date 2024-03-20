//
// Created by RobinQu on 2024/3/20.
//

#ifndef ICHAINCONTEXT_HPP
#define ICHAINCONTEXT_HPP
#include <string>



template<typename Payload>
class IFunctionContext {
public:
    IFunctionContext()=default;
    virtual ~IFunctionContext()=default;
    IFunctionContext(IFunctionContext&&)=delete;
    IFunctionContext(const IFunctionContext&)=delete;

    virtual Payload& MutablePayload() = 0;

    virtual void MapPayload(const std::string& name, const Payload& payload)  = 0;

    // virtual std::string GetString(const std::string& name, const std::string& default_value) = 0;
    // virtual void SetString(const std::string& name, const std::string& value) = 0;

    // virtual void MapContext(const std::string& name, const ContextPtr& child_context) = 0;
    virtual std::shared_ptr<IFunctionContext<Payload>> Clone() const = 0;
};

template<typename Payload>
using ContextPtr = std::shared_ptr<IFunctionContext<Payload>>;


#endif //ICHAINCONTEXT_HPP
