//
// Created by RobinQu on 2024/3/20.
//

#ifndef ICHAINCONTEXT_HPP
#define ICHAINCONTEXT_HPP
#include <string>

class IChainContext;
using ContextPtr = std::shared_ptr<IChainContext>;


class IChainContext {
public:
    IChainContext()=default;
    virtual ~IChainContext()=default;
    IChainContext(IChainContext&&)=delete;
    IChainContext(const IChainContext&)=delete;

    virtual std::string GetString(const std::string& name, const std::string& default_value) = 0;
    virtual void SetString(const std::string& name, const std::string& value) = 0;

    virtual void MapContext(const std::string& name, const ContextPtr& child_context) = 0;
    virtual ContextPtr Clone() = 0;
};


#endif //ICHAINCONTEXT_HPP
