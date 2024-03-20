//
// Created by RobinQu on 2024/3/20.
//

#ifndef CHAINABLE_HPP
#define CHAINABLE_HPP
#include <memory>

#include "IChainContext.hpp"

template<typename Context = ContextPtr>
class IChainable {
public:
    IChainable()=default;
    IChainable(IChainable&&)=delete;
    IChainable(const IChainable&)=delete;
    virtual ~IChainable() = default;

    virtual void Invoke(const Context& context) = 0;
};

using ChainablePtr = std::shared_ptr<IChainable<>>;


#endif //CHAINABLE_HPP
