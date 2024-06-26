//
// Created by RobinQu on 2024/3/20.
//

#ifndef IRUNNABLE_HPP
#define IRUNNABLE_HPP

template<typename Input, typename Output>
class IRunnable {
public:
    IRunnable()=default;
    virtual ~IRunnable()=default;
    IRunnable(IRunnable&&)=delete;
    IRunnable(const IRunnable&)=delete;

    virtual Output Invoke(const Input& input) = 0;
};

template<typename Input, typename Output>
using RunnablePtr = std::shared_ptr<IRunnable<Input,Output>>;

#endif //IRUNNABLE_HPP
