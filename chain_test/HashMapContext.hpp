//
// Created by RobinQu on 2024/3/20.
//

#ifndef HASHMAPCONTEXT_HPP
#define HASHMAPCONTEXT_HPP


#include <unordered_map>

#include "IChainContext.hpp"

class HashMapContext: public IChainContext {
    friend HashMapContext;
    std::unordered_map<std::string, std::string> data_;
public:
    explicit HashMapContext(const std::unordered_map<std::string, std::string>& data = {})
        : data_(data) {
    }

    std::string GetString(const std::string& name, const std::string& default_value) override {
        if (data_.contains(name)) {
            return data_.at(name);
        }
        return default_value;
    }

    void SetString(const std::string& name, const std::string& value) override {
        data_[name] = value;
    }

    void MapContext(const std::string& name, const ContextPtr& child_context) override {

    }

    ContextPtr Clone() override {
        return std::make_shared<HashMapContext>(data_);
    }
};

#endif //HASHMAPCONTEXT_HPP
