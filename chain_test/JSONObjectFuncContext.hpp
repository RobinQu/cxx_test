//
// Created by RobinQu on 2024/3/20.
//

#ifndef JSONOBJECTCHAINCONTEXT_HPP
#define JSONOBJECTCHAINCONTEXT_HPP

#include <nlohmann/json.hpp>

#include "IFunctionContext.hpp"
#include "IStepFunction.hpp"


class JSONObjectFuncContext final: public IFunctionContext<nlohmann::json>{
    nlohmann::json json_object_{};

public:
    template<typename T>
    static T Get(const ContextPtr<nlohmann::json>& context, const std::string& name) {
        return context->MutablePayload()[name].get<T>();
    }

    template<typename T>
    static void Set(const ContextPtr<nlohmann::json>& context, const std::string&name, T&& value) {
        context->MutablePayload()[name] = value;
    }

    template<typename T>
    static T GetPath(const ContextPtr<nlohmann::json>& context, const std::string& json_path) {
        return context->MutablePayload().at(nlohmann::json::json_pointer(json_path)).get<T>();
    }

    explicit JSONObjectFuncContext(nlohmann::json json_object = {})
        : json_object_(std::move(json_object)) {
    }

    nlohmann::basic_json<>& MutablePayload() override {
        return json_object_;
    }

    [[nodiscard]] std::shared_ptr<IFunctionContext<nlohmann::json>> Clone() const override {
        return std::make_shared<JSONObjectFuncContext>(json_object_);
    }

    void MapPayload(const std::string& name, const nlohmann::basic_json<>& payload) override {
        json_object_[name] = payload;
    }
};

using JSONObjectContextPtr = std::shared_ptr<IFunctionContext<nlohmann::json>>;



#endif //JSONOBJECTCHAINCONTEXT_HPP
