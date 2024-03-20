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
