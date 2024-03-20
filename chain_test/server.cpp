//
// Created by RobinQu on 2024/3/20.
//

#include <google/protobuf/util/json_util.h>
#include <httplib/httplib.h>

#include "server.pb.h"

#include "ChainBuilder.hpp"


class FooHandleStep: public IStepFunction<>{
public:
    void Invoke(std::shared_ptr<IFunctionContext<nlohmann::basic_json<>>>& context) override {
        const auto input = JSONObjectFuncContext::Get<std::string>(context, "prompt");
        JSONObjectFuncContext::Set(context, "answer", input + input);
    }
};

class JSONStringInputConvter: public IRunnable<std::string, ContextPtr<nlohmann::json>>{
public:
    std::shared_ptr<IFunctionContext<nlohmann::basic_json<>>> Invoke(const std::string& input) override {
        auto obj = nlohmann::json::parse(input);
        return std::make_shared<JSONObjectFuncContext>(obj);
    }
};

class JSONStringOutputConvter: public IRunnable<ContextPtr<nlohmann::json>, std::string> {
public:
    std::string Invoke(const std::shared_ptr<IFunctionContext<nlohmann::basic_json<>>>& input) override {
        return input->MutablePayload().dump();
    }
};

int main() {
    using namespace httplib;
    using namespace google::protobuf;


    auto builder = ChainBuilder<std::string, std::string>::Create();
    builder->WithInputConverter(std::make_shared<JSONStringInputConvter>());
    builder->WithOutputConvter(std::make_shared<JSONStringOutputConvter>());
    builder->WithStep(std::make_shared<FooHandleStep>());
    auto chain = builder->Build();

    Server server;
    server.set_tcp_nodelay(true);
    server.Post("/foo", [&](const Request& request, Response& response) {
        Arena arena;
        // auto foo_request = Arena::Create<FooRequest>(&arena);
        // util::JsonStringToMessage(request.body, foo_request);
        auto output = chain->Invoke(request.body);
        // util::MessageToJsonString(output, &response.body);
        response.status = 200;
        response.set_content(output, "text/plain");
    });

    server.Get("/bar", [](const Request& request, Response& response) {
        response.set_content("foobar", "text/plain");
    });

    int port = 9999;
    if(server.bind_to_port("0.0.0.0", port)) {
        std::cout << "server is up and running at " << port << std::endl;
        server.listen_after_bind();
        return 0;
    }
    std::cout << "server failed to start" << std::endl;
    return -1;
}
