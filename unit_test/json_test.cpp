//
// Created by RobinQu on 2024/4/23.
//

#include <nlohmann/json.hpp>
#include <iostream>

int main() {
    nlohmann::json data1 = "";
    std::cout << data1.empty() << std::endl;
    std::cout << data1.dump() << std::endl;

    nlohmann::json data2 = nullptr;
    std::cout << data2.empty() << std::endl;
    std::cout << data2.dump() << std::endl;

    int i = 1;
    nlohmann::json data3 = i;
    std::cout << data3.dump() << std::endl;

    auto data4 = nlohmann::json::parse(R"({"tool_calls":[{"function":{"arguments":"{\"bar\":1}","name":"foo"},"id":"call-1"}]})");
    std::cout << data4.dump() << std::endl;


}
