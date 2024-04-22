//
// Created by RobinQu on 2024/4/22.
//
#include <inja/inja.hpp>
#include <nlohmann/json.hpp>


int main() {
    using namespace inja;
    nlohmann::json data;
    data["name"] = "world";

    render("Hello {{ name }}!", data); // Returns std::string "Hello world!"
    render_to(std::cout, "Hello {{ name }}!", data);
}
