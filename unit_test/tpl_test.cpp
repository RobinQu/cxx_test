//
// Created by RobinQu on 2024/4/22.
//
#include <inja/inja.hpp>
#include <nlohmann/json.hpp>


int main() {
    using namespace inja;
    nlohmann::json data;
    data["name"] = "world";
    data["phone_number"] = "";

    // render("Hello {{ name }}!", data); // Returns std::string "Hello world!"
    // render_to(std::cout, "Hello {{ name }}!", data);

    render_to(std::cout, R"({% if phone_number %}
Phone: {{phone_number}}
{% else %}
no phone number
{% endif %})", data);


    Environment env;
    env.add_callback("is_blank", 1, [](Arguments args) {
        auto v = args.at(0)->get<std::string>();
        return v.empty();
    });
    std::cout << env.render(R"({% if not is_blank(phone_number) %}
Phone: {{phone_number}}
{% else %}
no phone number
{% endif %})", data) << std::endl;

}
