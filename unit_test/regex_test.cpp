//
// Created by RobinQu on 2024/4/10.
//
#include <iostream>
#include <regex>

int main() {
    const std::string paragraph = R"(" Math expression: 1 + 1

This expression evaluates to the answer of the question, which is 2. However, I am just providing the mathematical expression for you to input into a math library or calculator for evaluation. I cannot calculate or provide the result directly here.")";

    const std::string pattern_text = R"(Math expression:\s*(.+)\n)";
    const auto pattern = std::regex(pattern_text);
    // std::smatch text_match;
    // std::regex_match(str1, text_match, expression_regex);
    // for(const auto& match: text_match) {
    //     std::cout << match << std::endl;
    // }
    std::sregex_iterator iter(paragraph.begin(), paragraph.end(), pattern);
    std::sregex_iterator end;

    std::vector<std::smatch> matches;

    while (iter != end) {
        matches.push_back(*iter++);
    }

    if (!matches.empty()) {
        for (const auto& match : matches) {
            for(size_t i=0; i<match.size(); ++i) {
                std::cout << "Sub-match found: " << match.str(i) << '\n';
            }

        }
    } else {
        std::cout << "No match found.\n";
    }


}
