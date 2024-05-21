//
// Created by RobinQu on 2024/4/10.
//
#include <iostream>
#include <regex>

int main() {
    const std::string paragraph = "Thought: The observations indicate the responses to the quiz questions. The first response \"I don't know.\" is for the open-ended question about the favorite book, and the second response \"Carrot\" is for the multiple-choice question about identifying a fruit. I have sufficient information to provide feedback on these responses.\n\nAction: Finish(Feedback for the responses:\n1. For the question \"What is your favorite book?\", the response \"I don't know.\" suggests that you might not have a favorite book or are unsure at the moment. It's perfectly fine to not have a specific favorite.\n2. For the question \"Which of the following is a fruit?\", the response \"Carrot\" is incorrect. The correct answer is \"Apple\". Carrots are vegetables, not fruits.)";
    std::cout << paragraph << std::endl;


    std::regex ACTION_REGEX {R"(Action:\s*(.+)\(([^\)]*)\))"};
    std::smatch smatch;
    std::regex_search(paragraph, smatch, ACTION_REGEX);

    for(const auto& match: smatch) {
        std::cout << match.str() << std::endl;
    }



}
