//
// Created by RobinQu on 2024/4/10.
//
#include <iostream>
#include <regex>

int main() {
    const std::string paragraph = R"(1. display_quiz({
"type": "object",
"properties": {
"title": {"type": "string", "value": "Quiz Time!"},
"questions": {
"type": "array",
"items": {
"type": "object",
"properties": {
"question_type": {"type": "string", "enum": ["MULTIPLE_CHOICE"]},
"question_text": {"type": "string", "value": "What is your favorite programming language?"},
"choices": {
"type": "array",
"items": {"type": "string", "value": "Python"}
}
},
"required": ["question_type", "question_text", "choices"]
}
},
"required": ["title", "questions"]
})

2. display_quiz({
"type": "object",
"properties": {
"title": {"type": "string", "value": "Quiz Time!"},
"questions": {
"type": "array",
"items": {
"type": "object",
"properties": {
"question_type": {"type": "string", "enum": ["FREE_RESPONSE"]},
"question_text": {"type": "string", "value": "Why is it your favorite programming language?"}
},
"required": ["question_type", "question_text"]
}
},
"required": ["title", "questions"]
})

3. join())";

    static std::regex ACTION_PATTERN {R"((\d+)\.\s*(.+)\((.*)\))", std::regex::ECMAScript|std::regex::multiline};
    std::smatch action_match; std::regex_search(paragraph, action_match, ACTION_PATTERN);
    for (const auto& match: action_match) {
        std::cout << match.str() << std::endl;

    }
}
