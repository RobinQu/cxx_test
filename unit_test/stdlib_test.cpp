//
// Created by RobinQu on 2024/4/18.
//
#include <filesystem>
#include <iostream>

int main() {
    std::filesystem::path root = "/";
    std::cout << (root / "abc") << std::endl;

    std::cout << (root / "a b c") << std::endl;

    std::cout << (root / "a /b | c") << std::endl;
}
