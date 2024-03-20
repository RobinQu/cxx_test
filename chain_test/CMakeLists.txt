cmake_minimum_required(VERSION 3.28)
project(chain_test)

set(CMAKE_CXX_STANDARD 20)

find_package(nlohmann_json)

add_executable(main
        main.cpp
        IStepFunction.hpp
        IRunnable.hpp
        SequenceSteps.hpp
        MappingSteps.hpp
        IFunctionContext.hpp
        JSONObjectFuncContext.hpp
        RunnableChain.hpp)

target_link_libraries(main nlohmann_json::nlohmann_json)