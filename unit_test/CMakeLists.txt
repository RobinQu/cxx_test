cmake_minimum_required(VERSION 3.28)
project(unit_test)

set(CMAKE_CXX_STANDARD 20)

file(GLOB_RECURSE TEST_SRC_FILES *.cpp)
find_package(nlohmann_json)
find_package(protobuf)
find_package(inja REQUIRED)


# from list of files we'll create tests test_name.cpp -> test_name
foreach(_test_file ${TEST_SRC_FILES})
    get_filename_component(_test_name ${_test_file} NAME_WE)
    add_executable(${_test_name} ${_test_file})
    target_link_libraries(${_test_name} nlohmann_json::nlohmann_json protobuf::protobuf pantor::inja)
endforeach()
