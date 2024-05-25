cmake_minimum_required(VERSION 3.28)
project(unit_test)

set(CMAKE_CXX_STANDARD 20)

file(GLOB_RECURSE TEST_SRC_FILES *.cpp)
#set(TEST_SRC_FILES
#        json_test.cpp
#        regex_test.cpp
#        regex_test_2.cpp
#        regex_test_3.cpp
#        stdlib_test.cpp
#        test_any.cpp
#        test_timestamp.cpp
#        tpl_test.cpp
#)
find_package(nlohmann_json)
find_package(protobuf)
find_package(inja REQUIRED)

include(FetchContent)
FetchContent_Declare(
        ggml
        GIT_REPOSITORY https://github.com/ggerganov/ggml.git
        GIT_TAG f5c9599
)
FetchContent_MakeAvailable(ggml)


add_library(reranker INTERFACE reranker/bge_m3.hpp reranker/tokenizer.hpp
        reranker/config.hpp
        reranker/layers.hpp
        reranker/ops.hpp
        reranker/model.hpp)
target_link_libraries(reranker INTERFACE ggml)

add_library(proto-test OBJECT proto/test.proto)
target_link_libraries(proto-test PUBLIC
        protobuf::libprotoc
        protobuf::libprotobuf
)
set(PROTO_BINARY_DIR "${CMAKE_CURRENT_BINARY_DIR}")
target_include_directories(proto-test PUBLIC "$<BUILD_INTERFACE:${PROTO_BINARY_DIR}>")

protobuf_generate(
        TARGET proto-test
        IMPORT_DIRS "${CMAKE_CURRENT_LIST_DIR}/proto"
        PROTOC_OUT_DIR "${PROTO_BINARY_DIR}"
)


# from list of files we'll create tests test_name.cpp -> test_name
foreach(_test_file ${TEST_SRC_FILES})
    get_filename_component(_test_name ${_test_file} NAME_WE)
    add_executable(${_test_name} ${_test_file})
    target_link_libraries(${_test_name} PUBLIC nlohmann_json::nlohmann_json protobuf::protobuf pantor::inja proto-test reranker)
endforeach()
