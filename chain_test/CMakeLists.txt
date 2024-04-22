cmake_minimum_required(VERSION 3.26)
project(chain_test)

set(CMAKE_CXX_STANDARD 20)

find_package(nlohmann_json)
find_package(protobuf)
find_package(httplib)

add_library(proto-objects OBJECT proto/server.proto)
target_link_libraries(proto-objects PUBLIC
        protobuf::libprotoc
        protobuf::libprotobuf
)
set(PROTO_BINARY_DIR "${CMAKE_CURRENT_BINARY_DIR}")
target_include_directories(proto-objects PUBLIC "$<BUILD_INTERFACE:${PROTO_BINARY_DIR}>")

protobuf_generate(
        TARGET proto-objects
        IMPORT_DIRS "${CMAKE_CURRENT_LIST_DIR}/proto"
        PROTOC_OUT_DIR "${PROTO_BINARY_DIR}"
)


add_library(chain_lib INTERFACE
        IStepFunction.hpp
        IRunnable.hpp
        SequenceSteps.hpp
        MappingSteps.hpp
        IFunctionContext.hpp
        JSONObjectFuncContext.hpp
        RunnableChain.hpp
        ChainBuilder.hpp
)

target_link_libraries(chain_lib INTERFACE proto-objects nlohmann_json::nlohmann_json protobuf::protobuf)

add_executable(main main.cpp)
target_link_libraries(main chain_lib)

add_executable(server server.cpp)
target_link_libraries(server chain_lib httplib::httplib)
