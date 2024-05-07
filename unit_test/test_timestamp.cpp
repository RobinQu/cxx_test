//
// Created by RobinQu on 2024/5/6.
//
#include <google/protobuf/util/json_util.h>
#include <test.pb.h>

int main() {
    Foo foo;
    foo.mutable_created_at()->set_seconds(1715001463);
    foo.set_updated_at(1715001463);
    std::string output;
    google::protobuf::util::MessageToJsonString(foo, &output);
    std::cout << output << std::endl;
}