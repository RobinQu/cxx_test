//
// Created by RobinQu on 2024/5/17.
//
#include <test.pb.h>
#include <google/protobuf/util/json_util.h>

int main() {
    Cow cow;
    Foo foo;
    foo.set_bar("aa");
    cow.mutable_custom()->PackFrom(foo);
    std::string output;
    google::protobuf::util::MessageToJsonString(cow, &output);
    std::cout << output << std::endl;
}
