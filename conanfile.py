from conan import ConanFile
from conan.tools.build import check_min_cppstd
from conan.tools.cmake import cmake_layout


class CXXTestRecipe(ConanFile):
    name = "cxx_test"
    version = "0.1.0"
    # exports_sources = "src/*"
    # no_copy_source = True
    settings = "os", "compiler", "build_type", "arch"
    generators = "CMakeDeps", "CMakeToolchain"

    def validate(self):
        check_min_cppstd(self, 20)

    def build_requirements(self):
        self.tool_requires("cmake/3.27.9")

    def requirements(self):
        self.requires("nlohmann_json/3.11.3")

    def layout(self):
        cmake_layout(self)
