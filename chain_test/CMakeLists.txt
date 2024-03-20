cmake_minimum_required(VERSION 3.28)
project(chain_test)

set(CMAKE_CXX_STANDARD 20)

add_executable(main
        main.cpp
        IChainable.hpp
        IRunnable.hpp
        SequenceChain.hpp
        MappingChain.hpp
        IChainContext.hpp
        HashMapContext.hpp
        AlphaJob.hpp
        BetaJob.hpp)
