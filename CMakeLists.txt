cmake_minimum_required(VERSION 3.20)
project(llamad LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

# llama.cpp: only libllama + ggml, none of its tools/server/common.
set(LLAMA_BUILD_COMMON   OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_TESTS    OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_TOOLS    OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_SERVER   OFF CACHE BOOL "" FORCE)
add_subdirectory(third_party/llama.cpp EXCLUDE_FROM_ALL)

# Engine: libllama wrapper, no gRPC/protobuf.
add_library(llamad_engine STATIC src/engine.cpp)
target_include_directories(llamad_engine PUBLIC src)
target_link_libraries(llamad_engine PUBLIC llama)

add_executable(engine_smoke src/engine_smoke.cpp)
target_link_libraries(engine_smoke PRIVATE llamad_engine)

# Everything below needs gRPC; the engine can be built and tested without it.
find_package(Protobuf CONFIG)
find_package(gRPC CONFIG)
if(NOT (Protobuf_FOUND AND gRPC_FOUND))
    message(WARNING "gRPC/Protobuf not found: building the engine only (Arch: pacman -S grpc protobuf)")
    return()
endif()

add_library(llamad_proto STATIC proto/llamad/v1/llamad.proto)
target_link_libraries(llamad_proto PUBLIC protobuf::libprotobuf gRPC::grpc++)
target_include_directories(llamad_proto PUBLIC ${CMAKE_CURRENT_BINARY_DIR}/gen)
file(MAKE_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/gen)
protobuf_generate(TARGET llamad_proto
    IMPORT_DIRS proto
    PROTOC_OUT_DIR ${CMAKE_CURRENT_BINARY_DIR}/gen)
protobuf_generate(TARGET llamad_proto
    LANGUAGE grpc
    GENERATE_EXTENSIONS .grpc.pb.h .grpc.pb.cc
    PLUGIN "protoc-gen-grpc=$<TARGET_FILE:gRPC::grpc_cpp_plugin>"
    IMPORT_DIRS proto
    PROTOC_OUT_DIR ${CMAKE_CURRENT_BINARY_DIR}/gen)

add_executable(llamad src/main.cpp src/service.cpp)
target_link_libraries(llamad PRIVATE llamad_engine llamad_proto gRPC::grpc++_reflection)

add_subdirectory(client)
