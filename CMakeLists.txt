cmake_minimum_required(VERSION 3.20)
project(llamad LANGUAGES C CXX)

# llama.cpp builds at the standard it is written for. The project's own targets are declared
# after it and use C++26 (see below).
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

# llama.cpp: libllama + ggml, plus `common` (the Jinja chat templates, PEG chat parsers and
# schema->grammar converter), which only the chat layer below uses. None of its tools/server.
# OPENSSL/SUBPROCESS off: common's vendored cpp-httplib is for model downloads only, and turning
# them off is what keeps llamad free of an OpenSSL dependency and of process spawning.
set(LLAMA_BUILD_COMMON   ON  CACHE BOOL "" FORCE)
set(LLAMA_OPENSSL        OFF CACHE BOOL "" FORCE)
set(LLAMA_SUBPROCESS     OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_TESTS    OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_TOOLS    OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_SERVER   OFF CACHE BOOL "" FORCE)
add_subdirectory(third_party/llama.cpp EXCLUDE_FROM_ALL)

# llamad is written in C++26 and relies on static reflection (<meta>), which GCC enables with
# -freflection. Checked here so an unsupported compiler fails at configure time with a reason.
set(CMAKE_CXX_STANDARD 26)
include(CheckCXXSourceCompiles)
set(CMAKE_REQUIRED_FLAGS "-std=c++26 -freflection")
check_cxx_source_compiles("
    #include <meta>
    struct S { int a; };
    static_assert(std::meta::nonstatic_data_members_of(^^S, std::meta::access_context::current()).size() == 1);
    int main() {}" LLAMAD_HAVE_REFLECTION)
unset(CMAKE_REQUIRED_FLAGS)
if(NOT LLAMAD_HAVE_REFLECTION)
    message(FATAL_ERROR "llamad needs a compiler with C++26 static reflection (GCC 16 or later): "
                        "${CMAKE_CXX_COMPILER_ID} ${CMAKE_CXX_COMPILER_VERSION} does not provide it.")
endif()
add_compile_options($<$<COMPILE_LANGUAGE:CXX>:-freflection>)

# Engine: libllama wrapper, no gRPC/protobuf.
add_library(llamad_engine STATIC src/engine.cpp)
target_include_directories(llamad_engine PUBLIC src)
target_link_libraries(llamad_engine PUBLIC llama)

# Chat layer: the only place llama.cpp's unstable `common` chat API is used. llama-common is
# PRIVATE so none of its headers (or nlohmann/json) leak to consumers of chat_format.h; CMake
# still propagates the archive itself through $<LINK_ONLY:> because llamad_chat is static.
# common's headers (nlohmann/json among them) are written for C++17 and are included as system
# headers, so what C++26 deprecates in them is not reported against this project.
add_library(llamad_chat STATIC src/chat_format.cpp)
set_target_properties(llama-common nlohmann PROPERTIES SYSTEM ON)
target_include_directories(llamad_chat PUBLIC src)
target_link_libraries(llamad_chat PRIVATE llama-common)

add_executable(engine_smoke src/engine_smoke.cpp)
target_link_libraries(engine_smoke PRIVATE llamad_engine llamad_chat)

enable_testing()

# Everything below needs gRPC; the engine can be built and tested without it.
find_package(Protobuf CONFIG)
find_package(gRPC CONFIG)
if(NOT (Protobuf_FOUND AND gRPC_FOUND))
    message(WARNING "gRPC/Protobuf not found: building the engine only (see README.md for dependencies)")
    add_subdirectory(tests)
    return()
endif()

# `proto` is on the include path alongside the generated sources, so convert.h sits next to the
# .proto it converts and both sides spell it "llamad/v1/convert.h".
add_library(llamad_proto STATIC proto/llamad/v1/llamad.proto)
target_link_libraries(llamad_proto PUBLIC protobuf::libprotobuf gRPC::grpc++)
target_include_directories(llamad_proto PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/proto ${CMAKE_CURRENT_BINARY_DIR}/gen)
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
target_link_libraries(llamad PRIVATE llamad_engine llamad_chat llamad_proto gRPC::grpc++_reflection)

add_subdirectory(client)

# Last: the contract test sees the proto, the engine, the chat layer and the client together.
add_subdirectory(tests)
