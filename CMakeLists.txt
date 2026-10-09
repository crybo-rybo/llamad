cmake_minimum_required(VERSION 3.25)
project(llamad LANGUAGES C CXX)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    set(CMAKE_BUILD_TYPE Release)
endif()

option(LLAMAD_WARNINGS_AS_ERRORS "Fail the build on a warning in llamad's own sources" OFF)
option(LLAMAD_BUILD_DOCS "Enable the Doxygen documentation target" OFF)
if(LLAMAD_BUILD_DOCS)
    add_subdirectory(docs)
endif()

# llama.cpp: libllama + ggml, plus `common` (the Jinja chat templates, PEG chat parsers and
# schema->grammar converter), which only the chat layer uses. None of its tools/server.
# OPENSSL/SUBPROCESS off: common's vendored cpp-httplib is for model downloads only, and turning
# them off keeps llamad free of an OpenSSL dependency and of process spawning. llama.cpp sets the
# language standard of its own targets.
set(LLAMA_BUILD_COMMON   ON  CACHE BOOL "" FORCE)
set(LLAMA_OPENSSL        OFF CACHE BOOL "" FORCE)
set(LLAMA_SUBPROCESS     OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_TESTS    OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_TOOLS    OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_SERVER   OFF CACHE BOOL "" FORCE)
add_subdirectory(third_party/llama.cpp EXCLUDE_FROM_ALL)
# Their headers are someone else's code: llamad's warning flags do not apply to them.
set_target_properties(llama ggml llama-common nlohmann PROPERTIES SYSTEM ON)

# Protobuf through its CMake config where it ships one (Arch, Homebrew), else CMake's module
# (Debian and Ubuntu).
find_package(Protobuf CONFIG QUIET)
if(NOT Protobuf_FOUND)
    find_package(Protobuf REQUIRED)
endif()
find_package(gRPC CONFIG REQUIRED)

# What every llamad target compiles with.
add_library(llamad_options INTERFACE)
target_compile_features(llamad_options INTERFACE cxx_std_23)
target_compile_options(llamad_options INTERFACE
    -Wall -Wextra -Wpedantic
    $<$<BOOL:${LLAMAD_WARNINGS_AS_ERRORS}>:-Werror>)

# The wire contract, generated: target llamad_proto.
add_subdirectory(proto)

# Engine: libllama wrapper, no gRPC or protobuf.
add_library(llamad_engine STATIC src/engine.cpp)
target_include_directories(llamad_engine PUBLIC src)
target_link_libraries(llamad_engine PUBLIC llama PRIVATE llamad_options)

# Chat layer: the only place llama.cpp's unstable `common` chat API is used. llama-common is
# PRIVATE so none of its headers (nlohmann/json among them) leak to users of chat_format.h; CMake
# still propagates the archive itself because llamad_chat is static.
add_library(llamad_chat STATIC src/chat_format.cpp)
target_link_libraries(llamad_chat PUBLIC llamad_engine PRIVATE llama-common llamad_options)

# The gRPC service and the conversion between wire messages and engine types.
add_library(llamad_service STATIC src/service.cpp src/wire.cpp)
target_link_libraries(llamad_service PUBLIC llamad_engine llamad_chat llamad_proto PRIVATE llamad_options)

add_executable(llamad src/main.cpp)
target_link_libraries(llamad PRIVATE llamad_service gRPC::grpc++_reflection llamad_options)

# A standalone example client: only generated stubs, with no engine or chat-layer dependency.
add_executable(llamad_playground examples/playground_main.cpp examples/playground.cpp)
target_include_directories(llamad_playground PRIVATE src)
target_link_libraries(llamad_playground PRIVATE llamad_proto llamad_options)

enable_testing()
add_subdirectory(tests)
