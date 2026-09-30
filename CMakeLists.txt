cmake_minimum_required(VERSION 3.21)
project(llamad LANGUAGES CXX)

# Included by another project, the client is the default. Proto-only takes precedence and
# needs neither the reflected client nor a C compiler or a llama.cpp checkout.
option(LLAMAD_PROTO_ONLY "Build only llamad_proto, without reflection or llama.cpp" OFF)
if(PROJECT_IS_TOP_LEVEL)
    option(LLAMAD_CLIENT_ONLY "Build only llamad_proto and llamad_client, without llama.cpp" OFF)
else()
    option(LLAMAD_CLIENT_ONLY "Build only llamad_proto and llamad_client, without llama.cpp" ON)
endif()

option(LLAMAD_BUILD_DOCS "Enable the Doxygen documentation target" OFF)
if(LLAMAD_BUILD_DOCS)
    add_subdirectory(docs)
endif()

# Only for the project being built, never for one that includes llamad.
if(PROJECT_IS_TOP_LEVEL)
    set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
    if(NOT CMAKE_BUILD_TYPE)
        set(CMAKE_BUILD_TYPE Release)
    endif()
endif()

if(NOT (LLAMAD_CLIENT_ONLY OR LLAMAD_PROTO_ONLY))
    enable_language(C)

    # llama.cpp builds at the standard it is written for. Our targets declare their own level.
    set(CMAKE_CXX_STANDARD 17)
    set(CMAKE_CXX_STANDARD_REQUIRED ON)

    # llama.cpp: libllama + ggml, plus `common` (the Jinja chat templates, PEG chat parsers and
    # schema->grammar converter), which only the chat layer below uses. None of its tools/server.
    # OPENSSL/SUBPROCESS off: common's vendored cpp-httplib is for model downloads only, and
    # turning them off is what keeps llamad free of an OpenSSL dependency and of process spawning.
    set(LLAMA_BUILD_COMMON   ON  CACHE BOOL "" FORCE)
    set(LLAMA_OPENSSL        OFF CACHE BOOL "" FORCE)
    set(LLAMA_SUBPROCESS     OFF CACHE BOOL "" FORCE)
    set(LLAMA_BUILD_TESTS    OFF CACHE BOOL "" FORCE)
    set(LLAMA_BUILD_TOOLS    OFF CACHE BOOL "" FORCE)
    set(LLAMA_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
    set(LLAMA_BUILD_SERVER   OFF CACHE BOOL "" FORCE)
    add_subdirectory(third_party/llama.cpp EXCLUDE_FROM_ALL)

    # On macOS, build.sh points ggml's BLAS backend at vecLib's cblas.h, whose classic interface
    # Apple marks deprecated in favour of one GCC cannot compile. The warning is about
    # llama.cpp's source, not this project's.
    if(TARGET ggml-blas)
        target_compile_options(ggml-blas PRIVATE -Wno-deprecated-declarations)
    endif()
endif()

# The client and flag parser reflect over application structs, so every build but proto-only
# checks for reflection and fails at configure time with a reason. The function scope keeps the
# probe's C++26 standard from reaching llama.cpp or the stubs; each target that reflects
# declares C++26 and -freflection itself.
function(llamad_check_reflection)
    include(CheckCXXSourceCompiles)
    set(CMAKE_CXX_STANDARD 26)
    set(CMAKE_REQUIRED_FLAGS "-freflection")
    check_cxx_source_compiles("
        #include <meta>
        struct S { int a; };
        static_assert(std::meta::nonstatic_data_members_of(^^S, std::meta::access_context::current()).size() == 1);
        int main() {}" LLAMAD_HAVE_REFLECTION)
endfunction()

if(NOT LLAMAD_PROTO_ONLY)
    llamad_check_reflection()
    if(NOT LLAMAD_HAVE_REFLECTION)
        message(FATAL_ERROR "llamad needs a compiler with C++26 static reflection (GCC 16 or later): "
                            "${CMAKE_CXX_COMPILER_ID} ${CMAKE_CXX_COMPILER_VERSION} does not provide it.")
    endif()
endif()

if(NOT (LLAMAD_CLIENT_ONLY OR LLAMAD_PROTO_ONLY))
    # Command-line flags, from a struct whose members are the flags. Header-only and standard
    # library only; its one include directory is src/cli, so llamad-chat parses its flags without
    # any daemon header becoming reachable from the client tree.
    add_library(llamad_flags INTERFACE)
    target_include_directories(llamad_flags INTERFACE ${CMAKE_CURRENT_SOURCE_DIR}/src/cli)
    target_compile_features(llamad_flags INTERFACE cxx_std_26)
    target_compile_options(llamad_flags INTERFACE $<$<COMPILE_LANGUAGE:CXX>:-freflection>)

    # Engine: libllama wrapper, no gRPC/protobuf.
    add_library(llamad_engine STATIC src/engine.cpp)
    target_include_directories(llamad_engine PUBLIC src)
    target_link_libraries(llamad_engine PUBLIC llama)
    target_compile_features(llamad_engine PUBLIC cxx_std_26)

    # Chat layer: the only place llama.cpp's unstable `common` chat API is used. llama-common is
    # PRIVATE so none of its headers (or nlohmann/json) leak to consumers of chat_format.h; CMake
    # still propagates the archive itself through $<LINK_ONLY:> because llamad_chat is static.
    # common's headers (nlohmann/json among them) are written for C++17 and are included as
    # system headers, so what C++26 deprecates in them is not reported against this project.
    add_library(llamad_chat STATIC src/chat_format.cpp)
    set_target_properties(llama-common nlohmann PROPERTIES SYSTEM ON)
    target_include_directories(llamad_chat PUBLIC src)
    target_link_libraries(llamad_chat PRIVATE llama-common)
    target_compile_features(llamad_chat PUBLIC cxx_std_26)
endif()

# Registered with a project that includes llamad, the tests would be its tests.
if(PROJECT_IS_TOP_LEVEL AND NOT LLAMAD_PROTO_ONLY)
    enable_testing()
endif()

# Everything below needs gRPC; the engine can be built and tested without it, but the client
# cannot.
find_package(Protobuf CONFIG QUIET)
if(NOT Protobuf_FOUND)
    find_package(Protobuf MODULE QUIET)
endif()
find_package(gRPC CONFIG QUIET)
if(NOT (Protobuf_FOUND AND gRPC_FOUND))
    if(LLAMAD_CLIENT_ONLY OR LLAMAD_PROTO_ONLY)
        message(FATAL_ERROR "gRPC/Protobuf not found, and client-only/proto-only builds require both: "
                            "`pacman -S grpc protobuf` on Arch Linux; "
                            "`apt install libgrpc++-dev libprotobuf-dev protobuf-compiler protobuf-compiler-grpc` "
                            "on Ubuntu; "
                            "on macOS, build them with llamad's scripts/build-deps-macos.sh "
                            "and put its build-deps/prefix on CMAKE_PREFIX_PATH (see docs/building.md)")
    endif()
    message(WARNING "gRPC/Protobuf not found: building the engine only (see README.md for dependencies)")
    add_subdirectory(tests)
    return()
endif()

add_subdirectory(proto)
if(LLAMAD_PROTO_ONLY)
    return()
endif()

add_subdirectory(client)
if(LLAMAD_CLIENT_ONLY)
    return()
endif()

add_executable(llamad src/main.cpp src/service.cpp)
target_link_libraries(llamad PRIVATE llamad_engine llamad_chat llamad_flags llamad_proto gRPC::grpc++_reflection)

# Last: the contract test sees the proto, the engine, the chat layer and the client together.
add_subdirectory(tests)
