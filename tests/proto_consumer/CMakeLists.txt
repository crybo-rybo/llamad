# A direct-stub consumer with an ordinary C++ compiler and no submodules or reflected client.
cmake_minimum_required(VERSION 3.21)
project(llamad_proto_consumer LANGUAGES CXX)

set(LLAMAD_GIT_REPOSITORY "https://github.com/crybo-rybo/llamad.git" CACHE STRING "Repository to fetch llamad from")
set(LLAMAD_GIT_TAG        "main"                                    CACHE STRING "llamad commit, branch or tag")

set(LLAMAD_PROTO_ONLY ON CACHE BOOL "" FORCE)
include(FetchContent)
FetchContent_Declare(llamad
    GIT_REPOSITORY ${LLAMAD_GIT_REPOSITORY}
    GIT_TAG        ${LLAMAD_GIT_TAG}
    GIT_SUBMODULES "")
FetchContent_MakeAvailable(llamad)

if(TARGET llamad_client OR TARGET llama OR CMAKE_C_COMPILER_LOADED)
    message(FATAL_ERROR "Proto-only builds only the wire stubs and enables only C++")
endif()

add_executable(consumer consumer.cpp)
target_link_libraries(consumer PRIVATE llamad::proto)

enable_testing()
add_test(NAME consumer COMMAND consumer)
