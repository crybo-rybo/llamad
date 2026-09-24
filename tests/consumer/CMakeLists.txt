# A project that uses llamad the way an application does: FetchContent without the submodules,
# then the client, and the proto for a fake of the daemon to test against. Not part of llamad's
# own build; CI configures, builds and runs it against the commit under test.
cmake_minimum_required(VERSION 3.21)
project(llamad_consumer LANGUAGES CXX)

set(LLAMAD_GIT_REPOSITORY "https://github.com/crybo-rybo/llamad.git" CACHE STRING "Repository to fetch llamad from")
set(LLAMAD_GIT_TAG        "main"                                    CACHE STRING "llamad commit, branch or tag")

include(FetchContent)
FetchContent_Declare(llamad
    GIT_REPOSITORY ${LLAMAD_GIT_REPOSITORY}
    GIT_TAG        ${LLAMAD_GIT_TAG}
    GIT_SUBMODULES "")  # the client needs nothing from llama.cpp
FetchContent_MakeAvailable(llamad)

add_executable(consumer consumer.cpp)
target_link_libraries(consumer PRIVATE llamad::client llamad::proto)

enable_testing()
add_test(NAME consumer COMMAND consumer)
