# llamad_client: the only thing an application needs to talk to the daemon.
# gRPC and protobuf stay private. The reflected tool adapter includes nlohmann's header: in the
# full build llama.cpp's pinned vendor tree supplies it, without linking any llama.cpp library.
# Client-only has no llama.cpp, so it takes the including project's nlohmann_json target if it
# has one, and an installed package otherwise; one copy of the header either way.
if(LLAMAD_CLIENT_ONLY)
    if(NOT TARGET nlohmann_json::nlohmann_json)
        find_package(nlohmann_json CONFIG REQUIRED)
    endif()
    set(llamad_nlohmann nlohmann_json::nlohmann_json)
else()
    set(llamad_nlohmann vendor::nlohmann)
endif()

add_library(llamad_client STATIC src/client.cpp)
add_library(llamad::client ALIAS llamad_client)
target_include_directories(llamad_client PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_link_libraries(llamad_client PUBLIC ${llamad_nlohmann} PRIVATE llamad_proto)

# client.h reflects over the application's own tool functions, so the language level and the
# reflection flag are part of what including it costs.
target_compile_features(llamad_client PUBLIC cxx_std_26)
target_compile_options(llamad_client PUBLIC $<$<COMPILE_LANGUAGE:CXX>:-freflection>)

if(NOT LLAMAD_CLIENT_ONLY)
    add_executable(llamad-chat examples/chat_cli.cpp)
    target_link_libraries(llamad-chat PRIVATE llamad_client llamad_flags)
endif()
