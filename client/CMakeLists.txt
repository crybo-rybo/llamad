# llamad_client: the only thing an application needs to talk to the daemon.
# gRPC and protobuf stay private. The reflected tool adapter includes nlohmann's header,
# supplied by llama.cpp's pinned vendor tree without linking any llama.cpp library.
add_library(llamad_client STATIC src/client.cpp)
target_include_directories(llamad_client PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_link_libraries(llamad_client PUBLIC vendor::nlohmann PRIVATE llamad_proto)

# client.h reflects over the application's own tool functions, so the language level and the
# reflection flag are part of what including it costs.
target_compile_features(llamad_client PUBLIC cxx_std_26)
target_compile_options(llamad_client PUBLIC $<$<COMPILE_LANGUAGE:CXX>:-freflection>)

add_executable(llamad-chat examples/chat_cli.cpp)
target_link_libraries(llamad-chat PRIVATE llamad_client llamad_flags)
