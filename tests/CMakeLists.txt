# Plain executables, no test framework: a CHECK macro and a non-zero exit code are enough.

add_executable(chat_format_test chat_format_test.cpp)
target_link_libraries(chat_format_test PRIVATE llamad_chat)

# The parser tests need a real Jinja template but no GGUF: ChatFormat takes the template source.
target_compile_definitions(chat_format_test PRIVATE
    QWEN25_TEMPLATE_PATH="${CMAKE_SOURCE_DIR}/third_party/llama.cpp/models/templates/Qwen-Qwen2.5-7B-Instruct.jinja")

add_test(NAME chat_format COMMAND chat_format_test)

# The one place the proto, the engine, the chat layer and the client are visible at once, which
# is what lets it check that every struct mirroring a message still matches it.
if(TARGET llamad_proto)
    add_executable(contract_test contract_test.cpp)
    target_link_libraries(contract_test PRIVATE llamad_proto llamad_engine llamad_chat)
    target_include_directories(contract_test PRIVATE ${CMAKE_SOURCE_DIR}/client/include)

    add_test(NAME contract COMMAND contract_test)
endif()
