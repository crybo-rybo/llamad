# Plain executables, no test framework: a CHECK macro and a non-zero exit code are enough.

add_executable(chat_format_test chat_format_test.cpp)
target_link_libraries(chat_format_test PRIVATE llamad_chat)

# The parser tests need a real Jinja template but no GGUF: ChatFormat takes the template source.
# Qwen3.5 is the thinking template: its generation prompt opens a <think> block, which is what
# makes a response schema and the prompt's assistant opening liable to disagree.
target_compile_definitions(chat_format_test PRIVATE
    QWEN25_TEMPLATE_PATH="${CMAKE_SOURCE_DIR}/third_party/llama.cpp/models/templates/Qwen-Qwen2.5-7B-Instruct.jinja"
    QWEN35_TEMPLATE_PATH="${CMAKE_SOURCE_DIR}/third_party/llama.cpp/models/templates/Qwen3.5-4B.jinja")

add_test(NAME chat_format COMMAND chat_format_test)

# The flag parser is header-only and needs neither gRPC nor a model.
add_executable(flags_test flags_test.cpp)
target_link_libraries(flags_test PRIVATE llamad_flags)

add_test(NAME flags COMMAND flags_test)

# The one place the proto, the engine, the chat layer and the client are visible at once, which
# is what lets it check that every struct mirroring a message still matches it.
if(TARGET llamad_proto)
    add_executable(contract_test contract_test.cpp)
    target_link_libraries(contract_test PRIVATE llamad_proto llamad_engine llamad_chat llamad_client)

    add_test(NAME contract COMMAND contract_test)
endif()

# Reflected JSON conversion and tool dispatch: no daemon or model.
if(TARGET llamad_client)
    add_executable(json_test json_test.cpp)
    target_link_libraries(json_test PRIVATE llamad_client)
    add_test(NAME json COMMAND json_test)

    add_executable(client_tools_test client_tools_test.cpp)
    target_link_libraries(client_tools_test PRIVATE llamad_client)
    add_test(NAME client_tools COMMAND client_tools_test)

    # The tool loop against a scripted fake daemon, which is why this one also links llamad_proto:
    # the fake implements the generated service base class.
    add_executable(client_chat_test client_chat_test.cpp)
    target_link_libraries(client_chat_test PRIVATE llamad_client llamad_proto)
    add_test(NAME client_chat COMMAND client_chat_test)
endif()

# The engine and the chat layer in-process, without the daemon. It needs a model file, so it is
# a CLI to run by hand and not a CTest test; it needs no gRPC, so it builds on every path.
add_executable(engine_smoke engine_smoke.cpp)
target_link_libraries(engine_smoke PRIVATE llamad_engine llamad_chat llamad_flags)
