# Plain executables, no test framework: a CHECK macro and a non-zero exit code are enough.

add_executable(chat_format_test chat_format_test.cpp)
target_link_libraries(chat_format_test PRIVATE llamad_chat llamad_options)

# The parser tests need a real Jinja template but no GGUF: ChatFormat takes the template source.
# Qwen3.5 is the thinking template: its generation prompt opens a <think> block, which is what
# makes a response schema and the prompt's assistant opening liable to disagree.
target_compile_definitions(chat_format_test PRIVATE
    QWEN25_TEMPLATE_PATH="${PROJECT_SOURCE_DIR}/third_party/llama.cpp/models/templates/Qwen-Qwen2.5-7B-Instruct.jinja"
    QWEN35_TEMPLATE_PATH="${PROJECT_SOURCE_DIR}/third_party/llama.cpp/models/templates/Qwen3.5-4B.jinja")

add_test(NAME chat_format COMMAND chat_format_test)

# The engine's prompt-cache bookkeeping, reached without loading a model.
add_executable(engine_test engine_test.cpp)
target_link_libraries(engine_test PRIVATE llamad_engine llamad_options)
add_test(NAME engine COMMAND engine_test)

# The flag reader and the shared engine flags; no model.
add_executable(flags_test flags_test.cpp)
target_link_libraries(flags_test PRIVATE llamad_engine llamad_options)
add_test(NAME flags COMMAND flags_test)

# Conversion between wire messages and engine types, checked against the messages' descriptors.
add_executable(wire_test wire_test.cpp)
target_link_libraries(wire_test PRIVATE llamad_service llamad_options)
add_test(NAME wire COMMAND wire_test)

# The engine and the chat layer in-process, without the daemon. It needs a model file, so it is
# a CLI to run by hand and not a CTest test.
add_executable(engine_smoke engine_smoke.cpp)
target_link_libraries(engine_smoke PRIVATE llamad_chat llamad_options)

# Prefill and decode throughput, for scripts/bench.sh. Like engine_smoke it needs a model file.
add_executable(engine_bench engine_bench.cpp)
target_link_libraries(engine_bench PRIVATE llamad_chat llamad_options)
