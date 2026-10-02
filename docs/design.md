# Design notes

- **The proto is the interface.** Applications in all programming languages integrate against
  `proto/llamad/v1/llamad.proto`. The proto file contains the guarantees in this list. There is
  no client library. The application does all the work that must know what a tool does or what a
  reply means. For example, the application runs the tools, does the loop over tool rounds and
  changes a reply into a type.
- **One compiler and no special toolchain.** The daemon is C++23, and the compiler of each
  platform builds it. For this reason, the daemon links against the gRPC from Homebrew on macOS.
  Also, llama.cpp gets its usual Apple settings (Accelerate, native CPU features, Metal).
- **No TCP listener.** In llamad, gRPC is HTTP/2 on a Unix domain socket. There is no port to
  protect with a firewall. The file permissions of the socket (0600) control access.
- **llama.cpp is a pinned submodule that is not modified.** It is not a fork. `src/engine.cpp`
  uses only the public `llama.h` C API. `src/chat_format.cpp` is the only file that uses the
  `common` library of llama.cpp. This library contains the Jinja chat templates, the tool-call
  parsers for each model and the converter from JSON Schema to grammar. The `common` library is
  not stable and it is not a public API. For this reason, a submodule update can break only that
  one file.
- **The daemon is stateless for each request.** It keeps no sessions and no registries. It keeps
  no state between calls that changes the meaning of a request. Applications send the history
  again in each request. Tools are opaque data in each request. The daemon does not run tools and
  does not check them.
- **The KV cache stays after a request, but only as a cache.** After each request, the engine
  records the tokens that the KV cache holds. These tokens are the prompt and each generated
  token that the engine decoded. The next request keeps the longest prefix that its prompt shares
  with these tokens, and decodes only the remaining tokens. Thus, a chat turn or a tool round
  uses time only for its new messages, and not for the full history.

  The engine always decodes at minimum the last token of the prompt, because sampling must have
  its logits. A decode error empties the cache and the record together. Some models cannot
  remove part of a sequence from their memory (recurrent and hybrid models). Some caches do not
  keep the start of the sequence (sliding-window caches). For these models, the engine uses an
  empty cache.

  The cache changes the time that a prompt takes. It never changes which prompt runs. But
  llama.cpp does not guarantee that the logits are identical bit for bit for different batch
  splits. Thus, a greedy reply from a warm cache can be different from the reply of a daemon that
  just started. The `cached_prompt_tokens` field in the stats shows how many prompt tokens came
  from the cache.
- **Version v1 serves one generation at a time.** The engine serializes `generate`. Thus,
  concurrent requests wait in a queue and do not share the context.
- **Stream shape.** A stream has zero or more `text` chunks, and then exactly one `finish` chunk.
  The `finish` chunk contains the finish reason and the stats. `GenerateChunk` is a `oneof` of
  these two types, so the types enforce the shape.

  Tool calls are complete and are in the `finish` chunk. `tool_calls` is not empty if and only if
  the finish reason is `TOOL_CALLS`. Text chunks contain only the content that the user sees.
  They never contain tool-call markup, partial UTF-8 or part of a matched stop string.
- **Chat uses the Jinja template of the model.** This template is in the GGUF file. The daemon
  renders it through the `common` library of llama.cpp, which is the same path that
  `llama-server` uses. Thus, the model sees the scaffolding that it was trained on: tool blocks,
  role markers and its default system prompt. A model can have no template, or a template that
  does not parse. Such a model still serves `Generate`, `Tokenize` and `GetModelInfo`. The daemon
  rejects only `Chat`, with `FAILED_PRECONDITION`.

  The template controls the prompt, and its defaults also. For example, Qwen2.5 adds its own
  default system prompt when the request has no `system` message. A response schema on `Chat`
  uses the same path from JSON Schema to grammar as tool arguments. Thus, the reply is a JSON
  object that agrees with the schema, and it streams as usual text. On a schema turn, the daemon
  tells the template to close its `<think>` block. Thus, a reasoning model replies with only the
  JSON.

  The schema also goes into the prompt, as a system instruction to reply with one JSON object
  that agrees with the schema. This instruction gives the property descriptions to the model. It
  replaces the default system prompt of the template.
- **One model, one type of work.** If the GGUF of a model declares a pooling type, the model is
  an embedding model. With an embedding model, the daemon serves `Embed` and rejects `Generate`
  and `Chat`. With all other models, the daemon does the opposite. The daemon rejects a request
  of the incorrect type with `FAILED_PRECONDITION`.

  Pooling operates on the micro-batch that decodes a sequence. Also, the non-causal attention of
  an encoder cannot split a sequence. For these reasons, the engine decodes each input complete,
  with one `llama_decode` call for each input. On the CPU, this is as fast as one decode that
  packs many inputs as separate sequences. That alternative also must have a context with space
  for many sequences. The vectors are always L2-normalised.
- **Errors have types.** Errors in the request map to `INVALID_ARGUMENT`. A missing capability
  maps to `FAILED_PRECONDITION`. All other errors map to `INTERNAL`.

`AGENTS.md` gives the layer rules that the code must follow.
