# Design notes

- **No TCP listener.** gRPC here is HTTP/2 over a Unix domain socket. There is
  no port to firewall; access control is the socket's file permissions (0600).
- **llama.cpp is an unmodified pinned submodule**, not a fork. `src/engine.cpp`
  uses only the public `llama.h` C API. `src/chat_format.cpp` is the one file that
  touches llama.cpp's `common` library, its Jinja chat templates, its per-model
  tool-call parsers and its JSON-Schema-to-grammar converter, which is unstable
  and not a public API, so a submodule bump can break at most that single file.
- **The daemon is stateless per request.** No sessions, no registries, no state carried
  between calls. Clients resend history; tools travel with each request as opaque data and
  are never executed or validated by the daemon.
- **v1 serves one generation at a time.** The engine serializes `generate`, so
  concurrent clients queue rather than sharing the context. Each request starts
  from an empty KV cache.
- **Stream shape.** Zero or more text chunks, then exactly one final chunk carrying
  `finish_reason` and `stats`. Tool calls arrive whole on that final chunk, and `tool_calls`
  is non-empty if and only if the finish reason is `TOOL_CALLS`. Text chunks carry
  user-visible content only: never tool-call markup, never partial UTF-8, never part of a
  matched stop string.
- **Chat renders the model's own Jinja template**, the one stored in the GGUF,
  through llama.cpp's `common` library, the same path `llama-server` takes. So
  whatever scaffolding the model was trained on (tool blocks, role markers, its
  default system prompt) is what it actually sees. A model with no template, or
  with one that will not parse, still serves `Generate`, `Tokenize` and
  `GetModelInfo`; only `Chat` is refused, with `FAILED_PRECONDITION`.

  The template is in charge of the prompt, defaults included: Qwen2.5, for
  instance, injects its own default system prompt when the client sends no
  `system` message. A response schema on `Chat` takes the same
  JSON-Schema-to-grammar path as tool arguments, so the reply is a JSON object
  fitting the schema and streams as ordinary text. A schema turn asks the template
  to close its `<think>` block, so a thinking model answers with the JSON and
  nothing else. The schema goes into the prompt too, as a system instruction to
  reply with one matching JSON object; that is what carries its property
  descriptions to the model, and it takes the place of a template's own default
  system prompt.
- **Errors are typed.** Caller mistakes map to `INVALID_ARGUMENT`, a missing capability to
  `FAILED_PRECONDITION`, everything else to `INTERNAL`.

The wire contract itself is `proto/llamad/v1/llamad.proto`, and `AGENTS.md` lists the
layering rules the code is held to.
