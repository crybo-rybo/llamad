# Design notes

- **No TCP listener.** gRPC here is HTTP/2 over a Unix domain socket. There is
  no port to firewall; access control is the socket's file permissions (0600).
- **llama.cpp is an unmodified pinned submodule**, not a fork. `src/engine.cpp`
  uses only the public `llama.h` C API. `src/chat_format.cpp` is the one file that
  touches llama.cpp's `common` library, its Jinja chat templates, its per-model
  tool-call parsers and its JSON-Schema-to-grammar converter, which is unstable
  and not a public API, so a submodule bump can break at most that single file.
- **The daemon is stateless per request.** No sessions, no registries, no state carried
  between calls that a request's meaning depends on. Clients resend history; tools travel
  with each request as opaque data and are never executed or validated by the daemon.
- **The KV cache outlives a request, as a cache.** The engine records the tokens the KV
  cache holds after each request: the prompt and every generated token it decoded. The next
  request keeps the longest prefix its prompt shares with them and decodes only the rest, so
  a chat turn or a tool round pays for its new messages rather than the whole history; at
  least the prompt's last token is always decoded, because sampling needs its logits. Any
  failed decode empties the cache and the record together, and a model whose memory cannot
  drop part of a sequence (recurrent and hybrid models) or no longer holds its start (a
  sliding-window cache) falls back to an empty cache. The cache changes how long a prompt
  takes, never which prompt runs; but llama.cpp does not promise bit-for-bit identical logits
  for different batch splits, so a greedy reply from a warm cache can differ from the one a
  freshly started daemon gives. `cached_prompt_tokens` in the stats says how much was reused.
- **v1 serves one generation at a time.** The engine serializes `generate`, so
  concurrent clients queue rather than sharing the context.
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
- **One model, one kind of work.** A model whose GGUF declares a pooling type is an embedding
  model: the daemon serves `Embed` and refuses `Generate` and `Chat`; any other model is the
  reverse, and the refusal is `FAILED_PRECONDITION`. Pooling runs over the micro-batch a
  sequence is decoded in, and an encoder's non-causal attention cannot split a sequence at all,
  so each input is decoded whole, one per `llama_decode`. On the CPU that is as fast as packing
  several inputs into one decode as separate sequences, which would also mean a context sized
  for many sequences. Vectors are always L2-normalised.
- **Errors are typed.** Caller mistakes map to `INVALID_ARGUMENT`, a missing capability to
  `FAILED_PRECONDITION`, everything else to `INTERNAL`.

The wire contract itself is `proto/llamad/v1/llamad.proto`, and `AGENTS.md` lists the
layering rules the code is held to.
