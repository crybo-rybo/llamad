#!/usr/bin/env python3
"""A terminal chat client for llamad. It shares only llamad.proto with the daemon.

Set it up from the repository root:

    pip install -r examples/python/requirements.txt
    python -m grpc_tools.protoc -I proto --python_out=examples/python \\
        --grpc_python_out=examples/python llamad/v1/llamad.proto
"""

import argparse
import os
import sys

import grpc

try:
    from llamad.v1 import llamad_pb2 as pb, llamad_pb2_grpc as rpc
except ModuleNotFoundError:
    sys.exit("error: generate the llamad.v1 stubs first, as the top of chat.py shows.")

try:
    import readline  # line editing and input history for input()
except ImportError:
    readline = None

SYSTEM = pb.ChatMessage(role="system", content="You are a concise, friendly assistant.")
SAMPLING = pb.SamplingParams(temperature=0, max_tokens=512)
CHAT_TIMEOUT_S = 120
HELP = """
Type a message to chat with the model.
/clear  Reset chat history.
/help   Show this help.
/quit   Exit (or press Ctrl+D).
"""

# Escape codes only for a person at a terminal, so piped output stays plain text.
COLOR = sys.stdin.isatty() and sys.stdout.isatty()


def paint(text, code, prompt=False):
    if not COLOR:
        return text
    start, end = f"\033[{code}m", "\033[0m"
    if prompt and readline:
        # readline must not count the escape codes in the width of the prompt.
        start, end = f"\001{start}\002", f"\001{end}\002"
    return start + text + end


def default_socket():
    runtime_dir = os.environ.get("XDG_RUNTIME_DIR")
    if runtime_dir:
        return os.path.join(runtime_dir, "llamad.sock")
    return f"/tmp/llamad-{os.getuid()}.sock"


def describe(error):
    if isinstance(error, grpc.RpcError):
        return f"{error.code().name}: {error.details()}"
    return str(error)


def check_model(stub):
    info = stub.GetModelInfo(pb.GetModelInfoRequest(), timeout=5)
    if info.serves_embeddings:
        raise RuntimeError("the daemon serves an embedding model, not a chat model")
    if not info.has_chat_template:
        raise RuntimeError("the model has no chat template")
    return info


def stream_reply(stub, messages):
    """Print the reply as it arrives. Return its text and the Finish chunk."""
    stream = stub.Chat(pb.ChatRequest(messages=messages, sampling=SAMPLING), timeout=CHAT_TIMEOUT_S)
    text, finish = [], None
    try:
        for chunk in stream:
            if chunk.WhichOneof("chunk") == "text":
                text.append(chunk.text)
                print(chunk.text, end="", flush=True)
            else:
                finish = chunk.finish
    except KeyboardInterrupt:
        stream.cancel()
        raise
    finally:
        print()
    if finish is None:
        raise RuntimeError("the stream ended without a finish chunk")
    return "".join(text), finish


def stats_line(finish):
    stats = finish.stats
    reason = pb.FinishReason.Name(finish.reason).removeprefix("FINISH_REASON_")
    parts = [f"finish {reason}",
             f"prompt {stats.prompt_tokens} (cached {stats.cached_prompt_tokens})",
             f"output {stats.completion_tokens}",
             f"prefill {stats.prompt_ms:.1f} ms"]
    if stats.completion_ms > 0:
        parts.append(f"decode {1000 * stats.completion_tokens / stats.completion_ms:.1f} tok/s")
    return "[" + " | ".join(parts) + "]"


def main():
    parser = argparse.ArgumentParser(
        description="Chat with a running llamad daemon. This client loads no model.",
        epilog="Type /help for commands. Ctrl+D or /quit exits.")
    parser.add_argument("--socket", metavar="PATH", default=default_socket(),
                        help="the daemon's socket. The default matches llamad: %(default)s")
    args = parser.parse_args()

    stub = rpc.LlamaStub(grpc.insecure_channel("unix:" + args.socket))
    try:
        info = check_model(stub)
    except (grpc.RpcError, RuntimeError) as error:
        print(paint("error> ", "1;31") + describe(error))
        print("Start llamad with a chat model and check --socket.")
        return 1
    print(paint(f"llamad | {info.description} | context {info.n_ctx} tokens", "1"))
    print(HELP)

    history = [SYSTEM]  # The daemon keeps no conversation, so the client sends all of it each turn.
    failed = False
    while True:
        try:
            line = input(paint("you> ", "1;32", prompt=True)).strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not line:
            continue
        if line == "/quit":
            break
        if line == "/help":
            print(HELP)
            continue
        if line == "/clear":
            history = [SYSTEM]
            print("Chat history cleared.")
            continue
        if line.startswith("/"):
            print("Use /help, /clear or /quit.")
            continue

        user = pb.ChatMessage(role="user", content=line)
        print(paint("assistant> ", "1;34"), end="", flush=True)
        try:
            text, finish = stream_reply(stub, history + [user])
        except KeyboardInterrupt:
            print(paint("[reply cancelled]", "2"))
            continue
        except (grpc.RpcError, RuntimeError) as error:
            print(paint("error> ", "1;31") + describe(error))
            print("Use /clear if the conversation exceeds the model's context.")
            failed = True
            continue
        print(paint(stats_line(finish), "2"))
        history += [user, pb.ChatMessage(role="assistant", content=text)]
    print()
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
