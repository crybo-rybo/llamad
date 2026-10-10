#!/usr/bin/env python3
# /// script
# requires-python = ">=3.9"
# dependencies = ["grpcio>=1.84", "grpcio-tools>=1.84"]
# ///
"""A terminal chat client for llamad. It shares only llamad.proto with the daemon.

Run it from a llamad checkout, with pip:

    pip install -r examples/python/requirements.txt
    python examples/python/chat.py

or with uv, which reads the dependencies above:

    uv run examples/python/chat.py
"""

import argparse
import os
import sys
from datetime import datetime
from pathlib import Path

import grpc

# Compile llamad.proto at start-up, so there are no generated stubs to keep in step with it.
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "proto"))
try:
    pb, rpc = grpc.protos_and_services("llamad/v1/llamad.proto")
except (ImportError, NotImplementedError) as error:
    sys.exit(f"error: cannot load llamad.proto ({error}). Install the requirements, "
             "and run chat.py from a llamad checkout.")

try:
    import readline  # line editing and input history for input()
except ImportError:
    readline = None

SYSTEM = pb.ChatMessage(role="system", content="You are a concise, friendly assistant.")
SAMPLING = pb.SamplingParams(temperature=0, max_tokens=512)
CHAT_TIMEOUT_S = 120
# The client runs the tools itself. The daemon only offers them to the model.
TOOLS = [pb.Tool(name="get_current_time",
                 description="Get the current local date and time on this computer.")]
MAX_TOOL_ROUNDS = 4  # replies per turn, so a model that keeps calling tools cannot loop forever
HELP = """
Type a message to chat with the model. Ask the time to see a tool call.
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
    request = pb.ChatRequest(messages=messages, sampling=SAMPLING, tools=TOOLS)
    stream = stub.Chat(request, timeout=CHAT_TIMEOUT_S)
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
        # A reply with only tool calls streams no text, so its tool lines can share this line.
        if text or finish is None:
            print()
    if finish is None:
        raise RuntimeError("the stream ended without a finish chunk")
    return "".join(text), finish


def run_tool(call):
    if call.name == "get_current_time":
        # Models often cannot work out the weekday from a date, so the result gives it.
        now = datetime.now().astimezone()
        return f"{now:%A} {now.isoformat(timespec='seconds')}"
    return f"error: unknown tool {call.name}"


def chat_turn(stub, history, user):
    """Answer one user message, running tool calls between replies.

    Return the messages of the turn, or None if the model still calls tools after MAX_TOOL_ROUNDS.
    """
    turn = [user]
    for _ in range(MAX_TOOL_ROUNDS):
        print(paint("assistant> ", "1;34"), end="", flush=True)
        text, finish = stream_reply(stub, history + turn)
        turn.append(pb.ChatMessage(role="assistant", content=text, tool_calls=finish.tool_calls))
        for call in finish.tool_calls:
            result = run_tool(call)
            print(paint(f"[tool {call.name}] {result}", "2"))
            turn.append(pb.ChatMessage(role="tool", tool_call_id=call.id, content=result))
        print(paint(stats_line(finish), "2"))
        if finish.reason != pb.FINISH_REASON_TOOL_CALLS:
            return turn
    return None


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
        try:
            turn = chat_turn(stub, history, user)
        except KeyboardInterrupt:
            print(paint("[reply cancelled]", "2"))
            continue
        except (grpc.RpcError, RuntimeError) as error:
            print(paint("error> ", "1;31") + describe(error))
            print("Use /clear if the conversation exceeds the model's context.")
            failed = True
            continue
        if turn is None:
            # Drop the turn, so the history never ends on tool results without a reply.
            print(paint(f"[still calling tools after {MAX_TOOL_ROUNDS} replies, turn dropped]", "2"))
            continue
        history += turn  # only a complete turn, so Ctrl+C or an error leaves the history clean
    print()
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
