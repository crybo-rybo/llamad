#!/usr/bin/env python3
"""Writes the site's wire protocol page from llamad.proto.

Usage: proto_reference.py PROTO OUTPUT

Reads the subset of proto3 that llamad.proto uses: a header comment, one service, messages with
plain, optional, repeated and oneof fields, and enums, each with leading and trailing comments.
Any other line stops the build with its line number, so a new construct in the proto fails the
documentation build rather than disappearing from the page.
"""

import re
import sys
from dataclasses import dataclass, field

SCALARS = {
    "double", "float", "int32", "int64", "uint32", "uint64", "sint32", "sint64",
    "fixed32", "fixed64", "sfixed32", "sfixed64", "bool", "string", "bytes",
}

NAME = r"[A-Za-z_][A-Za-z0-9_]*"
TRAILING = r"\s*(?://\s?(?P<trailing>.*))?$"

SYNTAX_RE  = re.compile(r'syntax\s*=\s*"proto3"\s*;$')
PACKAGE_RE = re.compile(rf"package\s+(?P<name>{NAME}(?:\.{NAME})*)\s*;$")
SERVICE_RE = re.compile(rf"service\s+(?P<name>{NAME})\s*\{{$")
RPC_RE     = re.compile(rf"rpc\s+(?P<name>{NAME})\s*\(\s*(?P<request>{NAME})\s*\)\s*returns\s*"
                        rf"\(\s*(?P<stream>stream\s+)?(?P<response>{NAME})\s*\)\s*;{TRAILING}")
MESSAGE_RE = re.compile(rf"message\s+(?P<name>{NAME})\s*\{{(?P<empty>\s*\}})?$")
ENUM_RE    = re.compile(rf"enum\s+(?P<name>{NAME})\s*\{{$")
ONEOF_RE   = re.compile(rf"oneof\s+(?P<name>{NAME})\s*\{{$")
FIELD_RE   = re.compile(rf"(?:(?P<label>optional|repeated)\s+)?(?P<type>{NAME})\s+(?P<name>{NAME})"
                        rf"\s*=\s*(?P<number>\d+)\s*;{TRAILING}")
VALUE_RE   = re.compile(rf"(?P<name>{NAME})\s*=\s*(?P<number>\d+)\s*;{TRAILING}")


class ProtoError(Exception):
    pass


@dataclass
class Rpc:
    name: str
    request: str
    response: str
    streaming: bool
    comment: str


@dataclass
class Field:
    number: int
    name: str
    type: str
    label: str  # "", "optional", "repeated" or "oneof <name>"
    comment: str


@dataclass
class EnumValue:
    number: int
    name: str
    comment: str


@dataclass
class Type:
    kind: str  # "message" or "enum"
    name: str
    comment: str
    fields: list = field(default_factory=list)  # Field for a message, EnumValue for an enum


@dataclass
class Proto:
    header: str = ""
    package: str = ""
    service: str = ""
    service_comment: str = ""
    rpcs: list = field(default_factory=list)
    types: list = field(default_factory=list)


# Parsing

def parse(text):
    """Parse the proto into a Proto, raising ProtoError at the first line it does not understand."""
    proto = Proto()
    comment = []      # leading comment lines waiting for the declaration they precede
    scope = []        # open blocks, innermost last: ("service"|"message"|"enum"|"oneof", name)
    header_open = True

    for number, raw in enumerate(text.splitlines(), start=1):
        line = raw.strip()
        try:
            if line.startswith("//"):
                comment.append(line[2:].removeprefix(" "))
                continue
            if not line:
                if header_open and comment:
                    proto.header = "\n".join(comment)
                    header_open = False
                comment = []
                continue
            header_open = False
            parse_line(proto, scope, line, " ".join(c.strip() for c in comment))
            comment = []
        except ProtoError as error:
            raise ProtoError(f"line {number}: {error}: {raw.strip()}") from None

    if scope:
        raise ProtoError(f"end of file inside {scope[-1][0]} {scope[-1][1]}")
    check_types(proto)
    return proto


def parse_line(proto, scope, line, leading):
    inside = scope[-1][0] if scope else None

    if line == "}" and scope:
        scope.pop()
    elif inside is None and SYNTAX_RE.match(line):
        pass
    elif inside is None and (m := PACKAGE_RE.match(line)):
        proto.package = m["name"]
    elif inside is None and (m := SERVICE_RE.match(line)):
        if proto.service:
            raise ProtoError("a second service")
        proto.service, proto.service_comment = m["name"], leading
        scope.append(("service", m["name"]))
    elif inside == "service" and (m := RPC_RE.match(line)):
        proto.rpcs.append(Rpc(m["name"], m["request"], m["response"], bool(m["stream"]),
                              join(leading, m["trailing"])))
    elif inside is None and (m := MESSAGE_RE.match(line)):
        proto.types.append(Type("message", m["name"], leading))
        if not m["empty"]:
            scope.append(("message", m["name"]))
    elif inside is None and (m := ENUM_RE.match(line)):
        proto.types.append(Type("enum", m["name"], leading))
        scope.append(("enum", m["name"]))
    elif inside == "message" and (m := ONEOF_RE.match(line)):
        scope.append(("oneof", m["name"]))
    elif inside in ("message", "oneof") and (m := FIELD_RE.match(line)):
        label = f"oneof {scope[-1][1]}" if inside == "oneof" else (m["label"] or "")
        if inside == "oneof" and m["label"]:
            raise ProtoError("a labelled field inside a oneof")
        proto.types[-1].fields.append(Field(int(m["number"]), m["name"], m["type"], label,
                                            join(leading, m["trailing"])))
    elif inside == "enum" and (m := VALUE_RE.match(line)):
        proto.types[-1].fields.append(EnumValue(int(m["number"]), m["name"],
                                                join(leading, m["trailing"])))
    else:
        raise ProtoError(f"cannot document this line inside {inside or 'the file'}")


def join(*parts):
    return " ".join(part.strip() for part in parts if part and part.strip())


def check_types(proto):
    """Every type a field or RPC names must be a scalar or defined in this file, so every link resolves."""
    defined = {t.name for t in proto.types}
    used = [(r.name, r.request) for r in proto.rpcs] + [(r.name, r.response) for r in proto.rpcs]
    used += [(f"{t.name}.{f.name}", f.type) for t in proto.types if t.kind == "message" for f in t.fields]
    for where, type_name in used:
        if type_name not in SCALARS and type_name not in defined:
            raise ProtoError(f"{where} uses {type_name}, which this file does not define")
    if not proto.service:
        raise ProtoError("no service")


# Rendering

def escape(text):
    """Doxygen-escape prose. Doxygen takes `code spans` and "quoted text" literally, so those stay
    as they are."""
    pieces = re.split(r'(`[^`]*`|"[^"\n]*")', text)
    return "".join(piece if piece[:1] in ('`', '"') else re.sub(r"([\\@&$#<>%|])", r"\\\1", piece)
                   for piece in pieces)


def type_ref(type_name):
    if type_name in SCALARS:
        return f"<code>{type_name}</code>"
    return f'@ref wire_{type_name} "{type_name}"'


def label_badge(label):
    if not label:
        return ""
    kind = label.split()[0]
    return f'<span class="badge badge-{kind}">{label}</span>'


def users_of(proto, type_name):
    """Where a type appears: as an RPC's request or response, or as a message's field."""
    uses = []
    for rpc in proto.rpcs:
        if type_name in (rpc.request, rpc.response):
            uses.append(f"@ref wire_rpc_{rpc.name} \"{rpc.name}\"")
    for t in proto.types:
        if t.kind == "message":
            uses += [f'@ref wire_{t.name} "{t.name}.{f.name}"' for f in t.fields if f.type == type_name]
    return uses


def render_service(proto):
    lines = [f"@section wire_service The {proto.service} service", "",
             f"`{proto.package}.{proto.service}`. {escape(proto.service_comment)}", "",
             '<table class="api-table rpc-table">',
             "<tr><th>RPC</th><th>Request</th><th>Response</th><th>Description</th></tr>"]
    for rpc in proto.rpcs:
        stream = ' <span class="badge badge-stream">stream</span>' if rpc.streaming else ""
        lines.append(f"<tr><td>@anchor wire_rpc_{rpc.name}<code>{rpc.name}</code></td>"
                     f"<td>{type_ref(rpc.request)}</td>"
                     f"<td>{type_ref(rpc.response)}{stream}</td>"
                     f"<td>{escape(rpc.comment)}</td></tr>")
    lines += ["</table>", ""]
    return lines


def render_message(proto, message):
    lines = [f"@subsection wire_{message.name} {message.name}", "", meta(proto, message), ""]
    if message.comment:
        lines += [escape(message.comment), ""]
    if not message.fields:
        lines += ["No fields.", ""]
    else:
        lines += ['<table class="api-table field-table">',
                  "<tr><th>#</th><th>Field</th><th>Type</th><th>Description</th></tr>"]
        for f in message.fields:
            lines.append(f'<tr><td class="num">{f.number}</td><td><code>{f.name}</code></td>'
                         f"<td>{type_ref(f.type)} {label_badge(f.label)}</td>"
                         f"<td>{escape(f.comment)}</td></tr>")
        lines += ["</table>", ""]
    return lines


def render_enum(proto, enum):
    lines = [f"@subsection wire_{enum.name} {enum.name}", "", meta(proto, enum), ""]
    if enum.comment:
        lines += [escape(enum.comment), ""]
    lines += ['<table class="api-table enum-table">',
              "<tr><th>#</th><th>Value</th><th>Description</th></tr>"]
    for value in enum.fields:
        lines.append(f'<tr><td class="num">{value.number}</td><td><code>{value.name}</code></td>'
                     f"<td>{escape(value.comment)}</td></tr>")
    lines += ["</table>", ""]
    return lines


def meta(proto, t):
    """The line under a type's heading: its kind, and where it is used."""
    uses = users_of(proto, t.name)
    used_by = f' <span class="used-by">used by {", ".join(uses)}</span>' if uses else ""
    return f'<span class="meta"><span class="kind">{t.kind}</span>{used_by}</span>'


def render(proto, source_url):
    lines = ["/**", "@page wire_protocol Wire protocol", "",
             "@tableofcontents", "",
             # "\\." keeps JAVADOC_AUTOBRIEF from ending a brief, and the page, inside the span.
             '<span class="lead">The `llamad.v1` contract, generated from '
             f"[proto/llamad/v1/llamad.proto]({source_url})\\. Generate stubs from that file "
             'with `protoc` in any language; the @ref md_docs_2protocol "integration guide" '
             "walks through using them\\.</span>", "",
             "@section wire_guarantees Guarantees", ""]
    lines += [escape(proto.header), ""]
    lines += render_service(proto)
    lines += ["@section wire_flow Streams and tool calls", "",
              "@htmlinclude[block] docs/site/stream-frame.html", "",
              "@htmlinclude[block] docs/site/tool-round.html", "",
              "@section wire_types Messages and enums", ""]
    for t in proto.types:
        lines += render_message(proto, t) if t.kind == "message" else render_enum(proto, t)
    lines.append("*/")
    return "\n".join(lines) + "\n"


def main(argv):
    if len(argv) != 3:
        print(__doc__.strip(), file=sys.stderr)
        return 2
    proto_path, output_path = argv[1], argv[2]
    try:
        with open(proto_path, encoding="utf-8") as f:
            proto = parse(f.read())
    except ProtoError as error:
        print(f"{proto_path}: {error}", file=sys.stderr)
        print("proto_reference.py does not know how to document this; extend it.", file=sys.stderr)
        return 1
    source_url = "https://github.com/crybo-rybo/llamad/blob/main/proto/llamad/v1/llamad.proto"
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(render(proto, source_url))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
