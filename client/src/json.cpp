#include "llamad/json.h"

#include <cstdio>

namespace llamad {
namespace client {
namespace json {
namespace detail {
namespace {

// Nesting that skip_value() will walk through before calling the input malformed. The daemon's
// output is grammar-constrained, so this only ever fires on something hand-written; it is here
// because the alternative is a stack overflow.
constexpr std::size_t kMaxDepth = 64;

bool is_whitespace(char c) {
    return c == ' ' || c == '\t' || c == '\n' || c == '\r';
}

void append_utf8(std::string & out, unsigned code) {
    if (code < 0x80) {
        out += static_cast<char>(code);
    } else if (code < 0x800) {
        out += static_cast<char>(0xC0 | (code >> 6));
        out += static_cast<char>(0x80 | (code & 0x3F));
    } else if (code < 0x10000) {
        out += static_cast<char>(0xE0 | (code >> 12));
        out += static_cast<char>(0x80 | ((code >> 6) & 0x3F));
        out += static_cast<char>(0x80 | (code & 0x3F));
    } else {
        out += static_cast<char>(0xF0 | (code >> 18));
        out += static_cast<char>(0x80 | ((code >> 12) & 0x3F));
        out += static_cast<char>(0x80 | ((code >> 6) & 0x3F));
        out += static_cast<char>(0x80 | (code & 0x3F));
    }
}

}  // namespace

void Reader::fail(const std::string & message) const {
    throw Error(at_, "json: " + message + " at offset " + std::to_string(at_));
}

void Reader::skip_whitespace() {
    while (at_ < text_.size() && is_whitespace(text_[at_])) {
        ++at_;
    }
}

bool Reader::consume(char c) {
    skip_whitespace();
    if (at_ >= text_.size() || text_[at_] != c) {
        return false;
    }
    ++at_;
    return true;
}

void Reader::expect(char c) {
    if (!consume(c)) {
        fail(std::string("expected '") + c + "'");
    }
}

void Reader::expect_end() {
    skip_whitespace();
    if (at_ != text_.size()) {
        fail("trailing text");
    }
}

bool Reader::match(std::string_view word) {
    skip_whitespace();
    if (text_.compare(at_, word.size(), word) != 0) {
        return false;
    }
    at_ += word.size();
    return true;
}

unsigned Reader::read_hex4() {
    if (at_ + 4 > text_.size()) {
        fail("truncated \\u escape");
    }
    unsigned code = 0;
    for (int i = 0; i < 4; ++i) {
        const char c     = text_[at_++];
        const int  digit = c >= '0' && c <= '9'   ? c - '0'
                           : c >= 'a' && c <= 'f' ? c - 'a' + 10
                           : c >= 'A' && c <= 'F' ? c - 'A' + 10
                                                  : -1;
        if (digit < 0) {
            fail("bad \\u escape");
        }
        code = code * 16 + static_cast<unsigned>(digit);
    }
    return code;
}

std::string Reader::read_string() {
    expect('"');
    std::string out;
    while (true) {
        if (at_ >= text_.size()) {
            fail("unterminated string");
        }
        const char c = text_[at_++];
        if (c == '"') {
            return out;
        }
        if (c != '\\') {
            if (static_cast<unsigned char>(c) < 0x20) {
                fail("control character in string");
            }
            out += c;
            continue;
        }
        if (at_ >= text_.size()) {
            fail("unterminated escape");
        }
        const char escape = text_[at_++];
        switch (escape) {
            case '"':
            case '\\':
            case '/': out += escape; break;
            case 'b': out += '\b';   break;
            case 'f': out += '\f';   break;
            case 'n': out += '\n';   break;
            case 'r': out += '\r';   break;
            case 't': out += '\t';   break;
            case 'u': {
                unsigned code = read_hex4();
                // A code point outside the BMP arrives as a high surrogate followed by a low one.
                // Either half alone has no UTF-8 encoding, so it is an error rather than output.
                if (code >= 0xD800 && code <= 0xDBFF) {
                    if (at_ + 1 >= text_.size() || text_[at_] != '\\' || text_[at_ + 1] != 'u') {
                        fail("lone surrogate");
                    }
                    at_ += 2;
                    const unsigned low = read_hex4();
                    if (low < 0xDC00 || low > 0xDFFF) {
                        fail("lone surrogate");
                    }
                    code = 0x10000 + ((code - 0xD800) << 10) + (low - 0xDC00);
                } else if (code >= 0xDC00 && code <= 0xDFFF) {
                    fail("lone surrogate");
                }
                append_utf8(out, code);
                break;
            }
            default: fail(std::string("unknown escape '\\") + escape + "'");
        }
    }
}

std::string_view Reader::scan_number() {
    skip_whitespace();
    const std::size_t start = at_;
    if (at_ < text_.size() && text_[at_] == '-') {
        ++at_;
    }
    while (at_ < text_.size()) {
        const char c = text_[at_];
        if ((c < '0' || c > '9') && c != '.' && c != 'e' && c != 'E' && c != '+' && c != '-') {
            break;
        }
        ++at_;
    }
    if (at_ == start) {
        fail("expected a number");
    }
    return text_.substr(start, at_ - start);
}

double Reader::read_number() {
    const std::string_view token = scan_number();
    double                 value = 0;
    const auto             end   = std::from_chars(token.data(), token.data() + token.size(), value);
    if (end.ec != std::errc{} || end.ptr != token.data() + token.size()) {
        fail("expected a number");
    }
    return value;
}

long long Reader::read_integer() {
    const std::string_view token = scan_number();
    long long              value = 0;
    const auto             end   = std::from_chars(token.data(), token.data() + token.size(), value);
    if (end.ec != std::errc{} || end.ptr != token.data() + token.size()) {
        fail("expected an integer");
    }
    return value;
}

bool Reader::read_bool() {
    if (match("true")) {
        return true;
    }
    if (match("false")) {
        return false;
    }
    fail("expected true or false");
}

bool Reader::read_null() {
    return match("null");
}

void Reader::skip_value() {
    skip_value(0);
}

void Reader::skip_value(std::size_t depth) {
    if (depth > kMaxDepth) {
        fail("nested too deeply");
    }
    skip_whitespace();
    if (at_ >= text_.size()) {
        fail("expected a value");
    }

    const char c = text_[at_];
    if (c == '"') {
        read_string();
    } else if (c == '{' || c == '[') {
        const bool  object = c == '{';
        const char  close  = object ? '}' : ']';
        ++at_;
        if (!consume(close)) {
            do {
                if (object) {
                    read_string();
                    expect(':');
                }
                skip_value(depth + 1);
            } while (consume(','));
            expect(close);
        }
    } else if (c == 't' || c == 'f') {
        read_bool();
    } else if (c == 'n') {
        if (!read_null()) {
            fail("expected a value");
        }
    } else {
        read_number();
    }
}

void write_string(std::string & out, std::string_view text) {
    out += '"';
    for (const char c : text) {
        switch (c) {
            case '"':  out += "\\\""; break;
            case '\\': out += "\\\\"; break;
            case '\b': out += "\\b";  break;
            case '\f': out += "\\f";  break;
            case '\n': out += "\\n";  break;
            case '\r': out += "\\r";  break;
            case '\t': out += "\\t";  break;
            default:
                if (static_cast<unsigned char>(c) < 0x20) {
                    char escape[7];
                    std::snprintf(escape, sizeof(escape), "\\u%04x", static_cast<unsigned char>(c));
                    out += escape;
                } else {
                    out += c;
                }
        }
    }
    out += '"';
}

void write_key(std::string & out, bool & first, std::string_view name) {
    if (!first) {
        out += ',';
    }
    first = false;
    write_string(out, name);
    out += ':';
}

}  // namespace detail
}  // namespace json
}  // namespace client
}  // namespace llamad
