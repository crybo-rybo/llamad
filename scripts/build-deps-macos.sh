#!/usr/bin/env bash
set -euo pipefail

grpc_ref=v1.84.0

if [[ ${1:-} == --help ]]; then
    echo "Usage: $0"
    echo "Builds gRPC $grpc_ref, Protobuf and Abseil into build-deps/prefix, where build.sh finds them."
    echo "Homebrew's packages are built against libc++, which GCC's libstdc++ cannot link against."
    echo "Takes about fifteen minutes and a gigabyte; re-running reuses the checkout."
    echo "CMAKE_BUILD_PARALLEL_LEVEL defaults to 4."
    exit 0
fi

if (( $# )); then
    echo "Expected no arguments; see $0 --help" >&2
    exit 2
fi

if [[ $(uname -s) != Darwin ]]; then
    echo "This script is macOS-only; elsewhere install gRPC and Protobuf from your package manager." >&2
    exit 2
fi

if ! command -v g++-16 >/dev/null; then
    echo "g++-16 not found; install it with: brew install gcc" >&2
    exit 1
fi

# --installed: a bare `brew --prefix` prints a path and succeeds for a formula that is not there.
if ! openssl_root=$(brew --prefix --installed openssl@3 2>/dev/null); then
    echo "openssl@3 not found; install it with: brew install openssl@3" >&2
    exit 1
fi

root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
deps="$root/build-deps"
src="$deps/grpc-src"
build="$deps/grpc-build"
prefix="$deps/prefix"

if [[ ! -d "$src" ]]; then
    git clone --depth 1 --branch "$grpc_ref" --recurse-submodules --shallow-submodules \
        https://github.com/grpc/grpc "$src"
fi

# The same hybrid toolchain llamad itself uses: Apple clang for C, g++-16 for C++, so the
# std::string crossing gRPC's API is libstdc++ on both sides.
# -D_DARWIN_C_SOURCE: abseil's cctz defines _XOPEN_SOURCE 500, under which the macOS SDK hides
# quick_exit and at_quick_exit, and GCC's <cstdlib> then fails because it expects them.
# gRPC_SSL_PROVIDER=package: the bundled BoringSSL derives warning flags from the C compiler and
# hands clang-only ones to g++ with -Werror. OpenSSL is a C ABI, so Homebrew's build fits here.
cmake -S "$src" -B "$build" -G Ninja -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_C_COMPILER=/usr/bin/clang -DCMAKE_CXX_COMPILER=g++-16 -DCMAKE_CXX_STANDARD=17 \
    -DCMAKE_CXX_FLAGS=-D_DARWIN_C_SOURCE -DCMAKE_INSTALL_PREFIX="$prefix" \
    -DgRPC_INSTALL=ON -DgRPC_BUILD_TESTS=OFF -DABSL_PROPAGATE_CXX_STD=ON \
    -DgRPC_SSL_PROVIDER=package -DOPENSSL_ROOT_DIR="$openssl_root" \
    -DgRPC_BUILD_CSHARP_EXT=OFF -DgRPC_BUILD_GRPC_CSHARP_PLUGIN=OFF \
    -DgRPC_BUILD_GRPC_NODE_PLUGIN=OFF -DgRPC_BUILD_GRPC_OBJECTIVE_C_PLUGIN=OFF \
    -DgRPC_BUILD_GRPC_PHP_PLUGIN=OFF -DgRPC_BUILD_GRPC_PYTHON_PLUGIN=OFF \
    -DgRPC_BUILD_GRPC_RUBY_PLUGIN=OFF
cmake --build "$build" --parallel "${CMAKE_BUILD_PARALLEL_LEVEL:-4}" --target install

echo "Installed gRPC $grpc_ref into $prefix"
