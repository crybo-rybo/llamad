#pragma once

#include <iosfwd>

#include "llamad/v1/llamad.grpc.pb.h"

// The example owns its history and tools. The daemon sees only ordinary wire requests.
int run_playground(llamad::v1::Llama::StubInterface & stub, std::istream & input, std::ostream & output);
