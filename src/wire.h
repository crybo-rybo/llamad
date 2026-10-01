/** @file
 * @brief Conversion between llamad.proto's messages and the engine and chat layer's plain types.
 *
 * Written out field by field. tests/wire_test.cpp checks every message these functions handle
 * against the message's descriptor, so a field added to llamad.proto and not carried here fails
 * a test rather than being dropped silently.
 */

#pragma once

#include "chat_format.h"
#include "engine.h"
#include "llamad/v1/llamad.pb.h"

#include <vector>

namespace llamad::wire {

/// Engine sampling parameters, with the engine's defaults wherever the request left a field unset.
SamplingParams from_proto(const v1::SamplingParams & params);

/// One history turn.
ChatMessage from_proto(const v1::ChatMessage & message);

/// One tool offered for a request.
Tool from_proto(const v1::Tool & tool);

/// One tool call replayed from history.
ToolCall from_proto(const v1::ToolCall & call);

/// Model metadata.
void to_proto(const ModelInfo & info, v1::ModelInfo * out);

/// The vectors of one Embed request.
void to_proto(const EmbedResult & result, v1::EmbedResponse * out);

/// The last chunk of a Generate or Chat stream. `tool_calls` are what the chat parser found; they
/// are kept, and the reason becomes TOOL_CALLS, only when generation ran to completion (EOG or a
/// stop string). A reply cut short by the token limit or a cancel stopped mid-thought, so whatever
/// the parser salvaged from it is not a call anyone should execute.
v1::GenerateChunk finish_chunk(FinishReason reason, const GenerateStats & stats, std::vector<ToolCall> tool_calls);

}  // namespace llamad::wire
