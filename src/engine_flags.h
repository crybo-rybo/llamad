/** @file
 * @brief Shared context and offload options for the daemon and engine smoke CLI.
 *
 * The context and offload flags the daemon and engine_smoke both take, the EngineConfig they
 * describe, and the device table --list-devices prints. The comma-separated lists arrive as text
 * and are split here, so both binaries accept the same spelling and report the same errors.
 */

#pragma once

#include "engine.h"
#include "flags.h"

#include <algorithm>
#include <charconv>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <optional>
#include <string>
#include <system_error>
#include <vector>

namespace llamad {

/// Shared CLI spelling of model context and offload configuration.
struct EngineFlags {
    [[=cli::help{"N", "context size (default 4096)"}]]
    int32_t ctx = 4096;  ///< Positive context length in tokens.

    [[=cli::help{"N", "layers to offload to the GPU (default 99)"}]]
    int32_t ngl = 99;  ///< Number of model layers to offload; zero disables offload.

    [[=cli::help{"N", "threads for inference, 0 = auto, half the hardware threads (default 0)"}]]
    int32_t threads = 0;  ///< Non-negative thread count; zero selects the engine default.

    [[=cli::help{"NAMES", "comma-separated devices to offload to (default: every discrete GPU)"}]]
    std::optional<std::string> devices;  ///< Comma-separated backend device names, unset for automatic selection.

    [[=cli::help{"S", "comma-separated share per device, e.g. 3,1 (default: by free memory)"}]]
    std::optional<std::string> tensor_split;  ///< Comma-separated non-negative device shares, unset for free-memory weighting.

    [[=cli::help{"list the devices this build can offload to, and exit"}]]
    bool list_devices = false;  ///< Print available backend devices and exit without loading a model.
};

namespace detail {

/// Splits "a,b,c" on commas, or nothing at all when an element is empty, so "a,,b" and "" are
/// rejected rather than turned into a list with a hole in it.
inline std::optional<std::vector<std::string>> split_list(const std::string & text) {
    std::vector<std::string> parts;
    for (size_t start = 0;;) {
        const size_t comma = text.find(',', start);
        const size_t len   = comma == std::string::npos ? std::string::npos : comma - start;
        std::string  part  = text.substr(start, len);
        if (part.empty()) {
            return std::nullopt;
        }
        parts.push_back(std::move(part));
        if (comma == std::string::npos) {
            return parts;
        }
        start = comma + 1;
    }
}

/// The same message whichever part of the value is wrong: the shape of the whole is what the
/// reader needs to see.
constexpr const char * tensor_split_error =
    "--tensor-split needs a comma-separated list of non-negative numbers, e.g. 3,1";

}  // namespace detail

/// The EngineConfig these flags describe, model path aside. Throws cli::FlagError on a value the
/// engine cannot take; the engine owns the remaining tensor-split checks (all-zero, more entries
/// than devices), which need the device list.
inline EngineConfig to_config(const EngineFlags & flags) {
    EngineConfig config;

    if (flags.ctx <= 0) {
        throw cli::FlagError("--ctx needs a positive integer");
    }
    if (flags.threads < 0) {
        throw cli::FlagError("--threads needs a non-negative integer");
    }
    config.n_ctx        = static_cast<uint32_t>(flags.ctx);
    config.n_gpu_layers = flags.ngl;
    config.n_threads    = flags.threads;

    if (flags.devices) {
        const std::optional<std::vector<std::string>> names = detail::split_list(*flags.devices);
        if (!names) {
            throw cli::FlagError("--devices needs a comma-separated list of device names"
                                 " (see --list-devices)");
        }
        config.devices = *names;
    }

    if (flags.tensor_split) {
        const std::optional<std::vector<std::string>> parts = detail::split_list(*flags.tensor_split);
        if (!parts) {
            throw cli::FlagError(detail::tensor_split_error);
        }
        for (const std::string & part : *parts) {
            const char * end   = part.data() + part.size();
            float        value = 0;
            const std::from_chars_result parsed = std::from_chars(part.data(), end, value);
            if (parsed.ec != std::errc{} || parsed.ptr != end || !(value >= 0.0f)) {
                throw cli::FlagError(detail::tensor_split_error);
            }
            config.tensor_split.push_back(value);
        }
    }

    return config;
}

/// Prints the --list-devices table to `out`: one row per backend device with its free and total
/// memory, found without loading a model.
inline void print_device_table(std::FILE * out) {
    const std::vector<DeviceInfo> devices = Engine::list_devices();
    if (devices.empty()) {
        std::fprintf(out, "no devices (this build has no ggml backend registered)\n");
        return;
    }

    const auto format_gib = [](uint64_t bytes) {
        char buf[32];
        std::snprintf(buf, sizeof(buf), "%.1f GiB", static_cast<double>(bytes) / (1024.0 * 1024.0 * 1024.0));
        return std::string(buf);
    };

    size_t w_name = std::strlen("NAME");
    size_t w_type = std::strlen("TYPE");
    for (const DeviceInfo & device : devices) {
        w_name = std::max(w_name, device.name.size());
        w_type = std::max(w_type, device.type.size());
    }

    std::fprintf(out, "%-*s  %-*s  %9s  %9s  %s\n", static_cast<int>(w_name), "NAME", static_cast<int>(w_type),
                 "TYPE", "FREE", "TOTAL", "DESCRIPTION");
    for (const DeviceInfo & device : devices) {
        std::fprintf(out, "%-*s  %-*s  %9s  %9s  %s\n", static_cast<int>(w_name), device.name.c_str(),
                     static_cast<int>(w_type), device.type.c_str(), format_gib(device.memory_free).c_str(),
                     format_gib(device.memory_total).c_str(), device.description.c_str());
    }
}

}  // namespace llamad
