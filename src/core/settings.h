// Copyright 2014 Citra Emulator Project
// Licensed under GPLv2 or any later version
// Refer to the license.txt file included.

#pragma once

#include <array>
#include <memory>
#include <string>
#include <tuple>

#include "common/common_types.h"
#include "common/string_util.h"

namespace Settings {
namespace NativeInput {
enum Values {
    // directly mapped keys
    A, B, X, Y,
    L, R, ZL, ZR,
    START, SELECT, HOME,
    DUP, DDOWN, DLEFT, DRIGHT,
    CUP, CDOWN, CLEFT, CRIGHT,

    // indirectly mapped keys
    CIRCLE_UP, CIRCLE_DOWN, CIRCLE_LEFT, CIRCLE_RIGHT,

    NUM_INPUTS
};
static const std::array<const char*, NUM_INPUTS> Mapping = { {
        // directly mapped keys
        "pad_a", "pad_b", "pad_x", "pad_y",
        "pad_l", "pad_r", "pad_zl", "pad_zr",
        "pad_start", "pad_select", "pad_home",
        "pad_dup", "pad_ddown", "pad_dleft", "pad_dright",
        "pad_cup", "pad_cdown", "pad_cleft", "pad_cright",

        // indirectly mapped keys
        "pad_circle_up", "pad_circle_down", "pad_circle_left", "pad_circle_right"
} };
static const std::array<Values, NUM_INPUTS> All = { {
    A, B, X, Y,
    L, R, ZL, ZR,
    START, SELECT, HOME,
    DUP, DDOWN, DLEFT, DRIGHT,
    CUP, CDOWN, CLEFT, CRIGHT,
    CIRCLE_UP, CIRCLE_DOWN, CIRCLE_LEFT, CIRCLE_RIGHT
} };
}

enum class DeviceFramework {
    Qt, SDL
};
enum class Device {
    Keyboard, Gamepad
};
struct InputDeviceMapping {
    DeviceFramework framework = DeviceFramework::Qt;
    int number = 0;
    Device device = Device::Keyboard;
    std::string key;

    InputDeviceMapping() = default;
    explicit InputDeviceMapping(int keyboardKey) {
        key = std::to_string(keyboardKey);
    }
    explicit InputDeviceMapping(const std::string& input) {
        std::vector<std::string> parts;
        Common::SplitString(input, '/', parts);
        if (parts.size() != 4)
            return;

        if (parts[0] == "Qt")
            framework = DeviceFramework::Qt;
        else if (parts[0] == "SDL")
            framework = DeviceFramework::SDL;

        number = std::stoi(parts[1]);

        if (parts[2] == "Keyboard")
            device = Device::Keyboard;
        else if (parts[2] == "Gamepad")
            device = Device::Gamepad;
        key = parts[3];
    }

    bool operator==(const InputDeviceMapping& rhs) const {
        return std::tie(device, framework, number) == std::tie(rhs.device, rhs.framework, rhs.number);
    }
    bool operator==(const std::string& rhs) const {
        return ToString() == rhs;
    }
    std::string ToString() const {
        std::string result;
        if (framework == DeviceFramework::Qt)
            result = "Qt";
        else if (framework == DeviceFramework::SDL)
            result = "SDL";

        result += "/";
        result += std::to_string(this->number);
        result += "/";

        if (device == Device::Keyboard)
            result += "Keyboard";
        else if (device == Device::Gamepad)
            result += "Gamepad";

        result += "/";
        result += key;
        return result;
    }
};

struct Values {
    // CheckNew3DS
    bool is_new_3ds;

    // Controls
    std::array<InputDeviceMapping, NativeInput::NUM_INPUTS> input_mappings;
    InputDeviceMapping pad_circle_modifier;
    float pad_circle_modifier_scale;

    // Core
    bool use_cpu_jit;
    int frame_skip;

    // Data Storage
    bool use_virtual_sd;

    // System Region
    int region_value;

    // Renderer
    bool use_hw_renderer;
    bool use_shader_jit;
    bool use_scaled_resolution;
    bool use_vsync;

    float bg_red;
    float bg_green;
    float bg_blue;

    std::string log_filter;

    // Audio
    std::string sink_id;
    bool enable_audio_stretching;

    // Debugging
    bool use_gdbstub;
    u16 gdbstub_port;
};
extern Values values;

void Apply();
}
