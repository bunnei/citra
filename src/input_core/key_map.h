// Copyright 2016 Citra Emulator Project
// Licensed under GPLv2 or any later version
// Refer to the license.txt file included.

#pragma once

#include <array>
#include <tuple>

#include "core/hle/service/hid/hid.h"

class EmuWindow;

namespace KeyMap {
extern const std::array<Service::HID::PadState, Settings::NativeInput::NUM_INPUTS> mapping_targets;
extern const std::array<Service::HID::PadState, 4> analog_inputs;

///Handles the pressing of a key and modifies InputCore state
void PressKey(Service::HID::PadState target, float strength);

///Handles the releasing of a key and modifies InputCore state
void ReleaseKey(Service::HID::PadState target);
}
