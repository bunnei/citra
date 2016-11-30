// Copyright 2016 Citra Emulator Project
// Licensed under GPLv2 or any later version
// Refer to the license.txt file included.

#include <algorithm>
#include <iterator>
#include <map>

#include "common/emu_window.h"

#include "input_core/input_core.h"
#include "input_core/key_map.h"

namespace KeyMap {
constexpr int MAX_CIRCLEPAD_POS = 0x9C; /// Max value for a circle pad position
const std::array<Service::HID::PadState, Settings::NativeInput::NUM_INPUTS> mapping_targets = { {
    Service::HID::PAD_A, Service::HID::PAD_B, Service::HID::PAD_X, Service::HID::PAD_Y,
    Service::HID::PAD_L, Service::HID::PAD_R, Service::HID::PAD_ZL, Service::HID::PAD_ZR,
    Service::HID::PAD_START, Service::HID::PAD_SELECT, Service::HID::PAD_TOUCH,
    Service::HID::PAD_UP, Service::HID::PAD_DOWN, Service::HID::PAD_LEFT, Service::HID::PAD_RIGHT,
    Service::HID::PAD_C_UP, Service::HID::PAD_C_DOWN, Service::HID::PAD_C_LEFT, Service::HID::PAD_C_RIGHT,

    Service::HID::PAD_CIRCLE_UP,
    Service::HID::PAD_CIRCLE_DOWN,
    Service::HID::PAD_CIRCLE_LEFT,
    Service::HID::PAD_CIRCLE_RIGHT,
} };
///Array of inputs that are analog only, and require a strength when set
const std::array<Service::HID::PadState, 4> analog_inputs = {
    Service::HID::PAD_CIRCLE_UP,
    Service::HID::PAD_CIRCLE_DOWN,
    Service::HID::PAD_CIRCLE_LEFT,
    Service::HID::PAD_CIRCLE_RIGHT
};

void PressKey(const Service::HID::PadState target, const float strength) {
    auto pad_state = InputCore::GetPadState();
    // If is digital keytarget
    if (std::find(std::begin(analog_inputs), std::end(analog_inputs), target) == std::end(analog_inputs)) {
        pad_state.hex |= target.hex;
        InputCore::SetPadState(pad_state);
    }
    else { // it is analog input
        auto circle_pad = InputCore::GetCirclePad();
        if (target == Service::HID::PAD_CIRCLE_UP || target == Service::HID::PAD_CIRCLE_DOWN) {
            std::get<1>(circle_pad) = MAX_CIRCLEPAD_POS * strength * -1;
        }
        else if (target == Service::HID::PAD_CIRCLE_LEFT || target == Service::HID::PAD_CIRCLE_RIGHT) {
            std::get<0>(circle_pad) = MAX_CIRCLEPAD_POS * strength;
        }
        InputCore::SetCirclePad(circle_pad);
    }
}

void ReleaseKey(const Service::HID::PadState target) {
    auto pad_state = InputCore::GetPadState();
    // If is digital keytarget
    if (std::find(std::begin(analog_inputs), std::end(analog_inputs), target) == std::end(analog_inputs)) {
        pad_state.hex &= ~target.hex;
        InputCore::SetPadState(pad_state);
    }
    else { // it is analog input
        auto circle_pad = InputCore::GetCirclePad();
        if (target == Service::HID::PAD_CIRCLE_UP || target == Service::HID::PAD_CIRCLE_DOWN) {
            std::get<1>(circle_pad) = 0;
        }
        else if (target == Service::HID::PAD_CIRCLE_LEFT || target == Service::HID::PAD_CIRCLE_RIGHT) {
            std::get<0>(circle_pad) = 0;
        }
        InputCore::SetCirclePad(circle_pad);
    }
}
}
