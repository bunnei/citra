// Copyright 2016 Citra Emulator Project
// Licensed under GPLv2 or any later version
// Refer to the license.txt file included.

#pragma once

#include <functional>
#include <memory>
#include <tuple>

#include "core/hle/service/hid/hid.h"
#include "core/settings.h"
#include "input_core/devices/device.h"

class Keyboard;

namespace InputCore {

void Init();
void Shutdown();

/**
 * Threadsafe getter to the current PadState
 * @return Service::HID::PadState instance
 */
Service::HID::PadState GetPadState();

/**
 * Threadsafe setter for the current PadState
 * @param state New PadState to overwrite current PadState.
 */
void SetPadState(const Service::HID::PadState& state);

/**
 * Getter for current CirclePad
 * @return std::tuple<s16, s16> CirclePad state
 */
std::tuple<s16, s16> GetCirclePad();

/**
 * Setter for current CirclePad
 * @param circle New CirclePad state
 */
void SetCirclePad(std::tuple<s16, s16> circle);

/**
 * Getter for Citra's main keyboard input handler
 * @return std::shared_ptr<Keyboard> Device Keyboard instance
 */
std::shared_ptr<Keyboard> GetKeyboard();

/**
 * Gets the current touch screen state (touch X/Y coordinates and whether or not it is pressed). Threadsafe.
 * @note This should be called by the core emu thread to get a state set by the window thread.
 * @return std::tuple of (x, y, pressed) where `x` and `y` are the touch coordinates and
 *         `pressed` is true if the touch screen is currently being pressed
 */
std::tuple<u16, u16, bool> GetTouchState();

/**
 * Threadsafe setter for the current touch screen state.
 * @param value New Touch State
 */
void SetTouchState(std::tuple<u16, u16, bool> value);

/**
 * Reload input key mapping settings during game-play
 */
void ReloadSettings();

/**
 * Loops through all devices and detects the first device that produces an input
 * @param max_time: maximum amount of time to wait until input detected, in milliseconds.
 * @param update_GUI: function to run in while loop to process any gui events.
 * @return Settings::InputDeviceMapping of input device
 */
Settings::InputDeviceMapping DetectInput(int max_time, std::function<void(void)> update_GUI);
}
