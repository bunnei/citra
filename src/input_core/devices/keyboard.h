// Copyright 2016 Citra Emulator Project
// Licensed under GPLv2 or any later version
// Refer to the license.txt file included.

#pragma once

#include <map>
#include <memory>
#include <mutex>

#include "input_core/devices/device.h"

struct KeyboardKey {
    uint32_t key;
    std::string character;

    KeyboardKey() = default;
    KeyboardKey(uint32_t key_, std::string character_)
        : key(key_), character(std::move(character_)) {
    }
    bool operator==(const KeyboardKey& other) const {
        return key == other.key;
    }
    bool operator==(uint32_t other) const {
        return key == other;
    }
    bool operator<(const KeyboardKey& other)  const {
        return key < other.key;
    }
};
class Keyboard : public IDevice {
public:
    Keyboard();
    ~Keyboard();
    bool InitDevice(int number, const std::map<std::string, std::vector<Service::HID::PadState>>& keyMap) override;
    void ProcessInput() override;
    bool CloseDevice() override;
    void KeyPressed(KeyboardKey key);
    void KeyReleased(KeyboardKey key);
    void Clear() override;
    Settings::InputDeviceMapping GetInput() override;
private:
    std::map<KeyboardKey, bool> keys_pressed;
    std::map<int, bool> keys_pressed_last;
    std::mutex m; ///< Keys pressed from frontend is on a separate thread.
    std::map<int, float> circle_pad_directions; ///< Inverts the strength of key press if it is a circle pad direction that needs it.
    KeyboardKey circle_pad_modifier;
};
