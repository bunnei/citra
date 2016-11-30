// Copyright 2016 Citra Emulator Project
// Licensed under GPLv2 or any later version
// Refer to the license.txt file included.

#include <algorithm>
#include <SDL_keyboard.h>

#include "input_core/devices/keyboard.h"

Keyboard::Keyboard() {
}

Keyboard::~Keyboard() {
}

bool Keyboard::InitDevice(int number, const std::map<std::string, std::vector<Service::HID::PadState>>& keyMap) {
    key_mapping = keyMap;

    //Check if keyboard is mapped for circle up or left. if so, set modifier to -1
    for (const auto& entry : key_mapping) {
        if (entry.first == "")
            continue;
        for (const auto& padstate : entry.second) {
            if (padstate == Service::HID::PAD_CIRCLE_UP || padstate == Service::HID::PAD_CIRCLE_LEFT) {
                circle_pad_directions[stoi(entry.first)] = -1.0;
            }
            else if (padstate == Service::HID::PAD_CIRCLE_DOWN || padstate == Service::HID::PAD_CIRCLE_RIGHT) {
                circle_pad_directions[stoi(entry.first)] = 1.0;
            }
        }
    }
    //Check if responsible for circle pad modifier
    auto mapping = Settings::values.pad_circle_modifier;
    if (mapping.device == Settings::Device::Keyboard && mapping.key != "")
        circle_pad_modifier = KeyboardKey(stoi(mapping.key),"");
    return true;
}

void Keyboard::ProcessInput() {
    std::map<KeyboardKey, bool> keysPressedCopy;
    {
        std::lock_guard<std::mutex> lock(m);
        keysPressedCopy = keys_pressed;
    }
    bool circlePadModPressed = keysPressedCopy[circle_pad_modifier];
    for (const auto& entry : key_mapping) {
        if (entry.first == "")
            continue;
        int keycode = std::stoi(entry.first);
        KeyboardKey proxy = KeyboardKey(keycode, "");

        //if key is pressed when prev state is unpressed, or if key pressed and is a circle pad direction
        if ((keysPressedCopy[proxy] == true && keys_pressed_last[keycode] == false) || (keysPressedCopy[proxy] == true && circle_pad_directions.count(keycode))) {
            for (const auto& key : entry.second) {
                if (circle_pad_directions.count(keycode)) { //If is analog key press
                    float modifier = (circlePadModPressed) ? Settings::values.pad_circle_modifier_scale : 1;
                    KeyMap::PressKey(key, circle_pad_directions[keycode] * modifier);
                }
                else // Is digital key press
                    KeyMap::PressKey(key, 1.0);
            }
            keys_pressed_last[keycode] = true;
        }
        else if (keysPressedCopy[proxy] == false && keys_pressed_last[keycode] == true) {
            for (const auto& key : entry.second) {
                KeyMap::ReleaseKey(key);
            }
            keys_pressed_last[keycode] = false;
        }
    }
}

bool Keyboard::CloseDevice() {
    return true;
}

void Keyboard::KeyPressed(KeyboardKey key) {
    std::lock_guard<std::mutex> lock(m);
    keys_pressed[key] = true;
}

void Keyboard::KeyReleased(KeyboardKey key) {
    std::lock_guard<std::mutex> lock(m);
    keys_pressed[key] = false;
}

void Keyboard::Clear() {
    std::lock_guard<std::mutex> lock(m);
    keys_pressed.clear();
}

Settings::InputDeviceMapping Keyboard::GetInput() {
    std::map<KeyboardKey, bool> keysPressedCopy;
    {
        std::lock_guard<std::mutex> lock(m);
        keysPressedCopy = keys_pressed;
    }
    for (const auto& entry : keysPressedCopy) {
        int keycode = entry.first.key;
        if (keysPressedCopy[entry.first] == true && keys_pressed_last[keycode] == false) {
            return Settings::InputDeviceMapping("QT/0/Keyboard/" + std::to_string(keycode));
        }
    }
    return Settings::InputDeviceMapping("");
}
