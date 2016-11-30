// Copyright 2016 Citra Emulator Project
// Licensed under GPLv2 or any later version
// Refer to the license.txt file included.

#include <cmath>
#include <memory>
#include <SDL.h>
#include <SDL_gamecontroller.h>

#include "common/assert.h"
#include "common/logging/log.h"
#include "common/string_util.h"

#include "input_core/devices/sdl_gamepad.h"
#include "input_core/devices/gamecontrollerdb.h"

bool SDLGamepad::SDLInitialized = false;

SDLGamepad::SDLGamepad() {
}
SDLGamepad::SDLGamepad(int number_, _SDL_GameController* gamepad_)
    : number(number_), gamepad(gamepad_) {

}
SDLGamepad::~SDLGamepad() {
    CloseDevice();
}

bool SDLGamepad::InitDevice(int number, const std::map<std::string, std::vector<Service::HID::PadState>>& keyMap) {
    if (!SDLGamepad::SDLInitialized && SDL_Init(SDL_INIT_GAMECONTROLLER) < 0) {
        LOG_CRITICAL(Input, "SDL_Init(SDL_INIT_GAMECONTROLLER) failed");
        return false;
    }
    SDL_GameControllerEventState(SDL_IGNORE);
    SDLGamepad::SDLInitialized = true;
    LoadGameControllerDB();

    if (SDL_IsGameController(number)) {
        gamepad = SDL_GameControllerOpen(number);
        if (gamepad == nullptr) {
            LOG_INFO(Input, "Controller found but unable to open connection.");
            return false;
        }
    }
    key_mapping = keyMap;
    for (const auto& entry : key_mapping) {
        keys_pressed[entry.first] = false;
    }

    return true;
}

void SDLGamepad::ProcessInput() {
    if (gamepad == nullptr)
        return;
    SDL_GameControllerUpdate();
    for (const auto& entry : key_mapping) {
        SDL_GameControllerButton button = SDL_GameControllerGetButtonFromString(gamepadinput_to_sdlname_mapping[static_cast<GamepadInputs>(stoi(entry.first))].c_str());
        if (button != SDL_GameControllerButton::SDL_CONTROLLER_BUTTON_INVALID) {
            Uint8 pressed = SDL_GameControllerGetButton(gamepad, button);
            if (pressed == 1 && keys_pressed[entry.first] == false) {
                for (const auto& padstate : entry.second) {
                    KeyMap::PressKey(padstate, 1.0);
                    keys_pressed[entry.first] = true;
                }
            }
            else if (pressed == 0 && keys_pressed[entry.first] == true) {
                for (const auto& padstate : entry.second) {
                    KeyMap::ReleaseKey(padstate);
                    keys_pressed[entry.first] = false;
                }
            }
        }
        else {
            // Try axis if button isn't valid
            SDL_GameControllerAxis axis = SDL_GameControllerGetAxisFromString(gamepadinput_to_sdlname_mapping[static_cast<GamepadInputs>(stoi(entry.first))].c_str());
            if (axis != SDL_GameControllerAxis::SDL_CONTROLLER_AXIS_INVALID) {
                Sint16 value = SDL_GameControllerGetAxis(gamepad, axis);
                for (const auto& padstate : entry.second) {
                    // TODO: calculate deadzone by radial field rather than axial field. (sqrt(x^2 + y^2) > deadzone)
                    // dont process if in deadzone. Replace later with settings for deadzone.
                    if (abs(value) < 0.2 * 32767.0)
                        KeyMap::ReleaseKey(padstate);
                    else
                        KeyMap::PressKey(padstate, (float)value / 32767.0);
                }
            }
        }
    }
}

bool SDLGamepad::CloseDevice() {
    if (gamepad != nullptr) {
        SDL_GameControllerClose(gamepad);
    }
    return true;
}

std::vector<std::shared_ptr<IDevice>> SDLGamepad::GetAllDevices() {
    std::vector<std::shared_ptr<IDevice>> devices;
    if (!SDLGamepad::SDLInitialized && SDL_Init(SDL_INIT_GAMECONTROLLER) < 0) {
        LOG_CRITICAL(Input, "SDL_Init(SDL_INIT_GAMECONTROLLER) failed");
        return devices;
    }
    LoadGameControllerDB();
    SDL_GameControllerEventState(SDL_IGNORE);
    for (int i = 0; i < 8; i++) {
        SDL_GameController* gamecontroller;
        if (SDL_IsGameController(i)) {
            gamecontroller = SDL_GameControllerOpen(i);
            if (gamecontroller != nullptr) {
                devices.push_back(std::make_shared<SDLGamepad>(i, gamecontroller));
            }
        }
    }
    return devices;
}

void SDLGamepad::LoadGameControllerDB() {
    std::vector<std::string> lines1,lines2,lines3,lines4;
    Common::SplitString(SDLGameControllerDB::db_file1, '\n', lines1);
    Common::SplitString(SDLGameControllerDB::db_file2, '\n', lines2);
    Common::SplitString(SDLGameControllerDB::db_file3, '\n', lines3);
    Common::SplitString(SDLGameControllerDB::db_file4, '\n', lines4);
    lines1.insert(lines1.end(), lines2.begin(), lines2.end());
    lines1.insert(lines1.end(), lines3.begin(), lines3.end());
    lines1.insert(lines1.end(), lines4.begin(), lines4.end());
    for (std::string s : lines1) {
        SDL_GameControllerAddMapping(s.c_str());
    }
}

Settings::InputDeviceMapping SDLGamepad::GetInput() {
    if (gamepad == nullptr)
        return Settings::InputDeviceMapping("");
    SDL_GameControllerUpdate();
    for (int i = 0; i < SDL_GameControllerButton::SDL_CONTROLLER_BUTTON_MAX; i++)
    {
        Uint8 pressed = SDL_GameControllerGetButton(gamepad, SDL_GameControllerButton(i));
        if (pressed == 0)
            continue;

        auto buttonName = SDL_GameControllerGetStringForButton(SDL_GameControllerButton(i));
        for (const auto& mapping : gamepadinput_to_sdlname_mapping) {
            if (mapping.second == buttonName) {
                return Settings::InputDeviceMapping("SDL/" + std::to_string(number) + "/" + "Gamepad/" + std::to_string(static_cast<int>(mapping.first)));
            }
        }
    }
    for (int i = 0; i < SDL_GameControllerAxis::SDL_CONTROLLER_AXIS_MAX; i++) {
        Sint16 value = SDL_GameControllerGetAxis(gamepad, SDL_GameControllerAxis(i));
        // TODO: calculate deadzone by radial field rather than axial field. (sqrt(x^2 + y^2) > deadzone)
        // dont process if in deadzone. Replace later with settings for deadzone.
        if (abs(value) < 0.2 * 32767.0)
            continue;
        std::string modifier;
        if (value > 0)
            modifier = "+";
        else
            modifier = "-";
        std::string axisName = SDL_GameControllerGetStringForAxis(SDL_GameControllerAxis(i));
        for (const auto& mapping : gamepadinput_to_sdlname_mapping) {
            if (mapping.second == axisName) {
                if ((mapping.first == GamepadInputs::LeftXMinus ||
                    mapping.first == GamepadInputs::LeftYMinus ||
                    mapping.first == GamepadInputs::RightXMinus ||
                    mapping.first == GamepadInputs::RightYMinus) && modifier == "+") {
                    continue;
                }
                else if ((mapping.first == GamepadInputs::LeftXPlus ||
                    mapping.first == GamepadInputs::LeftYPlus ||
                    mapping.first == GamepadInputs::RightXPlus ||
                    mapping.first == GamepadInputs::RightYPlus) && modifier == "-") {
                    continue;
                }
                return Settings::InputDeviceMapping("SDL/" + std::to_string(this->number) + "/" + "Gamepad/" + std::to_string(static_cast<int>(mapping.first)));
            }
        }
    }
    return Settings::InputDeviceMapping("");
}

void SDLGamepad::Clear() {
}
