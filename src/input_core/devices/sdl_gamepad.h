// Copyright 2016 Citra Emulator Project
// Licensed under GPLv2 or any later version
// Refer to the license.txt file included.

#pragma once

#include "input_core/devices/device.h"

struct _SDL_GameController;
class SDLGamepad : public IDevice {
public:
    SDLGamepad();
    SDLGamepad(int number_, _SDL_GameController* gamepad_);
    ~SDLGamepad();

    bool InitDevice(int number, const std::map<std::string, std::vector<Service::HID::PadState>>& keyMap) override;
    void ProcessInput() override;
    bool CloseDevice() override;
    Settings::InputDeviceMapping GetInput() override;
    void Clear() override;

    ///Returns vector of all gamepads connected to computer. Used for keybinding setup
    static std::vector<std::shared_ptr<IDevice>> GetAllDevices();
    enum class GamepadInputs {
        ButtonA,ButtonB,ButtonX,ButtonY,LeftShoulder,RightShoulder,
        Start,Back,DPadUp,DpadDown,DpadLeft,DpadRight,L3,R3,LeftTrigger,RightTrigger,
        LeftYPlus,LeftYMinus,LeftXPlus,LeftXMinus,RightYPlus,RightYMinus,
        RightXPlus,RightXMinus, MAX
    };
private:
    /// Maps the friendly name shown on GUI with the string name for getting the SDL button instance.
    std::map<GamepadInputs, std::string> gamepadinput_to_sdlname_mapping = {
        { GamepadInputs::ButtonA, "a" },
        { GamepadInputs::ButtonB, "b" },
        { GamepadInputs::ButtonX, "x" },
        { GamepadInputs::ButtonY, "y" },
        { GamepadInputs::LeftShoulder, "leftshoulder" },
        { GamepadInputs::RightShoulder, "rightshoulder" },
        { GamepadInputs::Start, "start" },
        { GamepadInputs::Back, "back" },
        { GamepadInputs::DPadUp, "dpup" },
        { GamepadInputs::DpadDown, "dpdown" },
        { GamepadInputs::DpadLeft, "dpleft" },
        { GamepadInputs::DpadRight, "dpright" },
        { GamepadInputs::L3, "leftstick" },
        { GamepadInputs::R3, "rightstick" },
        { GamepadInputs::LeftTrigger, "lefttrigger" },
        { GamepadInputs::RightTrigger, "righttrigger" },
        { GamepadInputs::LeftYPlus, "lefty" },
        { GamepadInputs::LeftYMinus, "lefty" },
        { GamepadInputs::LeftXPlus, "leftx" },
        { GamepadInputs::LeftXMinus, "leftx" },
        { GamepadInputs::RightYPlus, "righty" },
        { GamepadInputs::RightYMinus, "righty" },
        { GamepadInputs::RightXPlus, "rightx" },
        { GamepadInputs::RightXMinus, "rightx" },
    };
    static bool SDLInitialized;
    std::map<std::string, bool> keys_pressed; ///< Map of keys that were pressed on previous iteration
    _SDL_GameController* gamepad = nullptr;
    int number; ///< Index of gamepad connection

    static void LoadGameControllerDB();
};
