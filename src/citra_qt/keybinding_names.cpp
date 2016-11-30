// Copyright 2016 Citra Emulator Project
// Licensed under GPLv2 or any later version
// Refer to the license.txt file included.

#include "citra_qt/keybinding_names.h"

const std::array <QString, static_cast<int>(SDLGamepad::GamepadInputs::MAX)> KeyBindingNames::sdl_gamepad_names = {
    {
      tr("Button A"),
      tr("Button B"),
      tr("Button X"),
      tr("Button Y"),
      tr("Left Shoulder"),
      tr("Right Shoulder"),
      tr("Start"),
      tr("Back"),
      tr("Dpad Up"),
      tr("Dpad Down"),
      tr("Dpad Left"),
      tr("Dpad Right"),
      tr("L3"),
      tr("R3"),
      tr("Left Trigger"),
      tr("Right Trigger"),
      tr("Left Y+"),
      tr("Left Y-"),
      tr("Left X+"),
      tr("Left X-"),
      tr("Right Y+"),
      tr("Right Y-"),
      tr("Right X+"),
      tr("Right X-")
    }
};
