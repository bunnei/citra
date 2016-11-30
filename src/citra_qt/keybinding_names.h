// Copyright 2016 Citra Emulator Project
// Licensed under GPLv2 or any later version
// Refer to the license.txt file included.

#pragma once

#include <array>
#include <QCoreApplication>

#include "input_core/devices/sdl_gamepad.h"

///Map translatable, user-friendly names to input device buttons for use in the Qt input configuration GUI.
class KeyBindingNames {
    Q_DECLARE_TR_FUNCTIONS(KeyBindingNames)
public:
    static const std::array<QString, static_cast<int>(SDLGamepad::GamepadInputs::MAX)> sdl_gamepad_names;
};
