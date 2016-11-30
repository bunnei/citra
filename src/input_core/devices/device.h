// Copyright 2016 Citra Emulator Project
// Licensed under GPLv2 or any later version
// Refer to the license.txt file included.

#pragma once

#include <map>

#include "core/hle/service/hid/hid.h"
#include "core/settings.h"
#include "input_core/key_map.h"

class IDevice {
public:
    virtual ~IDevice();

    /**
     * Initialize IDevice object with device's index and the map of keys that it will listen to.
     * @param number: device number as ordered connected to computer.
     * @param keymap: vector of PadStates for device to listen for
     * @return true if successful
     */
    virtual bool InitDevice(int number, const std::map<std::string, std::vector<Service::HID::PadState>>& keyMap) = 0;

    /**
     * Process inputs that were pressed since last frame
     */
    virtual void ProcessInput() = 0;

    /**
     * Close connection to device
     * @return true if successful
     */
    virtual bool CloseDevice() = 0;

    /**
     * Gets Input for a single frame. Used only in Qt gui for key binding.
     * @return Settings::InputDeviceMapping instance that captures what button was pressed
     */
    virtual Settings::InputDeviceMapping GetInput() = 0;

    /**
     * Clears info from last frame.
     */
    virtual void Clear() = 0;
protected:
    std::map<std::string, std::vector<Service::HID::PadState>> key_mapping; ///< Maps the string in the settings file to the HID Padstate object
};
