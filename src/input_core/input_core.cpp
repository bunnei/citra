// Copyright 2016 Citra Emulator Project
// Licensed under GPLv2 or any later version
// Refer to the license.txt file included.

#include <algorithm>
#include <chrono>
#include <map>
#include <mutex>

#include "core/core_timing.h"

#include "input_core/input_core.h"
#include "input_core/devices/keyboard.h"
#include "input_core/devices/sdl_gamepad.h"

namespace InputCore {
constexpr u64 frame_ticks = 268123480ull / 60;
static int tick_event;
static Service::HID::PadState pad_state;
static std::tuple<s16, s16> circle_pad = { 0,0 };
static std::shared_ptr<Keyboard> main_keyboard; ///< Keyboard is always active for Citra
static std::vector<std::shared_ptr<IDevice>> devices; ///< Devices that are handling input for the game
static std::mutex pad_state_mutex;
static std::mutex touch_mutex;
static u16 touch_x;        ///< Touchpad X-position in native 3DS pixel coordinates (0-320)
static u16 touch_y;        ///< Touchpad Y-position in native 3DS pixel coordinates (0-240)
static bool touch_pressed; ///< True if touchpad area is currently pressed, otherwise false

static void InputTickCallback(u64, int cycles_late) {
    for (auto& device : devices)
        device->ProcessInput();

    Service::HID::Update();

    // Reschedule recurrent event
    CoreTiming::ScheduleEvent(frame_ticks - cycles_late, tick_event);
}

Service::HID::PadState GetPadState() {
    std::lock_guard<std::mutex> lock(pad_state_mutex);
    return pad_state;
}

void SetPadState(const Service::HID::PadState& state) {
    std::lock_guard<std::mutex> lock(pad_state_mutex);
    pad_state.hex = state.hex;
}

std::tuple<s16, s16> GetCirclePad() {
    return circle_pad;
}

void SetCirclePad(std::tuple<s16, s16> pad) {
    circle_pad = pad;
}

std::shared_ptr<Keyboard> GetKeyboard() {
    if (main_keyboard == nullptr)
        main_keyboard = std::make_shared<Keyboard>();
    return main_keyboard;
}

std::tuple<u16, u16, bool> GetTouchState() {
    std::lock_guard<std::mutex> lock(touch_mutex);
    return std::make_tuple(touch_x, touch_y, touch_pressed);
}

void SetTouchState(std::tuple<u16, u16, bool> value) {
    std::lock_guard<std::mutex> lock(touch_mutex);
    std::tie(touch_x, touch_y, touch_pressed) = value;
}

/// Helper method to check if device was already initialized
bool CheckIfMappingExists(const std::vector<Settings::InputDeviceMapping>& uniqueMapping, Settings::InputDeviceMapping mappingToCheck) {
    return std::any_of(uniqueMapping.begin(), uniqueMapping.end(), [mappingToCheck](const auto& mapping) {
        return mapping == mappingToCheck;
    });
}

/// Get Unique input mappings from settings
static std::vector<Settings::InputDeviceMapping> GatherUniqueMappings() {
    std::vector<Settings::InputDeviceMapping> uniqueMappings;

    for (const auto& mapping : Settings::values.input_mappings) {
        if (!CheckIfMappingExists(uniqueMappings, mapping)) {
            uniqueMappings.push_back(mapping);
        }
    }
    return uniqueMappings;
}

/// Builds map of input keys to 3ds buttons for unique device
static std::map<std::string, std::vector<Service::HID::PadState>> BuildKeyMapping(Settings::InputDeviceMapping mapping) {
    std::map<std::string, std::vector<Service::HID::PadState>> keyMapping;
    for (size_t i = 0; i < Settings::values.input_mappings.size(); i++) {
        Service::HID::PadState val = KeyMap::mapping_targets[i];
        std::string key = Settings::values.input_mappings[i].key;
        if (Settings::values.input_mappings[i] == mapping) {
            keyMapping[key].push_back(val);
        }
    }
    return keyMapping;
}

/// Generate a device for each unique mapping
static void GenerateUniqueDevices(const std::vector<Settings::InputDeviceMapping>& uniqueMappings) {
    devices.clear();
    std::shared_ptr<IDevice> input;
    for (const auto& mapping : uniqueMappings) {
        switch (mapping.framework) {
        case Settings::DeviceFramework::Qt:
        {
            main_keyboard = std::make_shared<Keyboard>();
            input = main_keyboard;
            break;
        }
        case Settings::DeviceFramework::SDL:
        {
            if (mapping.device == Settings::Device::Keyboard) {
                main_keyboard = std::make_shared<Keyboard>();
                input = main_keyboard;
                break;
            }
            else if (mapping.device == Settings::Device::Gamepad) {
                input = std::make_shared<SDLGamepad>();
                break;
            }
        }
        }
        devices.push_back(input);

        // Build map of inputs to listen for, for this device
        auto keyMapping = BuildKeyMapping(mapping);

        input->InitDevice(mapping.number, keyMapping);
    }
}

/// Read settings to initialize devices
void ParseSettings() {
    auto uniqueMappings = GatherUniqueMappings();
    GenerateUniqueDevices(uniqueMappings);
}

void ReloadSettings() {
    if (devices.empty())
        return;
    devices.clear();
    ParseSettings();
}

/// Returns all available input devices. Used for key binding in GUI
std::vector<std::shared_ptr<IDevice>> GetAllDevices() {
    auto all_devices = SDLGamepad::GetAllDevices();
    all_devices.push_back(InputCore::GetKeyboard());

    return all_devices;
}

Settings::InputDeviceMapping DetectInput(int max_time, std::function<void(void)> update_GUI) {
    auto devices = GetAllDevices();
    for (auto& device : devices) {
        device->Clear();
    }
    Settings::InputDeviceMapping input_device;
    auto start = std::chrono::high_resolution_clock::now();
    while (input_device.key == "") {
        update_GUI();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::high_resolution_clock::now() - start).count();
        if (duration >= max_time) {
            break;
        }
        for (auto& device : devices) {
            input_device = device->GetInput();
            if (input_device.key != "")
                break;
        }
    };
    return input_device;
}

void Init() {
    ParseSettings();
    tick_event = CoreTiming::RegisterEvent("InputCore::tick_event", InputTickCallback);
    CoreTiming::ScheduleEvent(frame_ticks, tick_event);
}

void Shutdown() {
    CoreTiming::UnscheduleEvent(tick_event, 0);
    devices.clear();
}
}
