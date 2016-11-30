// Copyright 2016 Citra Emulator Project
// Licensed under GPLv2 or any later version
// Refer to the license.txt file included.

#pragma once
#include <memory>
#include <QWidget>
#include <QKeyEvent>

#include "citra_qt/config.h"
#include "core/settings.h"
#include "input_core/devices/device.h"
#include "ui_configure_input.h"

class QPushButton;
class QString;

namespace Ui {
    class ConfigureInput;
}

class ConfigureInput : public QWidget {
    Q_OBJECT

public:
    explicit ConfigureInput(QWidget* parent = nullptr);

    /// Save all button configurations to settings file
    void applyConfiguration();

private:
    std::unique_ptr<Ui::ConfigureInput> ui;
    std::map<Settings::NativeInput::Values, QPushButton*> qt_buttons;
    std::map<QPushButton*, Settings::InputDeviceMapping> button_mapping;
    QPushButton* changing_button = nullptr; ///< button currently waiting for key press.
    QString previous_mapping;

    /// Load configuration settings into button text
    void setConfiguration();

    /// Check all inputs for duplicate keys. Clears out any other button with the same value as this button's new value.
    void removeDuplicates(const Settings::InputDeviceMapping newValue);

    /// Handle keykoard key press event for input tab when a button is 'waiting'.
    void keyPressEvent(QKeyEvent* event) override;

    /// Convert key ASCII value to its' letter/name
    QString getKeyName(Settings::InputDeviceMapping mapping) const;

    /// Set button text to name of key pressed.
    void setKey(Settings::InputDeviceMapping keyPressed);

private slots:
    /// Event handler for all button released() event.
    void handleClick();

    /// Restore all buttons to their default values.
    void restoreDefaults();
};
