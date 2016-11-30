// Copyright 2016 Citra Emulator Project
// Licensed under GPLv2 or any later version
// Refer to the license.txt file included.

#include <utility>

#include "citra_qt/configure_input.h"
#include "citra_qt/keybinding_names.h"
#include "common/string_util.h"

#include "input_core/input_core.h"
#include "input_core/devices/keyboard.h"


ConfigureInput::ConfigureInput(QWidget* parent) : QWidget(parent), ui(std::make_unique<Ui::ConfigureInput>()) {
    ui->setupUi(this);

    // Initialize mapping of input enum to UI button.
    qt_buttons = {
        { std::make_pair(Settings::NativeInput::Values::A, ui->buttonA) },
        { std::make_pair(Settings::NativeInput::Values::B, ui->buttonB) },
        { std::make_pair(Settings::NativeInput::Values::X, ui->buttonX) },
        { std::make_pair(Settings::NativeInput::Values::Y, ui->buttonY) },
        { std::make_pair(Settings::NativeInput::Values::L, ui->buttonL) },
        { std::make_pair(Settings::NativeInput::Values::R, ui->buttonR) },
        { std::make_pair(Settings::NativeInput::Values::ZL, ui->buttonZL) },
        { std::make_pair(Settings::NativeInput::Values::ZR, ui->buttonZR) },
        { std::make_pair(Settings::NativeInput::Values::START, ui->buttonStart) },
        { std::make_pair(Settings::NativeInput::Values::SELECT, ui->buttonSelect) },
        { std::make_pair(Settings::NativeInput::Values::HOME, ui->buttonHome) },
        { std::make_pair(Settings::NativeInput::Values::DUP, ui->buttonDpadUp) },
        { std::make_pair(Settings::NativeInput::Values::DDOWN, ui->buttonDpadDown) },
        { std::make_pair(Settings::NativeInput::Values::DLEFT, ui->buttonDpadLeft) },
        { std::make_pair(Settings::NativeInput::Values::DRIGHT, ui->buttonDpadRight) },
        { std::make_pair(Settings::NativeInput::Values::CUP, ui->buttonCStickUp) },
        { std::make_pair(Settings::NativeInput::Values::CDOWN, ui->buttonCStickDown) },
        { std::make_pair(Settings::NativeInput::Values::CLEFT, ui->buttonCStickLeft) },
        { std::make_pair(Settings::NativeInput::Values::CRIGHT, ui->buttonCStickRight) },
        { std::make_pair(Settings::NativeInput::Values::CIRCLE_UP, ui->buttonCircleUp) },
        { std::make_pair(Settings::NativeInput::Values::CIRCLE_DOWN, ui->buttonCircleDown) },
        { std::make_pair(Settings::NativeInput::Values::CIRCLE_LEFT, ui->buttonCircleLeft) },
        { std::make_pair(Settings::NativeInput::Values::CIRCLE_RIGHT, ui->buttonCircleRight) }
    };

    // Attach handle click method to each button click.
    for (const auto& entry : qt_buttons) {
        connect(entry.second, SIGNAL(released()), this, SLOT(handleClick()));
    }
    connect(ui->buttonCircleMod, SIGNAL(released()), this, SLOT(handleClick()));
    connect(ui->buttonRestoreDefaults, SIGNAL(released()), this, SLOT(restoreDefaults()));

    setFocusPolicy(Qt::ClickFocus);

    this->setConfiguration();
}

void ConfigureInput::handleClick() {
    QPushButton* sender = qobject_cast<QPushButton*>(QObject::sender());
    previous_mapping = sender->text();
    sender->setText(tr("[waiting]"));
    sender->setFocus();
    grabKeyboard();
    grabMouse();
    changing_button = sender;
    auto update = []() {
        QCoreApplication::processEvents();
    };
    auto input_device = InputCore::DetectInput(5000, update);

    setKey(input_device);
}

void ConfigureInput::keyPressEvent(QKeyEvent* event) {
    if (!changing_button)
        return;
    if (!event || event->key() == Qt::Key_unknown)
        return;

    auto keyboard = InputCore::GetKeyboard();
    KeyboardKey param = KeyboardKey(event->key(), QKeySequence(event->key()).toString().toStdString());
    keyboard->KeyPressed(param);
}

void ConfigureInput::applyConfiguration() {
    for (int i = 0; i < Settings::NativeInput::NUM_INPUTS; ++i) {
        Settings::values.input_mappings[Settings::NativeInput::All[i]] = button_mapping[qt_buttons[Settings::NativeInput::Values(i)]];
    }
    Settings::values.pad_circle_modifier = button_mapping[ui->buttonCircleMod];
    Settings::Apply();
}

void ConfigureInput::setConfiguration() {
    for (int i = 0; i < Settings::NativeInput::NUM_INPUTS; ++i) {
        Settings::InputDeviceMapping mapping = Settings::values.input_mappings[i];
        button_mapping[qt_buttons[Settings::NativeInput::Values(i)]] = mapping;

        qt_buttons[Settings::NativeInput::Values(i)]->setText(getKeyName(mapping));
    }
    button_mapping[ui->buttonCircleMod] = Settings::values.pad_circle_modifier;
    ui->buttonCircleMod->setText(getKeyName(Settings::values.pad_circle_modifier));
}

void ConfigureInput::setKey(Settings::InputDeviceMapping keyPressed) {
    if (keyPressed.key == "" || keyPressed.key == std::to_string(Qt::Key_Escape))
        changing_button->setText(previous_mapping);
    else {
        changing_button->setText(getKeyName(keyPressed));
        button_mapping[changing_button] = keyPressed;
        removeDuplicates(keyPressed);
    }
    releaseKeyboard();
    releaseMouse();
    changing_button = nullptr;
    previous_mapping = nullptr;
}

QString ConfigureInput::getKeyName(Settings::InputDeviceMapping mapping) const {
    if (mapping.key.empty())
        return "";
    int key_code;
    if (!Common::TryParse(mapping.key, &key_code))
        key_code = -1;
    if (mapping.device == Settings::Device::Gamepad) {
        if (KeyBindingNames::sdl_gamepad_names.size() > key_code && key_code >= 0)
            return KeyBindingNames::sdl_gamepad_names[key_code];
        else
            return "";
    }
    if (key_code == Qt::Key_Shift)
        return tr("Shift");
    if (key_code == Qt::Key_Control)
        return tr("Ctrl");
    if (key_code == Qt::Key_Alt)
        return tr("Alt");
    if (key_code == Qt::Key_Meta)
        return "";
    if (key_code < 0)
        return "";

    return QKeySequence(key_code).toString();
}

void ConfigureInput::removeDuplicates(const Settings::InputDeviceMapping newValue) {
    for (auto& entry : button_mapping) {
        if (changing_button != entry.first) {
            if (newValue == entry.second && newValue.key == entry.second.key) {
                entry.first->setText("");
                entry.second = Settings::InputDeviceMapping();
            }
        }
    }
}

void ConfigureInput::restoreDefaults() {
    for (int i = 0; i < Settings::NativeInput::NUM_INPUTS; ++i) {
        Settings::InputDeviceMapping mapping = Settings::InputDeviceMapping(Config::defaults[i].toInt());
        button_mapping[qt_buttons[Settings::NativeInput::Values(i)]] = mapping;
        const QString keyValue = getKeyName(Settings::InputDeviceMapping(Config::defaults[i].toInt()));
        qt_buttons[Settings::NativeInput::Values(i)]->setText(keyValue);
    }
    button_mapping[ui->buttonCircleMod] = Settings::InputDeviceMapping(Qt::Key_F);
    ui->buttonCircleMod->setText(getKeyName(Settings::InputDeviceMapping(Qt::Key_F)));
}
