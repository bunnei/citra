// Copyright 2016 Citra Emulator Project
// Licensed under GPLv2 or any later version
// Refer to the license.txt file included.
#include "configure_input.h"

ConfigureInput::ConfigureInput(QWidget *parent) :
    QWidget(parent),
    ui(new Ui::ConfigureInput)
{
    ui->setupUi(this);

    //Initialize mapping of input enum to UI button.
    input_mapping = {
        { std::make_pair(Settings::NativeInput::Values::A, ui->btnFaceA) },
        { std::make_pair(Settings::NativeInput::Values::B, ui->btnFaceB) },
        { std::make_pair(Settings::NativeInput::Values::X, ui->btnFaceX) },
        { std::make_pair(Settings::NativeInput::Values::Y, ui->btnFaceY) },
        { std::make_pair(Settings::NativeInput::Values::L, ui->btnShdrL) },
        { std::make_pair(Settings::NativeInput::Values::R, ui->btnShdrR) },
        { std::make_pair(Settings::NativeInput::Values::ZL, ui->btnShdrZL) },
        { std::make_pair(Settings::NativeInput::Values::ZR, ui->btnShdrZR) },
        { std::make_pair(Settings::NativeInput::Values::START, ui->btnStart) },
        { std::make_pair(Settings::NativeInput::Values::SELECT, ui->btnSelect) },
        { std::make_pair(Settings::NativeInput::Values::HOME, ui->btnHome) },
        { std::make_pair(Settings::NativeInput::Values::DUP, ui->btnDirUp) },
        { std::make_pair(Settings::NativeInput::Values::DDOWN, ui->btnDirDown) },
        { std::make_pair(Settings::NativeInput::Values::DLEFT, ui->btnDirLeft) },
        { std::make_pair(Settings::NativeInput::Values::DRIGHT, ui->btnDirRight) },
        { std::make_pair(Settings::NativeInput::Values::CUP, ui->btnStickUp) },
        { std::make_pair(Settings::NativeInput::Values::CDOWN, ui->btnStickDown) },
        { std::make_pair(Settings::NativeInput::Values::CLEFT, ui->btnStickLeft) },
        { std::make_pair(Settings::NativeInput::Values::CRIGHT, ui->btnStickRight) },
        { std::make_pair(Settings::NativeInput::Values::CIRCLE_UP, ui->btnCircleUp) },
        { std::make_pair(Settings::NativeInput::Values::CIRCLE_DOWN, ui->btnCircleDown) },
        { std::make_pair(Settings::NativeInput::Values::CIRCLE_LEFT, ui->btnCircleLeft) },
        { std::make_pair(Settings::NativeInput::Values::CIRCLE_RIGHT, ui->btnCircleRight) },
    };

    //Attach handle click method to each button click.
    for (auto &ent1 : input_mapping) {
        connect(ent1.second, &QPushButton::released, this, &ConfigureInput::HandleClick);
    }
    connect(ui->btnRestoreDefaults, &QPushButton::released, this, &ConfigureInput::RestoreDefaults);
    setFocusPolicy(Qt::ClickFocus);
    this->setConfiguration();
}

ConfigureInput::~ConfigureInput()
{
}

///Event handler for all button released() event.
void ConfigureInput::HandleClick()
{
    QPushButton* sender = qobject_cast<QPushButton*>(QObject::sender());
    sender->setText("[waiting]");
    sender->setFocus();
    grabKeyboard();
    grabMouse();
    changingButton = sender;
}

///Save all button configurations to settings file
void ConfigureInput::applyConfiguration()
{
    for (int i = 0; i < Settings::NativeInput::NUM_INPUTS - 1; ++i) {
        int value = GetKeyValue(input_mapping[Settings::NativeInput::Values(i)]->text());
        Settings::values.input_mappings[Settings::NativeInput::All[i]] = value;
    }
    Settings::Apply();
}

///Load configuration settings into button text
void ConfigureInput::setConfiguration()
{
    for (int i = 0; i < Settings::NativeInput::NUM_INPUTS - 1; ++i) {
        QString keyValue = GetKeyName(Settings::values.input_mappings[i]);
        input_mapping[Settings::NativeInput::Values(i)]->setText(keyValue);
    }
}

///Handle key press event for input tab when a button is 'waiting'.
void ConfigureInput::keyPressEvent(QKeyEvent *event)
{
    if (changingButton != nullptr && event->key() > 0)
    {
        keysPressed.push_back(event->key());

        //Can't save Modifier + Keys yet as input. Will re-enable after settings refactor
        /*if (event->key() == Qt::Key_Shift)
            return;

        else if (event->key() == Qt::Key_Control)
            return;

        else if (event->key() == Qt::Key_Alt)
            return;

        else if (event->key() == Qt::Key_Meta)
            return;
        else*/
        SetKey();
    }
}

///Set button text to name of key pressed.
void ConfigureInput::SetKey()
{
    QString keyValue = "";
    for (int i : keysPressed) // Will only contain one key until settings refactor
    {
        keyValue += GetKeyName(i);
    }
    //RemoveDuplicates(keyValue);
    changingButton->setText(keyValue);

    keysPressed.clear();
    releaseKeyboard();
    releaseMouse();
    changingButton = nullptr;
}

///Convert key ASCII value to its' letter/name
QString ConfigureInput::GetKeyName(int key_code)
{
    if (key_code == Qt::Key_Shift)
        return tr("Shift");

    else if (key_code == Qt::Key_Control)
        return tr("Ctrl");

    else if (key_code == Qt::Key_Alt)
        return tr("Alt");

    else if (key_code == Qt::Key_Meta)
        return "";
    else if (key_code == -1)
        return "";

    return QKeySequence(key_code).toString();
}

///Convert letter/name of key to its ASCII value.
int ConfigureInput::GetKeyValue(QString text)
{
    if (text == "Shift")
        return Qt::Key_Shift;
    else if (text == "Ctrl")
        return Qt::Key_Control;
    else if (text == "Alt")
        return Qt::Key_Alt;
    else if (text == "Meta")
        return -1;
    else if (text == "")
        return -1;
    return QKeySequence(text)[0];
}

///Check all inputs for duplicate keys. Clears out any other button with same key as new button.
void ConfigureInput::RemoveDuplicates(QString newValue)
{
    for (int i = 0; i < Settings::NativeInput::NUM_INPUTS - 1; ++i) {
        if (changingButton != input_mapping[Settings::NativeInput::Values(i)]) {
            QString oldValue = input_mapping[Settings::NativeInput::Values(i)]->text();
            if (newValue == oldValue)
                input_mapping[Settings::NativeInput::Values(i)]->setText("");
        }
    }
}

///Restore all buttons to their default values.
void ConfigureInput::RestoreDefaults() {
    for (int i = 0; i < Settings::NativeInput::NUM_INPUTS - 1; ++i)
    {
        QString keyValue = GetKeyName(defaults[i].toInt());
        input_mapping[Settings::NativeInput::Values(i)]->setText(keyValue);
    }
}