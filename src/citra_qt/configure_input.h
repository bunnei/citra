// Copyright 2016 Citra Emulator Project
// Licensed under GPLv2 or any later version
// Refer to the license.txt file included.
#pragma once
#include <memory>

#include <QKeyEvent>
#include <QObject>
#include <QPushButton>
#include <QSettings>
#include <QString>
#include <QWidget>

#include "citra_qt/config.h"
#include "core/settings.h"
#include "ui_configure_input.h"

namespace Ui {
    class ConfigureInput;
}

class ConfigureInput : public QWidget
{
    Q_OBJECT

public:
    explicit ConfigureInput(QWidget *parent = 0);
    ~ConfigureInput();
    void applyConfiguration();

private:
    std::unique_ptr<Ui::ConfigureInput> ui;
    std::map<Settings::NativeInput::Values, QPushButton*> input_mapping;
    std::vector<int> keysPressed;
    QPushButton* changingButton = NULL; // button currently waiting for key press.

    void HandleClick();
    void setConfiguration();
    void SetKey();
    void RemoveDuplicates(QString newValue);
    void RestoreDefaults();
    virtual void  keyPressEvent(QKeyEvent *event);
    QString GetKeyName(int key_code);
    int GetKeyValue(QString text);
};
