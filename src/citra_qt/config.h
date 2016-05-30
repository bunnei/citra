// Copyright 2014 Citra Emulator Project
// Licensed under GPLv2 or any later version
// Refer to the license.txt file included.

#pragma once

#include <string>
#include "core/settings.h"
#include <QSettings>
#include <QString>
#include <QStringList>

class QSettings;

class Config {
    QSettings* qt_config;
    std::string qt_config_loc;

    void ReadValues();
    void SaveValues();
public:
    Config();
    ~Config();

    void Reload();
    void Save();
};
extern const std::array<QVariant, Settings::NativeInput::NUM_INPUTS> defaults;
