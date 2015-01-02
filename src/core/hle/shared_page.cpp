// Copyright 2014 Citra Emulator Project
// Licensed under GPLv2 or any later version
// Refer to the license.txt file included.

#include "common/common_types.h"
#include "common/log.h"

#include "core/hle/config_mem.h"

////////////////////////////////////////////////////////////////////////////////////////////////////

namespace SharedPage {

static float three_d_slider_value = 0.0f;

enum {
    DATETIME_SELECTOR       = 0x1FF81000,
    RUNNING_HW              = 0x1FF81004,
    ICU_HW_INFO             = 0x1FF81005,
    DATETIME_0              = 0x1FF81020,
    DATETIME_1              = 0x1FF81040,
    WIFI_MACADDR            = 0x1FF81060,
    WIFI_MYSTERY1           = 0x1FF81066,
    WIFI_MYSTERY2           = 0x1FF81067,
    THREED_SLIDERSTATE      = 0x1FF81080,
    THREED_LEDSTATE         = 0x1FF81084,
    MENUTID                 = 0x1FF810A0,
    ACTIVEMENUTID           = 0x1FF810A8,
    MYSTERY_3               = 0x1FF810C0,
};

template <typename T>
inline void Read(T &var, const u32 addr) {
    switch (addr) {

    case DATETIME_SELECTOR:
        var = 0x00000001;
        break;
    case RUNNING_HW:
        var = 0x00000001;
        break;
    // TODO: mcu, datetime, wifi_mac
    case WIFI_MYSTERY1:
        var = 0x00000000;
        break;
    case WIFI_MYSTERY2:
        var = 0x00000000;
        break;
    case THREED_SLIDERSTATE:
        var = *((u32*) &three_d_slider_value);
        break;
    case THREED_LEDSTATE:
        var = (three_d_slider_value == 0.0f)? 0x00000001: 0x00000000; // off when 0.0f
        break;
    // TODO: menu/active menu TID, mystery 1
    default:
        LOG_ERROR(Kernel, "unknown addr=0x%08X", addr);
    }
}

// Explicitly instantiate template functions because we aren't defining this in the header:

template void Read<u64>(u64 &var, const u32 addr);
template void Read<u32>(u32 &var, const u32 addr);
template void Read<u16>(u16 &var, const u32 addr);
template void Read<u8>(u8 &var, const u32 addr);

void Set3DSlider(float amount) {
    three_d_slider_value = amount;
}

} // namespace
