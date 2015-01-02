// Copyright 2014 Citra Emulator Project
// Licensed under GPLv2 or any later version
// Refer to the license.txt file included.

#pragma once

// The shared page stores various runtime configuration settings. This memory page is
// read-only for ARM11 processes. I'm guessing this would normally be written to by the firmware/
// bootrom. Because we're not emulating this, and essentially just "stubbing" the functionality, I'm
// putting this as a subset of HLE for now.

#include "common/common_types.h"

////////////////////////////////////////////////////////////////////////////////////////////////////

namespace SharedPage {

template <typename T>
void Read(T &var, const u32 addr);

void Set3DSlider(float amount);

} // namespace
