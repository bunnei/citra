// Copyright 2008 Dolphin Emulator Project
// Licensed under GPLv2+
// Refer to the license.txt file included.

#pragma once

#include "common/common_types.h"

namespace Common {

u64 GetHash64(const u8 *src, u32 len, u32 samples);
void SetHash64Function();

} // namespace Common
