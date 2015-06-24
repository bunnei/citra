// Copyright 2015 Citra Emulator Project
// Licensed under GPLv2 or any later version
// Refer to the license.txt file included.

#pragma once

#include "common/common_types.h"
#include "nihstro/shader_bytecode.h"

#include <string>

// TODO: Add function to construct fragment shader to avoid branches and such
std::string PICAVertexShaderToGLSL(u32 main_offset, const u32* shader_data, const u32* swizzle_raw_data);
