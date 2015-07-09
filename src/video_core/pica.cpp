// Copyright 2015 Citra Emulator Project
// Licensed under GPLv2 or any later version
// Refer to the license.txt file included.

#include <string.h>

#include "pica.h"

#include "vertex_shader_simd.h"

namespace Pica {

State g_state;

void Init() {
#if _M_SSE >= 0x401
    VertexShaderSIMD::Init();
#endif
}

void Shutdown() {
    memset(&g_state, 0, sizeof(State));
}

}
