// Copyright 2015 Citra Emulator Project
// Licensed under GPLv2 or any later version
// Refer to the license.txt file included.

#include "common/common_types.h"
#include "common/intrinsics.h"

#include "vertex_shader.h"

namespace Pica {

namespace VertexShaderSIMD {

struct Reg {
    union {
        struct {
            f32 x, y, z, w;
        };
#if _M_SSE >= 0x401
        __m128 raw_f;
        __m128i raw_i;
#endif
    };

    f32 operator [] (int i) {
        return *(&x + i);
    }
};

struct CoreState {
    u32 pc;
    Reg address_offset;
    bool conditional_code[2];

    struct CallStackElement {
        u32 final_address;  // Address upon which we jump to return_address
        u32 return_address; // Where to jump when leaving scope
        u8 repeat_counter;  // How often to repeat until this call stack element is removed
        u8 loop_increment;  // Which value to add to the loop counter after an iteration
        // TODO: Should this be a signed value? Does it even matter?
        u32 loop_address;   // The address where we'll return to after each loop iteration
    };

    // TODO: Is there a maximal size for this?
    std::vector<CallStackElement> call_stack;

    union {
        struct {
            Reg output[0x10];
            Reg input[0x10];
            Reg temporary[0x10];
            Reg uniform[0x60];
        };
        Reg regs[0x90];
    };

    Reg InputReg(int index) const {
        return input[index];
    }

    Reg& OutputReg(int index) {
        return regs[(index & 0xf) | ((index & 0x10) << 1)];
    }

};

/// Initializes lookup tables used by the SIMD vertex shader core
void Init();

/// Initializes a SIMD vertex shader core for the current shader in Pica memory
void InitCore(CoreState& state);

/// Runs a vertex shader core for the current shader in Pica memory
VertexShader::OutputVertex RunShader(CoreState& state, const VertexShader::InputVertex& input, int num_attributes);

} // namespace VertexShaderSIMD

} // namespace Pica
