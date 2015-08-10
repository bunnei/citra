// Copyright 2015 Citra Emulator Project
// Licensed under GPLv2 or any later version
// Refer to the license.txt file included.

#pragma once

#include <boost/container/static_vector.hpp>

#include "common/common_types.h"
#include "common/vector_math.h"

#include "video_core/pica.h"

#include "shader_bytecode.h"

namespace Pica {

namespace Shader {

struct InputVertex {
    Math::Vec4<float24> attr[16];
};

struct OutputVertex {
    OutputVertex() = default;

    // VS output attributes
    Math::Vec4<float24> pos;
    Math::Vec4<float24> dummy; // quaternions (not implemented, yet)
    Math::Vec4<float24> color;
    Math::Vec2<float24> tc0;
    Math::Vec2<float24> tc1;
    float24 pad[6];
    Math::Vec2<float24> tc2;

    // Padding for optimal alignment
    float24 pad2[4];

    // Attributes used to store intermediate results

    // position after perspective divide
    Math::Vec3<float24> screenpos;
    float24 pad3;

    // Linear interpolation
    // factor: 0=this, 1=vtx
    void Lerp(float24 factor, const OutputVertex& vtx) {
        pos = pos * factor + vtx.pos * (float24::FromFloat32(1) - factor);

        // TODO: Should perform perspective correct interpolation here...
        tc0 = tc0 * factor + vtx.tc0 * (float24::FromFloat32(1) - factor);
        tc1 = tc1 * factor + vtx.tc1 * (float24::FromFloat32(1) - factor);
        tc2 = tc2 * factor + vtx.tc2 * (float24::FromFloat32(1) - factor);

        screenpos = screenpos * factor + vtx.screenpos * (float24::FromFloat32(1) - factor);

        color = color * factor + vtx.color * (float24::FromFloat32(1) - factor);
    }

    // Linear interpolation
    // factor: 0=v0, 1=v1
    static OutputVertex Lerp(float24 factor, const OutputVertex& v0, const OutputVertex& v1) {
        OutputVertex ret = v0;
        ret.Lerp(factor, v1);
        return ret;
    }
};
static_assert(std::is_pod<OutputVertex>::value, "Structure is not POD");
static_assert(sizeof(OutputVertex) == 32 * sizeof(float), "OutputVertex has invalid size");

struct UnitState {
    // The registers are accessed by the JIT using SSE instructions, and are required to be 16-byte
    // aligned. Declaring these at the beginning of the struct ensures this.
    Math::Vec4<float24> input_registers[16];
    Math::Vec4<float24> output_registers[16];
    Math::Vec4<float24> temporary_registers[16];

    u32 program_counter;
    bool conditional_code[2];

    // Two Address registers and one loop counter
    // TODO: How many bits do these actually have?
    s32 address_registers[3];

    enum {
        INVALID_ADDRESS = 0xFFFFFFFF
    };

    struct CallStackElement {
        u32 final_address;  // Address upon which we jump to return_address
        u32 return_address; // Where to jump when leaving scope
        u8 repeat_counter;  // How often to repeat until this call stack element is removed
        u8 loop_increment;  // Which value to add to the loop counter after an iteration
                            // TODO: Should this be a signed value? Does it even matter?
        u32 loop_address;   // The address where we'll return to after each loop iteration
    };

    // TODO: Is there a maximal size for this?
    boost::container::static_vector<CallStackElement, 16> call_stack;

    struct {
        u32 max_offset; // maximum program counter ever reached
        u32 max_opdesc_id; // maximum swizzle pattern index ever used
    } debug;

    static size_t InputOffset(const SourceRegister& reg) {
        switch (reg.GetRegisterType()) {
        case RegisterType::Input:
            return offsetof(UnitState, input_registers[0]) + reg.GetIndex()*sizeof(input_registers[0]);

        case RegisterType::Temporary:
            return offsetof(UnitState, temporary_registers[0]) + reg.GetIndex()*sizeof(temporary_registers[0]);

        default:
            UNREACHABLE();
            return 0;
        }
    }

    static size_t OutputOffset(const DestRegister& reg) {
        switch (reg.GetRegisterType()) {
        case RegisterType::Output:
            return offsetof(UnitState, output_registers[0]) + reg.GetIndex()*sizeof(output_registers[0]);

        case RegisterType::Temporary:
            return offsetof(UnitState, temporary_registers[0]) + reg.GetIndex()*sizeof(temporary_registers[0]);

        default:
            UNREACHABLE();
            return 0;
        }
    }
};

/**
 * Performs any shader unit setup that only needs to happen once per shader (as opposed to once per
 * vertex, which would happen within the `Run` function).
 * @param state Shader unit state, must be setup per shader and per shader unit
 */
void Setup(UnitState& state);

/**
 * Runs the currently setup shader
 * @param state Shader unit state, must be setup per shader and per shader unit
 * @param input Input vertex into the shader
 * @param num_attributes The number of vertex shader attributes
 * @return The output vertex, after having been processed by the vertex shader
 */
OutputVertex Run(UnitState& state, const InputVertex& input, int num_attributes);

} // namespace Shader

} // namespace Pica
