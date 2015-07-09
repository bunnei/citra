// Copyright 2015 Citra Emulator Project
// Licensed under GPLv2 or any later version
// Refer to the license.txt file included.

#include <cstring>
#include <vector>

#include <nihstro/shader_bytecode.h>

#include "video_core/vertex_shader_simd.h"

using nihstro::OpCode;
using nihstro::SwizzlePattern;

namespace Pica {

namespace VertexShaderSIMD {

#if _M_SSE >= 0x401
static __m128i swizzle_dst_mask[16];
static __m128i swizzle_src_selector[256];
#endif

void Init() {
#if _M_SSE >= 0x401
    for (int i = 0; i < 16; ++i) {
        swizzle_dst_mask[i].m128i_u32[0] = ((i >> 3) & 1) * 0xffffffff;
        swizzle_dst_mask[i].m128i_u32[1] = ((i >> 2) & 1) * 0xffffffff;
        swizzle_dst_mask[i].m128i_u32[2] = ((i >> 1) & 1) * 0xffffffff;
        swizzle_dst_mask[i].m128i_u32[3] = (i & 1) * 0xffffffff;
    }

#define SELECT(n) (((((n) << 2) + 3) << 24) | ((((n) << 2) + 2) << 16) | ((((n) << 2) + 1) << 8) | ((n) << 2))
    for (int i = 0; i < 256; ++i) {
        swizzle_src_selector[i].m128i_u32[0] = SELECT((i >> 6) & 0x3);
        swizzle_src_selector[i].m128i_u32[1] = SELECT((i >> 4) & 0x3);
        swizzle_src_selector[i].m128i_u32[2] = SELECT((i >> 2) & 0x3);
        swizzle_src_selector[i].m128i_u32[3] = SELECT( i       & 0x3);
    }
#undef SELECT
#endif
}

#pragma pack(1)
union Instruction {
    Instruction& operator =(const Instruction& instr) {
        hex = instr.hex;
        return *this;
    }

    u32 hex;

    BitField<0x1a, 0x6, OpCode::Id> opcode;

    // General notes:
    //
    // When two input registers are used, one of them uses a 5-bit index while the other
    // one uses a 7-bit index. This is because at most one floating point uniform may be used
    // as an input.

    enum CompareOp : u32 {
        Equal = 0,
        NotEqual = 1,
        LessThan = 2,
        LessEqual = 3,
        GreaterThan = 4,
        GreaterEqual = 5,
        Unk6 = 6,
        Unk7 = 7
    };

    // Format used e.g. by arithmetic instructions and comparisons
    union Common { // TODO: Remove name
        BitField<0x00, 0x7, u32> operand_desc_id;

        /**
        * Source inputs may be reordered for certain instructions.
        * Use GetSrc1 and GetSrc2 instead to access the input register indices hence.
        */
        BitField<0x07, 0x5, u32> src2;
        BitField<0x0c, 0x7, u32> src1;

        BitField<0x07, 0x7, u32> src2i;
        BitField<0x0e, 0x5, u32> src1i;

        // Address register value is used for relative addressing of src1
        BitField<0x13, 0x2, u32> address_register_index;

        union {
            BitField<0x15, 0x3, CompareOp> y;
            BitField<0x18, 0x3, CompareOp> x;
        } compare_op;

        BitField<0x15, 0x5, u32> dest;
    } common;

    union FlowControlType {  // TODO: Make nameless once MSVC supports it
        enum Op : u32 {
            Or = 0,
            And = 1,
            JustX = 2,
            JustY = 3
        };

        BitField<0x00, 0x8, u32> num_instructions;
        BitField<0x0a, 0xc, u32> dest_offset;

        BitField<0x16, 0x2, Op> op;
        BitField<0x16, 0x4, u32> bool_uniform_id;
        BitField<0x16, 0x2, u32> int_uniform_id; // TODO: Verify that only this many bits are used...

        BitFlag<0x18, u32> refy;
        BitFlag<0x19, u32> refx;
    } flow_control;

    union {
        BitField<0x00, 0x5, u32> operand_desc_id;

        BitField<0x05, 0x5, u32> src3;
        BitField<0x0a, 0x7, u32> src2;
        BitField<0x11, 0x7, u32> src1;

        BitField<0x05, 0x7, u32> src3i;
        BitField<0x0c, 0x5, u32> src2i;

        BitField<0x18, 0x5, u32> dest;
    } cmp;

    union {
        BitField<0x00, 0x5, u32> operand_desc_id;

        BitField<0x05, 0x5, u32> src3;
        BitField<0x0a, 0x7, u32> src2;
        BitField<0x11, 0x7, u32> src1;

        BitField<0x05, 0x7, u32> src3i;
        BitField<0x0c, 0x5, u32> src2i;

        BitField<0x18, 0x5, u32> dest;
    } mad;
};
static_assert(sizeof(Instruction) == 0x4, "Incorrect structure size");
static_assert(std::is_standard_layout<Instruction>::value, "Structure does not have standard layout");

static inline void Call(CoreState& state, u32 offset, u32 num_instructions, u32 return_offset, u8 repeat_count, u8 loop_increment) {
    state.pc = offset;
    state.call_stack.push_back({ offset + num_instructions, return_offset, repeat_count, loop_increment, offset });
};

static inline bool Compare(Instruction::CompareOp op, f32 src1, f32 src2) {
    switch (op) {
    case Instruction::CompareOp::Equal:
        return src1 == src2;

    case Instruction::CompareOp::NotEqual:
        return src1 != src2;

    case Instruction::CompareOp::LessThan:
        return src1 < src2;

    case Instruction::CompareOp::LessEqual:
        return src1 <= src2;

    case Instruction::CompareOp::GreaterThan:
        return src1 > src2;

    case Instruction::CompareOp::GreaterEqual:
        return src1 >= src2;
    }

    return false;
}

static inline bool EvaluateCondition(const CoreState& state, bool refx, bool refy, Instruction::FlowControlType flow_control) {
    const bool results[2] = { refx == state.conditional_code[0],
                              refy == state.conditional_code[1] };

    switch (flow_control.op) {
    case flow_control.Or:
        return results[0] || results[1];

    case flow_control.And:
        return results[0] && results[1];

    case flow_control.JustX:
        return results[0];

    case flow_control.JustY:
        return results[1];
    }

    return false;
};

void InitCore(CoreState& state) {
    std::memcpy(&state.uniform[0].x, g_state.vs.uniforms.f, sizeof(state.uniform));
}

VertexShader::OutputVertex RunShader(CoreState& state, const VertexShader::InputVertex& input, int num_attributes) {
    const auto& regs = g_state.regs;
    auto& vs = g_state.vs;
    const auto& swizzle_data = vs.swizzle_data;
    const auto& program_code = vs.program_code;
    bool exit_loop = false;

    state.pc = regs.vs_main_offset;
    state.address_offset.raw_i.m128i_i32[0] = 0;
    state.conditional_code[0] = false;
    state.conditional_code[1] = false;

    const auto& reg_map = regs.vs_input_register_map;

    for (int i = 0; i < num_attributes; ++i) {
        Reg& reg = state.input[reg_map.GetRegisterForAttribute(i)];
        reg.x = input.attr[i].x.ToFloat32();
        reg.y = input.attr[i].y.ToFloat32();
        reg.z = input.attr[i].z.ToFloat32();
        reg.w = input.attr[i].w.ToFloat32();
    }

#if _M_SSE >= 0x401
    while (true) {
        if (!state.call_stack.empty()) {
            auto& top = state.call_stack.back();
            if (state.pc == top.final_address) {
                state.address_offset.raw_i.m128i_i32[3] += top.loop_increment;

                if (top.repeat_counter-- == 0) {
                    state.pc = top.return_address;
                    state.call_stack.pop_back();
                } else {
                    state.pc = top.loop_address;
                }

                // TODO: Is "trying again" accurate to hardware?
                continue;
            }
        }

        const Instruction& instr = *(const Instruction*)&program_code[state.pc];

        #define NEGATE(value, negate) _mm_xor_si128(value, _mm_set1_epi32(negate << 31))

        #define SWIZZLE(reg, selector) _mm_shuffle_epi8(state.InputReg(reg).raw_i, swizzle_src_selector[selector])

        #define FORMAT1(operation_) { \
            Reg src1, src2, temp; \
            const SwizzlePattern& swizzle = *(SwizzlePattern*)&swizzle_data[instr.common.operand_desc_id]; \
            int offset = state.address_offset.raw_i.m128i_i32[instr.common.address_register_index]; \
            src1.raw_i = NEGATE(SWIZZLE(instr.common.src1 + offset, swizzle.src1_selector), swizzle.negate_src1); \
            src2.raw_i = NEGATE(SWIZZLE(instr.common.src2, swizzle.src2_selector), swizzle.negate_src2); \
            temp.raw_f = operation_; \
            __m128i dst_mask = swizzle_dst_mask[swizzle.dest_mask]; \
            auto& dest = state.OutputReg(instr.common.dest); \
            dest.raw_i = _mm_or_si128(_mm_andnot_si128(dst_mask, dest.raw_i), _mm_and_si128(dst_mask, temp.raw_i)); \
        }

        #define FORMAT1I(operation_) { \
            Reg src1, src2, temp; \
            const SwizzlePattern& swizzle = *(SwizzlePattern*)&swizzle_data[instr.common.operand_desc_id]; \
            int offset = state.address_offset.raw_i.m128i_i32[instr.common.address_register_index]; \
            src1.raw_i = NEGATE(SWIZZLE(instr.common.src1i, swizzle.src1_selector), swizzle.negate_src1); \
            src2.raw_i = NEGATE(SWIZZLE(instr.common.src2i + offset, swizzle.src2_selector), swizzle.negate_src2); \
            temp.raw_f = operation_; \
            __m128i dst_mask = swizzle_dst_mask[swizzle.dest_mask]; \
            auto& dest = state.OutputReg(instr.common.dest); \
            dest.raw_i = _mm_or_si128(_mm_andnot_si128(dst_mask, dest.raw_i), _mm_and_si128(dst_mask, temp.raw_i)); \
        }

        #define FORMAT1U(operation_) { \
            Reg src1, temp; \
            const SwizzlePattern& swizzle = *(SwizzlePattern*)&swizzle_data[instr.common.operand_desc_id]; \
            int offset = state.address_offset.raw_i.m128i_i32[instr.common.address_register_index]; \
            src1.raw_i = NEGATE(SWIZZLE(instr.common.src1 + offset, swizzle.src1_selector), swizzle.negate_src1); \
            temp.raw_f = operation_; \
            __m128i dst_mask = swizzle_dst_mask[swizzle.dest_mask]; \
            auto& dest = state.OutputReg(instr.common.dest); \
            dest.raw_i = _mm_or_si128(_mm_andnot_si128(dst_mask, dest.raw_i), _mm_and_si128(dst_mask, temp.raw_i)); \
        }

        #define FORMAT5(operation_) { \
            Reg src1, src2, src3, temp; \
            const SwizzlePattern& swizzle = *(SwizzlePattern*)&swizzle_data[instr.mad.operand_desc_id]; \
            src1.raw_i = NEGATE(SWIZZLE(instr.mad.src1, swizzle.src1_selector), swizzle.negate_src1); \
            src2.raw_i = NEGATE(SWIZZLE(instr.mad.src2, swizzle.src2_selector), swizzle.negate_src2); \
            src3.raw_i = NEGATE(SWIZZLE(instr.mad.src3, swizzle.src3_selector), swizzle.negate_src3); \
            temp.raw_f = operation_; \
            __m128i dst_mask = swizzle_dst_mask[swizzle.dest_mask]; \
            auto& dest = state.OutputReg(instr.mad.dest); \
            dest.raw_i = _mm_or_si128(_mm_andnot_si128(dst_mask, dest.raw_i), _mm_and_si128(dst_mask, temp.raw_i)); \
        }

        #define FORMAT5I(operation_) { \
            Reg src1, src2, src3, temp; \
            const SwizzlePattern& swizzle = *(SwizzlePattern*)&swizzle_data[instr.mad.operand_desc_id]; \
            src1.raw_i = NEGATE(SWIZZLE(instr.mad.src1, swizzle.src1_selector), swizzle.negate_src1); \
            src2.raw_i = NEGATE(SWIZZLE(instr.mad.src2i, swizzle.src2_selector), swizzle.negate_src2); \
            src3.raw_i = NEGATE(SWIZZLE(instr.mad.src3i, swizzle.src3_selector), swizzle.negate_src3); \
            temp.raw_f = operation_; \
            __m128i dst_mask = swizzle_dst_mask[swizzle.dest_mask]; \
            auto& dest = state.OutputReg(instr.mad.dest); \
            dest.raw_i = _mm_or_si128(_mm_andnot_si128(dst_mask, dest.raw_i), _mm_and_si128(dst_mask, temp.raw_i)); \
        }

        switch (instr.opcode) {
        case OpCode::Id::ADD:
            FORMAT1(_mm_add_ps(src1.raw_f, src2.raw_f));
            break;

        case OpCode::Id::DP3:
            FORMAT1(_mm_dp_ps(src1.raw_f, src2.raw_f, 0x7f));
            break;

        case OpCode::Id::DP4:
            FORMAT1(_mm_dp_ps(src1.raw_f, src2.raw_f, 0xff));
            break;

        case OpCode::Id::MUL:
            FORMAT1(_mm_mul_ps(src1.raw_f, src2.raw_f));
            break;

        case OpCode::Id::SLT:
            FORMAT1(_mm_and_ps(_mm_cmplt_ps(src1.raw_f, src2.raw_f), _mm_set1_ps(1.f)));
            break;

        case OpCode::Id::SLTI:
            FORMAT1I(_mm_and_ps(_mm_cmplt_ps(src1.raw_f, src2.raw_f), _mm_set1_ps(1.f)));
            break;

        case OpCode::Id::FLR:
            FORMAT1U(_mm_floor_ps(src1.raw_f));
            break;

        case OpCode::Id::MAX:
            FORMAT1(_mm_max_ps(src1.raw_f, src2.raw_f));
            break;

        case OpCode::Id::MIN:
            FORMAT1(_mm_min_ps(src1.raw_f, src2.raw_f));
            break;

        case OpCode::Id::RCP:
            FORMAT1U(_mm_rcp_ps(src1.raw_f));
            break;

        case OpCode::Id::RSQ:
            FORMAT1U(_mm_rsqrt_ps(src1.raw_f));
            break;

        case OpCode::Id::MOVA:
        {
            Reg src1, temp;
            const SwizzlePattern& swizzle = *(SwizzlePattern*)&swizzle_data[instr.common.operand_desc_id];
            int offset = state.address_offset.raw_i.m128i_i32[instr.common.address_register_index];
            src1.raw_i = NEGATE(SWIZZLE(instr.common.src1 + offset, swizzle.src1_selector), swizzle.negate_src1);
            temp.raw_i.m128i_i32[1] = static_cast<s32>(src1[0]);
            temp.raw_i.m128i_i32[2] = static_cast<s32>(src1[1]);
            __m128i dst_mask = swizzle_dst_mask[(swizzle.dest_mask & 0xc) >> 1];
            state.address_offset.raw_i = _mm_or_si128(_mm_andnot_si128(dst_mask, state.address_offset.raw_i),
                _mm_and_si128(dst_mask, temp.raw_i));
            break;
        }

        case OpCode::Id::MOV:
            FORMAT1U(src1.raw_f);
            break;

        case OpCode::Id::CALL:
            Call(state,
                 instr.flow_control.dest_offset,
                 instr.flow_control.num_instructions,
                 state.pc + 1, 0, 0);
            continue;

        case OpCode::Id::CALLU:
            if (vs.uniforms.b[instr.flow_control.bool_uniform_id]) {
                Call(state,
                     instr.flow_control.dest_offset,
                     instr.flow_control.num_instructions,
                     state.pc + 1, 0, 0);
                continue;
            }
            break;

        case OpCode::Id::CALLC:
            if (EvaluateCondition(state, instr.flow_control.refx, instr.flow_control.refy, instr.flow_control)) {
                Call(state,
                     instr.flow_control.dest_offset,
                     instr.flow_control.num_instructions,
                     state.pc + 1, 0, 0);
                continue;
            }
            break;

        case OpCode::Id::NOP:
            break;

        case OpCode::Id::END:
            exit_loop = true;
            break;

        case OpCode::Id::IFU:
            if (vs.uniforms.b[instr.flow_control.bool_uniform_id]) {
                Call(state,
                     state.pc + 1,
                     instr.flow_control.dest_offset - state.pc - 1,
                     instr.flow_control.dest_offset + instr.flow_control.num_instructions, 0, 0);
            } else {
                Call(state,
                     instr.flow_control.dest_offset,
                     instr.flow_control.num_instructions,
                     instr.flow_control.dest_offset + instr.flow_control.num_instructions, 0, 0);
            }
            continue;

        case OpCode::Id::IFC:
            // TODO: Do we need to consider swizzlers here?
            if (EvaluateCondition(state, instr.flow_control.refx, instr.flow_control.refy, instr.flow_control)) {
                Call(state,
                     state.pc + 1,
                     instr.flow_control.dest_offset - state.pc - 1,
                     instr.flow_control.dest_offset + instr.flow_control.num_instructions, 0, 0);
            } else {
                Call(state,
                     instr.flow_control.dest_offset,
                     instr.flow_control.num_instructions,
                     instr.flow_control.dest_offset + instr.flow_control.num_instructions, 0, 0);
            }
            continue;

        case OpCode::Id::LOOP:
            state.address_offset.raw_i.m128i_i32[3] = vs.uniforms.i[instr.flow_control.int_uniform_id].y;

            Call(state,
                state.pc + 1,
                instr.flow_control.dest_offset - state.pc + 1,
                instr.flow_control.dest_offset + 1,
                vs.uniforms.i[instr.flow_control.int_uniform_id].x,
                vs.uniforms.i[instr.flow_control.int_uniform_id].z);

            continue;

        case OpCode::Id::JMPC:
            if (EvaluateCondition(state, instr.flow_control.refx, instr.flow_control.refy, instr.flow_control)) {
                state.pc = instr.flow_control.dest_offset;
                continue;
            }
            break;

        case OpCode::Id::JMPU:
            if (vs.uniforms.b[instr.flow_control.bool_uniform_id]) {
                state.pc = instr.flow_control.dest_offset;
                continue;
            }
            break;

        case OpCode::Id::CMP:
        case OpCode::Id::CMP + 1:
        {
            Reg src1, src2;
            const SwizzlePattern& swizzle = *(SwizzlePattern*)&swizzle_data[instr.common.operand_desc_id];
            int offset = state.address_offset.raw_i.m128i_i32[instr.common.address_register_index];
            src1.raw_i = NEGATE(SWIZZLE(instr.common.src1 + offset, swizzle.src1_selector), swizzle.negate_src1);
            src2.raw_i = NEGATE(SWIZZLE(instr.common.src2, swizzle.src2_selector), swizzle.negate_src2);
            state.conditional_code[0] = Compare(instr.common.compare_op.x, src1.x, src2.x);
            state.conditional_code[1] = Compare(instr.common.compare_op.y, src1.y, src2.y);
            break;
        }

        case OpCode::Id::MADI:
        case OpCode::Id::MADI + 1:
        case OpCode::Id::MADI + 2:
        case OpCode::Id::MADI + 3:
        case OpCode::Id::MADI + 4:
        case OpCode::Id::MADI + 5:
        case OpCode::Id::MADI + 6:
        case OpCode::Id::MADI + 7:
            FORMAT5I(_mm_add_ps(_mm_mul_ps(src1.raw_f, src2.raw_f), src3.raw_f));
            break;

        case OpCode::Id::MAD:
        case OpCode::Id::MAD + 1:
        case OpCode::Id::MAD + 2:
        case OpCode::Id::MAD + 3:
        case OpCode::Id::MAD + 4:
        case OpCode::Id::MAD + 5:
        case OpCode::Id::MAD + 6:
        case OpCode::Id::MAD + 7:
            FORMAT5(_mm_add_ps(_mm_mul_ps(src1.raw_f, src2.raw_f), src3.raw_f));
            break;

        default:
            LOG_CRITICAL(HW_GPU, "Unhandled opcode: 0x%02x", instr.opcode.Value());
            UNIMPLEMENTED();
        }

        state.pc += 1;

        if (exit_loop)
            break;
    }
#else
    UNREACHABLE();
#endif

    // Setup output data
    VertexShader::OutputVertex ret;
    // TODO(neobrain): Under some circumstances, up to 16 attributes may be output. We need to
    // figure out what those circumstances are and enable the remaining outputs then.
    for (int i = 0; i < 7; ++i) {
        const auto& output_register_map = regs.vs_output_attributes[i];

        u32 semantics[4] = {
            output_register_map.map_x, output_register_map.map_y,
            output_register_map.map_z, output_register_map.map_w
        };

        for (int comp = 0; comp < 4; ++comp) {
            f32* out = ((f32*)&ret) + semantics[comp];
            *out = state.output[i][comp];
        }
    }

    return ret;
}

} // namespace VertexShaderSIMD

} // namespace Pica
