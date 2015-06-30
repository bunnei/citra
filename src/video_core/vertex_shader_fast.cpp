//#include "video_core/vertex_shader.h"

#include <stack>

#include <nihstro/shader_bytecode.h>

#include "video_core/vertex_shader_fast.h"

namespace Pica {

enum OpCode : u32 {
    ADD = 0x00,
    DP3 = 0x01,
    DP4 = 0x02,
    DPH = 0x03,
    EX2 = 0x05,
    LG2 = 0x06,
    MUL = 0x08,
    SGE = 0x09,
    SLT = 0x0A,
    FLR = 0x0B,
    MAX = 0x0C,
    MIN = 0x0D,
    RCP = 0x0E,
    RSQ = 0x0F,
    MOVA = 0x12,
    MOV = 0x13,
    DPHI = 0x18,
    SGEI = 0x1A,
    SLTI = 0x1B,
    NOP = 0x21,
    END = 0x22,
    BREAKC = 0x23,
    CALL = 0x24,
    CALLC = 0x25,
    CALLU = 0x26,
    IFU = 0x27,
    IFC = 0x28,
    LOOP = 0x29,
    EMIT = 0x2A,
    SETEMIT = 0x2B,
    JMPC = 0x2C,
    JMPU = 0x2D,
    CMP = 0x2E,
    MADI = 0x30,
    MAD = 0x38,
};

#pragma pack(1)
union Instruction {
    Instruction& operator =(const Instruction& instr) {
        hex = instr.hex;
        return *this;
    }

    u32 hex;

    BitField<0x1a, 0x6, OpCode> opcode;


    // General notes:
    //
    // When two input registers are used, one of them uses a 5-bit index while the other
    // one uses a 7-bit index. This is because at most one floating point uniform may be used
    // as an input.


    // Format used e.g. by arithmetic instructions and comparisons
    union Common { // TODO: Remove name
        BitField<0x00, 0x7, u32> operand_desc_id;

        const u32 GetSrc1(bool is_inverted) const {
            if (!is_inverted) {
                return src1;
            } else {
                return src1i;
            }
        }

        const u32 GetSrc2(bool is_inverted) const {
            if (!is_inverted) {
                return src2;
            } else {
                return src2i;
            }
        }

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

        union CompareOpType {  // TODO: Make nameless once MSVC supports it
            enum Op : u32 {
                Equal        = 0,
                NotEqual     = 1,
                LessThan     = 2,
                LessEqual    = 3,
                GreaterThan  = 4,
                GreaterEqual = 5,
                Unk6         = 6,
                Unk7         = 7
            };

            BitField<0x15, 0x3, Op> y;
            BitField<0x18, 0x3, Op> x;

            const std::string ToString(Op op) const {
                switch (op) {
                case Equal:        return "==";
                case NotEqual:     return "!=";
                case LessThan:     return "<";
                case LessEqual:    return "<=";
                case GreaterThan:  return ">";
                case GreaterEqual: return ">=";
                case Unk6:         return "UNK6";
                case Unk7:         return "UNK7";
                default:           return "";
                };
            }
        } compare_op;

        std::string AddressRegisterName() const {
            if (address_register_index == 0) return "";
            else if (address_register_index == 1) return "a0.x";
            else if (address_register_index == 2) return "a0.y";
            else /*if (address_register_index == 3)*/ return "aL";
        }

        BitField<0x15, 0x5, u32> dest;
    } common;

    union FlowControlType {  // TODO: Make nameless once MSVC supports it
        enum Op : u32 {
            Or    = 0,
            And   = 1,
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
    } mad;
};
static_assert(sizeof(Instruction) == 0x4, "Incorrect structure size");
static_assert(std::is_standard_layout<Instruction>::value, "Structure does not have standard layout");

union SwizzlePattern {
    SwizzlePattern& operator =(const SwizzlePattern& instr) {
        hex = instr.hex;
        return *this;
    }

    u32 hex;

    bool DestComponentEnabled(unsigned int i) const {
        return (dest_mask & (0x8 >> i)) != 0;
    }

    // Components of "dest" that should be written to: LSB=dest.w, MSB=dest.x
    BitField< 0, 4, u32> dest_mask;

    BitField< 4, 1, u32> negate_src1;
    BitField< 5, 2, u32> src1_selector_3;
    BitField< 7, 2, u32> src1_selector_2;
    BitField< 9, 2, u32> src1_selector_1;
    BitField<11, 2, u32> src1_selector_0;

    BitField<13, 1, u32> negate_src2;
    BitField<14, 2, u32> src2_selector_3;
    BitField<16, 2, u32> src2_selector_2;
    BitField<18, 2, u32> src2_selector_1;
    BitField<20, 2, u32> src2_selector_0;

    BitField<22, 1, u32> negate_src3;
    BitField<23, 2, u32> src3_selector_3;
    BitField<25, 2, u32> src3_selector_2;
    BitField<27, 2, u32> src3_selector_1;
    BitField<29, 2, u32> src3_selector_0;
};
static_assert(sizeof(SwizzlePattern) == 0x4, "Incorrect structure size");

namespace VertexShaderFast {

enum RegisterOffset {
    Input = 0,
    Temporary = 16,
    Uniform = 32,
};

struct State {
    const u32* pc;

    s32 address_registers[3];
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
    std::stack<CallStackElement> call_stack;
};

VertexShader::OutputVertex RunShader(const VertexShader::InputVertex& input, int num_attributes) {
    const auto& regs = g_state.regs;
    auto& vs = g_state.vs;
    const auto& swizzle_data = g_state.vs.swizzle_data;
    const auto& program_code = g_state.vs.program_code;
    bool exit_loop = false;
    State state;

    //for (int i = 0; i < 96; i++) {
    //    state.regs[RegisterOffset::Uniform + i] = Register(uniforms.f[i].x,
    //                                                       uniforms.f[i].y,
    //                                                       uniforms.f[i].z,
    //                                                       uniforms.f[i].w);
    //}

    const u32* main = &vs.program_code[regs.vs_main_offset];
    state.pc = (u32*)main;
    state.conditional_code[0] = false;
    state.conditional_code[1] = false;

    const auto& reg_map = regs.vs_input_register_map;

    for (int i = 0; i < num_attributes; ++i) {
        auto& reg = vs.regs[reg_map.GetRegisterForAttribute(i)];
        reg[0] = input.attr[i].x.ToFloat32();
        reg[1] = input.attr[i].y.ToFloat32();
        reg[2] = input.attr[i].z.ToFloat32();
        reg[3] = input.attr[i].w.ToFloat32();
    }

#define SWIZZLE_SRC_1_(x) {x[swizzle.src1_selector_0], x[swizzle.src1_selector_1], x[swizzle.src1_selector_2], x[swizzle.src1_selector_3]}
#define SWIZZLE_SRC_2_(x) {x[swizzle.src2_selector_0], x[swizzle.src2_selector_1], x[swizzle.src2_selector_2], x[swizzle.src2_selector_3]}
#define SWIZZLE_SRC_3_(x) {x[swizzle.src3_selector_0], x[swizzle.src3_selector_1], x[swizzle.src3_selector_2], x[swizzle.src3_selector_3]}

#define SWIZZLE_MAD_SRC_1_(x) {x[swizzle_mad.src1_selector_0], x[swizzle_mad.src1_selector_1], x[swizzle_mad.src1_selector_2], x[swizzle_mad.src1_selector_3]}
#define SWIZZLE_MAD_SRC_2_(x) {x[swizzle_mad.src2_selector_0], x[swizzle_mad.src2_selector_1], x[swizzle_mad.src2_selector_2], x[swizzle_mad.src2_selector_3]}
#define SWIZZLE_MAD_SRC_3_(x) {x[swizzle_mad.src3_selector_0], x[swizzle_mad.src3_selector_1], x[swizzle_mad.src3_selector_2], x[swizzle_mad.src3_selector_3]}

#define ADDR_OFFS_ ((instr.common.address_register_index == 0) ? 0 : state.address_registers[instr.common.address_register_index - 1])
#define SRC1 SWIZZLE_SRC_1_(vs.regs[instr.common.src1 + ADDR_OFFS_])
#define SRC2 SWIZZLE_SRC_2_(vs.regs[instr.common.src2])

#define SRC1_INV SWIZZLE_SRC_1_(vs.regs[instr.common.src1i])
#define SRC2_INV SWIZZLE_SRC_2_(vs.regs[instr.common.src2i + ADDR_OFFS_])

#define SRC1_MAD SWIZZLE_MAD_SRC_1_(vs.regs[instr.mad.src1])
#define SRC2_MAD SWIZZLE_MAD_SRC_2_(vs.regs[instr.mad.src2])
#define SRC3_MAD SWIZZLE_MAD_SRC_3_(vs.regs[instr.mad.src3])

#define DST state.regs[instr.common.dest];

    static auto evaluate_condition = [](const State& state, bool refx, bool refy, Instruction::FlowControlType flow_control) {
        bool results[2] = { refx == state.conditional_code[0],
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
    };

    while (true) {

        if (!state.call_stack.empty()) {
            auto& top = state.call_stack.top();
            if (state.pc - program_code.data() == top.final_address) {
                state.address_registers[2] += top.loop_increment;

                if (top.repeat_counter-- == 0) {
                    state.pc = &program_code[top.return_address];
                    state.call_stack.pop();
                }
                else {
                    state.pc = &program_code[top.loop_address];
                }

                // TODO: Is "trying again" accurate to hardware?
                continue;
            }
        }

        static auto call = [&program_code](State& state, u32 offset, u32 num_instructions,
            u32 return_offset, u8 repeat_count, u8 loop_increment) {
            state.pc = &program_code[offset];
            state.call_stack.push({ offset + num_instructions, return_offset, repeat_count, loop_increment, offset });
        };

        const Instruction& instr = *(const Instruction*)state.pc;
        const SwizzlePattern& swizzle = *(SwizzlePattern*)&swizzle_data[instr.common.operand_desc_id];
        const SwizzlePattern& swizzle_mad = *(SwizzlePattern*)&swizzle_data[instr.mad.operand_desc_id];

        u32 binary_offset = state.pc - program_code.data();

        switch (instr.opcode) {
        case OpCode::ADD:
        {
            f32 src1[] = SRC1;
            f32 src2[] = SRC2;

            for (int i = 0; i < 4; ++i) {
                if (!swizzle.DestComponentEnabled(i))
                    continue;

                vs.regs[instr.common.dest][i] = src1[i] + src2[i];
            }
            break;
        }

        case OpCode::DP3:
        case OpCode::DP4:
        {
            f32 dot = 0.f;
            f32 src1[] = SRC1;
            f32 src2[] = SRC2;

            int num_components = (instr.opcode == OpCode::DP3) ? 3 : 4;

            for (int i = 0; i < num_components; ++i)
                dot = dot + src1[i] * src2[i];

            for (int i = 0; i < num_components; ++i) {
                if (!swizzle.DestComponentEnabled(i))
                    continue;

                vs.regs[instr.common.dest][i] = dot;
            }
            break;
        }

        case OpCode::MUL:
        {
            f32 src1[] = SRC1;
            f32 src2[] = SRC2;

            for (int i = 0; i < 4; ++i) {
                if (!swizzle.DestComponentEnabled(i))
                    continue;

                vs.regs[instr.common.dest][i] = src1[i] * src2[i];
            }
            break;
        }

        case OpCode::MAX:
        {
            f32 src1[] = SRC1;
            f32 src2[] = SRC2;

            for (int i = 0; i < 4; ++i) {
                if (!swizzle.DestComponentEnabled(i))
                    continue;

                vs.regs[instr.common.dest][i] = std::max(src1[i], src2[i]);
            }
            break;
        }

        case OpCode::RCP:
        {
            f32 src1[] = SRC1;

            for (int i = 0; i < 4; ++i) {
                if (!swizzle.DestComponentEnabled(i))
                    continue;

                // TODO: Be stable against division by zero!
                // TODO: I think this might be wrong... we should only use one component here
                vs.regs[instr.common.dest][i] = 1.0f / src1[i];
            }

            break;
        }

        case OpCode::RSQ:
        {
            f32 src1[] = SRC1;

            for (int i = 0; i < 4; ++i) {
                if (!swizzle.DestComponentEnabled(i))
                    continue;

                // TODO: Be stable against division by zero!
                // TODO: I think this might be wrong... we should only use one component here
                vs.regs[instr.common.dest][i] = 1.0f / sqrt(src1[i]);
            }
            break;
        }

        case OpCode::MOVA:
        {
            f32 src1[] = SRC1;
            for (int i = 0; i < 2; ++i) {
                if (!swizzle.DestComponentEnabled(i))
                    continue;

                // TODO: Figure out how the rounding is done on hardware
                state.address_registers[i] = static_cast<s32>(src1[i]);
            }

            break;
        }

        case OpCode::MOV:
        {
            f32 src1[] = SRC1;

            for (int i = 0; i < 4; ++i) {
                if (!swizzle.DestComponentEnabled(i))
                    continue;

                vs.regs[instr.common.dest][i] = src1[i];
            }
            break;
        }

        case OpCode::CALL:
            call(state,
                instr.flow_control.dest_offset,
                instr.flow_control.num_instructions,
                binary_offset + 1, 0, 0);

            continue;

        case OpCode::NOP:
            break;

        case OpCode::END:
            exit_loop = true;
            break;

        case OpCode::IFU:
            if (vs.uniforms_b[instr.flow_control.bool_uniform_id]) {
                call(state,
                    binary_offset + 1,
                    instr.flow_control.dest_offset - binary_offset - 1,
                    instr.flow_control.dest_offset + instr.flow_control.num_instructions, 0, 0);
            }
            else {
                call(state,
                    instr.flow_control.dest_offset,
                    instr.flow_control.num_instructions,
                    instr.flow_control.dest_offset + instr.flow_control.num_instructions, 0, 0);
            }

            continue;

        case OpCode::IFC:
            // TODO: Do we need to consider swizzlers here?

            if (evaluate_condition(state, instr.flow_control.refx, instr.flow_control.refy, instr.flow_control)) {
                call(state,
                    binary_offset + 1,
                    instr.flow_control.dest_offset - binary_offset - 1,
                    instr.flow_control.dest_offset + instr.flow_control.num_instructions, 0, 0);
            } else {
                call(state,
                    instr.flow_control.dest_offset,
                    instr.flow_control.num_instructions,
                    instr.flow_control.dest_offset + instr.flow_control.num_instructions, 0, 0);
            }

            continue;

        case OpCode::CMP:
        case OpCode::CMP + 1:
        {
            f32 src1[] = SRC1;
            f32 src2[] = SRC2;
            for (int i = 0; i < 2; ++i) {
                // TODO: Can you restrict to one compare via dest masking?

                auto compare_op = instr.common.compare_op;
                auto op = (i == 0) ? compare_op.x.Value() : compare_op.y.Value();

                switch (op) {
                case compare_op.Equal:
                    state.conditional_code[i] = (src1[i] == src2[i]);
                    break;

                case compare_op.NotEqual:
                    state.conditional_code[i] = (src1[i] != src2[i]);
                    break;

                case compare_op.LessThan:
                    state.conditional_code[i] = (src1[i] < src2[i]);
                    break;

                case compare_op.LessEqual:
                    state.conditional_code[i] = (src1[i] <= src2[i]);
                    break;

                case compare_op.GreaterThan:
                    state.conditional_code[i] = (src1[i] > src2[i]);
                    break;

                case compare_op.GreaterEqual:
                    state.conditional_code[i] = (src1[i] >= src2[i]);
                    break;

                default:
                    LOG_ERROR(HW_GPU, "Unknown compare mode %x", static_cast<int>(op));
                    break;
                }
            }

            break;
        }

        case OpCode::MAD:
        case OpCode::MAD + 1:
        case OpCode::MAD + 2:
        case OpCode::MAD + 3:
        case OpCode::MAD + 4:
        case OpCode::MAD + 5:
        case OpCode::MAD + 6:
        case OpCode::MAD + 7:
        {
            f32 src1[] = SRC1_MAD;
            f32 src2[] = SRC2_MAD;
            f32 src3[] = SRC3_MAD;

            for (int i = 0; i < 4; ++i) {
                if (!swizzle_mad.DestComponentEnabled(i))
                    continue;

                vs.regs[instr.mad.dest][i] = src1[i] * src2[i] + src3[i];
            }

            break;
        }

        default:
            LOG_CRITICAL(HW_GPU, "Unhandled opcode: 0x%02x", instr.opcode.Value());
            UNIMPLEMENTED();
        }

        ++state.pc;

        if (exit_loop)
            break;
    }

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
            float24* out = ((float24*)&ret) + semantics[comp];
            if (semantics[comp] != Regs::VSOutputAttributes::INVALID) {
                *out = float24::FromFloat32(vs.regs[i][comp]);
            } else {
                // Zero output so that attributes which aren't output won't have denormals in them,
                // which would slow us down later.
                memset(out, 0, sizeof(*out));
            }
        }
    }

    //LOG_ERROR(Render_Software, "2: output vertex: pos (%.2f, %.2f, %.2f, %.2f), col(%.2f, %.2f, %.2f, %.2f), tc0(%.2f, %.2f)",
    //    ret.pos.x.ToFloat32(), ret.pos.y.ToFloat32(), ret.pos.z.ToFloat32(), ret.pos.w.ToFloat32(),
    //    ret.color.x.ToFloat32(), ret.color.y.ToFloat32(), ret.color.z.ToFloat32(), ret.color.w.ToFloat32(),
    //    ret.tc0.u().ToFloat32(), ret.tc0.v().ToFloat32());


    return ret;
}

} // namespace VertexShaderFast

} // namespace Pica
