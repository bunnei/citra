#include <array>
#include <memory>
#include <unordered_map>
#include <vector>

#include "common/hash.h"
#include "common/make_unique.h"

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
        } compare_op;

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

////////////////////////////////////////////////////////////////////////////////////////////////////

struct State {
    u32 pc;
    s32 address_offset[4];
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
};


////////////////////////////////////////////////////////////////////////////////////////////////////

struct SrcOperand {
    const f32* x;
    const f32* y;
    const f32* z;
    const f32* w;

    f32 sign;

    const f32* operator [] (int i) {
        return *(&x + i);
    }
};

struct DstOperand {
    f32* x;
    f32* y;
    f32* z;
    f32* w;

    f32* operator [] (int i) {
        return *(&x + i);
    }
};

struct InstructionInfo {
    SrcOperand src1;
    SrcOperand src2;
    SrcOperand src3;
    DstOperand dest;

    f32 sign; // src1 * src2 sign, minor optimization for some instructions
};

struct ShaderInfo {
    std::array<InstructionInfo, 1024> instructions;
};

static inline void Call(State& state, u32 offset, u32 num_instructions, u32 return_offset, u8 repeat_count, u8 loop_increment) {
    state.pc = offset;
    state.call_stack.push_back({ offset + num_instructions, return_offset, repeat_count, loop_increment, offset });
};

static inline bool EvaluateCondition(const State& state, bool refx, bool refy, Instruction::FlowControlType flow_control) {
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
};

void DecodeShader(ShaderInfo* shader) {
    static f32 dummy_reg = 0.f;
    auto& vs = g_state.vs;
    const auto& swizzle_data = vs.swizzle_data;
    const auto& program_code = vs.program_code;

    for (int i = 0; i < program_code.size(); i++) {
        const Instruction& instr = *(const Instruction*)&program_code[i];

        switch (instr.opcode) {

        // Format 1
        case OpCode::ADD:
        case OpCode::DP4:
        case OpCode::DPH:
        case OpCode::MUL:
        case OpCode::SGE:
        case OpCode::SLT:
        case OpCode::MAX:
        case OpCode::MIN:
        {
            auto swizzle = *(SwizzlePattern*)&swizzle_data[instr.common.operand_desc_id];

            SrcOperand src1 = { &vs.InputReg(instr.common.src1)[swizzle.src1_selector_0],
                                &vs.InputReg(instr.common.src1)[swizzle.src1_selector_1],
                                &vs.InputReg(instr.common.src1)[swizzle.src1_selector_2],
                                &vs.InputReg(instr.common.src1)[swizzle.src1_selector_3],
                                swizzle.negate_src1 ? -1.f : 1.f };

            SrcOperand src2 = { &vs.InputReg(instr.common.src2)[swizzle.src2_selector_0],
                                &vs.InputReg(instr.common.src2)[swizzle.src2_selector_1],
                                &vs.InputReg(instr.common.src2)[swizzle.src2_selector_2],
                                &vs.InputReg(instr.common.src2)[swizzle.src2_selector_3],
                                swizzle.negate_src2 ? -1.f : 1.f };

            DstOperand dest = { swizzle.DestComponentEnabled(0) ? &vs.regs[instr.common.dest][0] : &dummy_reg,
                                swizzle.DestComponentEnabled(1) ? &vs.regs[instr.common.dest][1] : &dummy_reg,
                                swizzle.DestComponentEnabled(2) ? &vs.regs[instr.common.dest][2] : &dummy_reg,
                                swizzle.DestComponentEnabled(3) ? &vs.regs[instr.common.dest][3] : &dummy_reg };

            shader->instructions[i] = { src1, src2, {}, dest, src1.sign * src2.sign };

            continue;
        }

        case OpCode::DP3:
        {
            auto swizzle = *(SwizzlePattern*)&swizzle_data[instr.common.operand_desc_id];

            SrcOperand src1 = { &vs.InputReg(instr.common.src1)[swizzle.src1_selector_0],
                                &vs.InputReg(instr.common.src1)[swizzle.src1_selector_1],
                                &vs.InputReg(instr.common.src1)[swizzle.src1_selector_2],
                                &dummy_reg,
                                swizzle.negate_src1 ? -1.f : 1.f };

            SrcOperand src2 = { &vs.InputReg(instr.common.src2)[swizzle.src2_selector_0],
                                &vs.InputReg(instr.common.src2)[swizzle.src2_selector_1],
                                &vs.InputReg(instr.common.src2)[swizzle.src2_selector_2],
                                &dummy_reg,
                                swizzle.negate_src2 ? -1.f : 1.f };

            DstOperand dest = { swizzle.DestComponentEnabled(0) ? &vs.regs[instr.common.dest][0] : &dummy_reg,
                                swizzle.DestComponentEnabled(1) ? &vs.regs[instr.common.dest][1] : &dummy_reg,
                                swizzle.DestComponentEnabled(2) ? &vs.regs[instr.common.dest][2] : &dummy_reg,
                                &dummy_reg };

            shader->instructions[i] = { src1, src2, {}, dest, src1.sign * src2.sign };

            continue;
        }

        // Format 1u
        case OpCode::EX2:
        case OpCode::LG2:
        case OpCode::FLR:
        case OpCode::RCP:
        case OpCode::RSQ:
        case OpCode::MOVA:
        case OpCode::MOV:
        {
            auto swizzle = *(SwizzlePattern*)&swizzle_data[instr.common.operand_desc_id];

            SrcOperand src1 = { &vs.InputReg(instr.common.src1)[swizzle.src1_selector_0],
                                &vs.InputReg(instr.common.src1)[swizzle.src1_selector_1],
                                &vs.InputReg(instr.common.src1)[swizzle.src1_selector_2],
                                &vs.InputReg(instr.common.src1)[swizzle.src1_selector_3],
                                swizzle.negate_src1 ? -1.f : 1.f };

            DstOperand dest = { swizzle.DestComponentEnabled(0) ? &vs.regs[instr.common.dest][0] : &dummy_reg,
                                swizzle.DestComponentEnabled(1) ? &vs.regs[instr.common.dest][1] : &dummy_reg,
                                swizzle.DestComponentEnabled(2) ? &vs.regs[instr.common.dest][2] : &dummy_reg,
                                swizzle.DestComponentEnabled(3) ? &vs.regs[instr.common.dest][3] : &dummy_reg };

            shader->instructions[i] = { src1, {}, {}, dest };

            continue;
        }

        // Format 1i
        case OpCode::DPHI:
        case OpCode::SGEI:
        case OpCode::SLTI:
        {
            auto swizzle = *(SwizzlePattern*)&swizzle_data[instr.common.operand_desc_id];

            SrcOperand src1 = { &vs.InputReg(instr.common.src1i)[swizzle.src1_selector_0],
                                &vs.InputReg(instr.common.src1i)[swizzle.src1_selector_1],
                                &vs.InputReg(instr.common.src1i)[swizzle.src1_selector_2],
                                &vs.InputReg(instr.common.src1i)[swizzle.src1_selector_3],
                                swizzle.negate_src1 ? -1.f : 1.f };

            SrcOperand src2 = { &vs.InputReg(instr.common.src2i)[swizzle.src2_selector_0],
                                &vs.InputReg(instr.common.src2i)[swizzle.src2_selector_1],
                                &vs.InputReg(instr.common.src2i)[swizzle.src2_selector_2],
                                &vs.InputReg(instr.common.src2i)[swizzle.src2_selector_3],
                                swizzle.negate_src2 ? -1.f : 1.f };

            DstOperand dest = { swizzle.DestComponentEnabled(0) ? &vs.regs[instr.common.dest][0] : &dummy_reg,
                                swizzle.DestComponentEnabled(1) ? &vs.regs[instr.common.dest][1] : &dummy_reg,
                                swizzle.DestComponentEnabled(2) ? &vs.regs[instr.common.dest][2] : &dummy_reg,
                                swizzle.DestComponentEnabled(3) ? &vs.regs[instr.common.dest][3] : &dummy_reg };

            shader->instructions[i] = { src1, src2, {}, dest };

            continue;
        }

        // Format 0
        case OpCode::NOP:
        case OpCode::END:
        case OpCode::EMIT:

        // Format 2
        case OpCode::BREAKC:
        case OpCode::CALL:
        case OpCode::CALLC:
        case OpCode::IFC:
        case OpCode::JMPC:

        // Format 3
        case OpCode::CALLU:
        case OpCode::IFU:
        case OpCode::LOOP:
        case OpCode::JMPU:

        // Format 4
        case OpCode::SETEMIT:
            continue;

        // Format 1c
        case OpCode::CMP:
        case OpCode::CMP + 1:
        {
            auto swizzle = *(SwizzlePattern*)&swizzle_data[instr.common.operand_desc_id];

            SrcOperand src1 = { &vs.InputReg(instr.common.src1)[swizzle.src1_selector_0],
                                &vs.InputReg(instr.common.src1)[swizzle.src1_selector_1],
                                &vs.InputReg(instr.common.src1)[swizzle.src1_selector_2],
                                &vs.InputReg(instr.common.src1)[swizzle.src1_selector_3],
                                swizzle.negate_src1 ? -1.f : 1.f };

            SrcOperand src2 = { &vs.InputReg(instr.common.src2)[swizzle.src2_selector_0],
                                &vs.InputReg(instr.common.src2)[swizzle.src2_selector_1],
                                &vs.InputReg(instr.common.src2)[swizzle.src2_selector_2],
                                &vs.InputReg(instr.common.src2)[swizzle.src2_selector_3],
                                swizzle.negate_src2 ? -1.f : 1.f };

            shader->instructions[i] = { src1, src2, {}, {} };
            continue;
        }

        // Format 5i
        case OpCode::MADI:
        case OpCode::MADI + 1:
        case OpCode::MADI + 2:
        case OpCode::MADI + 3:
        case OpCode::MADI + 4:
        case OpCode::MADI + 5:
        case OpCode::MADI + 6:
        case OpCode::MADI + 7:
        {
            const SwizzlePattern& swizzle = *(SwizzlePattern*)&swizzle_data[instr.mad.operand_desc_id];

            SrcOperand src1 = { &vs.InputReg(instr.mad.src1)[swizzle.src1_selector_0],
                                &vs.InputReg(instr.mad.src1)[swizzle.src1_selector_1],
                                &vs.InputReg(instr.mad.src1)[swizzle.src1_selector_2],
                                &vs.InputReg(instr.mad.src1)[swizzle.src1_selector_3],
                                swizzle.negate_src1 ? -1.f : 1.f };

            SrcOperand src2 = { &vs.InputReg(instr.mad.src2i)[swizzle.src2_selector_0],
                                &vs.InputReg(instr.mad.src2i)[swizzle.src2_selector_1],
                                &vs.InputReg(instr.mad.src2i)[swizzle.src2_selector_2],
                                &vs.InputReg(instr.mad.src2i)[swizzle.src2_selector_3],
                                swizzle.negate_src2 ? -1.f : 1.f };

            SrcOperand src3 = { &vs.InputReg(instr.mad.src3i)[swizzle.src3_selector_0],
                                &vs.InputReg(instr.mad.src3i)[swizzle.src3_selector_1],
                                &vs.InputReg(instr.mad.src3i)[swizzle.src3_selector_2],
                                &vs.InputReg(instr.mad.src3i)[swizzle.src3_selector_3],
                                swizzle.negate_src3 ? -1.f : 1.f };

            DstOperand dest = { swizzle.DestComponentEnabled(0) ? &vs.regs[instr.mad.dest][0] : &dummy_reg,
                                swizzle.DestComponentEnabled(1) ? &vs.regs[instr.mad.dest][1] : &dummy_reg,
                                swizzle.DestComponentEnabled(2) ? &vs.regs[instr.mad.dest][2] : &dummy_reg,
                                swizzle.DestComponentEnabled(3) ? &vs.regs[instr.mad.dest][3] : &dummy_reg };

            shader->instructions[i] = { src1, src2, src3, dest, src1.sign * src2.sign };

            continue;
        }

        // Format 5
        case OpCode::MAD:
        case OpCode::MAD + 1:
        case OpCode::MAD + 2:
        case OpCode::MAD + 3:
        case OpCode::MAD + 4:
        case OpCode::MAD + 5:
        case OpCode::MAD + 6:
        case OpCode::MAD + 7:
        {
            const SwizzlePattern& swizzle = *(SwizzlePattern*)&swizzle_data[instr.mad.operand_desc_id];

            SrcOperand src1 = { &vs.InputReg(instr.mad.src1)[swizzle.src1_selector_0],
                                &vs.InputReg(instr.mad.src1)[swizzle.src1_selector_1],
                                &vs.InputReg(instr.mad.src1)[swizzle.src1_selector_2],
                                &vs.InputReg(instr.mad.src1)[swizzle.src1_selector_3],
                                swizzle.negate_src1 ? -1.f : 1.f };

            SrcOperand src2 = { &vs.InputReg(instr.mad.src2)[swizzle.src2_selector_0],
                                &vs.InputReg(instr.mad.src2)[swizzle.src2_selector_1],
                                &vs.InputReg(instr.mad.src2)[swizzle.src2_selector_2],
                                &vs.InputReg(instr.mad.src2)[swizzle.src2_selector_3],
                                swizzle.negate_src2 ? -1.f : 1.f };

            SrcOperand src3 = { &vs.InputReg(instr.mad.src3)[swizzle.src3_selector_0],
                                &vs.InputReg(instr.mad.src3)[swizzle.src3_selector_1],
                                &vs.InputReg(instr.mad.src3)[swizzle.src3_selector_2],
                                &vs.InputReg(instr.mad.src3)[swizzle.src3_selector_3],
                                swizzle.negate_src3 ? -1.f : 1.f };

            DstOperand dest = { swizzle.DestComponentEnabled(0) ? &vs.regs[instr.mad.dest][0] : &dummy_reg,
                                swizzle.DestComponentEnabled(1) ? &vs.regs[instr.mad.dest][1] : &dummy_reg,
                                swizzle.DestComponentEnabled(2) ? &vs.regs[instr.mad.dest][2] : &dummy_reg,
                                swizzle.DestComponentEnabled(3) ? &vs.regs[instr.mad.dest][3] : &dummy_reg };

            shader->instructions[i] = { src1, src2, src3, dest, src1.sign * src2.sign };

            continue;
        }

        default:
            continue;
        }
    }
}

std::unordered_map<u64, std::unique_ptr<ShaderInfo>> vertex_shader_cache;

VertexShader::OutputVertex RunShader(const VertexShader::InputVertex& input, int num_attributes) {
    const auto& regs = g_state.regs;
    auto& vs = g_state.vs;
    const auto& swizzle_data = g_state.vs.swizzle_data;
    const auto& program_code = g_state.vs.program_code;
    bool exit_loop = false;
    State state;

    state.pc = regs.vs_main_offset;
    state.address_offset[0] = 0;
    state.conditional_code[0] = false;
    state.conditional_code[1] = false;

    const auto& reg_map = regs.vs_input_register_map;

    for (int i = 0; i < num_attributes; ++i) {
        f32* reg = vs.inputs[reg_map.GetRegisterForAttribute(i)];
        reg[0] = input.attr[i].x.ToFloat32();
        reg[1] = input.attr[i].y.ToFloat32();
        reg[2] = input.attr[i].z.ToFloat32();
        reg[3] = input.attr[i].w.ToFloat32();
    }

    u64 cache_key = Common::GetCRC32((const u8*)vs.program, 8192, 128);
    auto cached_shader = vertex_shader_cache.find(cache_key);
    ShaderInfo* shader_info;

    if (cached_shader != vertex_shader_cache.end()) {
        shader_info = cached_shader->second.get();
    } else {
        std::unique_ptr<ShaderInfo> new_shader_info = Common::make_unique<ShaderInfo>();
        shader_info = new_shader_info.get();
        DecodeShader(shader_info);
        vertex_shader_cache.emplace(cache_key, std::move(new_shader_info));
    }

    while (true) {

        if (!state.call_stack.empty()) {
            auto& top = state.call_stack.back();
            if (&program_code[state.pc] - program_code.data() == top.final_address) {
                state.address_offset[2] += top.loop_increment << 2;

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
        InstructionInfo instr_info = shader_info->instructions[state.pc];

        switch (instr.opcode) {
        case OpCode::ADD:
        {
            int offset = state.address_offset[instr.common.address_register_index];

            instr_info.dest.x[0] = instr_info.src1.sign * instr_info.src1.x[offset] + instr_info.src2.sign * instr_info.src2.x[0];
            instr_info.dest.y[0] = instr_info.src1.sign * instr_info.src1.y[offset] + instr_info.src2.sign * instr_info.src2.y[0];
            instr_info.dest.z[0] = instr_info.src1.sign * instr_info.src1.z[offset] + instr_info.src2.sign * instr_info.src2.z[0];
            instr_info.dest.w[0] = instr_info.src1.sign * instr_info.src1.w[offset] + instr_info.src2.sign * instr_info.src2.w[0];

            break;
        }

        case OpCode::DP3:
        {
            int offset = state.address_offset[instr.common.address_register_index];
            f32 dot = 0.f;

            for (int i = 0; i < 3; ++i)
                dot += instr_info.sign * instr_info.src1[i][offset] * instr_info.src2[i][0];

            instr_info.dest.x[0] = instr_info.dest.y[0] = instr_info.dest.z[0] = dot;

            break;
        }

        case OpCode::DP4:
        {
            int offset = state.address_offset[instr.common.address_register_index];
            f32 dot = 0.f;

            for (int i = 0; i < 4; ++i)
                dot += instr_info.sign * instr_info.src1[i][offset] * instr_info.src2[i][0];

            instr_info.dest.x[0] = instr_info.dest.y[0] = instr_info.dest.z[0] = instr_info.dest.w[0] = dot;

            break;
        }

        case OpCode::MUL:
        {
            int offset = state.address_offset[instr.common.address_register_index];

            instr_info.dest.x[0] = instr_info.sign * instr_info.src1.x[offset] * instr_info.src2.x[0];
            instr_info.dest.y[0] = instr_info.sign * instr_info.src1.y[offset] * instr_info.src2.y[0];
            instr_info.dest.z[0] = instr_info.sign * instr_info.src1.z[offset] * instr_info.src2.z[0];
            instr_info.dest.w[0] = instr_info.sign * instr_info.src1.w[offset] * instr_info.src2.w[0];

            break;
        }

        case OpCode::SLT:
        {
            int offset = state.address_offset[instr.common.address_register_index];

            instr_info.dest.x[0] = (instr_info.src1.x[offset] < instr_info.src2.x[0]) ? 1.f : 0.f;
            instr_info.dest.y[0] = (instr_info.src1.y[offset] < instr_info.src2.y[0]) ? 1.f : 0.f;
            instr_info.dest.z[0] = (instr_info.src1.z[offset] < instr_info.src2.z[0]) ? 1.f : 0.f;
            instr_info.dest.w[0] = (instr_info.src1.w[offset] < instr_info.src2.w[0]) ? 1.f : 0.f;

            break;
        }

        case OpCode::SLTI:
        {
            int offset = state.address_offset[instr.common.address_register_index];

            instr_info.dest.x[0] = (instr_info.src1.sign * instr_info.src1.x[0] < instr_info.src2.sign * instr_info.src2.x[offset]) ? 1.f : 0.f;
            instr_info.dest.y[0] = (instr_info.src1.sign * instr_info.src1.y[0] < instr_info.src2.sign * instr_info.src2.y[offset]) ? 1.f : 0.f;
            instr_info.dest.z[0] = (instr_info.src1.sign * instr_info.src1.z[0] < instr_info.src2.sign * instr_info.src2.z[offset]) ? 1.f : 0.f;
            instr_info.dest.w[0] = (instr_info.src1.sign * instr_info.src1.w[0] < instr_info.src2.sign * instr_info.src2.w[offset]) ? 1.f : 0.f;

            break;
        }

        case OpCode::FLR:
        {
            int offset = state.address_offset[instr.common.address_register_index];

            instr_info.dest.x[0] = std::floor(instr_info.src1.sign * instr_info.src1.x[offset]);
            instr_info.dest.y[0] = std::floor(instr_info.src1.sign * instr_info.src1.y[offset]);
            instr_info.dest.z[0] = std::floor(instr_info.src1.sign * instr_info.src1.z[offset]);
            instr_info.dest.w[0] = std::floor(instr_info.src1.sign * instr_info.src1.w[offset]);

            break;
        }

        case OpCode::MAX:
        {
            int offset = state.address_offset[instr.common.address_register_index];

            instr_info.dest.x[0] = std::max(instr_info.src1.sign * instr_info.src1.x[offset], instr_info.src2.sign * instr_info.src2.x[0]);
            instr_info.dest.y[0] = std::max(instr_info.src1.sign * instr_info.src1.y[offset], instr_info.src2.sign * instr_info.src2.y[0]);
            instr_info.dest.z[0] = std::max(instr_info.src1.sign * instr_info.src1.z[offset], instr_info.src2.sign * instr_info.src2.z[0]);
            instr_info.dest.w[0] = std::max(instr_info.src1.sign * instr_info.src1.w[offset], instr_info.src2.sign * instr_info.src2.w[0]);

            break;
        }

        case OpCode::MIN:
        {
            int offset = state.address_offset[instr.common.address_register_index];

            instr_info.dest.x[0] = std::min(instr_info.src1.sign * instr_info.src1.x[offset], instr_info.src2.sign * instr_info.src2.x[0]);
            instr_info.dest.y[0] = std::min(instr_info.src1.sign * instr_info.src1.y[offset], instr_info.src2.sign * instr_info.src2.y[0]);
            instr_info.dest.z[0] = std::min(instr_info.src1.sign * instr_info.src1.z[offset], instr_info.src2.sign * instr_info.src2.z[0]);
            instr_info.dest.w[0] = std::min(instr_info.src1.sign * instr_info.src1.w[offset], instr_info.src2.sign * instr_info.src2.w[0]);

            break;
        }

        case OpCode::RCP:
        {
            int offset = state.address_offset[instr.common.address_register_index];

            // TODO: Be stable against division by zero!
            // TODO: I think this might be wrong... we should only use one component here
            instr_info.dest.x[0] = instr_info.src1.sign / instr_info.src1.x[offset];
            instr_info.dest.y[0] = instr_info.src1.sign / instr_info.src1.y[offset];
            instr_info.dest.z[0] = instr_info.src1.sign / instr_info.src1.z[offset];
            instr_info.dest.w[0] = instr_info.src1.sign / instr_info.src1.w[offset];

            break;
        }

        case OpCode::RSQ:
        {
            int offset = state.address_offset[instr.common.address_register_index];

            // TODO: Be stable against division by zero!
            // TODO: I think this might be wrong... we should only use one component here
            instr_info.dest.x[0] = instr_info.src1.sign / instr_info.src1.x[offset];
            instr_info.dest.y[0] = instr_info.src1.sign / instr_info.src1.y[offset];
            instr_info.dest.z[0] = instr_info.src1.sign / instr_info.src1.z[offset];
            instr_info.dest.w[0] = instr_info.src1.sign / instr_info.src1.w[offset];

            break;
        }

        case OpCode::MOVA:
        {
            int offset = state.address_offset[instr.common.address_register_index];
            const SwizzlePattern& swizzle = *(SwizzlePattern*)&swizzle_data[instr.common.operand_desc_id];

            for (int i = 0; i < 2; ++i) {
                if (!swizzle.DestComponentEnabled(i))
                    continue;

                // TODO: Figure out how the rounding is done on hardware
                state.address_offset[i + 1] = static_cast<s32>(instr_info.src1.sign * instr_info.src1[i][offset]) << 2;
            }
            break;
        }

        case OpCode::MOV:
        {
            int offset = state.address_offset[instr.common.address_register_index];

            instr_info.dest.x[0] = instr_info.src1.sign * instr_info.src1.x[offset];
            instr_info.dest.y[0] = instr_info.src1.sign * instr_info.src1.y[offset];
            instr_info.dest.z[0] = instr_info.src1.sign * instr_info.src1.z[offset];
            instr_info.dest.w[0] = instr_info.src1.sign * instr_info.src1.w[offset];

            break;
        }

        case OpCode::CALL:
            Call(state,
                 instr.flow_control.dest_offset,
                 instr.flow_control.num_instructions,
                 state.pc + 1, 0, 0);
            continue;

        case OpCode::CALLU:
            if (vs.uniforms_b[instr.flow_control.bool_uniform_id]) {
                Call(state,
                     instr.flow_control.dest_offset,
                     instr.flow_control.num_instructions,
                     state.pc + 1, 0, 0);
                continue;
            }
            break;

        case OpCode::CALLC:
            if (EvaluateCondition(state, instr.flow_control.refx, instr.flow_control.refy, instr.flow_control)) {
                Call(state,
                     instr.flow_control.dest_offset,
                     instr.flow_control.num_instructions,
                     state.pc + 1, 0, 0);
                continue;
            }
            break;

        case OpCode::NOP:
            break;

        case OpCode::END:
            exit_loop = true;
            break;

        case OpCode::IFU:
            if (vs.uniforms_b[instr.flow_control.bool_uniform_id]) {
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

        case OpCode::IFC:
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

        case OpCode::LOOP:
            state.address_offset[2] = vs.uniforms_i[instr.flow_control.int_uniform_id].y << 2;

            Call(state,
                state.pc + 1,
                instr.flow_control.dest_offset - state.pc + 1,
                instr.flow_control.dest_offset + 1,
                vs.uniforms_i[instr.flow_control.int_uniform_id].x,
                vs.uniforms_i[instr.flow_control.int_uniform_id].z);

            continue;

        case OpCode::JMPC:
            if (EvaluateCondition(state, instr.flow_control.refx, instr.flow_control.refy, instr.flow_control)) {
                state.pc = instr.flow_control.dest_offset;
                continue;
            }
            break;

        case OpCode::JMPU:
            if (vs.uniforms_b[instr.flow_control.bool_uniform_id]) {
                state.pc = instr.flow_control.dest_offset;
                continue;
            }
            break;

        case OpCode::CMP:
        case OpCode::CMP + 1:
        {
            int offset = state.address_offset[instr.common.address_register_index];
            for (int i = 0; i < 2; ++i) {
                // TODO: Can you restrict to one compare via dest masking?

                auto compare_op = instr.common.compare_op;
                auto op = (i == 0) ? compare_op.x.Value() : compare_op.y.Value();
                auto src1 = instr_info.src1.sign * instr_info.src1[i][offset];
                auto src2 = instr_info.src2.sign * instr_info.src2[i][0];

                switch (op) {
                case compare_op.Equal:
                    state.conditional_code[i] = (src1 == src2);
                    break;

                case compare_op.NotEqual:
                    state.conditional_code[i] = (src1 != src2);
                    break;

                case compare_op.LessThan:
                    state.conditional_code[i] = (src1 < src2);
                    break;

                case compare_op.LessEqual:
                    state.conditional_code[i] = (src1 <= src2);
                    break;

                case compare_op.GreaterThan:
                    state.conditional_code[i] = (src1 > src2);
                    break;

                case compare_op.GreaterEqual:
                    state.conditional_code[i] = (src1 >= src2);
                    break;

                default:
                    LOG_ERROR(HW_GPU, "Unknown compare mode %x", static_cast<int>(op));
                    break;
                }
            }
            break;
        }

        case OpCode::MADI:
        case OpCode::MADI + 1:
        case OpCode::MADI + 2:
        case OpCode::MADI + 3:
        case OpCode::MADI + 4:
        case OpCode::MADI + 5:
        case OpCode::MADI + 6:
        case OpCode::MADI + 7:
        case OpCode::MAD:
        case OpCode::MAD + 1:
        case OpCode::MAD + 2:
        case OpCode::MAD + 3:
        case OpCode::MAD + 4:
        case OpCode::MAD + 5:
        case OpCode::MAD + 6:
        case OpCode::MAD + 7:
        {
            instr_info.dest.x[0] = instr_info.sign * instr_info.src1.x[0] * instr_info.src2.x[0] + instr_info.src3.sign * instr_info.src3.x[0];
            instr_info.dest.y[0] = instr_info.sign * instr_info.src1.y[0] * instr_info.src2.y[0] + instr_info.src3.sign * instr_info.src3.y[0];
            instr_info.dest.z[0] = instr_info.sign * instr_info.src1.z[0] * instr_info.src2.z[0] + instr_info.src3.sign * instr_info.src3.z[0];
            instr_info.dest.w[0] = instr_info.sign * instr_info.src1.w[0] * instr_info.src2.w[0] + instr_info.src3.sign * instr_info.src3.w[0];
            break;
        }

        default:
            LOG_CRITICAL(HW_GPU, "Unhandled opcode: 0x%02x", instr.opcode.Value());
            UNIMPLEMENTED();
        }

        state.pc += 1;

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

    return ret;
}

} // namespace VertexShaderFast

} // namespace Pica
