// Copyright 2015 Citra Emulator Project
// Licensed under GPLv2 or any later version
// Refer to the license.txt file included.

#include "gl_shader_constructor.h"

#include "math.h"
#include "video_core/pica.h"
#include "video_core/video_core.h"

#include "nihstro/shader_bytecode.h"

#include "common/string_util.h"

const char g_glsl_shader_header[] = R"(
#version 150 core

#define NUM_ATTR 16
#define NUM_OUT 7

#define NUM_UNIFORM_FLOATVEC 96
#define NUM_UNIFORM_BOOL 16
#define NUM_UNIFORM_INTVEC 4

in vec4 v[NUM_ATTR];

out vec4 o[NUM_OUT];

uniform int num_attrs;
uniform int attr_map[NUM_ATTR];
uniform int out_map[NUM_ATTR * 4];
uniform vec4 c[NUM_UNIFORM_FLOATVEC];
uniform bool b[NUM_UNIFORM_BOOL];
uniform ivec4 i[NUM_UNIFORM_INTVEC];
uniform int aL;

float o_tmp[NUM_OUT * 4];
vec4 input_regs[NUM_ATTR];
vec4 output_regs[NUM_ATTR];
vec4 r[16];
ivec2 idx;
bvec2 cmp;

)";

const char g_glsl_shader_main[] = R"(
void main() {
	r[0] = vec4(0.0, 0.0, 0.0, 0.0);
	r[1] = vec4(0.0, 0.0, 0.0, 0.0);
	r[2] = vec4(0.0, 0.0, 0.0, 0.0);
	r[3] = vec4(0.0, 0.0, 0.0, 0.0);
	r[4] = vec4(0.0, 0.0, 0.0, 0.0);
	r[5] = vec4(0.0, 0.0, 0.0, 0.0);
	r[6] = vec4(0.0, 0.0, 0.0, 0.0);
	r[7] = vec4(0.0, 0.0, 0.0, 0.0);
	r[8] = vec4(0.0, 0.0, 0.0, 0.0);
	r[9] = vec4(0.0, 0.0, 0.0, 0.0);
	r[10] = vec4(0.0, 0.0, 0.0, 0.0);
	r[11] = vec4(0.0, 0.0, 0.0, 0.0);
	r[12] = vec4(0.0, 0.0, 0.0, 0.0);
	r[13] = vec4(0.0, 0.0, 0.0, 0.0);
	r[14] = vec4(0.0, 0.0, 0.0, 0.0);
	r[15] = vec4(0.0, 0.0, 0.0, 0.0);
	idx = ivec2(0, 0);
	cmp = bvec2(false, false);

	for (int i = 0; i < num_attrs; ++i) {
		input_regs[attr_map[i]] = v[i];
	}

)";

const char g_glsl_shader_main_end[] = R"(

	// o_tmp[] needed to allow for const-index into o[]
	for (int i = 0; i < 16 * 4; ++i) {
		o_tmp[out_map[i]] = output_regs[i / 4][i % 4];
	}

	o[0] = vec4(o_tmp[0], o_tmp[1], o_tmp[2], o_tmp[3]);
	o[1] = vec4(o_tmp[4], o_tmp[5], o_tmp[6], o_tmp[7]);
	o[2] = vec4(o_tmp[8], o_tmp[9], o_tmp[10], o_tmp[11]);
	o[3] = vec4(o_tmp[12], o_tmp[13], o_tmp[14], o_tmp[15]);
	o[4] = vec4(o_tmp[16], o_tmp[17], o_tmp[18], o_tmp[19]);
	o[5] = vec4(o_tmp[20], o_tmp[21], o_tmp[22], o_tmp[23]);
	o[6] = vec4(o_tmp[24], o_tmp[25], o_tmp[26], o_tmp[27]);
	gl_Position = vec4(output_regs[0].x, -output_regs[0].y, -output_regs[0].z, output_regs[0].w);
}

)";

class IfElseData
{
public:
    IfElseData(u32 num_if_instr, u32 num_else_instr) : stage(2), num_if_instr(num_if_instr), num_else_instr(num_else_instr) {

    }

    u32 stage;
    u32 num_if_instr;
    u32 num_else_instr;
};

// State used when translating
std::vector<IfElseData> g_if_else_offset_list;
std::map<u32, std::string> g_fn_offset_map;
u32 g_cur_fn_entry;

std::map<u32, std::string> g_block_divider_map;

u32 GetRegMaskLen(const nihstro::SwizzlePattern& swizzle)
{
    u32 length = 0;

    if (swizzle.DestComponentEnabled(0)) {
        length++;
    }
    if (swizzle.DestComponentEnabled(1)) {
        length++;
    }
    if (swizzle.DestComponentEnabled(2)) {
        length++;
    }
    if (swizzle.DestComponentEnabled(3)) {
        length++;
    }

    return length;
}

std::string DestMaskToString(const nihstro::SwizzlePattern& swizzle, u32 comp_num) {
    std::string out(".");

    if (swizzle.DestComponentEnabled(0) && (comp_num == 0 || comp_num == 1)) {
        out += "x";
    }
    if (swizzle.DestComponentEnabled(1) && (comp_num == 0 || comp_num == 2)) {
        out += "y";
    }
    if (swizzle.DestComponentEnabled(2) && (comp_num == 0 || comp_num == 3)) {
        out += "z";
    }
    if (swizzle.DestComponentEnabled(3) && (comp_num == 0 || comp_num == 4)) {
        out += "w";
    }

    if (out.compare(".") == 0 || out.compare(".xyzw") == 0) {
        return std::string();
    }

    return out;
}

std::string SwizzleToString(const nihstro::SwizzlePattern& swizzle, u32 srcidx, bool clamp_swizzle) {
    std::string out(".");
    
    static const char comp[] = { 'x', 'y', 'z', 'w' };
    if (srcidx == 0) {
        if (!clamp_swizzle || (clamp_swizzle && swizzle.DestComponentEnabled(0))) out += comp[(unsigned)swizzle.src1_selector_0.Value()];
        if (!clamp_swizzle || (clamp_swizzle && swizzle.DestComponentEnabled(1))) out += comp[(unsigned)swizzle.src1_selector_1.Value()];
        if (!clamp_swizzle || (clamp_swizzle && swizzle.DestComponentEnabled(2))) out += comp[(unsigned)swizzle.src1_selector_2.Value()];
        if (!clamp_swizzle || (clamp_swizzle && swizzle.DestComponentEnabled(3))) out += comp[(unsigned)swizzle.src1_selector_3.Value()];
    } else if (srcidx == 1) {
        if (!clamp_swizzle || (clamp_swizzle && swizzle.DestComponentEnabled(0))) out += comp[(unsigned)swizzle.src2_selector_0.Value()];
        if (!clamp_swizzle || (clamp_swizzle && swizzle.DestComponentEnabled(1))) out += comp[(unsigned)swizzle.src2_selector_1.Value()];
        if (!clamp_swizzle || (clamp_swizzle && swizzle.DestComponentEnabled(2))) out += comp[(unsigned)swizzle.src2_selector_2.Value()];
        if (!clamp_swizzle || (clamp_swizzle && swizzle.DestComponentEnabled(3))) out += comp[(unsigned)swizzle.src2_selector_3.Value()];
    } else if (srcidx == 2) {
        if (!clamp_swizzle || (clamp_swizzle && swizzle.DestComponentEnabled(0))) out += comp[(unsigned)swizzle.src3_selector_0.Value()];
        if (!clamp_swizzle || (clamp_swizzle && swizzle.DestComponentEnabled(1))) out += comp[(unsigned)swizzle.src3_selector_1.Value()];
        if (!clamp_swizzle || (clamp_swizzle && swizzle.DestComponentEnabled(2))) out += comp[(unsigned)swizzle.src3_selector_2.Value()];
        if (!clamp_swizzle || (clamp_swizzle && swizzle.DestComponentEnabled(3))) out += comp[(unsigned)swizzle.src3_selector_3.Value()];
    }

    if (out.compare(".") == 0 || out.compare(".xyzw") == 0) {
        return std::string();
    }

    return out;
}

std::string RegTxtSrc(nihstro::Instruction instr, bool is_mad, bool is_inverted, const nihstro::SwizzlePattern* swizzle_data, u32 srcidx, bool clamp_swizzle) {
    std::string reg_text;

    u32 swizzle_idx;
    if (is_mad) {
        swizzle_idx = instr.mad.operand_desc_id.Value();
    } else {
        swizzle_idx = instr.common.operand_desc_id.Value();
    }

    const nihstro::SwizzlePattern& swizzle = swizzle_data[swizzle_idx];
    bool is_negated = (srcidx == 0 && swizzle.negate_src1) ||
                      (srcidx == 1 && swizzle.negate_src2) ||
                      (srcidx == 2 && swizzle.negate_src3);

    nihstro::SourceRegister reg;
    if (is_mad) {
        if (srcidx == 0) {
            reg = instr.mad.GetSrc1(is_inverted);
        } else if (srcidx == 1) {
            reg = instr.mad.GetSrc2(is_inverted);
        } else if (srcidx == 2) {
            reg = instr.mad.GetSrc3(is_inverted);
        } else {
            // Should never happen
            reg = 0;
        }
    } else {
        if (srcidx == 0) {
            reg = instr.common.GetSrc1(is_inverted);
        } else if (srcidx == 1) {
            reg = instr.common.GetSrc2(is_inverted);
        } else {
            // Should never happen
            reg = 0;
        }
    }

    std::string index_string;

    if (srcidx == 0) {
        if (instr.common.address_register_index == 1) {
            index_string = " + idx.x";
        } else if (instr.common.address_register_index == 2) {
            index_string = " + idx.y";
        } else {
            // Bad offset, just don't use it
            index_string = "";
        }
    } else {
        index_string = "";
    }

    if (reg.GetRegisterType() == nihstro::RegisterType::Input) {
        reg_text += "input_regs[";
    } else if (reg.GetRegisterType() == nihstro::RegisterType::Temporary) {
        reg_text += "r[";
    } else {
        reg_text += "c[";
    }
    reg_text += std::to_string(reg.GetIndex());
    reg_text += index_string;
    reg_text += "]";

    return (is_negated ? "-" : "") + reg_text + SwizzleToString(swizzle, srcidx, clamp_swizzle);
}

std::string RegTxtDst(u8 v, const nihstro::SwizzlePattern& swizzle, u32 comp_num) {
    std::string reg_text;

    if (v < 0x10) {
        reg_text += "output_regs[";
        reg_text += std::to_string(v);
        reg_text += "]";
    } else if (v < 0x20) {
        reg_text += "r[";
        reg_text += std::to_string(v - 0x10);
        reg_text += "]";
    } else {
        reg_text += "r[";
        reg_text += std::to_string(v);
        reg_text += "]";
    }

    return reg_text + DestMaskToString(swizzle, comp_num);
}

std::string PICAInstrToGLSL(nihstro::Instruction instr, const nihstro::SwizzlePattern* swizzle_data) {
    std::string instr_text;

    nihstro::OpCode::Info info = instr.opcode.Value().GetInfo();

    if (info.type == nihstro::OpCode::Type::Arithmetic) {
        bool is_inverted = info.subtype & nihstro::OpCode::Info::SrcInversed;

        bool clamp_swizzle = instr.opcode.Value().EffectiveOpCode() == nihstro::OpCode::Id::ADD ||
            instr.opcode.Value().EffectiveOpCode() == nihstro::OpCode::Id::MUL  ||
            instr.opcode.Value().EffectiveOpCode() == nihstro::OpCode::Id::FLR  ||
            instr.opcode.Value().EffectiveOpCode() == nihstro::OpCode::Id::MAX  ||
            instr.opcode.Value().EffectiveOpCode() == nihstro::OpCode::Id::MIN  ||
            instr.opcode.Value().EffectiveOpCode() == nihstro::OpCode::Id::RCP  ||
            instr.opcode.Value().EffectiveOpCode() == nihstro::OpCode::Id::RSQ  ||
            instr.opcode.Value().EffectiveOpCode() == nihstro::OpCode::Id::MOV  ||
            instr.opcode.Value().EffectiveOpCode() == nihstro::OpCode::Id::MOVA ||
            instr.opcode.Value().EffectiveOpCode() == nihstro::OpCode::Id::SLT  ||
            instr.opcode.Value().EffectiveOpCode() == nihstro::OpCode::Id::SLTI;

        std::string dst = RegTxtDst(instr.common.dest.Value(), swizzle_data[instr.common.operand_desc_id.Value()], 0);
        std::string src1 = RegTxtSrc(instr, false, is_inverted, swizzle_data, 0, clamp_swizzle);
        std::string src2 = RegTxtSrc(instr, false, is_inverted, swizzle_data, 1, clamp_swizzle);

        switch (instr.opcode.Value().EffectiveOpCode())
        {
        case nihstro::OpCode::Id::ADD:
        {
            instr_text += dst;
            instr_text += " = ";
            instr_text += src1;
            instr_text += " + ";
            instr_text += src2;
            instr_text += ";\n";
            break;
        }

        case nihstro::OpCode::Id::DP3:
        case nihstro::OpCode::Id::DP4:
        {
            u32 mask_len = GetRegMaskLen(swizzle_data[instr.common.operand_desc_id.Value()]);
            if (mask_len == 1) {
                instr_text += dst;
            } else {
                for (u32 i = 1; i < mask_len; i++) {
                    instr_text += RegTxtDst(instr.common.dest.Value(), swizzle_data[instr.common.operand_desc_id.Value()], i);
                    instr_text += " = ";
                }

                instr_text += RegTxtDst(instr.common.dest.Value(), swizzle_data[instr.common.operand_desc_id.Value()], mask_len);
            }

            if (instr.opcode.Value().EffectiveOpCode() == nihstro::OpCode::Id::DP3) {
                instr_text += " = dot(vec3(";
                instr_text += src1;
                instr_text += "), vec3(";
                instr_text += src2;
                instr_text += "));\n";
            } else {
                instr_text += " = dot(";
                instr_text += src1;
                instr_text += ", ";
                instr_text += src2;
                instr_text += ");\n";
            }

            break;
        }

        case nihstro::OpCode::Id::MUL:
        {
            instr_text += dst;
            instr_text += " = ";
            instr_text += src1;
            instr_text += " * ";
            instr_text += src2;
            instr_text += ";\n";
            break;
        }

        case nihstro::OpCode::Id::FLR:
        {
            instr_text += dst;
            instr_text += " = floor(";
            instr_text += src1;
            instr_text += ");\n";
            break;
        }

        case nihstro::OpCode::Id::MAX:
        {
            instr_text += dst;
            instr_text += " = max(";
            instr_text += src1;
            instr_text += ", ";
            instr_text += src2;
            instr_text += ");\n";
            break;
        }

        case nihstro::OpCode::Id::MIN:
        {
            instr_text += dst;
            instr_text += " = min(";
            instr_text += src1;
            instr_text += ", ";
            instr_text += src2;
            instr_text += ");\n";
            break;
        }

        case nihstro::OpCode::Id::RCP:
        {
            instr_text += "if (length(";
            instr_text += src1;
            instr_text += ") > 0.0000001) {";
            instr_text += dst;
            instr_text += " = 1 / ";
            instr_text += src1;
            instr_text += ";}\n";
            break;
        }

        case nihstro::OpCode::Id::RSQ:
        {
            instr_text += "if (length(";
            instr_text += src1;
            instr_text += ") > 0.0000001) {";
            instr_text += dst;
            instr_text += " = inversesqrt(";
            instr_text += src1;
            instr_text += ");}\n";
            break;
        }

        case nihstro::OpCode::Id::MOVA:
        {
            u32 maskSz = GetRegMaskLen(swizzle_data[instr.common.operand_desc_id.Value()]);
            if (maskSz == 2) {
                instr_text += "idx.xy = ivec2(";
                instr_text += src1;
                instr_text += ");\n";
            } else if (maskSz == 3) {
                instr_text += "idx.xyz = ivec3(";
                instr_text += src1;
                instr_text += ");\n";
            } else if (maskSz == 4) {
                instr_text += "idx.xyzw = ivec4(";
                instr_text += src1;
                instr_text += ");\n";
            } else {
                instr_text += "idx.x = int(";
                instr_text += src1;
                instr_text += ");\n";
            }

            break;
        }

        case nihstro::OpCode::Id::MOV:
        {
            instr_text += dst;
            instr_text += " = ";
            instr_text += src1;
            instr_text += ";\n";
            break;
        }

        case nihstro::OpCode::Id::SLT:
        case nihstro::OpCode::Id::SLTI:
        {
            u32 maskSz = GetRegMaskLen(swizzle_data[instr.common.operand_desc_id.Value()]);
            if (maskSz > 1) {
                instr_text += dst;
                instr_text += " = lessThan(";
                instr_text += src1;
                instr_text += ", ";
                instr_text += src2;
                instr_text += ");\n";
            } else {
                instr_text += dst;
                instr_text += " = (";
                instr_text += src1;
                instr_text += " < ";
                instr_text += src2;
                instr_text += " ? 1.0 : 0.0);\n";
            }
            break;
        }

        case nihstro::OpCode::Id::CMP:
        {
            instr_text += "cmp.x = ";
            instr_text += src1;
            instr_text += ".x ";
            instr_text += instr.common.compare_op.ToString(instr.common.compare_op.x);
            instr_text += " ";
            instr_text += src2;
            instr_text += ".x; cmp.y = ";
            instr_text += src1;
            instr_text += ".y ";
            instr_text += instr.common.compare_op.ToString(instr.common.compare_op.y);
            instr_text += " ";
            instr_text += src2;
            instr_text += ".y;\n";
            break;
        }

        default:
        {
            instr_text = Common::StringFromFormat("// WARNING: Unknown Arithmetic instruction 0x%08X (%s)\n", *((u32*)&instr), info.name);
            break;
        }
        }
    } else if (info.type == nihstro::OpCode::Type::Conditional) {
        switch (instr.opcode.Value().EffectiveOpCode())
        {
        case nihstro::OpCode::Id::BREAKC:
        {
            instr_text = "break;\n";
            break;
        }

        case nihstro::OpCode::Id::CALL:
        {
            // Avoid recursive calls (occurs with multiple main()s)
            if (g_cur_fn_entry != instr.flow_control.dest_offset.Value()) {
                std::map<u32, std::string>::iterator call_offset = g_block_divider_map.find(instr.flow_control.dest_offset.Value());
                if (call_offset != g_block_divider_map.end()) {
                    instr_text += call_offset->second;
                    instr_text += "();\n";
                } else {
                    instr_text = "// WARNING: CALL to unknown offset\n";
                }
            } else {
                instr_text = "// WARNING: Culled recursive CALL\n";
            }

            break;
        }

        case nihstro::OpCode::Id::CALLC:
        {
            char negate_or_space_x = instr.flow_control.refx.Value() ? ' ' : '!';
            char negate_or_space_y = instr.flow_control.refy.Value() ? ' ' : '!';

            switch (instr.flow_control.op)
            {
            case nihstro::Instruction::FlowControlType::Or:
                instr_text += "if (";
                instr_text += negate_or_space_x;
                instr_text += "cmp.x || ";
                instr_text += negate_or_space_y;
                instr_text += "cmp.y) { bb";
                instr_text += std::to_string(instr.flow_control.dest_offset);
                instr_text += "(); }\n";
                break;

            case nihstro::Instruction::FlowControlType::And:
                instr_text += "if (";
                instr_text += negate_or_space_x;
                instr_text += "cmp.x && ";
                instr_text += negate_or_space_y;
                instr_text += "cmp.y) { bb";
                instr_text += std::to_string(instr.flow_control.dest_offset);
                instr_text += "(); }\n";
                break;

            case nihstro::Instruction::FlowControlType::JustX:
                instr_text += "if (";
                instr_text += negate_or_space_x;
                instr_text += "cmp.x) { bb";
                instr_text += std::to_string(instr.flow_control.dest_offset);
                instr_text += "(); }\n";
                break;

            case nihstro::Instruction::FlowControlType::JustY:
                instr_text += "if (";
                instr_text += negate_or_space_y;
                instr_text += "cmp.y) { bb";
                instr_text += std::to_string(instr.flow_control.dest_offset);
                instr_text += "(); }\n";
                break;

            default:
                instr_text = "// WARNING: Bad CALLC condition op\n";
                break;
            }

            break;
        }

        case nihstro::OpCode::Id::IFC:
        {
            char negate_or_space_x = instr.flow_control.refx.Value() ? ' ' : '!';
            char negate_or_space_y = instr.flow_control.refy.Value() ? ' ' : '!';

            switch (instr.flow_control.op)
            {
            case nihstro::Instruction::FlowControlType::Or:
                instr_text += "if (";
                instr_text += negate_or_space_x;
                instr_text += "cmp.x || ";
                instr_text += negate_or_space_y;
                instr_text += "cmp.y) {\n";
                instr_text = Common::StringFromFormat("if (%ccmp.x || %ccmp.y) {\n", negate_or_space_x, negate_or_space_y);
                break;

            case nihstro::Instruction::FlowControlType::And:
                instr_text += "if (";
                instr_text += negate_or_space_x;
                instr_text += "cmp.x && ";
                instr_text += negate_or_space_y;
                instr_text += "cmp.y) {\n";
                break;

            case nihstro::Instruction::FlowControlType::JustX:
                instr_text += "if (";
                instr_text += negate_or_space_x;
                instr_text += "cmp.x) {\n";
                break;

            case nihstro::Instruction::FlowControlType::JustY:
                instr_text += "if (";
                instr_text += negate_or_space_y;
                instr_text += "cmp.y) {\n";
                break;

            default:
                instr_text = "// WARNING: Bad IFC condition op\n";
                break;
            }

            break;
        }

        case nihstro::OpCode::Id::JMPC:
        {
            char negate_or_space_x = instr.flow_control.refx.Value() ? ' ' : '!';
            char negate_or_space_y = instr.flow_control.refy.Value() ? ' ' : '!';

            switch (instr.flow_control.op)
            {
            case nihstro::Instruction::FlowControlType::Or:
                instr_text += "if (";
                instr_text += negate_or_space_x;
                instr_text += "cmp.x || ";
                instr_text += negate_or_space_y;
                instr_text += "cmp.y) { return bb";
                instr_text += std::to_string(instr.flow_control.dest_offset);
                instr_text += "(); }\n";
                break;

            case nihstro::Instruction::FlowControlType::And:
                instr_text += "if (";
                instr_text += negate_or_space_x;
                instr_text += "cmp.x && ";
                instr_text += negate_or_space_y;
                instr_text += "cmp.y) { return bb";
                instr_text += std::to_string(instr.flow_control.dest_offset);
                instr_text += "(); }\n";
                break;

            case nihstro::Instruction::FlowControlType::JustX:
                instr_text += "if (";
                instr_text += negate_or_space_x;
                instr_text += "cmp.x) { return bb";
                instr_text += std::to_string(instr.flow_control.dest_offset);
                instr_text += "(); }\n";
                break;

            case nihstro::Instruction::FlowControlType::JustY:
                instr_text += "if (";
                instr_text += negate_or_space_y;
                instr_text += "cmp.y) { return bb";
                instr_text += std::to_string(instr.flow_control.dest_offset);
                instr_text += "(); }\n";
                break;

            default:
                instr_text = "// WARNING: Bad JMPC condition op\n";
                break;
            }

            break;
        }

        case nihstro::OpCode::Id::JMPU:
        {
            instr_text += "if (b[";
            instr_text += std::to_string(instr.flow_control.bool_uniform_id.Value());
            instr_text += "]) { return bb";
            instr_text += std::to_string(instr.flow_control.dest_offset);
            instr_text += "(); }\n";

            break;
        }

        default:
        {
            instr_text = Common::StringFromFormat("// WARNING: Unknown Conditional instruction 0x%08X (%s)\n", *((u32*)&instr), info.name);
            break;
        }
        }
    } else if (info.type == nihstro::OpCode::Type::UniformFlowControl) {
        switch (instr.opcode.Value().EffectiveOpCode())
        {
        case nihstro::OpCode::Id::CALLU:
        {
            instr_text += "if (b[";
            instr_text += std::to_string(instr.flow_control.bool_uniform_id.Value());
            instr_text += "]) { bb";
            instr_text += std::to_string(instr.flow_control.dest_offset);
            instr_text += "(); }\n";

            break;
        }

        case nihstro::OpCode::Id::IFU:
        {
            instr_text += "if (b[";
            instr_text += std::to_string(instr.flow_control.bool_uniform_id.Value());
            instr_text += "]) {\n";
            break;
        }

        case nihstro::OpCode::Id::LOOP:
        {
            // TODO: implement this
            // make it push an IfElseData(offset, 0) and the end brace will automatically be added
            // hopefully that doesnt get out of order with an if+else though - because it might put a bracket after the else?
            instr_text = "// WARNING: LOOP not yet implemented\n";
            break;
        }

        default:
        {
            instr_text = Common::StringFromFormat("// WARNING: Unknown UniformFlowControl instruction 0x%08X (%s)\n", *((u32*)&instr), info.name);
            break;
        }
        }
    } else if (info.type == nihstro::OpCode::Type::MultiplyAdd) {
        std::string dst = RegTxtDst(instr.mad.dest.Value(), swizzle_data[instr.mad.operand_desc_id.Value()], 0);
        std::string src1 = RegTxtSrc(instr, true, false, swizzle_data, 0, true);
        std::string src2 = RegTxtSrc(instr, true, false, swizzle_data, 1, true);
        std::string src3 = RegTxtSrc(instr, true, false, swizzle_data, 2, true);

        switch (instr.opcode.Value().EffectiveOpCode())
        {
        case nihstro::OpCode::Id::MAD:
        case nihstro::OpCode::Id::MADI:
            instr_text += dst;
            instr_text += " = ";
            instr_text += src1;
            instr_text += " * ";
            instr_text += src2;
            instr_text += " + ";
            instr_text += src3;
            instr_text += ";\n";
            break;

        default:
            instr_text = Common::StringFromFormat("// WARNING: Unknown MultiplyAdd instruction 0x%08X (%s)\n", *((u32*)&instr), info.name);
            break;
        }
    } else if (info.type == nihstro::OpCode::Type::Trivial) {
        instr_text = Common::StringFromFormat("// Ignored trivial 0x%08X (%s)\n", *((u32*)&instr), info.name);
    } else if (info.type == nihstro::OpCode::Type::SetEmit) {
        instr_text = "// WARNING: Unimplemented setemit\n";
    } else {
        instr_text = Common::StringFromFormat("// WARNING: Unknown instruction 0x%08X (%s)\n", *((u32*)&instr), info.name);
    }

    return instr_text;
}

void NewBasicBlockDivider(std::string* glsl_shader, uint32_t offset) {
    if (g_block_divider_map.find(offset) == g_block_divider_map.end()) {
        std::string fnName = std::string("bb") + std::to_string(offset);
        g_block_divider_map.emplace(offset, fnName);

        *glsl_shader += "int ";
        *glsl_shader += fnName;
        *glsl_shader += "();\n";
    }
}

std::string PICAVertexShaderToGLSL(u32 main_offset, const u32* shader_data, const u32* swizzle_raw_data) {
    const nihstro::SwizzlePattern* swizzle_data = (nihstro::SwizzlePattern*)swizzle_raw_data;

    std::string glsl_shader(g_glsl_shader_header);

    int fn_counter = 0;
    int nest_depth = 0;
    bool main_opened = false;
    bool last_was_nop = false;

    g_if_else_offset_list.clear();
    g_fn_offset_map.clear();
    g_cur_fn_entry = 0;

    g_block_divider_map.clear();

    NewBasicBlockDivider(&glsl_shader, main_offset);

    for (int i = 0; i < 1024; ++i) {
        nihstro::Instruction instr = ((nihstro::Instruction*)shader_data)[i];

        if (instr.opcode.Value().EffectiveOpCode() == nihstro::OpCode::Id::CALL  ||
                instr.opcode.Value().EffectiveOpCode() == nihstro::OpCode::Id::CALLC ||
                instr.opcode.Value().EffectiveOpCode() == nihstro::OpCode::Id::CALLU) {
            //Target
            NewBasicBlockDivider(&glsl_shader, instr.flow_control.dest_offset);
        } else if (instr.opcode.Value().EffectiveOpCode() == nihstro::OpCode::Id::JMPC ||
                   instr.opcode.Value().EffectiveOpCode() == nihstro::OpCode::Id::JMPU) {
            // Next
            NewBasicBlockDivider(&glsl_shader, i + 1);

            //Target
            NewBasicBlockDivider(&glsl_shader, instr.flow_control.dest_offset);
        }
    }

    glsl_shader += g_glsl_shader_main;

    glsl_shader += "\tint pc = ";
    glsl_shader += std::to_string(main_offset);
    glsl_shader += ";\n\twhile (pc != -1) {\n\t\t";

    for (const auto& block : g_block_divider_map) {
        if (block != *g_block_divider_map.begin())
        {
            glsl_shader += " else ";
        }

        glsl_shader += "if (pc == ";
        glsl_shader += std::to_string(block.first);
        glsl_shader += ") {\n\t\t\tpc = ";
        glsl_shader += block.second;
        glsl_shader += "();\n\t\t}";
    }

    glsl_shader += " else {\n\t\t\tbreak;\n\t\t}\n\t}";

    glsl_shader += g_glsl_shader_main_end;

    glsl_shader += "int junk0() {\n";

    for (int i = 0; i < 1024; ++i) {
        if (shader_data[i] == 0x0000000) {
            break;
        }

        nihstro::Instruction instr = ((nihstro::Instruction*)shader_data)[i];

        if (g_block_divider_map.find(i) != g_block_divider_map.end()) {
            std::map<u32, std::string>::iterator block_divider = g_block_divider_map.find(i);
            if (block_divider != g_block_divider_map.end()) {
                glsl_shader += "\treturn ";
                glsl_shader += std::to_string(i);
                glsl_shader += ";\n}\n\n";

                glsl_shader += "int " + block_divider->second + "() {\n";
                g_cur_fn_entry = i;
            }
        }

        // Consume ifelse offset if points to current offset
        for (IfElseData& ifelse_data : g_if_else_offset_list) {
            if (ifelse_data.stage == 2) {
                if (ifelse_data.num_if_instr == 1) {
                    nest_depth--;
                    glsl_shader += std::string(nest_depth, '\t') + "\t}";

                    if (ifelse_data.num_else_instr > 0) {
                        glsl_shader += " else {\n";
                        nest_depth++;

                        ifelse_data.stage--;
                    } else {
                        glsl_shader += "\n";

                        ifelse_data.stage -= 2;
                    }
                } else {
                    ifelse_data.num_if_instr--;
                }
            } else if (ifelse_data.stage == 1) {
                if (ifelse_data.num_else_instr == 1) {
                    ifelse_data.stage--;

                    nest_depth--;
                    glsl_shader += std::string(nest_depth, '\t') + "\t}\n";
                } else {
                    ifelse_data.num_else_instr--;
                }
            }
        }

        glsl_shader += "\t";
        glsl_shader += std::string(nest_depth, '\t') + PICAInstrToGLSL(instr, swizzle_data);

        if (instr.opcode.Value().EffectiveOpCode() == nihstro::OpCode::Id::IFU ||
                instr.opcode.Value().EffectiveOpCode() == nihstro::OpCode::Id::IFC) {
            g_if_else_offset_list.emplace_back(instr.flow_control.dest_offset.Value() - i, instr.flow_control.num_instructions.Value());
            nest_depth++;
        }

        if (instr.opcode.Value().EffectiveOpCode() == nihstro::OpCode::Id::END) {
            glsl_shader += "\treturn -1;\n}\n\n";

            glsl_shader += "int junk" + std::to_string(i) + "() {\n";
        }
    }

    glsl_shader += "\treturn -1;\n}";

    return glsl_shader;





    // First pass: scan for CALLs to determine function offsets
    for (int i = 0; i < 1024; ++i) {
        nihstro::Instruction instr = ((nihstro::Instruction*)shader_data)[i];

        if (instr.opcode.Value().EffectiveOpCode() == nihstro::OpCode::Id::CALL || instr.opcode.Value().EffectiveOpCode() == nihstro::OpCode::Id::CALLC || instr.opcode.Value().EffectiveOpCode() == nihstro::OpCode::Id::CALLU) {
            if (g_fn_offset_map.find(instr.flow_control.dest_offset.Value()) == g_fn_offset_map.end()) {
                std::string fnName = std::string("fn") + std::to_string(g_fn_offset_map.size());
                g_fn_offset_map.emplace(instr.flow_control.dest_offset.Value(), fnName);

                glsl_shader += "void ";
                glsl_shader += fnName;
                glsl_shader += "();\n";
            }
        }
    }

    int instr_index;

    // Second pass: translate instructions
    for (instr_index = 0; instr_index < 1024; ++instr_index) {
        if (shader_data[instr_index] == 0x0000000) {
            break;
        }

        // Consume ifelse offset if points to current offset
        for (IfElseData& ifelse_data : g_if_else_offset_list) {
            if (ifelse_data.stage == 2) {
                if (ifelse_data.num_if_instr == 1) {
                    nest_depth--;
                    glsl_shader += std::string(nest_depth, '\t') + "}\n";

                    if (ifelse_data.num_else_instr > 0) {
                        glsl_shader += std::string(nest_depth, '\t') + "else {\n";
                        nest_depth++;

                        ifelse_data.stage--;
                    } else {
                        ifelse_data.stage -= 2;
                    }
                } else {
                    ifelse_data.num_if_instr--;
                }
            } else if (ifelse_data.stage == 1) {
                if (ifelse_data.num_else_instr == 1) {
                    ifelse_data.stage--;

                    nest_depth--;
                    glsl_shader += std::string(nest_depth, '\t') + "}\n";
                } else {
                    ifelse_data.num_else_instr--;
                }
            }
        }

        // Consume function offset if points to current offset
        std::map<u32, std::string>::iterator fn_offset = g_fn_offset_map.find(instr_index);
        if (fn_offset != g_fn_offset_map.end()) {
            if (nest_depth > 0) {
                nest_depth--;
                glsl_shader += std::string(nest_depth, '\t') + "}\n\n";
            }

            g_cur_fn_entry = instr_index;

            glsl_shader += std::string(nest_depth, '\t') + "void " + fn_offset->second + "() {\n";
            nest_depth++;
        } else if (instr_index == main_offset) {
            // Hit entry point - get out of nested block if we're in one
            while (nest_depth > 0) {
                nest_depth--;
                glsl_shader += std::string(nest_depth, '\t') + "}\n\n";
            }

            g_cur_fn_entry = instr_index;

            glsl_shader += std::string(nest_depth, '\t') + "void main() {\n";
            nest_depth++;

            glsl_shader += std::string(nest_depth, '\t') + "r[0] = vec4(0.0, 0.0, 0.0, 0.0);\n";
            glsl_shader += std::string(nest_depth, '\t') + "r[1] = vec4(0.0, 0.0, 0.0, 0.0);\n";
            glsl_shader += std::string(nest_depth, '\t') + "r[2] = vec4(0.0, 0.0, 0.0, 0.0);\n";
            glsl_shader += std::string(nest_depth, '\t') + "r[3] = vec4(0.0, 0.0, 0.0, 0.0);\n";
            glsl_shader += std::string(nest_depth, '\t') + "r[4] = vec4(0.0, 0.0, 0.0, 0.0);\n";
            glsl_shader += std::string(nest_depth, '\t') + "r[5] = vec4(0.0, 0.0, 0.0, 0.0);\n";
            glsl_shader += std::string(nest_depth, '\t') + "r[6] = vec4(0.0, 0.0, 0.0, 0.0);\n";
            glsl_shader += std::string(nest_depth, '\t') + "r[7] = vec4(0.0, 0.0, 0.0, 0.0);\n";
            glsl_shader += std::string(nest_depth, '\t') + "r[8] = vec4(0.0, 0.0, 0.0, 0.0);\n";
            glsl_shader += std::string(nest_depth, '\t') + "r[9] = vec4(0.0, 0.0, 0.0, 0.0);\n";
            glsl_shader += std::string(nest_depth, '\t') + "r[10] = vec4(0.0, 0.0, 0.0, 0.0);\n";
            glsl_shader += std::string(nest_depth, '\t') + "r[11] = vec4(0.0, 0.0, 0.0, 0.0);\n";
            glsl_shader += std::string(nest_depth, '\t') + "r[12] = vec4(0.0, 0.0, 0.0, 0.0);\n";
            glsl_shader += std::string(nest_depth, '\t') + "r[13] = vec4(0.0, 0.0, 0.0, 0.0);\n";
            glsl_shader += std::string(nest_depth, '\t') + "r[14] = vec4(0.0, 0.0, 0.0, 0.0);\n";
            glsl_shader += std::string(nest_depth, '\t') + "r[15] = vec4(0.0, 0.0, 0.0, 0.0);\n";
            glsl_shader += std::string(nest_depth, '\t') + "idx = ivec2(0, 0);\n";
            glsl_shader += std::string(nest_depth, '\t') + "cmp = bvec2(false, false);\n";

            glsl_shader += std::string(nest_depth, '\t') + "for (int i = 0; i < num_attrs; ++i) {\n";
            nest_depth++;
            glsl_shader += std::string(nest_depth, '\t') + "input_regs[attr_map[i]] = v[i];\n";
            nest_depth--;
            glsl_shader += std::string(nest_depth, '\t') + "}\n";

            main_opened = true;
        }

        // Translate instruction at current offset
        // All instructions have to be in main or a function, else we've overrun actual shader data
        if (nest_depth > 0) {
            nihstro::Instruction instr = ((nihstro::Instruction*)shader_data)[instr_index];

            if (instr.opcode.Value().EffectiveOpCode() == nihstro::OpCode::Id::END) {
                if (nest_depth > 0) {
                    
                    //glsl_shader += std::string(nest_depth, '\t') + "int map;\n";
                    //so this all seems to work - issue is that it doesnt like variables in the leftmost array slot or something
                    //maybe it optimizes it out; or just doesn't support it?
                    //glsl_shader += std::string(nest_depth, '\t') + "int mapt = 3;int mapu=0; o[mapt ][mapu ] = output_regs[2][0];";
                    //glsl_shader += std::string(nest_depth, '\t') + "int mapy = 3;int mapi=1; o[mapy ][mapi ] = output_regs[2][1];";
                    //should be 7*4
                    /*for (int i = 2*4; i < 3 * 4-2; ++i) {
                        glsl_shader += std::string(nest_depth, '\t') + "map = out_map[";
                        glsl_shader += std::to_string(i / 4);
                        glsl_shader += "][";
                        glsl_shader += std::to_string(i % 4);
                        glsl_shader += "];\n";
                        glsl_shader += std::string(nest_depth, '\t') + "o[map / 4][map % 4] = output_regs[";
                        //glsl_shader += std::string(nest_depth, '\t') + "o[";
                        //glsl_shader += std::to_string(i / 4);
                        //glsl_shader += "][";
                        //glsl_shader += std::to_string(i % 4);
                        //glsl_shader += "]";
                        //glsl_shader += " = output_regs[";
                        glsl_shader += std::to_string(i / 4);
                        glsl_shader += "][";
                        glsl_shader += std::to_string(i % 4);
                        glsl_shader += "];\n";
                    }*/
                    //glsl_shader += std::string(nest_depth, '\t') + "o[3][0] = output_regs[2][0];";
                    //glsl_shader += std::string(nest_depth, '\t') + "o[3][1] = output_regs[2][1];";


                    //so i think this actually needs to be a 1d texture
                    //technically input_regs shouldnt be working either
                    //since non-const/non-loop-index index to array
                    ///so roll in and out maps into same 1d texture
                    //probably make it big enough to fit all 16 in maps and 7 out maps regarless of num_attrs so no resizing necessary



                    glsl_shader += std::string(nest_depth, '\t') + "// o_tmp[] needed to allow for const-index into o[]\n";
                    glsl_shader += std::string(nest_depth, '\t') + "for (int i = 0; i < 16 * 4; ++i) {\n";
                    nest_depth++;
                    glsl_shader += std::string(nest_depth, '\t') + "o_tmp[out_map[i]] = output_regs[i / 4][i % 4];\n";
                    nest_depth--;
                    glsl_shader += std::string(nest_depth, '\t') + "}\n";
                    glsl_shader += std::string(nest_depth, '\t') + "o[0] = vec4(o_tmp[0], o_tmp[1], o_tmp[2], o_tmp[3]);\n";
                    glsl_shader += std::string(nest_depth, '\t') + "o[1] = vec4(o_tmp[4], o_tmp[5], o_tmp[6], o_tmp[7]);\n";
                    glsl_shader += std::string(nest_depth, '\t') + "o[2] = vec4(o_tmp[8], o_tmp[9], o_tmp[10], o_tmp[11]);\n";
                    glsl_shader += std::string(nest_depth, '\t') + "o[3] = vec4(o_tmp[12], o_tmp[13], o_tmp[14], o_tmp[15]);\n";
                    glsl_shader += std::string(nest_depth, '\t') + "o[4] = vec4(o_tmp[16], o_tmp[17], o_tmp[18], o_tmp[19]);\n";
                    glsl_shader += std::string(nest_depth, '\t') + "o[5] = vec4(o_tmp[20], o_tmp[21], o_tmp[22], o_tmp[23]);\n";
                    glsl_shader += std::string(nest_depth, '\t') + "o[6] = vec4(o_tmp[24], o_tmp[25], o_tmp[26], o_tmp[27]);\n";
                    glsl_shader += std::string(nest_depth, '\t') + "gl_Position = vec4(output_regs[0].x, -output_regs[0].y, -output_regs[0].z, output_regs[0].w);\n";
                    nest_depth--;
                    glsl_shader += std::string(nest_depth, '\t') + "}// END\n";
                } else {
                    glsl_shader += "\n";
                }
            } else if (instr.opcode.Value().EffectiveOpCode() == nihstro::OpCode::Id::NOP) {
                glsl_shader += std::string(nest_depth, '\t') + "// NOP\n";
            } else {
                glsl_shader += std::string(nest_depth, '\t') + PICAInstrToGLSL(instr, swizzle_data);

                if (instr.opcode.Value().EffectiveOpCode() == nihstro::OpCode::Id::IFU || instr.opcode.Value().EffectiveOpCode() == nihstro::OpCode::Id::IFC) {
                    g_if_else_offset_list.emplace_back(instr.flow_control.dest_offset.Value() - instr_index, instr.flow_control.num_instructions.Value());

                    nest_depth++;
                }
            }
        }
    }

    // Close function if it was last thing
    if (nest_depth > 0) {
        glsl_shader += "}\n";
    }

    return glsl_shader;
}
