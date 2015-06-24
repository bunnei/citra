// Copyright 2015 Citra Emulator Project
// Licensed under GPLv2 or any later version
// Refer to the license.txt file included.

#include "common/color.h"

#include "core/settings.h"
#include "core/hw/gpu.h"

#include "video_core/pica.h"
#include "video_core/utils.h"
#include "video_core/renderer_opengl/gl_rasterizer.h"
#include "video_core/renderer_opengl/gl_shaders.h"
#include "video_core/renderer_opengl/gl_shader_util.h"
#include "video_core/renderer_opengl/pica_to_gl.h"

#include "generated/gl_3_2_core.h"

#include <memory>

static bool IsPassThroughTevStage(const Pica::Regs::TevStageConfig& stage) {
    return (stage.color_op == Pica::Regs::TevStageConfig::Operation::Replace &&
            stage.alpha_op == Pica::Regs::TevStageConfig::Operation::Replace &&
            stage.color_source1 == Pica::Regs::TevStageConfig::Source::Previous &&
            stage.alpha_source1 == Pica::Regs::TevStageConfig::Source::Previous &&
            stage.color_modifier1 == Pica::Regs::TevStageConfig::ColorModifier::SourceColor &&
            stage.alpha_modifier1 == Pica::Regs::TevStageConfig::AlphaModifier::SourceAlpha &&
            stage.GetColorMultiplier() == 1 &&
            stage.GetAlphaMultiplier() == 1);
}

RasterizerOpenGL::RasterizerOpenGL() : last_fb_color_addr(0), last_fb_depth_addr(0) { }
RasterizerOpenGL::~RasterizerOpenGL() { }

void RasterizerOpenGL::InitObjects() {
    // Create the hardware shader program
    shader.Create(GLShaders::g_vertex_shader_hw, GLShaders::g_fragment_shader_hw);

    // Generate VBO and VAO
    vertex_buffer.Create();
    vertex_array.Create();

    // Update OpenGL state
    state.draw.vertex_array = vertex_array.handle;
    state.draw.vertex_buffer = vertex_buffer.handle;

    state.Apply();

    // Create textures for OGL framebuffer that will be rendered to, initially 1x1 to succeed in framebuffer creation
    fb_color_texture.texture.Create();
    ReconfigureColorTexture(fb_color_texture, Pica::Regs::ColorFormat::RGBA8, 1, 1);

    state.texture_units[0].enabled_2d = true;
    state.texture_units[0].texture_2d = fb_color_texture.texture.handle;
    state.Apply();

    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAX_LEVEL, 0);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);

    state.texture_units[0].texture_2d = 0;
    state.Apply();

    fb_depth_texture.texture.Create();
    ReconfigureDepthTexture(fb_depth_texture, Pica::Regs::DepthFormat::D16, 1, 1);

    state.texture_units[0].enabled_2d = true;
    state.texture_units[0].texture_2d = fb_depth_texture.texture.handle;
    state.Apply();

    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAX_LEVEL, 0);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_COMPARE_FUNC, GL_LEQUAL);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_COMPARE_MODE, GL_NONE);

    state.texture_units[0].texture_2d = 0;
    state.Apply();

    // Configure OpenGL framebuffer
    framebuffer.Create();

    state.draw.framebuffer = framebuffer.handle;
    state.Apply();

    glActiveTexture(GL_TEXTURE0);
    glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, fb_color_texture.texture.handle, 0);
    glFramebufferTexture2D(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_TEXTURE_2D, fb_depth_texture.texture.handle, 0);

    ASSERT_MSG(glCheckFramebufferStatus(GL_FRAMEBUFFER) == GL_FRAMEBUFFER_COMPLETE,
               "OpenGL rasterizer framebuffer setup failed, status %X", glCheckFramebufferStatus(GL_FRAMEBUFFER));
}

void RasterizerOpenGL::Reset() {
    const auto& regs = Pica::g_state.regs;

    res_cache.FullFlush();

    SyncShader(true);
    SyncCullMode();
    SyncBlendEnabled();
    SyncBlendFuncs();
    SyncBlendColor();
    SyncLogicOp();
    SyncStencilTest();
    SyncDepthTest();
}

void RasterizerOpenGL::AddTriangle(const Pica::VertexShader::OutputVertex& v0,
                                   const Pica::VertexShader::OutputVertex& v1,
                                   const Pica::VertexShader::OutputVertex& v2) {
    vertex_batch.push_back(HardwareVertex(v0));
    vertex_batch.push_back(HardwareVertex(v1));
    vertex_batch.push_back(HardwareVertex(v2));
}

void RasterizerOpenGL::AddTriangleRaw(const RawVertex& v0,
                                      const RawVertex& v1,
                                      const RawVertex& v2) {
    raw_vertex_batch.push_back(v0);
    raw_vertex_batch.push_back(v1);
    raw_vertex_batch.push_back(v2);
}

void RasterizerOpenGL::DrawTriangles() {
    SyncShader(false);
    SyncFramebuffer();
    SyncDrawState();

    if (Settings::values.use_hw_vertex_shaders) {
        glBufferData(GL_ARRAY_BUFFER, raw_vertex_batch.size() * sizeof(RawVertex), raw_vertex_batch.data(), GL_STREAM_DRAW);
        glDrawArrays(GL_TRIANGLES, 0, (GLsizei)raw_vertex_batch.size());

        raw_vertex_batch.clear();
    } else {
        glBufferData(GL_ARRAY_BUFFER, vertex_batch.size() * sizeof(HardwareVertex), vertex_batch.data(), GL_STREAM_DRAW);
        glDrawArrays(GL_TRIANGLES, 0, (GLsizei)vertex_batch.size());

        vertex_batch.clear();
    }

    // Flush the resource cache at the current depth and color framebuffer addresses for render-to-texture
    const auto& regs = Pica::g_state.regs;

    PAddr cur_fb_color_addr = regs.framebuffer.GetColorBufferPhysicalAddress();
    u32 cur_fb_color_size = Pica::Regs::BytesPerColorPixel(regs.framebuffer.color_format)
                            * regs.framebuffer.GetWidth() * regs.framebuffer.GetHeight();

    PAddr cur_fb_depth_addr = regs.framebuffer.GetDepthBufferPhysicalAddress();
    u32 cur_fb_depth_size = Pica::Regs::BytesPerDepthPixel(regs.framebuffer.depth_format)
                            * regs.framebuffer.GetWidth() * regs.framebuffer.GetHeight();

    res_cache.NotifyFlush(cur_fb_color_addr, cur_fb_color_size);
    res_cache.NotifyFlush(cur_fb_depth_addr, cur_fb_depth_size);
}

void RasterizerOpenGL::CommitFramebuffer() {
    CommitColorBuffer();
    CommitDepthBuffer();
}

void RasterizerOpenGL::NotifyPicaRegisterChanged(u32 id) {
    const auto& regs = Pica::g_state.regs;

    if (!Settings::values.use_hw_renderer)
        return;

    switch(id) {
    // Vertex shader
    case PICA_REG_INDEX(vs_bool_uniforms):
        if (Settings::values.use_hw_vertex_shaders)
            SyncBoolUniforms();
        break;

    case PICA_REG_INDEX_WORKAROUND(vs_int_uniforms[0], 0x2b1):
    case PICA_REG_INDEX_WORKAROUND(vs_int_uniforms[1], 0x2b2):
    case PICA_REG_INDEX_WORKAROUND(vs_int_uniforms[2], 0x2b3):
    case PICA_REG_INDEX_WORKAROUND(vs_int_uniforms[3], 0x2b4):
    {
        if (Settings::values.use_hw_vertex_shaders) {
            int index = (id - PICA_REG_INDEX_WORKAROUND(vs_int_uniforms[0], 0x2b1));
            SyncIntUniform(index);
        }
        break;
    }

    case PICA_REG_INDEX(vs_input_register_map.attribute0_register):
    {
        if (Settings::values.use_hw_vertex_shaders)
            SyncFirstInMap();
        break;
    }

    // TODO: offset (+4) maybe wrong?
    case PICA_REG_INDEX(vs_input_register_map.attribute0_register)+1:
    {
        if (Settings::values.use_hw_vertex_shaders)
            SyncSecondInMap();
        break;
    }

    case PICA_REG_INDEX_WORKAROUND(vs_output_attributes[0], 0x50):
    case PICA_REG_INDEX_WORKAROUND(vs_output_attributes[1], 0x51):
    case PICA_REG_INDEX_WORKAROUND(vs_output_attributes[2], 0x52):
    case PICA_REG_INDEX_WORKAROUND(vs_output_attributes[3], 0x53):
    case PICA_REG_INDEX_WORKAROUND(vs_output_attributes[4], 0x54):
    case PICA_REG_INDEX_WORKAROUND(vs_output_attributes[5], 0x55):
    case PICA_REG_INDEX_WORKAROUND(vs_output_attributes[6], 0x56):
    case PICA_REG_INDEX_WORKAROUND(vs_output_attributes[7], 0x57):
    case PICA_REG_INDEX_WORKAROUND(vs_output_attributes[8], 0x58):
    case PICA_REG_INDEX_WORKAROUND(vs_output_attributes[9], 0x59):
    case PICA_REG_INDEX_WORKAROUND(vs_output_attributes[10], 0x5A):
    case PICA_REG_INDEX_WORKAROUND(vs_output_attributes[11], 0x5B):
    case PICA_REG_INDEX_WORKAROUND(vs_output_attributes[12], 0x5C):
    case PICA_REG_INDEX_WORKAROUND(vs_output_attributes[13], 0x5E):
    case PICA_REG_INDEX_WORKAROUND(vs_output_attributes[14], 0x5F):
    case PICA_REG_INDEX_WORKAROUND(vs_output_attributes[15], 0x60):
    {
        if (Settings::values.use_hw_vertex_shaders) {
            int index = (id - PICA_REG_INDEX_WORKAROUND(vs_output_attributes[0], 0x50));
            SyncOutMap(index);
        }
        break;
    }

    // Culling
    case PICA_REG_INDEX(cull_mode):
        SyncCullMode();
        break;

    // Blending
    case PICA_REG_INDEX(output_merger.alphablend_enable):
        SyncBlendEnabled();
        break;
    case PICA_REG_INDEX(output_merger.alpha_blending):
        SyncBlendFuncs();
        break;
    case PICA_REG_INDEX(output_merger.blend_const):
        SyncBlendColor();
        break;

    // Alpha test
    case PICA_REG_INDEX(output_merger.alpha_test):
        SyncAlphaTest();
        break;

    // Stencil test
    case PICA_REG_INDEX(output_merger.stencil_test):
        SyncStencilTest();
        break;

    // Depth test
    case PICA_REG_INDEX(output_merger.depth_test_enable):
        SyncDepthTest();
        break;

    // Logic op
    case PICA_REG_INDEX(output_merger.logic_op):
        SyncLogicOp();
        break;

    // TEV stage 0
    case PICA_REG_INDEX(tev_stage0.color_source1):
        SyncTevSources(0, regs.tev_stage0);
        break;
    case PICA_REG_INDEX(tev_stage0.color_modifier1):
        SyncTevModifiers(0, regs.tev_stage0);
        break;
    case PICA_REG_INDEX(tev_stage0.color_op):
        SyncTevOps(0, regs.tev_stage0);
        break;
    case PICA_REG_INDEX(tev_stage0.const_r):
        SyncTevColor(0, regs.tev_stage0);
        break;
    case PICA_REG_INDEX(tev_stage0.color_scale):
        SyncTevMultipliers(0, regs.tev_stage0);
        break;

    // TEV stage 1
    case PICA_REG_INDEX(tev_stage1.color_source1):
        SyncTevSources(1, regs.tev_stage1);
        break;
    case PICA_REG_INDEX(tev_stage1.color_modifier1):
        SyncTevModifiers(1, regs.tev_stage1);
        break;
    case PICA_REG_INDEX(tev_stage1.color_op):
        SyncTevOps(1, regs.tev_stage1);
        break;
    case PICA_REG_INDEX(tev_stage1.const_r):
        SyncTevColor(1, regs.tev_stage1);
        break;
    case PICA_REG_INDEX(tev_stage1.color_scale):
        SyncTevMultipliers(1, regs.tev_stage1);
        break;

    // TEV stage 2
    case PICA_REG_INDEX(tev_stage2.color_source1):
        SyncTevSources(2, regs.tev_stage2);
        break;
    case PICA_REG_INDEX(tev_stage2.color_modifier1):
        SyncTevModifiers(2, regs.tev_stage2);
        break;
    case PICA_REG_INDEX(tev_stage2.color_op):
        SyncTevOps(2, regs.tev_stage2);
        break;
    case PICA_REG_INDEX(tev_stage2.const_r):
        SyncTevColor(2, regs.tev_stage2);
        break;
    case PICA_REG_INDEX(tev_stage2.color_scale):
        SyncTevMultipliers(2, regs.tev_stage2);
        break;

    // TEV stage 3
    case PICA_REG_INDEX(tev_stage3.color_source1):
        SyncTevSources(3, regs.tev_stage3);
        break;
    case PICA_REG_INDEX(tev_stage3.color_modifier1):
        SyncTevModifiers(3, regs.tev_stage3);
        break;
    case PICA_REG_INDEX(tev_stage3.color_op):
        SyncTevOps(3, regs.tev_stage3);
        break;
    case PICA_REG_INDEX(tev_stage3.const_r):
        SyncTevColor(3, regs.tev_stage3);
        break;
    case PICA_REG_INDEX(tev_stage3.color_scale):
        SyncTevMultipliers(3, regs.tev_stage3);
        break;

    // TEV stage 4
    case PICA_REG_INDEX(tev_stage4.color_source1):
        SyncTevSources(4, regs.tev_stage4);
        break;
    case PICA_REG_INDEX(tev_stage4.color_modifier1):
        SyncTevModifiers(4, regs.tev_stage4);
        break;
    case PICA_REG_INDEX(tev_stage4.color_op):
        SyncTevOps(4, regs.tev_stage4);
        break;
    case PICA_REG_INDEX(tev_stage4.const_r):
        SyncTevColor(4, regs.tev_stage4);
        break;
    case PICA_REG_INDEX(tev_stage4.color_scale):
        SyncTevMultipliers(4, regs.tev_stage4);
        break;

    // TEV stage 5
    case PICA_REG_INDEX(tev_stage5.color_source1):
        SyncTevSources(5, regs.tev_stage5);
        break;
    case PICA_REG_INDEX(tev_stage5.color_modifier1):
        SyncTevModifiers(5, regs.tev_stage5);
        break;
    case PICA_REG_INDEX(tev_stage5.color_op):
        SyncTevOps(5, regs.tev_stage5);
        break;
    case PICA_REG_INDEX(tev_stage5.const_r):
        SyncTevColor(5, regs.tev_stage5);
        break;
    case PICA_REG_INDEX(tev_stage5.color_scale):
        SyncTevMultipliers(5, regs.tev_stage5);
        break;

    // TEV combiner buffer color
    case PICA_REG_INDEX(tev_combiner_buffer_color):
        SyncCombinerColor();
        break;

    // TEV combiner buffer write flags
    case PICA_REG_INDEX(tev_combiner_buffer_input):
        SyncCombinerWriteFlags();
        break;
    }
}

void RasterizerOpenGL::NotifyPreRead(PAddr addr, u32 size) {
    const auto& regs = Pica::g_state.regs;

    if (!Settings::values.use_hw_renderer)
        return;

    PAddr cur_fb_color_addr = regs.framebuffer.GetColorBufferPhysicalAddress();
    u32 cur_fb_color_size = Pica::Regs::BytesPerColorPixel(regs.framebuffer.color_format)
                            * regs.framebuffer.GetWidth() * regs.framebuffer.GetHeight();

    PAddr cur_fb_depth_addr = regs.framebuffer.GetDepthBufferPhysicalAddress();
    u32 cur_fb_depth_size = Pica::Regs::BytesPerDepthPixel(regs.framebuffer.depth_format)
                            * regs.framebuffer.GetWidth() * regs.framebuffer.GetHeight();

    // If source memory region overlaps 3DS framebuffers, commit them before the copy happens
    if (MathUtil::IntervalsIntersect(addr, size, cur_fb_color_addr, cur_fb_color_size))
        CommitColorBuffer();

    if (MathUtil::IntervalsIntersect(addr, size, cur_fb_depth_addr, cur_fb_depth_size))
        CommitDepthBuffer();
}

void RasterizerOpenGL::NotifyFlush(PAddr addr, u32 size) {
    const auto& regs = Pica::g_state.regs;

    if (!Settings::values.use_hw_renderer)
        return;

    PAddr cur_fb_color_addr = regs.framebuffer.GetColorBufferPhysicalAddress();
    u32 cur_fb_color_size = Pica::Regs::BytesPerColorPixel(regs.framebuffer.color_format)
                            * regs.framebuffer.GetWidth() * regs.framebuffer.GetHeight();

    PAddr cur_fb_depth_addr = regs.framebuffer.GetDepthBufferPhysicalAddress();
    u32 cur_fb_depth_size = Pica::Regs::BytesPerDepthPixel(regs.framebuffer.depth_format)
                            * regs.framebuffer.GetWidth() * regs.framebuffer.GetHeight();

    // If modified memory region overlaps 3DS framebuffers, reload their contents into OpenGL
    if (MathUtil::IntervalsIntersect(addr, size, cur_fb_color_addr, cur_fb_color_size))
        ReloadColorBuffer();

    if (MathUtil::IntervalsIntersect(addr, size, cur_fb_depth_addr, cur_fb_depth_size))
        ReloadDepthBuffer();

    // Notify cache of flush in case the region touches a cached resource
    res_cache.NotifyFlush(addr, size);
}

void RasterizerOpenGL::LocateUniforms(GLuint shader_handle) {
    uniform_alphatest_enabled = glGetUniformLocation(shader_handle, "alphatest_enabled");
    uniform_alphatest_func = glGetUniformLocation(shader_handle, "alphatest_func");
    uniform_alphatest_ref = glGetUniformLocation(shader_handle, "alphatest_ref");

    uniform_tex = glGetUniformLocation(shader_handle, "tex");

    uniform_tev_combiner_buffer_color = glGetUniformLocation(shader_handle, "tev_combiner_buffer_color");

    const auto tev_stages = Pica::g_state.regs.GetTevStages();
    for (unsigned tev_stage_index = 0; tev_stage_index < tev_stages.size(); ++tev_stage_index) {
        auto& uniform_tev_cfg = uniform_tev_cfgs[tev_stage_index];

        std::string tev_ref_str = "tev_cfgs[" + std::to_string(tev_stage_index) + "]";
        uniform_tev_cfg.enabled = glGetUniformLocation(shader_handle, (tev_ref_str + ".enabled").c_str());
        uniform_tev_cfg.color_sources = glGetUniformLocation(shader_handle, (tev_ref_str + ".color_sources").c_str());
        uniform_tev_cfg.alpha_sources = glGetUniformLocation(shader_handle, (tev_ref_str + ".alpha_sources").c_str());
        uniform_tev_cfg.color_modifiers = glGetUniformLocation(shader_handle, (tev_ref_str + ".color_modifiers").c_str());
        uniform_tev_cfg.alpha_modifiers = glGetUniformLocation(shader_handle, (tev_ref_str + ".alpha_modifiers").c_str());
        uniform_tev_cfg.color_alpha_op = glGetUniformLocation(shader_handle, (tev_ref_str + ".color_alpha_op").c_str());
        uniform_tev_cfg.color_alpha_multiplier = glGetUniformLocation(shader_handle, (tev_ref_str + ".color_alpha_multiplier").c_str());
        uniform_tev_cfg.const_color = glGetUniformLocation(shader_handle, (tev_ref_str + ".const_color").c_str());
        uniform_tev_cfg.updates_combiner_buffer_color_alpha = glGetUniformLocation(shader.handle, (tev_ref_str + ".updates_combiner_buffer_color_alpha").c_str());
    }

    uniform_num_attrs = glGetUniformLocation(shader_handle, "num_attrs");
    uniform_attr_map = glGetUniformLocation(shader_handle, "attr_map");
    uniform_out_map = glGetUniformLocation(shader_handle, "out_map");

    for (unsigned i = 0; i < 96; ++i)
        uniform_c[i] = glGetUniformLocation(shader_handle, ("c[" + std::to_string(i) + "]").c_str());
    for (unsigned i = 0; i < 16; ++i)
        uniform_b[i] = glGetUniformLocation(shader_handle, ("b[" + std::to_string(i) + "]").c_str());
    for (unsigned i = 0; i < 4; ++i)
        uniform_i[i] = glGetUniformLocation(shader_handle, ("i[" + std::to_string(i) + "]").c_str());
}

void RasterizerOpenGL::ReconfigureColorTexture(TextureInfo& texture, Pica::Regs::ColorFormat format, u32 width, u32 height) {
    GLint internal_format;

    texture.format = format;
    texture.width = width;
    texture.height = height;

    switch (format) {
    case Pica::Regs::ColorFormat::RGBA8:
        internal_format = GL_RGBA;
        texture.gl_format = GL_RGBA;
        texture.gl_type = GL_UNSIGNED_INT_8_8_8_8;
        break;

    case Pica::Regs::ColorFormat::RGB8:
        // This pixel format uses BGR since GL_UNSIGNED_BYTE specifies byte-order, unlike every
        // specific OpenGL type used in this function using native-endian (that is, little-endian
        // mostly everywhere) for words or half-words.
        // TODO: check how those behave on big-endian processors.
        internal_format = GL_RGB;
        texture.gl_format = GL_BGR;
        texture.gl_type = GL_UNSIGNED_BYTE;
        break;

    case Pica::Regs::ColorFormat::RGB5A1:
        internal_format = GL_RGBA;
        texture.gl_format = GL_RGBA;
        texture.gl_type = GL_UNSIGNED_SHORT_5_5_5_1;
        break;

    case Pica::Regs::ColorFormat::RGB565:
        internal_format = GL_RGB;
        texture.gl_format = GL_RGB;
        texture.gl_type = GL_UNSIGNED_SHORT_5_6_5;
        break;

    case Pica::Regs::ColorFormat::RGBA4:
        internal_format = GL_RGBA;
        texture.gl_format = GL_RGBA;
        texture.gl_type = GL_UNSIGNED_SHORT_4_4_4_4;
        break;

    default:
        LOG_CRITICAL(Render_OpenGL, "Unknown framebuffer texture color format %x", format);
        UNIMPLEMENTED();
        break;
    }

    state.texture_units[0].enabled_2d = true;
    state.texture_units[0].texture_2d = texture.texture.handle;
    state.Apply();

    glActiveTexture(GL_TEXTURE0);
    glTexImage2D(GL_TEXTURE_2D, 0, internal_format, texture.width, texture.height, 0,
                 texture.gl_format, texture.gl_type, nullptr);

    state.texture_units[0].texture_2d = 0;
    state.Apply();
}

void RasterizerOpenGL::ReconfigureDepthTexture(DepthTextureInfo& texture, Pica::Regs::DepthFormat format, u32 width, u32 height) {
    GLint internal_format;

    texture.format = format;
    texture.width = width;
    texture.height = height;

    switch (format) {
    case Pica::Regs::DepthFormat::D16:
        internal_format = GL_DEPTH_COMPONENT16;
        texture.gl_format = GL_DEPTH_COMPONENT;
        texture.gl_type = GL_UNSIGNED_SHORT;
        break;

    case Pica::Regs::DepthFormat::D24:
        internal_format = GL_DEPTH_COMPONENT24;
        texture.gl_format = GL_DEPTH_COMPONENT;
        texture.gl_type = GL_UNSIGNED_INT;
        break;

    case Pica::Regs::DepthFormat::D24S8:
        internal_format = GL_DEPTH24_STENCIL8;
        texture.gl_format = GL_DEPTH_STENCIL;
        texture.gl_type = GL_UNSIGNED_INT_24_8;
        break;

    default:
        LOG_CRITICAL(Render_OpenGL, "Unknown framebuffer texture depth format %x", format);
        UNIMPLEMENTED();
        break;
    }

    state.texture_units[0].enabled_2d = true;
    state.texture_units[0].texture_2d = texture.texture.handle;
    state.Apply();

    glActiveTexture(GL_TEXTURE0);
    glTexImage2D(GL_TEXTURE_2D, 0, internal_format, texture.width, texture.height, 0,
                 texture.gl_format, texture.gl_type, nullptr);

    state.texture_units[0].texture_2d = 0;
    state.Apply();
}

void RasterizerOpenGL::SyncFramebuffer() {
    const auto& regs = Pica::g_state.regs;

    PAddr cur_fb_color_addr = regs.framebuffer.GetColorBufferPhysicalAddress();
    Pica::Regs::ColorFormat new_fb_color_format = regs.framebuffer.color_format;

    PAddr cur_fb_depth_addr = regs.framebuffer.GetDepthBufferPhysicalAddress();
    Pica::Regs::DepthFormat new_fb_depth_format = regs.framebuffer.depth_format;

    bool fb_size_changed = fb_color_texture.width != regs.framebuffer.GetWidth() ||
                           fb_color_texture.height != regs.framebuffer.GetHeight();

    bool color_fb_prop_changed = fb_color_texture.format != new_fb_color_format ||
                                 fb_size_changed;

    bool depth_fb_prop_changed = fb_depth_texture.format != new_fb_depth_format ||
                                 fb_size_changed;

    bool color_fb_modified = last_fb_color_addr != cur_fb_color_addr ||
                             color_fb_prop_changed;

    bool depth_fb_modified = last_fb_depth_addr != cur_fb_depth_addr ||
                             depth_fb_prop_changed;

    // Commit if framebuffer modified in any way
    if (color_fb_modified)
        CommitColorBuffer();

    if (depth_fb_modified)
        CommitDepthBuffer();

    // Reconfigure framebuffer textures if any property has changed
    if (color_fb_prop_changed) {
        ReconfigureColorTexture(fb_color_texture, new_fb_color_format,
                                regs.framebuffer.GetWidth(), regs.framebuffer.GetHeight());
    }

    if (depth_fb_prop_changed) {
        ReconfigureDepthTexture(fb_depth_texture, new_fb_depth_format,
                                regs.framebuffer.GetWidth(), regs.framebuffer.GetHeight());

        // Only attach depth buffer as stencil if it supports stencil
        switch (new_fb_depth_format) {
        case Pica::Regs::DepthFormat::D16:
        case Pica::Regs::DepthFormat::D24:
            glFramebufferTexture2D(GL_FRAMEBUFFER, GL_STENCIL_ATTACHMENT, GL_TEXTURE_2D, 0, 0);
            break;

        case Pica::Regs::DepthFormat::D24S8:
            glFramebufferTexture2D(GL_FRAMEBUFFER, GL_STENCIL_ATTACHMENT, GL_TEXTURE_2D, fb_depth_texture.texture.handle, 0);
            break;

        default:
            LOG_CRITICAL(Render_OpenGL, "Unknown framebuffer depth format %x", new_fb_depth_format);
            UNIMPLEMENTED();
            break;
        }
    }

    // Load buffer data again if fb modified in any way
    if (color_fb_modified) {
        last_fb_color_addr = cur_fb_color_addr;

        ReloadColorBuffer();
    }

    if (depth_fb_modified) {
        last_fb_depth_addr = cur_fb_depth_addr;

        ReloadDepthBuffer();
    }
}

void RasterizerOpenGL::SyncShaderUniforms() {
    const auto& regs = Pica::g_state.regs;

    LocateUniforms(state.draw.shader_program);

    // Set the texture samplers to correspond to different texture units
    glUniform1i(uniform_tex, 0);
    glUniform1i(uniform_tex + 1, 1);
    glUniform1i(uniform_tex + 2, 2);

    if (Settings::values.use_hw_vertex_shaders) {
        SyncBoolUniforms();
        for (unsigned i = 0; i < 4; ++i)
            SyncIntUniform(i);
        for (unsigned i = 0; i < 96; ++i)
            SyncFloatUniform(i);

        SyncFirstInMap();
        SyncSecondInMap();

        for (unsigned i = 0; i < 16; ++i)
            SyncOutMap(i);
    }

    SyncAlphaTest();

    // TEV stage 0
    SyncTevSources(0, regs.tev_stage0);
    SyncTevModifiers(0, regs.tev_stage0);
    SyncTevOps(0, regs.tev_stage0);
    SyncTevColor(0, regs.tev_stage0);
    SyncTevMultipliers(0, regs.tev_stage0);

    // TEV stage 1
    SyncTevSources(1, regs.tev_stage1);
    SyncTevModifiers(1, regs.tev_stage1);
    SyncTevOps(1, regs.tev_stage1);
    SyncTevColor(1, regs.tev_stage1);
    SyncTevMultipliers(1, regs.tev_stage1);

    // TEV stage 2
    SyncTevSources(2, regs.tev_stage2);
    SyncTevModifiers(2, regs.tev_stage2);
    SyncTevOps(2, regs.tev_stage2);
    SyncTevColor(2, regs.tev_stage2);
    SyncTevMultipliers(2, regs.tev_stage2);

    // TEV stage 3
    SyncTevSources(3, regs.tev_stage3);
    SyncTevModifiers(3, regs.tev_stage3);
    SyncTevOps(3, regs.tev_stage3);
    SyncTevColor(3, regs.tev_stage3);
    SyncTevMultipliers(3, regs.tev_stage3);

    // TEV stage 4
    SyncTevSources(4, regs.tev_stage4);
    SyncTevModifiers(4, regs.tev_stage4);
    SyncTevOps(4, regs.tev_stage4);
    SyncTevColor(4, regs.tev_stage4);
    SyncTevMultipliers(4, regs.tev_stage4);

    // TEV stage 5
    SyncTevSources(5, regs.tev_stage5);
    SyncTevModifiers(5, regs.tev_stage5);
    SyncTevOps(5, regs.tev_stage5);
    SyncTevColor(5, regs.tev_stage5);
    SyncTevMultipliers(5, regs.tev_stage5);

    SyncCombinerColor();
    SyncCombinerWriteFlags();
}

void RasterizerOpenGL::SyncShader(bool force_reload) {
    const auto& regs = Pica::g_state.regs;

    if (Settings::values.use_hw_vertex_shaders) {
        if (res_cache.LoadAndBindShader(force_reload || state.draw.shader_program == shader.handle, state, regs.vs_main_offset, Pica::g_state.vs.program_code.data(), Pica::g_state.vs.swizzle_data.data())) {
            attrib_v = glGetAttribLocation(state.draw.shader_program, "v");

            for (int i = 0; i < 16; i++) {
                glVertexAttribPointer(attrib_v + i, 4, GL_FLOAT, GL_FALSE, sizeof(RawVertex), (GLvoid*)(i * 4 * sizeof(float)));
                glEnableVertexAttribArray(attrib_v + i);
            }

            SyncShaderUniforms();
        }
    } else if (state.draw.shader_program != shader.handle) {
        state.draw.shader_program = shader.handle;

        state.Apply();

        attrib_position = glGetAttribLocation(shader.handle, "vert_position");
        attrib_color = glGetAttribLocation(shader.handle, "vert_color");
        attrib_texcoords = glGetAttribLocation(shader.handle, "vert_texcoords");

        glVertexAttribPointer(attrib_position, 4, GL_FLOAT, GL_FALSE, sizeof(HardwareVertex), (GLvoid*)offsetof(HardwareVertex, position));
        glVertexAttribPointer(attrib_color, 4, GL_FLOAT, GL_FALSE, sizeof(HardwareVertex), (GLvoid*)offsetof(HardwareVertex, color));
        glVertexAttribPointer(attrib_texcoords, 2, GL_FLOAT, GL_FALSE, sizeof(HardwareVertex), (GLvoid*)offsetof(HardwareVertex, tex_coord0));
        glVertexAttribPointer(attrib_texcoords + 1, 2, GL_FLOAT, GL_FALSE, sizeof(HardwareVertex), (GLvoid*)offsetof(HardwareVertex, tex_coord1));
        glVertexAttribPointer(attrib_texcoords + 2, 2, GL_FLOAT, GL_FALSE, sizeof(HardwareVertex), (GLvoid*)offsetof(HardwareVertex, tex_coord2));
        glEnableVertexAttribArray(attrib_position);
        glEnableVertexAttribArray(attrib_color);
        glEnableVertexAttribArray(attrib_texcoords);
        glEnableVertexAttribArray(attrib_texcoords + 1);
        glEnableVertexAttribArray(attrib_texcoords + 2);

        SyncShaderUniforms();
    }
}

void RasterizerOpenGL::SyncBoolUniforms() {
    for (unsigned i = 0; i < 16; ++i) {
        if (uniform_b[i] != -1)
            glUniform1i(uniform_b[i], Pica::g_state.vs.uniforms.b[i]);
        //else
        //    LOG_WARNING(Render_OpenGL, "Attempted to write to bool uniform not used in shader");
    }
}

void RasterizerOpenGL::SyncIntUniform(u32 uniform_index) {
    if (uniform_i[uniform_index] != -1)
        glUniform4iv(uniform_i[uniform_index], 1, (const GLint*)&Pica::g_state.vs.uniforms.i[uniform_index]);
    else
        LOG_WARNING(Render_OpenGL, "Attempted to write to int uniform not used in shader");
}

void RasterizerOpenGL::SyncFloatUniform(u32 uniform_index) {
    const auto& float24_values = Pica::g_state.vs.uniforms.f[uniform_index];
    GLfloat gl_float_values[] = { float24_values.x.ToFloat32(),
                                  float24_values.y.ToFloat32(),
                                  float24_values.z.ToFloat32(),
                                  float24_values.w.ToFloat32() };
    if (uniform_c[uniform_index] != -1)
        glUniform4fv(uniform_c[uniform_index], 1, gl_float_values);
    else
        LOG_WARNING(Render_OpenGL, "Attempted to write to float uniform not used in shader");
}

void RasterizerOpenGL::SyncFirstInMap() {
    const auto& regs = Pica::g_state.regs;
    GLint maps[] = { regs.vs_input_register_map.attribute0_register,
                     regs.vs_input_register_map.attribute1_register,
                     regs.vs_input_register_map.attribute2_register,
                     regs.vs_input_register_map.attribute3_register,
                     regs.vs_input_register_map.attribute4_register,
                     regs.vs_input_register_map.attribute5_register,
                     regs.vs_input_register_map.attribute6_register,
                     regs.vs_input_register_map.attribute7_register
    };

    // TODO: only need to actually upload up to num_attributes maps (clamp to it)
    glUniform1iv(uniform_attr_map, 8, maps);
}

void RasterizerOpenGL::SyncSecondInMap() {
    const auto& regs = Pica::g_state.regs;
    GLint maps[] = { regs.vs_input_register_map.attribute8_register,
                     regs.vs_input_register_map.attribute9_register,
                     regs.vs_input_register_map.attribute10_register,
                     regs.vs_input_register_map.attribute11_register,
                     regs.vs_input_register_map.attribute12_register,
                     regs.vs_input_register_map.attribute13_register,
                     regs.vs_input_register_map.attribute14_register,
                     regs.vs_input_register_map.attribute15_register
    };

    // TODO: only need to actually upload up to num_attributes maps (clamp to it MINUS 8)
    glUniform1iv(uniform_attr_map + 8, 8, maps);
}

void RasterizerOpenGL::SyncOutMap(u32 map_index) {
    const auto& regs = Pica::g_state.regs;
    GLint maps[] = { regs.vs_output_attributes[map_index].map_x,
                     regs.vs_output_attributes[map_index].map_y,
                     regs.vs_output_attributes[map_index].map_z,
                     regs.vs_output_attributes[map_index].map_w
    };

    glUniform1iv(uniform_out_map + map_index * 4, 4, maps);
}
void RasterizerOpenGL::SyncCullMode() {
    const auto& regs = Pica::g_state.regs;

    switch (regs.cull_mode) {
    case Pica::Regs::CullMode::KeepAll:
        state.cull.enabled = false;
        break;

    case Pica::Regs::CullMode::KeepClockWise:
        state.cull.enabled = true;
        state.cull.mode = GL_BACK;
        break;

    case Pica::Regs::CullMode::KeepCounterClockWise:
        state.cull.enabled = true;
        state.cull.mode = GL_FRONT;
        break;

    default:
        LOG_CRITICAL(Render_OpenGL, "Unknown cull mode %d", regs.cull_mode.Value());
        UNIMPLEMENTED();
        break;
    }
}

void RasterizerOpenGL::SyncBlendEnabled() {
    state.blend.enabled = (Pica::g_state.regs.output_merger.alphablend_enable == 1);
}

void RasterizerOpenGL::SyncBlendFuncs() {
    const auto& regs = Pica::g_state.regs;
    state.blend.src_rgb_func = PicaToGL::BlendFunc(regs.output_merger.alpha_blending.factor_source_rgb);
    state.blend.dst_rgb_func = PicaToGL::BlendFunc(regs.output_merger.alpha_blending.factor_dest_rgb);
    state.blend.src_a_func = PicaToGL::BlendFunc(regs.output_merger.alpha_blending.factor_source_a);
    state.blend.dst_a_func = PicaToGL::BlendFunc(regs.output_merger.alpha_blending.factor_dest_a);
}

void RasterizerOpenGL::SyncBlendColor() {
    auto blend_color = PicaToGL::ColorRGBA8((u8*)&Pica::g_state.regs.output_merger.blend_const.r);
    state.blend.color.red = blend_color[0];
    state.blend.color.green = blend_color[1];
    state.blend.color.blue = blend_color[2];
    state.blend.color.alpha = blend_color[3];
}

void RasterizerOpenGL::SyncAlphaTest() {
    const auto& regs = Pica::g_state.regs;
    glUniform1i(uniform_alphatest_enabled, regs.output_merger.alpha_test.enable);
    glUniform1i(uniform_alphatest_func, (GLint)regs.output_merger.alpha_test.func.Value());
    glUniform1f(uniform_alphatest_ref, regs.output_merger.alpha_test.ref / 255.0f);
}

void RasterizerOpenGL::SyncLogicOp() {
    state.logic_op = PicaToGL::LogicOp(Pica::g_state.regs.output_merger.logic_op);
}

void RasterizerOpenGL::SyncStencilTest() {
    // TODO: Implement stencil test, mask, and op
}

void RasterizerOpenGL::SyncDepthTest() {
    const auto& regs = Pica::g_state.regs;
    state.depth.test_enabled = (regs.output_merger.depth_test_enable == 1);
    state.depth.test_func = PicaToGL::CompareFunc(regs.output_merger.depth_test_func);
    state.color_mask.red_enabled = regs.output_merger.red_enable;
    state.color_mask.green_enabled = regs.output_merger.green_enable;
    state.color_mask.blue_enabled = regs.output_merger.blue_enable;
    state.color_mask.alpha_enabled = regs.output_merger.alpha_enable;
    state.depth.write_mask = regs.output_merger.depth_write_enable ? GL_TRUE : GL_FALSE;
}

void RasterizerOpenGL::SyncTevSources(unsigned stage_index, const Pica::Regs::TevStageConfig& config) {
    GLint color_srcs[3] = { (GLint)config.color_source1.Value(),
                            (GLint)config.color_source2.Value(),
                            (GLint)config.color_source3.Value() };
    GLint alpha_srcs[3] = { (GLint)config.alpha_source1.Value(),
                            (GLint)config.alpha_source2.Value(),
                            (GLint)config.alpha_source3.Value() };

    glUniform3iv(uniform_tev_cfgs[stage_index].color_sources, 1, color_srcs);
    glUniform3iv(uniform_tev_cfgs[stage_index].alpha_sources, 1, alpha_srcs);
}

void RasterizerOpenGL::SyncTevModifiers(unsigned stage_index, const Pica::Regs::TevStageConfig& config) {
    GLint color_mods[3] = { (GLint)config.color_modifier1.Value(),
                            (GLint)config.color_modifier2.Value(),
                            (GLint)config.color_modifier3.Value() };
    GLint alpha_mods[3] = { (GLint)config.alpha_modifier1.Value(),
                            (GLint)config.alpha_modifier2.Value(),
                            (GLint)config.alpha_modifier3.Value() };

    glUniform3iv(uniform_tev_cfgs[stage_index].color_modifiers, 1, color_mods);
    glUniform3iv(uniform_tev_cfgs[stage_index].alpha_modifiers, 1, alpha_mods);
}

void RasterizerOpenGL::SyncTevOps(unsigned stage_index, const Pica::Regs::TevStageConfig& config) {
    glUniform2i(uniform_tev_cfgs[stage_index].color_alpha_op, (GLint)config.color_op.Value(), (GLint)config.alpha_op.Value());
}

void RasterizerOpenGL::SyncTevColor(unsigned stage_index, const Pica::Regs::TevStageConfig& config) {
    auto const_color = PicaToGL::ColorRGBA8((u8*)&config.const_r);
    glUniform4fv(uniform_tev_cfgs[stage_index].const_color, 1, const_color.data());
}

void RasterizerOpenGL::SyncTevMultipliers(unsigned stage_index, const Pica::Regs::TevStageConfig& config) {
    glUniform2i(uniform_tev_cfgs[stage_index].color_alpha_multiplier, config.GetColorMultiplier(), config.GetAlphaMultiplier());
}

void RasterizerOpenGL::SyncCombinerColor() {
    auto combiner_color = PicaToGL::ColorRGBA8((u8*)&Pica::g_state.regs.tev_combiner_buffer_color.r);
    glUniform4fv(uniform_tev_combiner_buffer_color, 1, combiner_color.data());
}

void RasterizerOpenGL::SyncCombinerWriteFlags() {
    const auto& regs = Pica::g_state.regs;
    const auto tev_stages = regs.GetTevStages();
    for (unsigned tev_stage_index = 0; tev_stage_index < tev_stages.size(); ++tev_stage_index) {
        glUniform2i(uniform_tev_cfgs[tev_stage_index].updates_combiner_buffer_color_alpha,
                    regs.tev_combiner_buffer_input.TevStageUpdatesCombinerBufferColor(tev_stage_index),
                    regs.tev_combiner_buffer_input.TevStageUpdatesCombinerBufferAlpha(tev_stage_index));
    }
}

void RasterizerOpenGL::SyncDrawState() {
    const auto& regs = Pica::g_state.regs;

    // Sync the viewport
    GLsizei viewport_width = (GLsizei)Pica::float24::FromRawFloat24(regs.viewport_size_x).ToFloat32() * 2;
    GLsizei viewport_height = (GLsizei)Pica::float24::FromRawFloat24(regs.viewport_size_y).ToFloat32() * 2;

    // OpenGL uses different y coordinates, so negate corner offset and flip origin
    // TODO: Ensure viewport_corner.x should not be negated or origin flipped
    // TODO: Use floating-point viewports for accuracy if supported
    glViewport((GLsizei)static_cast<float>(regs.viewport_corner.x),
                -(GLsizei)static_cast<float>(regs.viewport_corner.y)
                    + regs.framebuffer.GetHeight() - viewport_height,
                viewport_width, viewport_height);

    // Sync bound texture(s), upload if not cached
    const auto pica_textures = regs.GetTextures();
    for (unsigned texture_index = 0; texture_index < pica_textures.size(); ++texture_index) {
        const auto& texture = pica_textures[texture_index];

        if (texture.enabled) {
            state.texture_units[texture_index].enabled_2d = true;
            res_cache.LoadAndBindTexture(state, texture_index, texture);
        } else {
            state.texture_units[texture_index].enabled_2d = false;
        }
    }

    // Skip processing TEV stages that simply pass the previous stage results through
    const auto tev_stages = regs.GetTevStages();
    for (unsigned tev_stage_index = 0; tev_stage_index < tev_stages.size(); ++tev_stage_index) {
        glUniform1i(uniform_tev_cfgs[tev_stage_index].enabled, !IsPassThroughTevStage(tev_stages[tev_stage_index]));
    }

    glUniform1i(uniform_num_attrs, Pica::g_state.regs.vertex_attributes.GetNumTotalAttributes());
    //LOG_CRITICAL(Render_OpenGL, "%d", Pica::g_state.regs.vertex_attributes.GetNumTotalAttributes());

    state.Apply();
}

void RasterizerOpenGL::ReloadColorBuffer() {
    u8* color_buffer = Memory::GetPhysicalPointer(Pica::g_state.regs.framebuffer.GetColorBufferPhysicalAddress());

    if (color_buffer == nullptr)
        return;

    u32 bytes_per_pixel = Pica::Regs::BytesPerColorPixel(fb_color_texture.format);

    std::unique_ptr<u8[]> temp_fb_color_buffer(new u8[fb_color_texture.width * fb_color_texture.height * bytes_per_pixel]);

    // Directly copy pixels. Internal OpenGL color formats are consistent so no conversion is necessary.
    for (int y = 0; y < fb_color_texture.height; ++y) {
        for (int x = 0; x < fb_color_texture.width; ++x) {
            const u32 coarse_y = y & ~7;
            u32 dst_offset = VideoCore::GetMortonOffset(x, y, bytes_per_pixel) + coarse_y * fb_color_texture.width * bytes_per_pixel;
            u32 gl_pixel_index = (x + y * fb_color_texture.width) * bytes_per_pixel;

            u8* pixel = color_buffer + dst_offset;
            memcpy(&temp_fb_color_buffer[gl_pixel_index], pixel, bytes_per_pixel);
        }
    }

    state.texture_units[0].enabled_2d = true;
    state.texture_units[0].texture_2d = fb_color_texture.texture.handle;
    state.Apply();

    glActiveTexture(GL_TEXTURE0);
    glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, fb_color_texture.width, fb_color_texture.height,
                    fb_color_texture.gl_format, fb_color_texture.gl_type, temp_fb_color_buffer.get());

    state.texture_units[0].texture_2d = 0;
    state.Apply();
}

void RasterizerOpenGL::ReloadDepthBuffer() {
    // TODO: Appears to work, but double-check endianness of depth values and order of depth-stencil
    u8* depth_buffer = Memory::GetPhysicalPointer(Pica::g_state.regs.framebuffer.GetDepthBufferPhysicalAddress());

    if (depth_buffer == nullptr) {
        return;
    }

    u32 bytes_per_pixel = Pica::Regs::BytesPerDepthPixel(fb_depth_texture.format);

    // OpenGL needs 4 bpp alignment for D24
    u32 gl_bpp = bytes_per_pixel == 3 ? 4 : bytes_per_pixel;

    std::unique_ptr<u8[]> temp_fb_depth_buffer(new u8[fb_depth_texture.width * fb_depth_texture.height * gl_bpp]);

    u8* temp_fb_depth_data = bytes_per_pixel == 3 ? (temp_fb_depth_buffer.get() + 1) : temp_fb_depth_buffer.get();

    if (fb_depth_texture.format == Pica::Regs::DepthFormat::D24S8) {
        for (int y = 0; y < fb_depth_texture.height; ++y) {
            for (int x = 0; x < fb_depth_texture.width; ++x) {
                const u32 coarse_y = y & ~7;
                u32 dst_offset = VideoCore::GetMortonOffset(x, y, bytes_per_pixel) + coarse_y * fb_depth_texture.width * bytes_per_pixel;
                u32 gl_pixel_index = (x + y * fb_depth_texture.width);

                u8* pixel = depth_buffer + dst_offset;
                u32 depth_stencil = *(u32*)pixel;
                ((u32*)temp_fb_depth_data)[gl_pixel_index] = (depth_stencil << 8) | (depth_stencil >> 24);
            }
        }
    } else {
        for (int y = 0; y < fb_depth_texture.height; ++y) {
            for (int x = 0; x < fb_depth_texture.width; ++x) {
                const u32 coarse_y = y & ~7;
                u32 dst_offset = VideoCore::GetMortonOffset(x, y, bytes_per_pixel) + coarse_y * fb_depth_texture.width * bytes_per_pixel;
                u32 gl_pixel_index = (x + y * fb_depth_texture.width) * gl_bpp;

                u8* pixel = depth_buffer + dst_offset;
                memcpy(&temp_fb_depth_data[gl_pixel_index], pixel, bytes_per_pixel);
            }
        }
    }

    state.texture_units[0].enabled_2d = true;
    state.texture_units[0].texture_2d = fb_depth_texture.texture.handle;
    state.Apply();

    glActiveTexture(GL_TEXTURE0);
    glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, fb_depth_texture.width, fb_depth_texture.height,
                    fb_depth_texture.gl_format, fb_depth_texture.gl_type, temp_fb_depth_buffer.get());

    state.texture_units[0].texture_2d = 0;
    state.Apply();
}

void RasterizerOpenGL::CommitColorBuffer() {
    if (last_fb_color_addr != 0) {
        u8* color_buffer = Memory::GetPhysicalPointer(last_fb_color_addr);

        if (color_buffer != nullptr) {
            u32 bytes_per_pixel = Pica::Regs::BytesPerColorPixel(fb_color_texture.format);

            std::unique_ptr<u8[]> temp_gl_color_buffer(new u8[fb_color_texture.width * fb_color_texture.height * bytes_per_pixel]);

            state.texture_units[0].enabled_2d = true;
            state.texture_units[0].texture_2d = fb_color_texture.texture.handle;
            state.Apply();

            glActiveTexture(GL_TEXTURE0);
            glGetTexImage(GL_TEXTURE_2D, 0, fb_color_texture.gl_format, fb_color_texture.gl_type, temp_gl_color_buffer.get());

            state.texture_units[0].texture_2d = 0;
            state.Apply();

            // Directly copy pixels. Internal OpenGL color formats are consistent so no conversion is necessary.
            for (int y = 0; y < fb_color_texture.height; ++y) {
                for (int x = 0; x < fb_color_texture.width; ++x) {
                    const u32 coarse_y = y & ~7;
                    u32 dst_offset = VideoCore::GetMortonOffset(x, y, bytes_per_pixel) + coarse_y * fb_color_texture.width * bytes_per_pixel;
                    u32 gl_pixel_index = x * bytes_per_pixel + y * fb_color_texture.width * bytes_per_pixel;

                    u8* pixel = color_buffer + dst_offset;
                    memcpy(pixel, &temp_gl_color_buffer[gl_pixel_index], bytes_per_pixel);
                }
            }
        }
    }
}

void RasterizerOpenGL::CommitDepthBuffer() {
    if (last_fb_depth_addr != 0) {
        // TODO: Output seems correct visually, but doesn't quite match sw renderer output. One of them is wrong.
        u8* depth_buffer = Memory::GetPhysicalPointer(last_fb_depth_addr);

        if (depth_buffer != nullptr) {
            u32 bytes_per_pixel = Pica::Regs::BytesPerDepthPixel(fb_depth_texture.format);

            // OpenGL needs 4 bpp alignment for D24
            u32 gl_bpp = bytes_per_pixel == 3 ? 4 : bytes_per_pixel;

            std::unique_ptr<u8[]> temp_gl_depth_buffer(new u8[fb_depth_texture.width * fb_depth_texture.height * gl_bpp]);

            state.texture_units[0].enabled_2d = true;
            state.texture_units[0].texture_2d = fb_depth_texture.texture.handle;
            state.Apply();

            glActiveTexture(GL_TEXTURE0);
            glGetTexImage(GL_TEXTURE_2D, 0, fb_depth_texture.gl_format, fb_depth_texture.gl_type, temp_gl_depth_buffer.get());

            state.texture_units[0].texture_2d = 0;
            state.Apply();

            u8* temp_gl_depth_data = bytes_per_pixel == 3 ? (temp_gl_depth_buffer.get() + 1) : temp_gl_depth_buffer.get();

            if (fb_depth_texture.format == Pica::Regs::DepthFormat::D24S8) {
                for (int y = 0; y < fb_depth_texture.height; ++y) {
                    for (int x = 0; x < fb_depth_texture.width; ++x) {
                        const u32 coarse_y = y & ~7;
                        u32 dst_offset = VideoCore::GetMortonOffset(x, y, bytes_per_pixel) + coarse_y * fb_depth_texture.width * bytes_per_pixel;
                        u32 gl_pixel_index = (x + y * fb_depth_texture.width);

                        u8* pixel = depth_buffer + dst_offset;
                        u32 depth_stencil = ((u32*)temp_gl_depth_data)[gl_pixel_index];
                        *(u32*)pixel = (depth_stencil >> 8) | (depth_stencil << 24);
                    }
                }
            } else {
                for (int y = 0; y < fb_depth_texture.height; ++y) {
                    for (int x = 0; x < fb_depth_texture.width; ++x) {
                        const u32 coarse_y = y & ~7;
                        u32 dst_offset = VideoCore::GetMortonOffset(x, y, bytes_per_pixel) + coarse_y * fb_depth_texture.width * bytes_per_pixel;
                        u32 gl_pixel_index = (x + y * fb_depth_texture.width) * gl_bpp;

                        u8* pixel = depth_buffer + dst_offset;
                        memcpy(pixel, &temp_gl_depth_data[gl_pixel_index], bytes_per_pixel);
                    }
                }
            }
        }
    }
}
