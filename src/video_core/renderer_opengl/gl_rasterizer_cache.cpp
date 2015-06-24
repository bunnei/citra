// Copyright 2015 Citra Emulator Project
// Licensed under GPLv2 or any later version
// Refer to the license.txt file included.

#include "common/make_unique.h"
#include "common/math_util.h"
#include "common/vector_math.h"

#include "core/memory.h"

#include "video_core/renderer_opengl/gl_shaders.h"
#include "video_core/renderer_opengl/gl_shader_constructor.h"
#include "video_core/renderer_opengl/gl_rasterizer_cache.h"
#include "video_core/renderer_opengl/pica_to_gl.h"
#include "video_core/debug_utils/debug_utils.h"

RasterizerCacheOpenGL::~RasterizerCacheOpenGL() {
    FullFlush();
}

void RasterizerCacheOpenGL::LoadAndBindTexture(OpenGLState& state, unsigned texture_unit, const Pica::Regs::FullTextureConfig& config) {
    PAddr texture_addr = config.config.GetPhysicalAddress();

    const auto cached_texture = texture_cache.find(texture_addr);

    if (cached_texture != texture_cache.end()) {
        state.texture_units[texture_unit].texture_2d = cached_texture->second->texture.handle;
        state.Apply();
    } else {
        std::unique_ptr<CachedTexture> new_texture = Common::make_unique<CachedTexture>();

        new_texture->texture.Create();
        state.texture_units[texture_unit].texture_2d = new_texture->texture.handle;
        state.Apply();

        // TODO: Need to choose filters that correspond to PICA once register is declared
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);

        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, PicaToGL::WrapMode(config.config.wrap_s));
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, PicaToGL::WrapMode(config.config.wrap_t));

        const auto info = Pica::DebugUtils::TextureInfo::FromPicaRegister(config.config, config.format);

        u32 bpp = Pica::Regs::NibblesPerPixel(info.format) / 2;
        if (bpp == 0) {
            bpp = 1;
        }

        new_texture->width = info.width;
        new_texture->height = info.height;
        new_texture->size = info.width * info.height * bpp;

        u8* texture_src_data = Memory::GetPhysicalPointer(texture_addr);
        std::unique_ptr<Math::Vec4<u8>[]> temp_texture_buffer_rgba(new Math::Vec4<u8>[info.width * info.height]);

        for (int y = 0; y < info.height; ++y) {
            for (int x = 0; x < info.width; ++x) {
                temp_texture_buffer_rgba[x + info.width * y] = Pica::DebugUtils::LookupTexture(texture_src_data, x, info.height - 1 - y, info);
            }
        }

        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, info.width, info.height, 0, GL_RGBA, GL_UNSIGNED_BYTE, temp_texture_buffer_rgba.get());

        Memory::SetTextureMem(texture_addr, new_texture->size);

        texture_cache.emplace(texture_addr, std::move(new_texture));
    }
}

bool RasterizerCacheOpenGL::LoadAndBindShader(bool force_reload, OpenGLState& state, u32 main_offset, const u32* shader_data, const u32* swizzle_data) {
    std::string cache_key = std::to_string(main_offset) + std::string(reinterpret_cast<const char*>(shader_data), 1024) + std::string(reinterpret_cast<const char*>(swizzle_data), 1024);

    if (!force_reload && cache_key == cur_shader_key) {
        return false;
    }

    cur_shader_key = cache_key;

    auto cached_shader = vertex_shader_cache.find(cache_key);

    if (cached_shader != vertex_shader_cache.end()) {
        state.draw.shader_program = cached_shader->second->handle;
    } else {
        std::unique_ptr<OGLShader> new_shader = Common::make_unique<OGLShader>();

        new_shader->Create(PICAVertexShaderToGLSL(main_offset, shader_data, swizzle_data).c_str(), GLShaders::g_fragment_shader_hw);
        std::string shader_string = PICAVertexShaderToGLSL(main_offset, shader_data, swizzle_data);
        //FILE *file = fopen("s.txt", "a");
        //fwrite(shader_string.c_str(), 1, shader_string.length(), file);
        //fclose(file);
        LOG_CRITICAL(Render_OpenGL, "%s", shader_string.c_str());
        state.draw.shader_program = new_shader->handle;

        vertex_shader_cache.emplace(cache_key, std::move(new_shader));
    }

    state.Apply();

    return true;
}

void RasterizerCacheOpenGL::NotifyFlush(PAddr addr, u32 size) {
    // Flush any texture that falls in the flushed region
    // TODO: Optimize by also inserting upper bound (addr + size) of each texture into the same map and also narrow using lower_bound
    auto cache_upper_bound = texture_cache.upper_bound(addr + size);
    for (auto it = texture_cache.begin(); it != cache_upper_bound;) {
        if (MathUtil::IntervalsIntersect(addr, size, it->first, it->second->size)) {
            Memory::UnSetTextureMem(it->first, it->second->size);
            it = texture_cache.erase(it);
        } else {
            ++it;
        }
    }
}

void RasterizerCacheOpenGL::FullFlush() {
    texture_cache.clear();
    vertex_shader_cache.clear();
}
