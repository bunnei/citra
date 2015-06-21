// Copyright 2014 Citra Emulator Project
// Licensed under GPLv2 or any later version
// Refer to the license.txt file included.

#include <thread>

#include "common/emu_window.h"
#include "common/logging/log.h"
#include "common/thread.h"

#include "core/core.h"
#include "core/settings.h"
#include "core/hle/service/gsp_gpu.h"
#include "core/core_timing.h"

#include "video_core.h"
#include "renderer_base.h"
#include "renderer_opengl/renderer_opengl.h"

#include "gpu_debugger.h"

#include "pica.h"

////////////////////////////////////////////////////////////////////////////////////////////////////
// Video Core namespace

extern GraphicsDebugger g_debugger;

namespace VideoCore {

EmuWindow*      g_emu_window     = nullptr; ///< Frontend emulator window
RendererBase*   g_renderer       = nullptr; ///< Renderer plugin
bool            g_multi_threaded = true;
std::thread*    g_thread         = nullptr;

std::atomic<bool> g_hw_renderer_enabled;
std::atomic<bool> g_hw_vertex_shaders_enabled;
std::atomic<bool> g_high_res_enabled;

////////////////////////////////////////////////////////////////////////////////////////////////////

static Common::Event render_start_event;
static Common::Event render_done_event;

static std::mutex write_mutex, run_mutex;
static std::atomic<bool> running = false;

static std::vector<std::pair<u32, u32>> write_queue;

static int thread_sync_event = 0;
static void ThreadSyncCallback(u64 userdata, int cycles_late) {
    VideoCore::WaitForRender_Done();
}

bool IsEmpty() {
    std::lock_guard<std::mutex> lock(write_mutex);
    return write_queue.empty();
}

void WriteGPURegister(u32 id, u32 data) {
    std::lock_guard<std::mutex> lock(write_mutex);
    write_queue.insert(write_queue.begin(), { id, data });
}


// Call from GPU thread
//void WaitForRender_Start() {
//    // Wait until a new batch of commands is ready to be processed...
//    render_start_event.Wait();
//}

// Call from Core thread
void WaitForRender_Done() {
    if (running) {
        std::lock_guard<std::mutex> lock(run_mutex);
    }
}

void RenderLoop() {
    //RenderDone();

    std::pair<u32, u32> next_write;
    while (true) {

        if (!IsEmpty())
        {
            std::lock_guard<std::mutex> lock(run_mutex);

            // Pause CPU thread after so many ticks for the GPU to catch up (4096 is totally arbitrary)
            CoreTiming::ScheduleEvent(4096, thread_sync_event, 0);
            
            running = true;
            g_emu_window->MakeCurrent();

            do {
                std::lock_guard<std::mutex> lock(write_mutex);

                next_write = write_queue.back();
                write_queue.pop_back();

                GPU::Write<u32>(next_write.first, next_write.second);

            } while (!IsEmpty());

            CoreTiming::UnscheduleEvent(thread_sync_event, 0);

            g_emu_window->DoneCurrent();
            running = false;
        }

    }
}

////////////////////////////////////////////////////////////////////////////////////////////////////

/// Initialize the video core
void Init(EmuWindow* emu_window) {
    Pica::Init();

    g_emu_window = emu_window;
    g_renderer = new RendererOpenGL();
    g_renderer->SetWindow(g_emu_window);
    g_renderer->Init();

    // Thread sync variables
    running = false;
    write_queue.clear();
    render_start_event.Reset();
    render_done_event.Reset();

    thread_sync_event = CoreTiming::RegisterEvent("ThreadSyncCallback", ThreadSyncCallback);

    g_thread = new std::thread(RenderLoop);

    LOG_DEBUG(Render, "initialized OK");
}

/// Shutdown the video core
void Shutdown() {
    Pica::Shutdown();

    delete g_renderer;

    LOG_DEBUG(Render, "shutdown OK");
}

} // namespace
