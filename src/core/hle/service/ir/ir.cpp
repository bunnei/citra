// Copyright 2015 Citra Emulator Project
// Licensed under GPLv2 or any later version
// Refer to the license.txt file included.

#include "common/string_util.h"
#include "core/hle/service/service.h"
#include "core/hle/service/ir/ir.h"
#include "core/hle/service/ir/ir_rst.h"
#include "core/hle/service/ir/ir_u.h"
#include "core/hle/service/ir/ir_user.h"

#include "core/hle/hle.h"
#include "core/hle/kernel/event.h"
#include "core/hle/kernel/shared_memory.h"
#include "core/hle/kernel/thread.h"

namespace Service {
namespace IR {

enum class ConnectionStatus : u8 {
    STOPPED,
    TRYING_TO_CONNECT,
    CONNECTED,
    DISCONNECTING,
    FATAL_ERROR
};

static bool already_connected = false;
static Kernel::SharedPtr<Kernel::Event> handle_event;
static Kernel::SharedPtr<Kernel::Event> status_event;
static Kernel::SharedPtr<Kernel::Event> receive_event;
static Kernel::SharedPtr<Kernel::SharedMemory> shared_memory;
static Kernel::SharedPtr<Kernel::SharedMemory> shared_memory_iruser;

void GetHandles(Service::Interface* self) {
    u32* cmd_buff = Kernel::GetCommandBuffer();

    cmd_buff[1] = RESULT_SUCCESS.raw;
    cmd_buff[2] = 0x4000000;
    cmd_buff[3] = Kernel::g_handle_table.Create(Service::IR::shared_memory).MoveFrom();
    cmd_buff[4] = Kernel::g_handle_table.Create(Service::IR::handle_event).MoveFrom();

    LOG_DEBUG(Service, "called");
}

static unsigned DecodeBitrate(u8 value) {
    static const unsigned divisor_lookup[] = {
        0, 0, 0, 10, 12, 16, 24, 32, 48, 64, 96, 120, 192, 384, 20, 30, 60, 160, 240
    };
    return 1152000 / divisor_lookup[value];
}

void InitializeIrNopShared(Service::Interface* self) {
    u32* cmd_buff = Kernel::GetCommandBuffer();

    unsigned size = cmd_buff[2];
    unsigned bitrate = DecodeBitrate(cmd_buff[6] & 0xFF);

    shared_memory_iruser = Kernel::g_handle_table.Get<Kernel::SharedMemory>(cmd_buff[8]);

    LOG_DEBUG(Service, "called size=%d, addr=0x%08X, bitrate=%d", shared_memory_iruser->size,
        shared_memory_iruser->base_address, bitrate);

    cmd_buff[1] = RESULT_SUCCESS.raw;

    u32* shared_mem = (u32*)shared_memory_iruser->GetPointer();
    shared_mem[2] = (shared_mem[2] & 0xffffff00) | (u32)ConnectionStatus::STOPPED;
    shared_mem[3] |= 0x00010000;
}

void FinalizeIrNop(Service::Interface* self) {
    u32* cmd_buff = Kernel::GetCommandBuffer();
    cmd_buff[1] = RESULT_SUCCESS.raw;
    LOG_DEBUG(Service, "called");
}

void RequireConnection(Service::Interface* self) {
    u32* cmd_buff = Kernel::GetCommandBuffer();
    cmd_buff[1] = RESULT_SUCCESS.raw;

    u32* shared_mem = (u32*)shared_memory_iruser->GetPointer();
    shared_mem[2] = 0x00000100 | (u32)(already_connected ? ConnectionStatus::CONNECTED : ConnectionStatus::TRYING_TO_CONNECT);
    shared_mem[3] = 0x00010000 | 0xa700;
    shared_mem[30] = 0x00010000;
    shared_mem[32] = 0x00080000;
    shared_mem[34] = 0x00a50000;
    shared_mem[35] = 0x00000184;
    shared_mem[36] = 0x000090a7;

    if (already_connected)
        status_event->Signal();

    LOG_DEBUG(Service, "called");
}

void Disconnect(Service::Interface* self) {
    u32* cmd_buff = Kernel::GetCommandBuffer();
    cmd_buff[1] = RESULT_SUCCESS.raw;
    cmd_buff[2] = 0x1000;
    cmd_buff[3] = 0x1000a;
    cmd_buff[4] = 0x4000304;

    status_event->Signal();

    u32* shared_mem = (u32*)shared_memory_iruser->GetPointer();
    shared_mem[2] = (shared_mem[2] & 0xffffff00) | (u32)ConnectionStatus::STOPPED;

    LOG_DEBUG(Service, "called");
}

void GetConnectionStatusEvent(Service::Interface* self) {
    u32* cmd_buff = Kernel::GetCommandBuffer();

    cmd_buff[1] = RESULT_SUCCESS.raw;
    cmd_buff[2] = 0;
    cmd_buff[3] = Kernel::g_handle_table.Create(status_event).MoveFrom();

    u32* shared_mem = (u32*)shared_memory_iruser->GetPointer();
    shared_mem[3] |= 0x00010000;

    LOG_DEBUG(Service, "called");
}

void GetReceiveEvent(Service::Interface* self) {
    u32* cmd_buff = Kernel::GetCommandBuffer();

    cmd_buff[1] = RESULT_SUCCESS.raw;
    cmd_buff[3] = Kernel::g_handle_table.Create(receive_event).MoveFrom();

    LOG_DEBUG(Service, "called");
}

void SendIrNop(Service::Interface* self) {
    u32* cmd_buff = Kernel::GetCommandBuffer();

    unsigned size = cmd_buff[2] >> 14;
    VAddr addr = cmd_buff[3];
    u32* buff = (u32*)Memory::GetPointer(addr);

    for (int i = 0; i < size; i++) {
        LOG_DEBUG(Service, "\t%08x", buff[i]);
    }

    cmd_buff[1] = RESULT_SUCCESS.raw; // 0xc8a10c0d;

    receive_event->Signal();

    LOG_DEBUG(Service, "called size=%d", size);
}

void Init() {
    using namespace Kernel;

    AddService(new IR_RST_Interface);
    AddService(new IR_U_Interface);
    AddService(new IR_User_Interface);

    using Kernel::MemoryPermission;
    shared_memory = SharedMemory::Create(0x1000, Kernel::MemoryPermission::ReadWrite,
                                         Kernel::MemoryPermission::ReadWrite, "IR:SharedMemory");

    // Create event handle(s)
    handle_event = Event::Create(RESETTYPE_ONESHOT, "IR:HandleEvent");
    status_event = Event::Create(RESETTYPE_ONESHOT, "IR:StatusEvent");
    receive_event = Event::Create(RESETTYPE_ONESHOT, "IR:ReceiveEvent");
}

void Shutdown() {
    shared_memory = nullptr;
    handle_event = nullptr;
    status_event = nullptr;
    receive_event = nullptr;
}

} // namespace IR

} // namespace Service
