// Copyright 2014 Citra Emulator Project
// Licensed under GPLv2
// Refer to the license.txt file included.

#include "common/log.h"
#include "core/hle/hle.h"
#include "core/hle/kernel/event.h"
#include "core/hle/service/dsp_dsp.h"

////////////////////////////////////////////////////////////////////////////////////////////////////
// Namespace DSP_DSP

namespace DSP_DSP {

static Handle semaphore_event;
static Handle interrupt_event;

/**
 * DSP_DSP::LoadComponent service function
 *  Outputs:
 *      1 : Result of function, 0 on success, otherwise error code
 */
void LoadComponent(Service::Interface* self) {
    u32* cmd_buff = Service::GetCommandBuffer();

    cmd_buff[1] = 0; // No error
    cmd_buff[2] = 1; // Pretend that we actually loaded the DSP firmware

    // TODO(bunnei): Implement real DSP firmware loading

    DEBUG_LOG(KERNEL, "(STUBBED) called");
}

/**
 * DSP_DSP::GetSemaphoreEventHandle service function
 *  Outputs:
 *      1 : Result of function, 0 on success, otherwise error code
 */
void GetSemaphoreEventHandle(Service::Interface* self) {
    u32* cmd_buff = Service::GetCommandBuffer();

    cmd_buff[1] = 0; // No error
    cmd_buff[3] = semaphore_event; // Event handle

    DEBUG_LOG(KERNEL, "(STUBBED) called");
}

/**
 * DSP_DSP::RegisterInterruptEvents service function
 *  Outputs:
 *      1 : Result of function, 0 on success, otherwise error code
 */
void RegisterInterruptEvents(Service::Interface* self) {
    u32* cmd_buff = Service::GetCommandBuffer();

    interrupt_event = static_cast<Handle>(cmd_buff[4]);

    cmd_buff[1] = 0; // No error

    DEBUG_LOG(KERNEL, "(STUBBED) called");
}

/**
 * DSP_DSP::WriteReg0x10 service function
 *  Outputs:
 *      1 : Result of function, 0 on success, otherwise error code
 */
void WriteReg0x10(Service::Interface* self) {
    u32* cmd_buff = Service::GetCommandBuffer();

    Kernel::SignalEvent(interrupt_event);

    cmd_buff[1] = 0; // No error

    DEBUG_LOG(KERNEL, "(STUBBED) called");
}

const Interface::FunctionInfo FunctionTable[] = {
    {0x00010040, nullptr,                 "RecvData"},
    {0x00020040, nullptr,                 "RecvDataIsReady"},
    {0x00030080, nullptr,                 "SendData"},
    {0x00040040, nullptr,                 "SendDataIsEmpty"},
    {0x00070040, WriteReg0x10,            "WriteReg0x10"},
    {0x00080000, nullptr,                 "GetSemaphore"},
    {0x00090040, nullptr,                 "ClearSemaphore"},
    {0x000B0000, nullptr,                 "CheckSemaphoreRequest"},
    {0x000C0040, nullptr,                 "ConvertProcessAddressFromDspDram"},
    {0x000D0082, nullptr,                 "WriteProcessPipe"},
    {0x001000C0, nullptr,                 "ReadPipeIfPossible"},
    {0x001100C2, LoadComponent,           "LoadComponent"},
    {0x00120000, nullptr,                 "UnloadComponent"},
    {0x00130082, nullptr,                 "FlushDataCache"},
    {0x00140082, nullptr,                 "InvalidateDCache"},
    {0x00150082, RegisterInterruptEvents, "RegisterInterruptEvents"},
    {0x00160000, GetSemaphoreEventHandle, "GetSemaphoreEventHandle"},
    {0x00170040, nullptr,                 "SetSemaphoreMask"},
    {0x00180040, nullptr,                 "GetPhysicalAddress"},
    {0x00190040, nullptr,                 "GetVirtualAddress"},
    {0x001A0042, nullptr,                 "SetIirFilterI2S1_cmd1"},
    {0x001B0042, nullptr,                 "SetIirFilterI2S1_cmd2"},
    {0x001C0082, nullptr,                 "SetIirFilterEQ"},
    {0x001F0000, nullptr,                 "GetHeadphoneStatus"},
    {0x00210000, nullptr,                 "GetIsDspOccupied"},
};

////////////////////////////////////////////////////////////////////////////////////////////////////
// Interface class

Interface::Interface() {
    semaphore_event = Kernel::CreateEvent(RESETTYPE_ONESHOT, "DSP_DSP::semaphore_event");
    interrupt_event = 0;

    Register(FunctionTable, ARRAY_SIZE(FunctionTable));
}

Interface::~Interface() {
}

} // namespace
