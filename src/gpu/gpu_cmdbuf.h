// D3D command buffers: while a device records, its draws and clears are kept
// with a copy of the device struct and the register tags the recording dirtied; RunCommandBuffer replays them over
// the live device, taking the recorded registers and inheriting everything else.
#pragma once

#include <array>
#include <cstdint>
#include <vector>

#include <rex/ppc/context.h>

#include "gpu/gpu_draw.h"
#include "gpu/gpu_records.h"

namespace ae::gpu::guest {
struct D3DDevice;
}

namespace ae::gpu::capture {

// D3DDevice_ClearF arguments, kept for a recorded clear.
struct ClearArgs {
  uint32_t flags = 0;
  std::array<float, 4> color{};  // RGBA
  float z = 1.0f;
  uint32_t stencil = 0;
  std::vector<Rect> rects;
};

// The clear record for these arguments against the targets `device` has bound.
ClearRecord MakeClear(const guest::D3DDevice* device, const ClearArgs& a);

}  // namespace ae::gpu::capture

namespace ae::gpu::cmdbuf {

bool Recording(uint32_t device);

// D3DDevice_BeginCommandBuffer / EndCommandBuffer on `device`; a buffer address is reused by re-recording.
void Begin(uint32_t device, uint32_t buffer, uint32_t flags);
void End(uint32_t device);

// D3DDevice_SetCommandBufferPredication: later state carries this run value (0 = every run).
void SetPredication(uint32_t device, uint32_t run);

// Called before the runtime flushes a draw or clear on a recording device: folds the pending register tags in.
void NotePending(uint32_t device);

void RecordDraw(const draw::DrawArgs& a);
void RecordClear(uint32_t device, const capture::ClearArgs& a);

// D3DDevice_RunCommandBuffer on the live `device`.
void Run(PPCContext& ctx, uint8_t* base, uint32_t device, uint32_t buffer, uint32_t select);

}  // namespace ae::gpu::cmdbuf
