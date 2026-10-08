// Draw capture entry points shared with the command-buffer replay: the D3D draw arguments, and building a DrawRecord
// from a device struct that is not the live one.
#pragma once

#include <cstdint>
#include <vector>

#include <rex/graphics/xenos.h>
#include <rex/ppc/context.h>

#include "gpu/gpu_records.h"

namespace ae::gpu::guest {
struct D3DDevice;
}

namespace ae::gpu::draw {

struct DrawArgs {
  uint32_t device = 0;
  rex::graphics::xenos::PrimitiveType type = rex::graphics::xenos::PrimitiveType::kNone;
  bool indexed = false;
  uint32_t start = 0;          // StartVertex, or StartIndex for indexed draws
  int32_t base_vertex = 0;     // VGT_INDX_OFFSET the draw packet writes
  uint32_t count = 0;          // vertices or indices
  bool inline_data = false;    // Begin/UP: stream data at fetch 95 and indices in the ring
  uint32_t stride = 0;
  uint32_t vertex_bytes = 0;
  bool index32 = false;
};

// The bound vertex shader as the game link-patched it at one moment: D3DVertexShader_Bind rewrites the object in
// place and flips its variant flag, so a recorded draw keeps the code it was recorded with.
struct VsSnapshot {
  uint32_t object = 0;      // D3DVertexShader address
  bool variant1 = false;
  std::vector<uint8_t> code;  // the selected variant's microcode
};

VsSnapshot SnapshotVs(const guest::D3DDevice* device);

// Completes owned preparation jobs before the guest publishes a packet to the host.
void FinishPreparation(Packet& packet);

// Builds and submits a draw from `device`, a merged copy of a recorded and a live device struct; a.device names the
// device whose shader cache applies. Inline (Begin/UP) draws are not replayed.
void Replay(PPCContext& ctx, uint8_t* base, const DrawArgs& a, const guest::D3DDevice* device, const VsSnapshot& vs);

}  // namespace ae::gpu::draw
