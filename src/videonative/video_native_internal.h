// Internal glue between video_native.cpp (guest API surface + shader
// resolution) and renderer.cpp (plume backend). GPL-3.0, see LICENSE.

#pragma once

#include <cstdint>
#include <vector>

namespace plume {
class RenderDevice;
class RenderCommandQueue;
}  // namespace plume

namespace rex::videonative {
struct ResolvedShader;
}

namespace rex::videonative::detail {

plume::RenderDevice* Device();
plume::RenderCommandQueue* Queue();

// The shaders resolved for the current draw (set by OnDraw in
// video_native.cpp before the renderer submits).
const ResolvedShader* CurrentResolvedVertexShader();
const ResolvedShader* CurrentResolvedPixelShader();
// Whether the guest has any pixel shader bound for the current draw: false =
// a deliberate depth-only draw (PS-less pipeline), true + null resolved PS =
// a pack miss (skip the draw).
bool CurrentDrawPixelShaderBound();
// Render-queue worker: installs the shader resolution captured at enqueue
// time before executing a deferred draw.
void OverrideResolvedShaders(const ResolvedShader* vs,
                             const ResolvedShader* ps, bool ps_bound);
void ClearResolvedShaderOverride();

// Built-in shader blobs from the shader pack (empty if absent). Valid once
// the pack is loaded, which precedes any pipeline creation.
const std::vector<uint8_t>& RectExpandGsDxil();
const std::vector<uint8_t>& PointExpandGsDxil();
const std::vector<uint8_t>& BlitVsDxil();
const std::vector<uint8_t>& BlitPsDxil();

}  // namespace rex::videonative::detail
