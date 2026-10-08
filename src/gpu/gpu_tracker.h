// Guest-thread resource identity: which host resource stands for a guest surface or texture.
// Surfaces are keyed by their EDRAM placement (base, format, size, MSAA); textures by exact guest base address.
#pragma once

#include <cstdint>

#include "gpu/gpu_records.h"

namespace ae::gpu::tracker {

// A render target or depth surface the title bound; kind is kColorTarget or kDepthTarget.
ResourceDesc Surface(uint32_t surface_address, ResourceKind kind);

// A surface bound inside a BeginTiling bracket is one tile's size in EDRAM; draws still arrive once over the whole
// area, so the host target behind it covers width x height (at least the surface's own size).
void RegisterTiledSurface(uint32_t surface_address, uint32_t width, uint32_t height);

// The texture whose fetch constant sits in the D3DBaseTexture at texture_address; kNoResource if the header is unusable.
ResourceDesc Texture(uint32_t texture_address);

// A host texture that a resolve or swap already created at this physical base address, or kNoResource.
ResourceDesc FindResolved(uint32_t base_address, uint32_t width, uint32_t height, uint32_t guest_format);

// A CPU write or new resource owner invalidates host-only resolve contents overlapping this physical range.
void InvalidateResolved(uint32_t base, uint32_t size);
// The host destroyed this resolve destination; its alias goes so the next resolve makes a new one.
void ForgetResolved(uint32_t id);
// Resolve destination ids that lost their alias since the last call, for the host to destroy.
void TakeRetiredResolves(std::vector<uint32_t>& ids);

// Bumps whenever a resolve or swap texture is created or retired, so cached FindResolved answers know when to ask again.
uint32_t ResolveEpoch();

// A fresh host resource id, shared with the surface and texture ids above.
uint32_t NewId();

}  // namespace ae::gpu::tracker
