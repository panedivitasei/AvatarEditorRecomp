// Ring-contract sampler semantics for the native renderer: native keeps
// executing, the ring contract decides the sampler state.
// GPL-3.0, see LICENSE in src/videonative.

#pragma once

#include <cstdint>

#include <plume_render_interface.h>

namespace rex::videonative::semantics {

// Decoded ring-contract sampler state for one fetch constant. Fetch-constant
// only: pack bindings carry no per-instruction attribute overrides.
struct SamplerSpec {
  plume::RenderFilter min_filter;
  plume::RenderFilter mag_filter;
  bool aniso;
  uint32_t max_aniso;
  bool w_matters;  // 3D texture: W axis participates in addressing
  plume::RenderTextureAddressMode u, v, w;
  bool border_used;
  plume::RenderBorderColor border;
  // Real on hardware, not expressible on the current host path (host
  // textures are single-mip; ring biases LOD in-shader, never in the
  // sampler).
  float lod_bias;
  bool mip_point;
  bool ycbcr_border;
};

// Decode 6 host-endian fetch-constant dwords per ring's contract:
// SamplerInfo::Prepare + texture_util::GetClampModesForDimension +
// D3D12TextureCache::NormalizeClampMode/WriteSampler, ported.
void DecodeSamplerSpec(const uint32_t dw[6], SamplerSpec* out);

// Spec -> sampler descriptor slot: static 0-12 when semantically exact,
// dynamic 13-15 (write-once) for specs the static set can't express,
// nearest-static under pressure.
uint32_t MapToSlot(const SamplerSpec& spec);

// Call once after the static samplers (slots 0-12) are populated.
void InitSamplerArm(plume::RenderDevice* device,
                    plume::RenderDescriptorSet* sampler_set);

}  // namespace rex::videonative::semantics
