// Sampler semantics, see sampler_semantics.h.
// GPL-3.0, see LICENSE in src/videonative.
//
// Ported ring contract:
//   src/graphics/sampler_info.cpp            SamplerInfo::Prepare
//   src/graphics/pipeline/texture/util.cpp   GetClampModesForDimension
//   src/graphics/d3d12/texture_cache.cpp     NormalizeClampMode, WriteSampler

#ifdef _WIN32
#define WIN32_LEAN_AND_MEAN
#define NOMINMAX  // windows.h min/max macros vs std::min in xenos.h
#endif

#include "semantics/sampler_semantics.h"

#include <algorithm>
#include <cstring>
#include <memory>
#include <vector>

#include <rex/graphics/xenos.h>
#include <rex/logging.h>

#include "video_native_internal.h"

namespace rex::videonative::semantics {
namespace {

namespace xen = rex::graphics::xenos;

// Ring's D3D12 kAddressModeMap ported to plume, indexed by xen::ClampMode.
// The two half-border modes have no D3D9 equivalent either; ring collapses
// them exactly like this.
constexpr plume::RenderTextureAddressMode kAddressModeMap[] = {
    plume::RenderTextureAddressMode::WRAP,         // kRepeat
    plume::RenderTextureAddressMode::MIRROR,       // kMirroredRepeat
    plume::RenderTextureAddressMode::CLAMP,        // kClampToEdge
    plume::RenderTextureAddressMode::MIRROR_ONCE,  // kMirrorClampToEdge
    plume::RenderTextureAddressMode::CLAMP,        // kClampToHalfway
    plume::RenderTextureAddressMode::MIRROR_ONCE,  // kMirrorClampToHalfway
    plume::RenderTextureAddressMode::BORDER,       // kClampToBorder
    plume::RenderTextureAddressMode::MIRROR_ONCE,  // kMirrorClampToBorder
};

xen::ClampMode Normalize(xen::ClampMode m) {
  if (m == xen::ClampMode::kClampToHalfway) {
    return xen::ClampMode::kClampToEdge;
  }
  if (m == xen::ClampMode::kMirrorClampToHalfway ||
      m == xen::ClampMode::kMirrorClampToBorder) {
    return xen::ClampMode::kMirrorClampToEdge;
  }
  return m;
}

// Static-grid axis index {0 wrap, 1 mirror, 2 clamp} for the pressure
// fallback; MIRROR_ONCE degrades to mirror, BORDER to clamp.
uint32_t AxisMode3(plume::RenderTextureAddressMode m) {
  switch (m) {
    case plume::RenderTextureAddressMode::WRAP:
      return 0;
    case plume::RenderTextureAddressMode::MIRROR:
    case plume::RenderTextureAddressMode::MIRROR_ONCE:
      return 1;
    default:
      return 2;  // CLAMP / BORDER
  }
}

bool GridExact(plume::RenderTextureAddressMode m) {
  return m == plume::RenderTextureAddressMode::WRAP ||
         m == plume::RenderTextureAddressMode::MIRROR ||
         m == plume::RenderTextureAddressMode::CLAMP;
}

// Dynamic slots 13-15: write-once samplers, matched by packed key. Never
// mutated after first write (descriptor writes race in-flight lists
// otherwise); pressure past 3 uniques falls back to nearest static.
struct DynKey {
  uint8_t min_linear, mag_linear, aniso, u, v, w, border, pad;
  uint32_t max_aniso;
};
static_assert(sizeof(DynKey) == 12, "memcmp key must be padding-free");

plume::RenderDevice* g_dev = nullptr;
plume::RenderDescriptorSet* g_set = nullptr;
std::vector<std::unique_ptr<plume::RenderSampler>> g_dynSamplers;
DynKey g_dynKeys[3] = {};
uint32_t g_dynSlots[3] = {};
uint32_t g_dynCount = 0;

}  // namespace

void DecodeSamplerSpec(const uint32_t dw[6], SamplerSpec* out) {
  xen::xe_gpu_texture_fetch_t fetch;
  static_assert(sizeof(fetch) == 24, "fetch constant is 6 dwords");
  std::memcpy(&fetch, dw, 24);

  // GetClampModesForDimension: unused axes forced to clamp-to-edge; cube
  // ignores all three.
  xen::ClampMode cx = xen::ClampMode::kClampToEdge;
  xen::ClampMode cy = xen::ClampMode::kClampToEdge;
  xen::ClampMode cz = xen::ClampMode::kClampToEdge;
  switch (fetch.dimension) {
    case xen::DataDimension::k3D:
      cz = fetch.clamp_z;
      [[fallthrough]];
    case xen::DataDimension::k2DOrStacked:
      cy = fetch.clamp_y;
      [[fallthrough]];
    case xen::DataDimension::k1D:
      cx = fetch.clamp_x;
      break;
    default:
      break;  // cube: clamp modes not applicable
  }
  cx = Normalize(cx);
  cy = Normalize(cy);
  cz = Normalize(cz);
  out->u = kAddressModeMap[uint32_t(cx)];
  out->v = kAddressModeMap[uint32_t(cy)];
  out->w = kAddressModeMap[uint32_t(cz)];
  out->w_matters = fetch.dimension == xen::DataDimension::k3D;

  // Border color is honored only when a normalized mode borders (ring forces
  // black otherwise). Plume can express white and transparent black; the two
  // YCbCr borders are recorded on the spec instead.
  out->border_used = cx == xen::ClampMode::kClampToBorder ||
                     cy == xen::ClampMode::kClampToBorder ||
                     cz == xen::ClampMode::kClampToBorder;
  out->border = plume::RenderBorderColor::TRANSPARENT_BLACK;
  out->ycbcr_border = false;
  if (out->border_used) {
    switch (fetch.border_color) {
      case xen::BorderColor::k_ABGR_White:
        out->border = plume::RenderBorderColor::OPAQUE_WHITE;
        break;
      case xen::BorderColor::k_ABGR_Black:
        break;  // ring writes (0,0,0,0)
      default:
        out->ycbcr_border = true;
        break;
    }
  }

  out->min_filter = fetch.min_filter == xen::TextureFilter::kLinear
                        ? plume::RenderFilter::LINEAR
                        : plume::RenderFilter::NEAREST;
  out->mag_filter = fetch.mag_filter == xen::TextureFilter::kLinear
                        ? plume::RenderFilter::LINEAR
                        : plume::RenderFilter::NEAREST;
  const uint32_t aniso_raw = uint32_t(fetch.aniso_filter);
  out->aniso = aniso_raw >= 1 && aniso_raw <= 5;  // kMax1To1..kMax16To1
  out->max_aniso = out->aniso ? std::min(16u, 1u << (aniso_raw - 1)) : 1u;

  out->lod_bias = float(int32_t(fetch.lod_bias)) / 32.0f;
  out->mip_point = fetch.mip_filter == xen::TextureFilter::kPoint &&
                   fetch.mip_max_level > fetch.mip_min_level;
}

uint32_t MapToSlot(const SamplerSpec& s) {
  const bool linear = s.min_filter == plume::RenderFilter::LINEAR &&
                      s.mag_filter == plume::RenderFilter::LINEAR;
  const bool point = s.min_filter == plume::RenderFilter::NEAREST &&
                     s.mag_filter == plume::RenderFilter::NEAREST;
  uint32_t slot = 0;
  bool exact = false;

  if (!s.aniso && !s.border_used) {
    if (linear && GridExact(s.u) && GridExact(s.v) &&
        (!s.w_matters || s.w == s.v)) {
      slot = 4 + AxisMode3(s.u) * 3 + AxisMode3(s.v);
      exact = true;
    } else if (point && s.u == s.v && (!s.w_matters || s.w == s.v)) {
      if (s.u == plume::RenderTextureAddressMode::WRAP) {
        slot = 2;
        exact = true;
      } else if (s.u == plume::RenderTextureAddressMode::CLAMP) {
        slot = 3;
        exact = true;
      }
    }
  }

  if (!exact) {
    DynKey key = {};
    key.min_linear = s.min_filter == plume::RenderFilter::LINEAR;
    key.mag_linear = s.mag_filter == plume::RenderFilter::LINEAR;
    key.aniso = s.aniso;
    key.u = uint8_t(s.u);
    key.v = uint8_t(s.v);
    key.w = uint8_t(s.w);
    key.border = s.border_used ? uint8_t(s.border) : 0;
    key.max_aniso = s.aniso ? s.max_aniso : 0;
    for (uint32_t i = 0; i < g_dynCount; i++) {
      if (std::memcmp(&g_dynKeys[i], &key, sizeof(key)) == 0) {
        slot = g_dynSlots[i];
        exact = true;
        break;
      }
    }
    if (!exact && g_dev && g_set && g_dynCount < 3) {
      plume::RenderSamplerDesc desc;
      desc.minFilter = s.min_filter;
      desc.magFilter = s.mag_filter;
      desc.mipmapMode = plume::RenderMipmapMode::LINEAR;
      desc.addressU = s.u;
      desc.addressV = s.v;
      desc.addressW = s.w;
      desc.anisotropyEnabled = s.aniso;
      desc.maxAnisotropy = s.aniso ? s.max_aniso : 16;
      desc.borderColor = s.border_used
                             ? s.border
                             : plume::RenderBorderColor::TRANSPARENT_BLACK;
      auto sampler = g_dev->createSampler(desc);
      slot = 13 + g_dynCount;
      g_set->setSampler(slot, sampler.get());
      g_dynSamplers.push_back(std::move(sampler));
      g_dynKeys[g_dynCount] = key;
      g_dynSlots[g_dynCount] = slot;
      g_dynCount++;
      exact = true;
    }
    if (!exact) {
      // Pressure: all 3 dynamic slots taken by other specs. Nearest static
      // keeps per-axis addressing; filter/aniso/border degrade.
      slot = 4 + AxisMode3(s.u) * 3 + AxisMode3(s.v);
    }
  }

  return slot;
}

void InitSamplerArm(plume::RenderDevice* device,
                    plume::RenderDescriptorSet* sampler_set) {
  g_dev = device;
  g_set = sampler_set;
  g_dynSamplers.clear();
  g_dynCount = 0;
}

}  // namespace rex::videonative::semantics
