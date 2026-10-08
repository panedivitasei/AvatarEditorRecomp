// Guest surface and texture formats mapped to the host formats the renderer creates.
// An unmapped format is reported once at error level and rendered as magenta.
#pragma once

#include <cstdint>

namespace ae::gpu {

enum class HostFormat : uint8_t {
  kUnsupported,  // magenta RGBA8 stand-in
  kRGBA8,
  kRGB10A2,
  kRGBA16Float,
  kRG16Float,
  kR32Float,
  kRG32Float,
  kR8,
  kRG8,
  kDepth32FloatStencil8,
  // Sampled-only formats for textures decoded from guest memory.
  kBC1,
  kBC2,
  kBC3,
  kBC4,
  kBC5,
  kRGBA16,
  kRG16,
  kR16Float,
  kRGBA32Float,
  kR16,
};

enum class ResourceKind : uint8_t {
  kColorTarget,   // EDRAM colour surface
  kDepthTarget,   // EDRAM depth/stencil surface
  kTexture,       // guest texture at a base address: resolve destination or front buffer
  kSampledTexture,  // guest texture decoded from memory, sampled only
};

// Host texture shape of a sampled texture; render targets and resolve destinations are always 2D.
enum class TextureShape : uint8_t { k2D, kCube, k3D, kStacked };

// Xenos ColorRenderTargetFormat (RB_COLOR_INFO bits 16..19).
HostFormat HostFormatForColorTarget(uint32_t color_format);
// Xenos DepthRenderTargetFormat (RB_DEPTH_INFO bit 16).
HostFormat HostFormatForDepthTarget(uint32_t depth_format);
// Xenos TextureFormat (fetch constant dword 1, bits 0..5) of a resolve destination or front buffer.
HostFormat HostFormatForTexture(uint32_t texture_format);

const char* HostFormatName(HostFormat format);

// Components the host format stores (1, 2 or 4); the rest are replicated by the view swizzle.
uint32_t HostChannelCount(HostFormat format);

}  // namespace ae::gpu
