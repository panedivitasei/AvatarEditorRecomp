#include "gpu/gpu_formats.h"

#include <atomic>

#include <rex/graphics/xenos.h>
#include <rex/logging.h>

namespace ae::gpu {

namespace xenos = rex::graphics::xenos;

namespace {

// One error per distinct guest format and class, so a new format shows up in the log exactly once.
void ReportUnsupported(const char* what, uint32_t value) {
  static std::atomic<uint64_t> seen[3] = {};
  const int slot = what[0] == 'c' ? 0 : what[0] == 'd' ? 1 : 2;
  const uint64_t bit = uint64_t(1) << (value & 63);
  if (seen[slot].fetch_or(bit, std::memory_order_relaxed) & bit) return;
  REXGPU_ERROR("[gpu] unsupported {} format {}: rendering it as magenta", what, value);
}

}  // namespace

HostFormat HostFormatForColorTarget(uint32_t color_format) {
  using F = xenos::ColorRenderTargetFormat;
  switch (static_cast<F>(color_format)) {
    case F::k_8_8_8_8:
    case F::k_8_8_8_8_GAMMA:
      return HostFormat::kRGBA8;
    case F::k_2_10_10_10:
    case F::k_2_10_10_10_AS_10_10_10_10:
      return HostFormat::kRGB10A2;
    // 7e3 and the -32..32 fixed-point formats keep their range in half floats.
    case F::k_2_10_10_10_FLOAT:
    case F::k_2_10_10_10_FLOAT_AS_16_16_16_16:
    case F::k_16_16_16_16:
    case F::k_16_16_16_16_FLOAT:
      return HostFormat::kRGBA16Float;
    case F::k_16_16:
    case F::k_16_16_FLOAT:
      return HostFormat::kRG16Float;
    case F::k_32_FLOAT:
      return HostFormat::kR32Float;
    case F::k_32_32_FLOAT:
      return HostFormat::kRG32Float;
  }
  ReportUnsupported("colour target", color_format);
  return HostFormat::kUnsupported;
}

HostFormat HostFormatForDepthTarget(uint32_t depth_format) {
  // D24S8 and D24FS8 both fit a 32-bit float depth plane with 8-bit stencil.
  if (depth_format <= 1) return HostFormat::kDepth32FloatStencil8;
  ReportUnsupported("depth target", depth_format);
  return HostFormat::kUnsupported;
}

// Every format a resolve can write (xenos::ColorFormat plus the depth and EDRAM copy formats); the host texture is a
// render target, so packed 16-bit formats widen to RGBA8 and the 10/11-bit ones to half floats.
HostFormat HostFormatForTexture(uint32_t texture_format) {
  using F = xenos::TextureFormat;
  switch (static_cast<F>(texture_format)) {
    case F::k_8_8_8_8:
    case F::k_8_8_8_8_A:
    case F::k_8_8_8_8_AS_16_16_16_16:
    case F::k_8_8_8_8_GAMMA_EDRAM:
    case F::k_1_5_5_5:
    case F::k_5_6_5:
    case F::k_6_5_5:
    case F::k_4_4_4_4:
      return HostFormat::kRGBA8;
    case F::k_2_10_10_10:
    case F::k_2_10_10_10_AS_16_16_16_16:
      return HostFormat::kRGB10A2;
    // 7e3 copied out of EDRAM keeps its [0, 32) range in half floats, as the render target does.
    case F::k_2_10_10_10_FLOAT_EDRAM:
    case F::k_10_11_11:
    case F::k_11_11_10:
    case F::k_10_11_11_AS_16_16_16_16:
    case F::k_11_11_10_AS_16_16_16_16:
    case F::k_16_16_16_16_FLOAT:
    case F::k_16_16_16_16_EDRAM:
      return HostFormat::kRGBA16Float;
    case F::k_16_16_FLOAT:
    case F::k_16_16_EDRAM:
      return HostFormat::kRG16Float;
    case F::k_16_FLOAT:
      return HostFormat::kR16Float;
    case F::k_16:
      return HostFormat::kR16;
    case F::k_16_16:
      return HostFormat::kRG16;
    case F::k_16_16_16_16:
      return HostFormat::kRGBA16;
    case F::k_32_FLOAT:
      return HostFormat::kR32Float;
    case F::k_32_32_FLOAT:
      return HostFormat::kRG32Float;
    case F::k_32_32_32_32_FLOAT:
      return HostFormat::kRGBA32Float;
    case F::k_8:
    case F::k_8_A:
    case F::k_8_B:
      return HostFormat::kR8;
    case F::k_8_8:
      return HostFormat::kRG8;
    // Depth resolves: the depth plane as a float, the value the shaders compare against.
    case F::k_24_8:
    case F::k_24_8_FLOAT:
      return HostFormat::kR32Float;
    default:
      break;
  }
  ReportUnsupported("texture", texture_format);
  return HostFormat::kUnsupported;
}

const char* HostFormatName(HostFormat format) {
  switch (format) {
    case HostFormat::kUnsupported: return "unsupported";
    case HostFormat::kRGBA8: return "RGBA8";
    case HostFormat::kRGB10A2: return "RGB10A2";
    case HostFormat::kRGBA16Float: return "RGBA16F";
    case HostFormat::kRG16Float: return "RG16F";
    case HostFormat::kR32Float: return "R32F";
    case HostFormat::kRG32Float: return "RG32F";
    case HostFormat::kR8: return "R8";
    case HostFormat::kRG8: return "RG8";
    case HostFormat::kDepth32FloatStencil8: return "D32FS8";
    case HostFormat::kBC1: return "BC1";
    case HostFormat::kBC2: return "BC2";
    case HostFormat::kBC3: return "BC3";
    case HostFormat::kBC4: return "BC4";
    case HostFormat::kBC5: return "BC5";
    case HostFormat::kRGBA16: return "RGBA16";
    case HostFormat::kRG16: return "RG16";
    case HostFormat::kR16Float: return "R16F";
    case HostFormat::kRGBA32Float: return "RGBA32F";
    case HostFormat::kR16: return "R16";
  }
  return "?";
}

uint32_t HostChannelCount(HostFormat format) {
  switch (format) {
    case HostFormat::kR8:
    case HostFormat::kR16:
    case HostFormat::kR16Float:
    case HostFormat::kR32Float:
    case HostFormat::kBC4:
      return 1;
    case HostFormat::kRG8:
    case HostFormat::kRG16:
    case HostFormat::kRG16Float:
    case HostFormat::kRG32Float:
    case HostFormat::kBC5:
      return 2;
    default:
      return 4;
  }
}

}  // namespace ae::gpu
