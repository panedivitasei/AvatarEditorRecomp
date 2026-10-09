#include "gpu/gpu_texture_decode.h"
#include "gpu/gpu_burst_timing.h"
#include "gpu/precise_sleep.h"

#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstring>
#include <deque>
#include <functional>
#include <thread>

#include <rex/graphics/pipeline/texture/info.h>
#include <rex/graphics/pipeline/texture/util.h>
#include <rex/logging.h>
#include <rex/system/kernel_state.h>

namespace ae::gpu {

namespace {

namespace xenos = rex::graphics::xenos;
namespace texture_util = rex::graphics::texture_util;
using F = xenos::TextureFormat;

// kCopy moves guest blocks unchanged, k1555/k565/k655/k4444 widen 16-bit texels to RGBA8, and kExpand unpacks every
// component to float with its fetch-constant sign applied.
enum class Convert : uint8_t { kNone, k1555, k565, k655, k4444, kExpand };

// How kExpand reads one component: bit offset and width in the swapped texel, and how the bits are interpreted.
enum class Kind : uint8_t { kUnorm, kHalf, kFloat, kDepthUnorm24, kDepthFloat24 };
struct Component {
  uint8_t shift = 0;
  uint8_t bits = 0;
};
struct Unpack {
  Kind kind = Kind::kUnorm;
  uint8_t count = 0;           // components the format stores; the rest replicate the last one
  Component c[4];
  uint8_t block = 0;           // 0 = one texel per block, else the compressed decoder below
};
enum Block : uint8_t { kBlockNone, kBlockBC1, kBlockBC2, kBlockBC3, kBlockBC4, kBlockBC5, kBlockDxt3a, kBlockCtx1 };

struct FormatMap {
  HostFormat host = HostFormat::kUnsupported;
  Convert convert = Convert::kNone;
  Unpack unpack;
};

constexpr Unpack Packed(uint8_t count, Component x, Component y = {}, Component z = {}, Component w = {}) {
  Unpack u;
  u.count = count;
  u.c[0] = x, u.c[1] = y, u.c[2] = z, u.c[3] = w;
  return u;
}
constexpr Unpack Typed(Kind kind, uint8_t count, uint8_t bits) {
  Unpack u;
  u.kind = kind;
  u.count = count;
  for (uint8_t i = 0; i < count; ++i) u.c[i] = {uint8_t(i * bits), bits};
  return u;
}
constexpr Unpack Compressed(uint8_t count, uint8_t block) {
  Unpack u;
  u.count = count;
  u.block = block;
  for (uint8_t i = 0; i < 4; ++i) u.c[i] = {0, block == kBlockDxt3a ? uint8_t(4) : uint8_t(8)};
  return u;
}

// Component layouts follow the SDK texture cache's load shaders: X sits in the low bits of the swapped texel.
FormatMap MapFormat(F format) {
  switch (format) {
    case F::k_8:
    case F::k_8_A:
    case F::k_8_B: return {HostFormat::kR8, Convert::kNone, Packed(1, {0, 8})};
    case F::k_8_8: return {HostFormat::kRG8, Convert::kNone, Packed(2, {0, 8}, {8, 8})};
    case F::k_1_5_5_5: return {HostFormat::kRGBA8, Convert::k1555, Packed(4, {0, 5}, {5, 5}, {10, 5}, {15, 1})};
    case F::k_5_6_5: return {HostFormat::kRGBA8, Convert::k565, Packed(3, {0, 5}, {5, 6}, {11, 5})};
    case F::k_6_5_5: return {HostFormat::kRGBA8, Convert::k655, Packed(3, {0, 5}, {5, 5}, {10, 6})};
    case F::k_4_4_4_4: return {HostFormat::kRGBA8, Convert::k4444, Packed(4, {0, 4}, {4, 4}, {8, 4}, {12, 4})};
    case F::k_8_8_8_8:
    case F::k_8_8_8_8_A:
    case F::k_8_8_8_8_AS_16_16_16_16: return {HostFormat::kRGBA8, Convert::kNone, Typed(Kind::kUnorm, 4, 8)};
    case F::k_2_10_10_10:
    case F::k_2_10_10_10_AS_16_16_16_16:
      return {HostFormat::kRGB10A2, Convert::kNone, Packed(4, {0, 10}, {10, 10}, {20, 10}, {30, 2})};
    case F::k_10_11_11:
    case F::k_10_11_11_AS_16_16_16_16:
      return {HostFormat::kRGBA16Float, Convert::kExpand, Packed(3, {0, 11}, {11, 11}, {22, 10})};
    case F::k_11_11_10:
    case F::k_11_11_10_AS_16_16_16_16:
      return {HostFormat::kRGBA16Float, Convert::kExpand, Packed(3, {0, 10}, {10, 11}, {21, 11})};
    case F::k_DXT1:
    case F::k_DXT1_AS_16_16_16_16: return {HostFormat::kBC1, Convert::kNone, Compressed(4, kBlockBC1)};
    case F::k_DXT2_3:
    case F::k_DXT2_3_AS_16_16_16_16: return {HostFormat::kBC2, Convert::kNone, Compressed(4, kBlockBC2)};
    case F::k_DXT4_5:
    case F::k_DXT4_5_AS_16_16_16_16: return {HostFormat::kBC3, Convert::kNone, Compressed(4, kBlockBC3)};
    case F::k_DXT5A: return {HostFormat::kBC4, Convert::kNone, Compressed(1, kBlockBC4)};
    case F::k_DXN: return {HostFormat::kBC5, Convert::kNone, Compressed(2, kBlockBC5)};
    case F::k_DXT3A: return {HostFormat::kRGBA16Float, Convert::kExpand, Compressed(1, kBlockDxt3a)};
    case F::k_CTX1: return {HostFormat::kRGBA16Float, Convert::kExpand, Compressed(2, kBlockCtx1)};
    case F::k_16: return {HostFormat::kR16, Convert::kNone, Typed(Kind::kUnorm, 1, 16)};
    case F::k_16_16: return {HostFormat::kRG16, Convert::kNone, Typed(Kind::kUnorm, 2, 16)};
    case F::k_16_16_16_16: return {HostFormat::kRGBA16, Convert::kNone, Typed(Kind::kUnorm, 4, 16)};
    // The EXPAND formats hold half floats, as the SDK texture cache loads them.
    case F::k_16_FLOAT:
    case F::k_16_EXPAND: return {HostFormat::kR16Float, Convert::kNone, Typed(Kind::kHalf, 1, 16)};
    case F::k_16_16_FLOAT:
    case F::k_16_16_EXPAND: return {HostFormat::kRG16Float, Convert::kNone, Typed(Kind::kHalf, 2, 16)};
    case F::k_16_16_16_16_FLOAT:
    case F::k_16_16_16_16_EXPAND: return {HostFormat::kRGBA16Float, Convert::kNone, Typed(Kind::kHalf, 4, 16)};
    case F::k_32_FLOAT: return {HostFormat::kR32Float, Convert::kNone, Typed(Kind::kFloat, 1, 32)};
    case F::k_32_32_FLOAT: return {HostFormat::kRG32Float, Convert::kNone, Typed(Kind::kFloat, 2, 32)};
    case F::k_32_32_32_32_FLOAT: return {HostFormat::kRGBA32Float, Convert::kNone, Typed(Kind::kFloat, 4, 32)};
    case F::k_32_32_32_FLOAT: return {HostFormat::kRGBA32Float, Convert::kExpand, Typed(Kind::kFloat, 3, 32)};
    // Depth read back from memory: the 24-bit depth above the stencil byte, as the float the shaders compare.
    case F::k_24_8: {
      Unpack u = Packed(1, {8, 24});
      u.kind = Kind::kDepthUnorm24;
      return {HostFormat::kR32Float, Convert::kExpand, u};
    }
    case F::k_24_8_FLOAT: {
      Unpack u = Packed(1, {8, 24});
      u.kind = Kind::kDepthFloat24;
      return {HostFormat::kR32Float, Convert::kExpand, u};
    }
    default: return {};
  }
}

bool IsBC(HostFormat f) {
  return f == HostFormat::kBC1 || f == HostFormat::kBC2 || f == HostFormat::kBC3 || f == HostFormat::kBC4 ||
         f == HostFormat::kBC5;
}

void ReportOnce(const char* what, uint32_t value) {
  static std::atomic<uint64_t> seen[3] = {};
  const uint64_t bit = uint64_t(1) << (value & 63);
  auto& word = seen[what[0] == 'd' ? 1 : what[0] == 's' ? 2 : 0];
  if (word.fetch_or(bit, std::memory_order_relaxed) & bit) return;
  REXGPU_ERROR("[gpu] texture {} {} is not decoded yet; it samples as magenta", what, value);
}

}  // namespace

void SwapBlock(xenos::Endian endian, uint8_t* out, const uint8_t* in, uint32_t size) {
  switch (endian) {
    case xenos::Endian::k8in16:
      for (uint32_t i = 0; i + 1 < size; i += 2) out[i] = in[i + 1], out[i + 1] = in[i];
      break;
    case xenos::Endian::k8in32:
      for (uint32_t i = 0; i + 3 < size; i += 4) {
        out[i] = in[i + 3], out[i + 1] = in[i + 2], out[i + 2] = in[i + 1], out[i + 3] = in[i];
      }
      break;
    case xenos::Endian::k16in32:
      for (uint32_t i = 0; i + 3 < size; i += 4) {
        out[i] = in[i + 2], out[i + 1] = in[i + 3], out[i + 2] = in[i], out[i + 3] = in[i + 1];
      }
      break;
    default:
      std::memcpy(out, in, size);
      break;
  }
}

namespace {

uint8_t Expand(uint32_t v, uint32_t bits) { return uint8_t((v * 255 + ((1u << bits) - 1) / 2) / ((1u << bits) - 1)); }

// Component X sits in the low bits of the swapped word; a three-component format repeats Z in W, as the SDK's
// texture cache swizzles it.
void ConvertTexel(Convert convert, const uint8_t* in, uint8_t* out) {
  const uint32_t v = uint32_t(in[0]) | (uint32_t(in[1]) << 8);
  switch (convert) {
    case Convert::k565:
      out[0] = Expand(v & 31, 5), out[1] = Expand((v >> 5) & 63, 6), out[2] = Expand((v >> 11) & 31, 5);
      out[3] = out[2];
      break;
    case Convert::k655:
      out[0] = Expand(v & 31, 5), out[1] = Expand((v >> 5) & 31, 5), out[2] = Expand((v >> 10) & 63, 6);
      out[3] = out[2];
      break;
    case Convert::k1555:
      out[0] = Expand(v & 31, 5), out[1] = Expand((v >> 5) & 31, 5), out[2] = Expand((v >> 10) & 31, 5);
      out[3] = (v >> 15) ? 255 : 0;
      break;
    case Convert::k4444:
      out[0] = Expand(v & 15, 4), out[1] = Expand((v >> 4) & 15, 4), out[2] = Expand((v >> 8) & 15, 4);
      out[3] = Expand(v >> 12, 4);
      break;
    default:
      break;
  }
}

// BC colour block (8 bytes) into a 4x4 RGBA8 tile; bc1 enables the punch-through mode.
void DecodeColorBlock(const uint8_t* b, uint8_t out[16][4], bool bc1) {
  const uint32_t c0 = b[0] | (b[1] << 8), c1 = b[2] | (b[3] << 8);
  uint8_t pal[4][4];
  auto unpack = [](uint32_t c, uint8_t* p) {
    p[0] = Expand((c >> 11) & 31, 5), p[1] = Expand((c >> 5) & 63, 6), p[2] = Expand(c & 31, 5), p[3] = 255;
  };
  unpack(c0, pal[0]);
  unpack(c1, pal[1]);
  if (!bc1 || c0 > c1) {
    for (int i = 0; i < 3; ++i) {
      pal[2][i] = uint8_t((2 * pal[0][i] + pal[1][i]) / 3);
      pal[3][i] = uint8_t((pal[0][i] + 2 * pal[1][i]) / 3);
    }
    pal[2][3] = pal[3][3] = 255;
  } else {
    for (int i = 0; i < 3; ++i) pal[2][i] = uint8_t((pal[0][i] + pal[1][i]) / 2), pal[3][i] = 0;
    pal[2][3] = 255;
    pal[3][3] = 0;
  }
  const uint32_t idx = b[4] | (b[5] << 8) | (b[6] << 16) | (uint32_t(b[7]) << 24);
  for (int i = 0; i < 16; ++i) std::memcpy(out[i], pal[(idx >> (2 * i)) & 3], 4);
}

// BC4-style interpolated single channel block (8 bytes).
void DecodeAlphaBlock(const uint8_t* b, uint8_t out[16]) {
  uint8_t pal[8];
  pal[0] = b[0];
  pal[1] = b[1];
  if (pal[0] > pal[1]) {
    for (int i = 1; i < 7; ++i) pal[i + 1] = uint8_t(((7 - i) * pal[0] + i * pal[1]) / 7);
  } else {
    for (int i = 1; i < 5; ++i) pal[i + 1] = uint8_t(((5 - i) * pal[0] + i * pal[1]) / 5);
    pal[6] = 0;
    pal[7] = 255;
  }
  uint64_t bits = 0;
  for (int i = 0; i < 6; ++i) bits |= uint64_t(b[2 + i]) << (8 * i);
  for (int i = 0; i < 16; ++i) out[i] = pal[(bits >> (3 * i)) & 7];
}

void DecodeBCBlock(HostFormat f, const uint8_t* b, uint8_t out[16][4]) {
  uint8_t a[16];
  switch (f) {
    case HostFormat::kBC1:
      DecodeColorBlock(b, out, true);
      break;
    case HostFormat::kBC2:
      DecodeColorBlock(b + 8, out, false);
      for (int i = 0; i < 16; ++i) out[i][3] = Expand((b[i / 2] >> ((i & 1) * 4)) & 15, 4);
      break;
    case HostFormat::kBC3:
      DecodeColorBlock(b + 8, out, false);
      DecodeAlphaBlock(b, a);
      for (int i = 0; i < 16; ++i) out[i][3] = a[i];
      break;
    case HostFormat::kBC4:
      DecodeAlphaBlock(b, a);
      for (int i = 0; i < 16; ++i) out[i][0] = a[i], out[i][1] = out[i][2] = 0, out[i][3] = 255;
      break;
    case HostFormat::kBC5: {
      uint8_t g[16];
      DecodeAlphaBlock(b, a);
      DecodeAlphaBlock(b + 8, g);
      for (int i = 0; i < 16; ++i) out[i][0] = a[i], out[i][1] = g[i], out[i][2] = 0, out[i][3] = 255;
      break;
    }
    default:
      break;
  }
}

// One compressed guest block into 16 texels of raw component values (8-bit, or 4-bit for DXT3A).
void DecodeRawBlock(uint8_t block, const uint8_t* b, uint32_t raw[16][4]) {
  uint8_t tile[16][4] = {};
  switch (block) {
    case kBlockBC1: DecodeBCBlock(HostFormat::kBC1, b, tile); break;
    case kBlockBC2: DecodeBCBlock(HostFormat::kBC2, b, tile); break;
    case kBlockBC3: DecodeBCBlock(HostFormat::kBC3, b, tile); break;
    case kBlockBC4: DecodeBCBlock(HostFormat::kBC4, b, tile); break;
    case kBlockBC5: DecodeBCBlock(HostFormat::kBC5, b, tile); break;
    case kBlockDxt3a:
      // The alpha half of a DXT3 block: sixteen 4-bit values.
      for (int i = 0; i < 16; ++i) tile[i][0] = uint8_t((b[i / 2] >> ((i & 1) * 4)) & 15);
      break;
    case kBlockCtx1: {
      // Two 8:8 endpoints and 2-bit indices; the middle entries interpolate at thirds, as the SDK's CTX1 load does.
      const uint8_t e[4][2] = {{b[0], b[1]},
                               {b[2], b[3]},
                               {uint8_t((2 * b[0] + b[2]) / 3), uint8_t((2 * b[1] + b[3]) / 3)},
                               {uint8_t((b[0] + 2 * b[2]) / 3), uint8_t((b[1] + 2 * b[3]) / 3)}};
      const uint32_t idx = b[4] | (b[5] << 8) | (b[6] << 16) | (uint32_t(b[7]) << 24);
      for (int i = 0; i < 16; ++i) {
        const uint32_t k = (idx >> (2 * i)) & 3;
        tile[i][0] = e[k][0], tile[i][1] = e[k][1];
      }
      break;
    }
    default:
      break;
  }
  for (int i = 0; i < 16; ++i) {
    for (int c = 0; c < 4; ++c) raw[i][c] = tile[i][c];
  }
}

float HalfToFloat(uint32_t h) {
  const uint32_t sign = (h >> 15) & 1, exponent = (h >> 10) & 31, mantissa = h & 1023;
  float v;
  if (exponent == 0) v = std::ldexp(float(mantissa), -24);
  else if (exponent == 31) v = mantissa ? NAN : INFINITY;
  else v = std::ldexp(float(mantissa | 1024), int(exponent) - 25);
  return sign ? -v : v;
}

uint16_t FloatToHalf(float f) {
  uint32_t x;
  std::memcpy(&x, &f, 4);
  const uint32_t sign = (x >> 16) & 0x8000;
  const int32_t exponent = int32_t((x >> 23) & 0xFF) - 127 + 15;
  uint32_t mantissa = x & 0x7FFFFF;
  if (((x >> 23) & 0xFF) == 0xFF) return uint16_t(sign | 0x7C00 | (mantissa ? 0x200 : 0));
  if (exponent >= 31) return uint16_t(sign | 0x7C00);
  if (exponent <= 0) {
    if (exponent < -10) return uint16_t(sign);
    mantissa |= 0x800000;
    const uint32_t shift = uint32_t(14 - exponent);
    uint32_t half = mantissa >> shift;
    if ((mantissa >> (shift - 1)) & 1) ++half;
    return uint16_t(sign | half);
  }
  uint32_t half = sign | (uint32_t(exponent) << 10) | (mantissa >> 13);
  if (mantissa & 0x1000) ++half;  // round half up; carries into the exponent correctly
  return uint16_t(half);
}

// One component to float under its fetch-constant sign (xenos::TextureSign); float formats ignore the sign.
float ConvertComponent(const Unpack& u, uint32_t raw, uint32_t bits, xenos::TextureSign sign) {
  switch (u.kind) {
    case Kind::kHalf: return HalfToFloat(raw);
    case Kind::kFloat: {
      float f;
      std::memcpy(&f, &raw, 4);
      return f;
    }
    case Kind::kDepthUnorm24: return xenos::UNorm24To32(raw);
    case Kind::kDepthFloat24: return xenos::Float20e4To32(raw);
    default: break;
  }
  const uint32_t max = bits >= 32 ? 0xFFFFFFFFu : (1u << bits) - 1;
  switch (sign) {
    case xenos::TextureSign::kSigned: {
      // Two's complement, with the most negative value clamped to -1 as the SDK's fetch notes.
      if (bits < 2) return float(raw);
      const int32_t v = (raw & (1u << (bits - 1))) ? int32_t(raw) - int32_t(1u << bits) : int32_t(raw);
      return std::max(float(v) / float((1u << (bits - 1)) - 1), -1.0f);
    }
    case xenos::TextureSign::kUnsignedBiased: return float(raw) / float(max) * 2.0f - 1.0f;
    case xenos::TextureSign::kGamma: return xenos::PWLGammaToLinear(float(raw) / float(max));
    default: return float(raw) / float(max);
  }
}

// Reads the texel's component bits from the swapped block (up to 128 bits).
uint32_t Bits(const uint8_t* texel, uint32_t shift, uint32_t bits) {
  uint32_t out = 0;
  for (uint32_t i = 0; i < bits; ++i) {
    const uint32_t at = shift + i;
    out |= uint32_t((texel[at >> 3] >> (at & 7)) & 1) << i;
  }
  return out;
}

void WriteExpanded(const float v[4], HostFormat host, uint8_t* out) {
  if (host == HostFormat::kRGBA32Float) {
    std::memcpy(out, v, 16);
  } else if (host == HostFormat::kR32Float) {
    std::memcpy(out, v, 4);
  } else {
    for (int c = 0; c < 4; ++c) {
      const uint16_t h = FloatToHalf(v[c]);
      std::memcpy(out + c * 2, &h, 2);
    }
  }
}

// Fetch constants in the device hold physical pages; resource headers can hold a CPU view, which is converted.
uint32_t PhysicalAddress(uint32_t page_field) {
  const uint32_t address = page_field << 12;
  if (address < 0x20000000) return address;
  const uint32_t physical = REX_KERNEL_MEMORY()->GetPhysicalAddress(address);
  return physical != UINT32_MAX ? physical : address & 0x1FFFFFFF;
}

xenos::TextureSign SignOf(const xenos::xe_gpu_texture_fetch_t& f, uint32_t component) {
  return xenos::TextureSign((f.dword_0 >> (2 + 2 * component)) & 3);
}

}  // namespace

void DecodeBlockRGBA(HostFormat format, const uint8_t* block, uint8_t out[16][4]) { DecodeBCBlock(format, block, out); }

uint32_t HostBlockSize(HostFormat format) { return IsBC(format) ? 4 : 1; }

uint32_t HostBlockBytes(HostFormat format) {
  switch (format) {
    case HostFormat::kR8: return 1;
    case HostFormat::kRG8:
    case HostFormat::kR16:
    case HostFormat::kR16Float: return 2;
    case HostFormat::kRGBA8:
    case HostFormat::kRGB10A2:
    case HostFormat::kRG16Float:
    case HostFormat::kR32Float:
    case HostFormat::kRG16:
    case HostFormat::kUnsupported: return 4;
    case HostFormat::kRGBA16Float:
    case HostFormat::kRG32Float:
    case HostFormat::kRGBA16:
    case HostFormat::kBC1:
    case HostFormat::kBC4: return 8;
    case HostFormat::kRGBA32Float:
    case HostFormat::kBC2:
    case HostFormat::kBC3:
    case HostFormat::kBC5: return 16;
    case HostFormat::kDepth32FloatStencil8: return 8;
  }
  return 4;
}

std::array<uint8_t, 4> ComposeSwizzle(HostFormat format, const std::array<uint8_t, 4>& fetch_swizzle) {
  const uint32_t channels = HostChannelCount(format);
  std::array<uint8_t, 4> out = fetch_swizzle;
  for (auto& s : out) {
    if (s < 4) s = uint8_t(std::min<uint32_t>(s, channels - 1));
  }
  return out;
}

bool DescribeTexture(const xenos::xe_gpu_texture_fetch_t& fetch, GuestTextureDesc& out) {
  BurstTiming timing(4);
  out = {};
  out.fetch = fetch;
  switch (fetch.dimension) {
    case xenos::DataDimension::k2DOrStacked:
      out.shape = fetch.stacked ? TextureShape::kStacked : TextureShape::k2D;
      break;
    case xenos::DataDimension::kCube: out.shape = TextureShape::kCube; break;
    case xenos::DataDimension::k3D: out.shape = TextureShape::k3D; break;
    default:
      ReportOnce("dimension", uint32_t(fetch.dimension));
      return false;
  }
  FormatMap map = MapFormat(fetch.format);
  if (map.host == HostFormat::kUnsupported) {
    ReportOnce("format", uint32_t(fetch.format));
    return false;
  }
  uint32_t w1, h1, d1, base_page, mip_page, mip_min, mip_max;
  texture_util::GetSubresourcesFromFetchConstant(fetch, &w1, &h1, &d1, &base_page, &mip_page, &mip_min, &mip_max);
  out.width = w1 + 1;
  out.height = h1 + 1;
  out.depth = out.shape == TextureShape::k3D ? d1 + 1 : 1;
  out.layers = out.shape == TextureShape::kCube ? 6 : out.shape == TextureShape::kStacked ? d1 + 1 : 1;
  out.mip_levels = mip_max + 1;
  out.mip_min = mip_min;
  out.mip_max = mip_max;
  out.base_address = PhysicalAddress(fetch.base_address);
  out.mip_address = mip_page ? PhysicalAddress(fetch.mip_address) : 0;
  out.host_format = map.host;
  // Any sign other than unsigned on a stored fixed-point component goes through the float conversion.
  const bool fixed = map.unpack.kind == Kind::kUnorm;
  for (uint32_t c = 0; c < map.unpack.count && fixed; ++c) {
    if (SignOf(fetch, c) != xenos::TextureSign::kUnsigned) out.expand = true;
  }
  if (map.convert == Convert::kExpand) out.expand = true;
  if (out.expand) {
    uint32_t widest = 0;
    for (uint32_t c = 0; c < map.unpack.count; ++c) widest = std::max<uint32_t>(widest, map.unpack.c[c].bits);
    const bool one = map.host == HostFormat::kR32Float && map.unpack.count == 1;
    out.host_format = one ? HostFormat::kR32Float
                          : (widest > 11 || map.unpack.kind == Kind::kFloat) ? HostFormat::kRGBA32Float
                                                                             : HostFormat::kRGBA16Float;
  } else if (IsBC(map.host) && ((out.width & 3) || (out.height & 3))) {
    out.decompress = true;
    out.host_format = HostFormat::kRGBA8;
  }
  const uint32_t array_size = out.shape == TextureShape::k3D ? out.depth : out.layers;
  const auto layout = texture_util::GetGuestTextureLayout(fetch.dimension, fetch.pitch, out.width, out.height,
                                                          array_size, fetch.tiled, fetch.format, fetch.packed_mips,
                                                          true, mip_max);
  out.base_size = layout.base.level_data_extent_bytes;
  out.mip_size = layout.mips_total_extent_bytes;
  if (!out.mip_address) out.mip_size = 0;
  return out.base_address != 0;
}

bool TextureMemory(const xenos::xe_gpu_texture_fetch_t& fetch, uint32_t& base, uint32_t& base_size, uint32_t& mip,
                   uint32_t& mip_size) {
  base = base_size = mip = mip_size = 0;
  if (fetch.type != xenos::FetchConstantType::kTexture) return false;
  if (fetch.dimension == xenos::DataDimension::k1D) return false;
  uint32_t w1, h1, d1, base_page, mip_page, mip_min, mip_max;
  texture_util::GetSubresourcesFromFetchConstant(fetch, &w1, &h1, &d1, &base_page, &mip_page, &mip_min, &mip_max);
  const auto layout = texture_util::GetGuestTextureLayout(fetch.dimension, fetch.pitch, w1 + 1, h1 + 1, d1 + 1,
                                                          fetch.tiled, fetch.format, fetch.packed_mips, true, mip_max);
  base_size = layout.base.level_data_extent_bytes;
  mip_size = layout.mips_total_extent_bytes;
  base = PhysicalAddress(fetch.base_address);
  mip = mip_page ? PhysicalAddress(fetch.mip_address) : 0;
  if (!mip) mip_size = 0;
  return base != 0;
}

namespace {

// The guest bytes a worker decodes from: the base range followed by the mip range, copied on the guest thread.
struct Snapshot {
  uint32_t base = 0, base_size = 0, mip = 0, mip_size = 0;
  std::vector<uint8_t> bytes;
  const uint8_t* At(uint32_t address) const {
    if (address >= base && address < base + base_size) return bytes.data() + (address - base);
    if (address >= mip && address < mip + mip_size) return bytes.data() + base_size + (address - mip);
    return nullptr;
  }
};

bool DecodeInto(const GuestTextureDesc& desc, const Snapshot* snap, TextureUpload& upload);

// Two workers drain the decode queue. The pool is leaked on purpose so process exit never races its threads.
struct DecodePool {
  std::mutex mutex;
  std::condition_variable cv;
  std::deque<std::function<void()>> jobs;
  DecodePool() {
    for (int i = 0; i < 2; ++i) {
      std::thread([this] {
        ae::thread::LowerWorkerThread();
        for (;;) {
          std::function<void()> job;
          {
            std::unique_lock lock(mutex);
            cv.wait(lock, [this] { return !jobs.empty(); });
            job = std::move(jobs.front());
            jobs.pop_front();
          }
          job();
        }
      }).detach();
    }
  }
  void Post(std::function<void()> job) {
    {
      std::lock_guard lock(mutex);
      jobs.push_back(std::move(job));
    }
    cv.notify_one();
  }
};

DecodePool& Pool() {
  static DecodePool* pool = new DecodePool;
  return *pool;
}

}  // namespace

bool UnresolvedDepth(const GuestTextureDesc& desc) {
  return desc.fetch.format == xenos::TextureFormat::k_24_8 || desc.fetch.format == xenos::TextureFormat::k_24_8_FLOAT;
}

// Zeroed host levels in the layout the decoder would produce.
std::shared_ptr<TextureUpload> ZeroTexture(const GuestTextureDesc& desc) {
  auto upload = std::make_shared<TextureUpload>();
  const uint32_t host_bpb = HostBlockBytes(desc.host_format);
  const uint32_t host_block = HostBlockSize(desc.host_format);
  const bool per_texel = desc.decompress || desc.expand;
  const bool is_3d = desc.shape == TextureShape::k3D;
  for (uint32_t layer = 0; layer < desc.layers; ++layer) {
    for (uint32_t level = 0; level < desc.mip_levels; ++level) {
      const uint32_t lw = std::max(desc.width >> level, 1u);
      const uint32_t lh = std::max(desc.height >> level, 1u);
      const uint32_t ld = is_3d ? std::max(desc.depth >> level, 1u) : 1u;
      const uint32_t out_w = per_texel ? lw : (lw + host_block - 1) / host_block;
      const uint32_t out_h = per_texel ? lh : (lh + host_block - 1) / host_block;
      const uint32_t out_pitch = out_w * host_bpb;
      upload->levels.emplace_back(size_t(out_pitch) * out_h * ld, uint8_t(0));
      upload->row_pitch.push_back(out_pitch);
    }
  }
  return upload;
}

std::shared_ptr<TextureUpload> DecodeTexture(const GuestTextureDesc& desc) {
  if (UnresolvedDepth(desc)) return ZeroTexture(desc);
  auto upload = std::make_shared<TextureUpload>();
  if (!DecodeInto(desc, nullptr, *upload)) return nullptr;
  return upload;
}

std::shared_ptr<TextureUpload> DecodeTextureAsync(const GuestTextureDesc& desc) {
  constexpr uint32_t kInlineBytes = 16 * 1024;
  const uint64_t total = uint64_t(desc.base_size) + desc.mip_size;
  if (UnresolvedDepth(desc) || total <= kInlineBytes || total > (64u << 20)) return DecodeTexture(desc);
  auto snap = std::make_shared<Snapshot>();
  snap->base = desc.base_address;
  snap->base_size = desc.base_size;
  snap->mip = desc.mip_size ? desc.mip_address : 0;
  snap->mip_size = desc.mip_size;
  snap->bytes.resize(size_t(total));
  auto* memory = REX_KERNEL_MEMORY();
  std::memcpy(snap->bytes.data(), memory->TranslatePhysical<const uint8_t*>(desc.base_address), desc.base_size);
  if (desc.mip_size) {
    std::memcpy(snap->bytes.data() + desc.base_size, memory->TranslatePhysical<const uint8_t*>(desc.mip_address),
                desc.mip_size);
  }
  auto upload = std::make_shared<TextureUpload>();
  upload->ready = false;
  Pool().Post([desc, snap, upload] {
    DecodeInto(desc, snap.get(), *upload);
    upload->Publish();
  });
  return upload;
}

namespace {

bool DecodeInto(const GuestTextureDesc& desc, const Snapshot* snap, TextureUpload& upload_ref) {
  TextureUpload* upload = &upload_ref;
  const auto& fetch = desc.fetch;
  const auto* info = rex::graphics::FormatInfo::Get(fetch.format);
  const FormatMap map = MapFormat(fetch.format);
  if (!info) return false;
  const uint32_t guest_bpb = info->bytes_per_block();
  const uint32_t bw = info->block_width, bh = info->block_height;
  uint32_t bpb_log2 = 0;
  while ((1u << (bpb_log2 + 1)) <= guest_bpb) ++bpb_log2;
  const uint32_t max_level = desc.mip_levels - 1;
  const bool is_3d = desc.shape == TextureShape::k3D;
  const auto layout = texture_util::GetGuestTextureLayout(fetch.dimension, fetch.pitch, desc.width, desc.height,
                                                          is_3d ? desc.depth : desc.layers, fetch.tiled, fetch.format,
                                                          fetch.packed_mips, true, max_level);
  auto* memory = REX_KERNEL_MEMORY();
  const uint32_t host_bpb = HostBlockBytes(desc.host_format);
  const uint32_t host_block = HostBlockSize(desc.host_format);
  const Unpack& u = map.unpack;
  xenos::TextureSign signs[4];
  for (uint32_t c = 0; c < 4; ++c) signs[c] = SignOf(fetch, c);
  std::vector<uint8_t> block(guest_bpb);
  uint8_t tile[16][4];
  uint32_t raw[16][4];
  for (uint32_t layer = 0; layer < desc.layers; ++layer) {
    for (uint32_t level = 0; level <= max_level; ++level) {
      const uint32_t lw = std::max(desc.width >> level, 1u);
      const uint32_t lh = std::max(desc.height >> level, 1u);
      const uint32_t ld = is_3d ? std::max(desc.depth >> level, 1u) : 1u;
      // Packed mips live inside the storage level of the tail, at an offset in blocks.
      uint32_t storage = level;
      uint32_t off_x = 0, off_y = 0, off_z = 0;
      if (layout.packed_level != UINT32_MAX && level >= layout.packed_level) {
        storage = layout.packed_level;
        texture_util::GetPackedMipOffset(desc.width, desc.height, is_3d ? desc.depth : 1, fetch.format, level, off_x,
                                         off_y, off_z);
      }
      const auto& lv = storage == 0 ? layout.base : layout.mips[storage];
      uint32_t address = storage == 0 ? desc.base_address : desc.mip_address + layout.mip_offsets_bytes[storage];
      address += layer * lv.array_slice_stride_bytes;
      const uint8_t* src = snap ? snap->At(address) : memory->TranslatePhysical<const uint8_t*>(address);
      if (!src) return false;
      const uint32_t pitch_blocks = lv.row_pitch_bytes / guest_bpb;
      const uint32_t blocks_x = (lw + bw - 1) / bw;
      const uint32_t blocks_y = (lh + bh - 1) / bh;
      // Host rows: blocks of the host format, or single texels when decompressing or expanding.
      const bool per_texel = desc.decompress || desc.expand;
      const uint32_t out_w = per_texel ? lw : (lw + host_block - 1) / host_block;
      const uint32_t out_h = per_texel ? lh : (lh + host_block - 1) / host_block;
      const uint32_t out_pitch = out_w * host_bpb;
      std::vector<uint8_t> out(size_t(out_pitch) * out_h * ld);
      for (uint32_t z = 0; z < ld; ++z) {
        uint8_t* slice = out.data() + size_t(out_pitch) * out_h * z;
        for (uint32_t by = 0; by < blocks_y; ++by) {
          for (uint32_t bx = 0; bx < blocks_x; ++bx) {
            const uint32_t x = bx + off_x, y = by + off_y, zz = z + off_z;
            uint32_t offset;
            if (is_3d) {
              offset = fetch.tiled ? uint32_t(texture_util::GetTiledOffset3D(int32_t(x), int32_t(y), int32_t(zz),
                                                                             pitch_blocks, lv.z_slice_stride_block_rows,
                                                                             bpb_log2))
                                   : (zz * lv.z_slice_stride_block_rows + y) * lv.row_pitch_bytes + x * guest_bpb;
            } else {
              offset = fetch.tiled ? uint32_t(texture_util::GetTiledOffset2D(int32_t(x), int32_t(y), pitch_blocks,
                                                                             bpb_log2))
                                   : y * lv.row_pitch_bytes + x * guest_bpb;
            }
            SwapBlock(fetch.endianness, block.data(), src + offset, guest_bpb);
            if (desc.expand) {
              // Every texel of the block to float, signs applied, missing components replicated from the last one.
              const bool compressed = u.block != kBlockNone;
              if (compressed) DecodeRawBlock(u.block, block.data(), raw);
              const uint32_t tw = compressed ? 4 : 1, th = compressed ? 4 : 1;
              for (uint32_t ty = 0; ty < th; ++ty) {
                for (uint32_t tx = 0; tx < tw; ++tx) {
                  const uint32_t px = bx * tw + tx, py = by * th + ty;
                  if (px >= lw || py >= lh) continue;
                  float v[4];
                  for (uint32_t c = 0; c < 4; ++c) {
                    const uint32_t src_c = std::min<uint32_t>(c, u.count - 1u);
                    const uint32_t bits = u.c[src_c].bits;
                    const uint32_t value = compressed ? raw[ty * 4 + tx][src_c]
                                                      : Bits(block.data(), u.c[src_c].shift, bits);
                    v[c] = ConvertComponent(u, value, bits, signs[src_c]);
                  }
                  WriteExpanded(v, desc.host_format, slice + size_t(py) * out_pitch + px * host_bpb);
                }
              }
            } else if (desc.decompress) {
              DecodeBCBlock(map.host, block.data(), tile);
              for (uint32_t ty = 0; ty < 4; ++ty) {
                for (uint32_t tx = 0; tx < 4; ++tx) {
                  const uint32_t px = bx * 4 + tx, py = by * 4 + ty;
                  if (px < lw && py < lh) std::memcpy(&slice[size_t(py) * out_pitch + px * 4], tile[ty * 4 + tx], 4);
                }
              }
            } else if (map.convert != Convert::kNone) {
              ConvertTexel(map.convert, block.data(), &slice[size_t(by) * out_pitch + bx * 4]);
            } else {
              std::memcpy(&slice[size_t(by) * out_pitch + bx * host_bpb], block.data(), std::min(guest_bpb, host_bpb));
            }
          }
        }
      }
      upload->levels.push_back(std::move(out));
      upload->row_pitch.push_back(out_pitch);
    }
  }
  return true;
}

}  // namespace

}  // namespace ae::gpu
