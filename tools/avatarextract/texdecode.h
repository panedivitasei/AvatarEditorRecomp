// texdecode.h: Xbox 360 avatar texture -> RGBA8 decode helpers shared by the
// avatarextract modes. Header-only (static inline); include freely.
//
// Avatar texture payloads (kTexture STRB blocks and the textures embedded in
// kModel blocks) are block-compressed (DXT1/3/5) or A8R8G8B8, stored with
// every 16-bit word byte-swapped (console order) and in linear block order
// (the is_tiled flag describes the GPU surface, not the on-disk layout).
// Rows use the Texture::data_stride pitch, which is padded for narrow shapes
// (64/32-wide DXT rows are 512/256 bytes, not the tight width-derived pitch),
// and the layers of a frame stack follow each other at data_rows * data_stride.
#ifndef AVATAREXTRACT_TEXDECODE_H_
#define AVATAREXTRACT_TEXDECODE_H_

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <vector>

#include "texture.h"

namespace texdec {

enum class Kind { kUnknown, kBC1, kBC2, kBC3, kRGBA };

struct Image {
  uint32_t width = 0;
  uint32_t height = 0;
  std::vector<uint8_t> rgba;  // width * height * 4, row-major, top row first
  bool ok = false;

  const uint8_t* at(uint32_t x, uint32_t y) const {
    return rgba.data() + ((size_t)y * width + x) * 4;
  }
  uint8_t* at(uint32_t x, uint32_t y) {
    return rgba.data() + ((size_t)y * width + x) * 4;
  }
};

static inline const char* KindName(Kind k) {
  switch (k) {
    case Kind::kBC1: return "DXT1/BC1";
    case Kind::kBC2: return "DXT3/BC2";
    case Kind::kBC3: return "DXT5/BC3";
    case Kind::kRGBA: return "A8R8G8B8";
    default: return "UNKNOWN";
  }
}

// The avatar texture "format" packs the Xbox D3D texture format in the low 6
// bits: 18=DXT1, 19=DXT3, 20=DXT5, 6=A8R8G8B8.
static inline Kind ClassifyFormat(uint32_t format) {
  switch (format & 0x3F) {
    case 18: return Kind::kBC1;
    case 19: return Kind::kBC2;
    case 20: return Kind::kBC3;
    case 6: return Kind::kRGBA;
    default: return Kind::kUnknown;
  }
}

static inline uint32_t BytesPerBlock(Kind k) {
  switch (k) {
    case Kind::kBC1: return 8;
    case Kind::kBC2: return 16;
    case Kind::kBC3: return 16;
    case Kind::kRGBA: return 4;
    default: return 0;
  }
}

static inline void DecodeBc1Colors(const uint8_t* block, uint8_t palette[4][4],
                                   bool allow_alpha) {
  uint16_t c0 = (uint16_t)(block[0] | (block[1] << 8));
  uint16_t c1 = (uint16_t)(block[2] | (block[3] << 8));
  auto expand = [](uint16_t c, uint8_t out[3]) {
    uint8_t r5 = (c >> 11) & 0x1F, g6 = (c >> 5) & 0x3F, b5 = c & 0x1F;
    out[0] = (uint8_t)((r5 << 3) | (r5 >> 2));
    out[1] = (uint8_t)((g6 << 2) | (g6 >> 4));
    out[2] = (uint8_t)((b5 << 3) | (b5 >> 2));
  };
  uint8_t e0[3], e1[3];
  expand(c0, e0);
  expand(c1, e1);
  for (int i = 0; i < 3; ++i) {
    palette[0][i] = e0[i];
    palette[1][i] = e1[i];
  }
  palette[0][3] = 255;
  palette[1][3] = 255;
  if (c0 > c1 || !allow_alpha) {
    for (int i = 0; i < 3; ++i) {
      palette[2][i] = (uint8_t)((2 * e0[i] + e1[i]) / 3);
      palette[3][i] = (uint8_t)((e0[i] + 2 * e1[i]) / 3);
    }
    palette[2][3] = 255;
    palette[3][3] = 255;
  } else {
    for (int i = 0; i < 3; ++i) {
      palette[2][i] = (uint8_t)((e0[i] + e1[i]) / 2);
      palette[3][i] = 0;
    }
    palette[2][3] = 255;
    palette[3][3] = 0;
  }
}

// Writes a 4x4 block into dst (RGBA8, dst_pitch bytes per row), clipped to
// (clip_w, clip_h) texels so non-multiple-of-4 images do not overrun.
static inline void DecodeBc1Block(const uint8_t* block, uint8_t* dst,
                                  uint32_t dst_pitch, int clip_w, int clip_h) {
  uint8_t pal[4][4];
  DecodeBc1Colors(block, pal, true);
  uint32_t bits = (uint32_t)block[4] | ((uint32_t)block[5] << 8) |
                  ((uint32_t)block[6] << 16) | ((uint32_t)block[7] << 24);
  for (int y = 0; y < 4 && y < clip_h; ++y) {
    for (int x = 0; x < 4 && x < clip_w; ++x) {
      uint32_t idx = (bits >> (2 * (y * 4 + x))) & 3;
      uint8_t* p = dst + y * dst_pitch + x * 4;
      p[0] = pal[idx][0];
      p[1] = pal[idx][1];
      p[2] = pal[idx][2];
      p[3] = pal[idx][3];
    }
  }
}

static inline void DecodeBc2Block(const uint8_t* block, uint8_t* dst,
                                  uint32_t dst_pitch, int clip_w, int clip_h) {
  uint8_t pal[4][4];
  DecodeBc1Colors(block + 8, pal, false);
  uint32_t bits = (uint32_t)block[12] | ((uint32_t)block[13] << 8) |
                  ((uint32_t)block[14] << 16) | ((uint32_t)block[15] << 24);
  for (int y = 0; y < 4 && y < clip_h; ++y) {
    uint16_t arow = (uint16_t)(block[y * 2] | (block[y * 2 + 1] << 8));
    for (int x = 0; x < 4 && x < clip_w; ++x) {
      uint32_t idx = (bits >> (2 * (y * 4 + x))) & 3;
      uint8_t a4 = (arow >> (x * 4)) & 0xF;
      uint8_t* p = dst + y * dst_pitch + x * 4;
      p[0] = pal[idx][0];
      p[1] = pal[idx][1];
      p[2] = pal[idx][2];
      p[3] = (uint8_t)(a4 * 17);
    }
  }
}

static inline void DecodeBc3Block(const uint8_t* block, uint8_t* dst,
                                  uint32_t dst_pitch, int clip_w, int clip_h) {
  uint8_t pal[4][4];
  DecodeBc1Colors(block + 8, pal, false);
  uint8_t a0 = block[0], a1 = block[1];
  uint8_t apal[8];
  apal[0] = a0;
  apal[1] = a1;
  if (a0 > a1) {
    for (int i = 1; i < 7; ++i) apal[i + 1] = (uint8_t)(((7 - i) * a0 + i * a1) / 7);
  } else {
    for (int i = 1; i < 5; ++i) apal[i + 1] = (uint8_t)(((5 - i) * a0 + i * a1) / 5);
    apal[6] = 0;
    apal[7] = 255;
  }
  uint64_t abits = 0;
  for (int i = 0; i < 6; ++i) abits |= (uint64_t)block[2 + i] << (8 * i);
  uint32_t bits = (uint32_t)block[12] | ((uint32_t)block[13] << 8) |
                  ((uint32_t)block[14] << 16) | ((uint32_t)block[15] << 24);
  for (int y = 0; y < 4 && y < clip_h; ++y) {
    for (int x = 0; x < 4 && x < clip_w; ++x) {
      uint32_t idx = (bits >> (2 * (y * 4 + x))) & 3;
      uint32_t aidx = (uint32_t)((abits >> (3 * (y * 4 + x))) & 7);
      uint8_t* p = dst + y * dst_pitch + x * 4;
      p[0] = pal[idx][0];
      p[1] = pal[idx][1];
      p[2] = pal[idx][2];
      p[3] = apal[aidx];
    }
  }
}

// Decode one layer (frame) of an avatar Texture to RGBA8.
static inline Image DecodeTextureLayer(const rex::avatars::Texture& tex,
                                       uint32_t layer) {
  Image out;
  out.width = tex.width;
  out.height = tex.height;
  if (tex.is_empty || tex.data_bytes.empty() || tex.width == 0 ||
      tex.height == 0) {
    return out;
  }
  Kind kind = ClassifyFormat(tex.format);
  if (kind == Kind::kUnknown) return out;

  const uint32_t w = tex.width, h = tex.height;
  const uint32_t block_dim = (kind == Kind::kRGBA) ? 1 : 4;
  const uint32_t bpb = BytesPerBlock(kind);
  const uint32_t wb = (w + block_dim - 1) / block_dim;
  const uint32_t hb = (h + block_dim - 1) / block_dim;

  // Row pitch: the declared stride when it covers a block row (padded narrow
  // shapes), else the tight pitch.
  size_t pitch = tex.data_stride;
  if (pitch < (size_t)wb * bpb) pitch = (size_t)wb * bpb;
  const size_t layer_bytes = (size_t)tex.data_rows * tex.data_stride;
  size_t base = (size_t)layer * layer_bytes;
  if (layer_bytes == 0 || base + (size_t)(hb - 1) * pitch + (size_t)wb * bpb >
                             tex.data_bytes.size()) {
    // Fall back to a tight layout guess if the declared geometry is short.
    pitch = (size_t)wb * bpb;
    base = (size_t)layer * pitch * hb;
    if (base + pitch * hb > tex.data_bytes.size()) return out;
  }

  // Undo the 16-bit byte swap into a local copy of this layer.
  std::vector<uint8_t> swapped(tex.data_bytes.begin() + base,
                               tex.data_bytes.begin() + base + pitch * hb);
  for (size_t i = 0; i + 1 < swapped.size(); i += 2) {
    std::swap(swapped[i], swapped[i + 1]);
  }

  out.rgba.assign((size_t)w * h * 4, 0);
  const uint32_t dst_pitch = w * 4;
  if (kind == Kind::kRGBA) {
    for (uint32_t y = 0; y < h; ++y) {
      for (uint32_t x = 0; x < w; ++x) {
        const uint8_t* s = swapped.data() + y * pitch + (size_t)x * 4;
        uint8_t* p = out.at(x, y);
        // little-endian A8R8G8B8 => bytes B,G,R,A
        p[0] = s[2];
        p[1] = s[1];
        p[2] = s[0];
        p[3] = s[3];
      }
    }
  } else {
    for (uint32_t by = 0; by < hb; ++by) {
      for (uint32_t bx = 0; bx < wb; ++bx) {
        const uint8_t* block = swapped.data() + by * pitch + (size_t)bx * bpb;
        uint8_t* dst = out.rgba.data() + (size_t)(by * 4) * dst_pitch + (bx * 4) * 4;
        int cw = (int)std::min<uint32_t>(4, w - bx * 4);
        int ch = (int)std::min<uint32_t>(4, h - by * 4);
        switch (kind) {
          case Kind::kBC1: DecodeBc1Block(block, dst, dst_pitch, cw, ch); break;
          case Kind::kBC2: DecodeBc2Block(block, dst, dst_pitch, cw, ch); break;
          case Kind::kBC3: DecodeBc3Block(block, dst, dst_pitch, cw, ch); break;
          default: break;
        }
      }
    }
  }
  out.ok = true;
  return out;
}

static inline Image DecodeTexture(const rex::avatars::Texture& tex) {
  return DecodeTextureLayer(tex, 0);
}

}  // namespace texdec

#endif  // AVATAREXTRACT_TEXDECODE_H_
