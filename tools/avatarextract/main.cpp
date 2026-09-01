/**
 * avatarextract: extract an Xbox 360 avatar (marketplace STFS "LIVE" package)
 * into .obj + .mtl + .png files.
 *
 * Reuses the avatar decode pipeline (src/kernel/xam/avatars): STRB block
 * extraction, LZX decompression, Model::Read (vertices/indices/triangle batches)
 * and the embedded ModelTexture data. Adds a self-contained STFS reader (so we do
 * not need to pull in the full filesystem device stack) and BC1/BC2/BC3 +
 * A8R8G8B8 decoders that emit RGBA8 PNGs.
 */

#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <map>
#include <set>
#include <string>
#include <vector>

#define NOMINMAX
#define WIN32_LEAN_AND_MEAN
#include <windows.h>  // SEH (__try/__except) for malformed-item isolation

#include <rex/filesystem/devices/stfs_xbox.h>
#include <rex/math.h>

#include "asset_pack.h"
#include "guest_asset.h"
#include "bit_stream.h"
#include "blend_shape.h"
#include "blend_shape_apply.h"
#include "compression.h"
#include "model.h"
#include "skeleton.h"
#include "animation.h"
#include "strb.h"
#include "texture.h"

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

namespace fs = std::filesystem;
using namespace rex::filesystem;

// ---------------------------------------------------------------------------
// Self-contained STFS reader (read-only; mirrors StfsContainerDevice math).
// ---------------------------------------------------------------------------
namespace {

constexpr uint32_t kBlockSize = 0x1000;
constexpr uint32_t kEndOfChain = 0xFFFFFF;
constexpr uint32_t kBlocksPerHashLevel[3] = {170, 28900, 4913000};

struct StfsFile {
  std::string name;
  uint32_t length = 0;
  uint32_t start_block = 0;
  bool is_dir = false;
};

class StfsReader {
 public:
  bool Load(const fs::path& path) {
    FILE* f = std::fopen(path.string().c_str(), "rb");
    if (!f) return false;
    std::fseek(f, 0, SEEK_END);
    long sz = std::ftell(f);
    std::fseek(f, 0, SEEK_SET);
    if (sz < (long)sizeof(StfsHeader)) {
      std::fclose(f);
      return false;
    }
    data_.resize(sz);
    size_t got = std::fread(data_.data(), 1, sz, f);
    std::fclose(f);
    if (got != (size_t)sz) return false;

    if (std::memcmp(data_.data(), "LIVE", 4) != 0 &&
        std::memcmp(data_.data(), "CON ", 4) != 0 &&
        std::memcmp(data_.data(), "PIRS", 4) != 0) {
      return false;
    }
    const auto& d = hdr()->metadata.volume_descriptor.stfs;
    blocks_per_hash_table_ = d.flags.bits.read_only_format ? 1u : 2u;
    secondary_offset_ = (!d.flags.bits.read_only_format && d.flags.bits.root_active_index)
                            ? kBlockSize : 0u;
    return hdr()->metadata.volume_type == XContentVolumeType::kStfs;
  }

  const StfsHeader* hdr() const {
    return reinterpret_cast<const StfsHeader*>(data_.data());
  }

  std::vector<StfsFile> ListFiles() const {
    std::vector<StfsFile> out;
    const auto& d = hdr()->metadata.volume_descriptor.stfs;
    uint32_t table_block = d.file_table_block_number();
    uint32_t count = d.file_table_block_count;
    for (uint32_t n = 0; n < count && table_block != kEndOfChain; ++n) {
      size_t off = BlockToOffset(table_block);
      if (off + kBlockSize > data_.size()) break;
      const auto* db = reinterpret_cast<const StfsDirectoryBlock*>(&data_[off]);
      for (uint32_t e = 0; e < 0x40; ++e) {
        const StfsDirectoryEntry& de = db->entries[e];
        uint8_t nl = de.flags.name_length;
        if (nl == 0 || nl > 40) continue;
        StfsFile it;
        it.name.assign(de.name, nl);
        it.length = de.length;
        it.start_block = de.start_block_number();
        it.is_dir = de.flags.directory != 0;
        out.push_back(std::move(it));
      }
      table_block = GetBlockHash(table_block)->level0_next_block();
    }
    return out;
  }

  bool ReadFile(const StfsFile& file, std::vector<uint8_t>& out) const {
    out.clear();
    out.reserve(file.length);
    uint32_t block = file.start_block;
    size_t remaining = file.length;
    while (remaining && block != kEndOfChain) {
      size_t off = BlockToOffset(block);
      if (off + kBlockSize > data_.size()) return false;
      size_t chunk = remaining < kBlockSize ? remaining : kBlockSize;
      out.insert(out.end(), &data_[off], &data_[off] + chunk);
      remaining -= chunk;
      block = GetBlockHash(block)->level0_next_block();
    }
    return remaining == 0;
  }

 private:
  uint32_t HeaderSizeAligned() const {
    uint32_t hs = hdr()->header.header_size;
    return ((hs + (kBlockSize - 1)) / kBlockSize) * kBlockSize;
  }

  size_t BlockToOffset(uint64_t block_index) const {
    uint64_t base = kBlocksPerHashLevel[0];
    uint64_t block = block_index;
    for (uint32_t i = 0; i < 3; ++i) {
      block += ((block_index + base) / base) * blocks_per_hash_table_;
      if (block_index < base) break;
      base *= kBlocksPerHashLevel[0];
    }
    return HeaderSizeAligned() + (block << 12);
  }

  uint32_t BlockToHashBlockNumber(uint32_t block_index, uint32_t hash_level) const {
    // Mirrors xenia's STFSContainerDevice::BlockToHashBlockNumberSTFS:
    // block_step[0] = 0xAB/0xAC, block_step[1] = 0x718F/0x723A
    // (read-only / read-write packages respectively).
    const uint32_t bpht = blocks_per_hash_table_;
    const uint32_t block_step0 = kBlocksPerHashLevel[0] + bpht;
    const uint32_t block_step1 = kBlocksPerHashLevel[1] + (kBlocksPerHashLevel[0] + 1) * bpht;
    uint32_t block = 0;
    if (hash_level == 0) {
      if (block_index < kBlocksPerHashLevel[0]) return 0;
      block = (block_index / kBlocksPerHashLevel[0]) * block_step0;
      block += ((block_index / kBlocksPerHashLevel[1]) + 1) * bpht;
      if (block_index < kBlocksPerHashLevel[1]) return block;
      return block + bpht;
    }
    if (hash_level == 1) {
      if (block_index < kBlocksPerHashLevel[1]) return block_step0;
      block = (block_index / kBlocksPerHashLevel[1]) * block_step1;
      return block + bpht;
    }
    return block_step1;
  }

  const StfsHashEntry* GetBlockHash(uint32_t block_index) const {
    uint64_t block = BlockToHashBlockNumber(block_index, 0);
    size_t hash_offset = HeaderSizeAligned() + (block << 12) + secondary_offset_;
    const auto* table = reinterpret_cast<const StfsHashTable*>(&data_[hash_offset]);
    uint32_t record = block_index % kBlocksPerHashLevel[0];
    return &table->entries[record];
  }

  std::vector<uint8_t> data_;
  uint32_t blocks_per_hash_table_ = 1;
  uint32_t secondary_offset_ = 0;
};

// ---------------------------------------------------------------------------
// BC (DXT) block decoders -> RGBA8
// ---------------------------------------------------------------------------
inline void DecodeBc1Colors(const uint8_t* block, uint8_t palette[4][4],
                            bool allow_alpha) {
  uint16_t c0 = block[0] | (block[1] << 8);
  uint16_t c1 = block[2] | (block[3] << 8);
  auto expand = [](uint16_t c, uint8_t out[3]) {
    uint8_t r5 = (c >> 11) & 0x1F, g6 = (c >> 5) & 0x3F, b5 = c & 0x1F;
    out[0] = (r5 << 3) | (r5 >> 2);
    out[1] = (g6 << 2) | (g6 >> 4);
    out[2] = (b5 << 3) | (b5 >> 2);
  };
  uint8_t e0[3], e1[3];
  expand(c0, e0);
  expand(c1, e1);
  palette[0][0] = e0[0]; palette[0][1] = e0[1]; palette[0][2] = e0[2]; palette[0][3] = 255;
  palette[1][0] = e1[0]; palette[1][1] = e1[1]; palette[1][2] = e1[2]; palette[1][3] = 255;
  if (c0 > c1 || !allow_alpha) {
    for (int i = 0; i < 3; ++i) {
      palette[2][i] = (uint8_t)((2 * e0[i] + e1[i]) / 3);
      palette[3][i] = (uint8_t)((e0[i] + 2 * e1[i]) / 3);
    }
    palette[2][3] = 255; palette[3][3] = 255;
  } else {
    for (int i = 0; i < 3; ++i) {
      palette[2][i] = (uint8_t)((e0[i] + e1[i]) / 2);
      palette[3][i] = 0;
    }
    palette[2][3] = 255; palette[3][3] = 0;
  }
}

// Decode a 4x4 BC1 block; writes to a 4x4 region in dst (row-major RGBA8).
void DecodeBc1Block(const uint8_t* block, uint8_t* dst, uint32_t dst_pitch) {
  uint8_t pal[4][4];
  DecodeBc1Colors(block, pal, true);
  uint32_t bits = block[4] | (block[5] << 8) | (block[6] << 16) | (block[7] << 24);
  for (int y = 0; y < 4; ++y) {
    for (int x = 0; x < 4; ++x) {
      uint8_t idx = (bits >> (2 * (y * 4 + x))) & 3;
      uint8_t* p = dst + y * dst_pitch + x * 4;
      p[0] = pal[idx][0]; p[1] = pal[idx][1]; p[2] = pal[idx][2]; p[3] = pal[idx][3];
    }
  }
}

// BC2 (DXT3): explicit 4-bit alpha + BC1 color (no 1-bit alpha mode).
void DecodeBc2Block(const uint8_t* block, uint8_t* dst, uint32_t dst_pitch) {
  uint8_t pal[4][4];
  DecodeBc1Colors(block + 8, pal, false);
  uint32_t bits = block[12] | (block[13] << 8) | (block[14] << 16) | (block[15] << 24);
  uint64_t alpha = 0;
  for (int i = 0; i < 8; ++i) alpha |= (uint64_t)block[i] << (8 * i);
  for (int y = 0; y < 4; ++y) {
    for (int x = 0; x < 4; ++x) {
      uint8_t idx = (bits >> (2 * (y * 4 + x))) & 3;
      uint8_t a4 = (alpha >> (4 * (y * 4 + x))) & 0xF;
      uint8_t* p = dst + y * dst_pitch + x * 4;
      p[0] = pal[idx][0]; p[1] = pal[idx][1]; p[2] = pal[idx][2];
      p[3] = (a4 << 4) | a4;
    }
  }
}

// BC3 (DXT5): interpolated alpha + BC1 color (no 1-bit alpha mode).
void DecodeBc3Block(const uint8_t* block, uint8_t* dst, uint32_t dst_pitch) {
  uint8_t pal[4][4];
  DecodeBc1Colors(block + 8, pal, false);
  uint32_t bits = block[12] | (block[13] << 8) | (block[14] << 16) | (block[15] << 24);
  uint8_t a[8];
  a[0] = block[0];
  a[1] = block[1];
  if (a[0] > a[1]) {
    for (int i = 1; i < 7; ++i)
      a[i + 1] = (uint8_t)(((7 - i) * a[0] + i * a[1]) / 7);
  } else {
    for (int i = 1; i < 5; ++i)
      a[i + 1] = (uint8_t)(((5 - i) * a[0] + i * a[1]) / 5);
    a[6] = 0;
    a[7] = 255;
  }
  uint64_t abits = 0;
  for (int i = 0; i < 6; ++i) abits |= (uint64_t)block[2 + i] << (8 * i);
  for (int y = 0; y < 4; ++y) {
    for (int x = 0; x < 4; ++x) {
      int t = y * 4 + x;
      uint8_t idx = (bits >> (2 * t)) & 3;
      uint8_t aidx = (abits >> (3 * t)) & 7;
      uint8_t* p = dst + y * dst_pitch + x * 4;
      p[0] = pal[idx][0]; p[1] = pal[idx][1]; p[2] = pal[idx][2];
      p[3] = a[aidx];
    }
  }
}

enum class TexKind { kUnknown, kBC1, kBC2, kBC3, kRGBA };

struct DecodedTexture {
  uint32_t width = 0;
  uint32_t height = 0;
  std::vector<uint8_t> rgba;  // width*height*4
  bool ok = false;
};

const char* KindName(TexKind k) {
  switch (k) {
    case TexKind::kBC1: return "DXT1/BC1";
    case TexKind::kBC2: return "DXT3/BC2";
    case TexKind::kBC3: return "DXT5/BC3";
    case TexKind::kRGBA: return "A8R8G8B8";
    default: return "UNKNOWN";
  }
}

TexKind ClassifyFormat(uint32_t format) {
  // The avatar texture "format" packs the Xbox D3D texture format in the low 6
  // bits (see model_save.cpp): 18=DXT1, 19=DXT3, 20=DXT5, 6=A8R8G8B8.
  switch (format & 0x3F) {
    case 18: return TexKind::kBC1;
    case 19: return TexKind::kBC2;
    case 20: return TexKind::kBC3;
    case 6:  return TexKind::kRGBA;
    default: return TexKind::kUnknown;
  }
}

// Decode one avatar Texture (untile if tiled, BC-decode / convert) -> RGBA8.
DecodedTexture DecodeTexture(const rex::avatars::Texture& tex) {
  DecodedTexture out;
  out.width = tex.width;
  out.height = tex.height;
  if (tex.is_empty || tex.data_bytes.empty()) return out;

  TexKind kind = ClassifyFormat(tex.format);
  if (kind == TexKind::kUnknown) return out;

  uint32_t w = tex.width, h = tex.height;
  uint32_t block_dim = (kind == TexKind::kRGBA) ? 1 : 4;
  uint32_t bytes_per_block = (kind == TexKind::kBC1) ? 8
                            : (kind == TexKind::kRGBA) ? 4 : 16;
  uint32_t wb = (w + block_dim - 1) / block_dim;
  uint32_t hb = (h + block_dim - 1) / block_dim;

  // Avatar texture data is stored byte-swapped in 16-bit words (model_save.cpp
  // swaps every pair on export). Undo that to get little-endian block bytes.
  std::vector<uint8_t> swapped(tex.data_bytes);
  for (size_t i = 0; i + 1 < swapped.size(); i += 2) {
    std::swap(swapped[i], swapped[i + 1]);
  }

  // The texture data is stored in linear block order even when is_tiled is
  // set: the flag describes the GPU surface the engine would upload to, not
  // the on-disk layout.
  const uint8_t* linear_src = swapped.data();
  size_t needed = (size_t)wb * hb * bytes_per_block;
  if (swapped.size() < needed) return out;

  out.rgba.assign((size_t)w * h * 4, 0);
  uint32_t pitch = w * 4;

  if (kind == TexKind::kRGBA) {
    // Stored as A8R8G8B8 (BGRA byte order in little-endian). Convert to RGBA8.
    for (uint32_t y = 0; y < h; ++y) {
      for (uint32_t x = 0; x < w; ++x) {
        const uint8_t* s = linear_src + ((size_t)y * w + x) * 4;
        uint8_t* p = out.rgba.data() + (size_t)y * pitch + x * 4;
        // little-endian A8R8G8B8 => bytes B,G,R,A
        p[0] = s[2]; p[1] = s[1]; p[2] = s[0]; p[3] = s[3];
      }
    }
  } else {
    for (uint32_t by = 0; by < hb; ++by) {
      for (uint32_t bx = 0; bx < wb; ++bx) {
        const uint8_t* block = linear_src + ((size_t)by * wb + bx) * bytes_per_block;
        uint8_t* dst = out.rgba.data() + (size_t)(by * 4) * pitch + (bx * 4) * 4;
        switch (kind) {
          case TexKind::kBC1: DecodeBc1Block(block, dst, pitch); break;
          case TexKind::kBC2: DecodeBc2Block(block, dst, pitch); break;
          case TexKind::kBC3: DecodeBc3Block(block, dst, pitch); break;
          default: break;
        }
      }
    }
  }
  out.ok = true;
  return out;
}

// UVs are stored as Xenos half-floats (per-channel u16).
inline float UvHalf(uint16_t h) { return rex::xenos_half_to_float(h); }

// ---------------------------------------------------------------------------
// Decode a Model (already loaded) and write <prefix>.obj + <prefix>.mtl + its
// texture PNGs into out_dir. Shared by the STFS and asset-pack code paths.
// Emits per-vertex computed normals (vn) and v/vt/vn faces. Returns false on a
// hard write error; fills out_verts/out_tris with the totals.
// ---------------------------------------------------------------------------
bool WriteModel(const rex::avatars::Model& model, const fs::path& out_dir,
                const std::string& prefix, size_t& out_verts, size_t& out_tris) {
  out_verts = 0;
  out_tris = 0;
  for (const auto& b : model.triangle_batches) {
    out_verts += b.vertices.size();
    out_tris += b.indices.size() / 3;
  }
  std::printf("\n=== Model (%s) ===\n", prefix.c_str());
  std::printf("  triangle batches: %zu\n", model.triangle_batches.size());
  std::printf("  total vertices  : %zu\n", out_verts);
  std::printf("  total triangles : %zu\n", out_tris);
  std::printf("  textures        : %zu\n", model.textures.size());

  fs::create_directories(out_dir);

  // ---- textures ----
  std::vector<std::string> texture_png_names(model.textures.size());
  std::printf("\n=== Textures (%s) ===\n", prefix.c_str());
  for (size_t i = 0; i < model.textures.size(); ++i) {
    const auto& mt = model.textures[i];
    const auto& tex = mt.texture;
    TexKind kind = ClassifyFormat(tex.format);
    std::printf("  [%zu] fmt=0x%X (%s) %ux%u tiled=%d layers=%u empty=%d bytes=%zu\n",
                i, tex.format, KindName(kind), tex.width, tex.height,
                tex.is_tiled ? 1 : 0, tex.layer_count, tex.is_empty ? 1 : 0,
                tex.data_bytes.size());
    DecodedTexture dec = DecodeTexture(tex);
    if (!dec.ok) {
      std::printf("       -> skipped (empty or unsupported format)\n");
      continue;
    }
    std::string png = prefix + "_texture" + std::to_string(i) + ".png";
    fs::path png_path = out_dir / png;
    if (stbi_write_png(png_path.string().c_str(), (int)dec.width, (int)dec.height, 4,
                       dec.rgba.data(), (int)dec.width * 4)) {
      texture_png_names[i] = png;
      std::printf("       -> %s\n", png.c_str());
    } else {
      std::printf("       -> FAILED to write %s\n", png.c_str());
    }
  }

  auto batch_texture_index = [&](const rex::avatars::TriangleBatch& b) -> int {
    for (const auto& sp : b.shader_parameters) {
      if (sp.type == rex::avatars::ShaderParameterType::kTexture) {
        return (int)sp.texture.index;
      }
    }
    return -1;
  };

  // ---- <prefix>.mtl ----
  fs::path obj_path = out_dir / (prefix + ".obj");
  fs::path mtl_path = out_dir / (prefix + ".mtl");

  FILE* mtl = std::fopen(mtl_path.string().c_str(), "wb");
  if (!mtl) {
    std::printf("ERROR: cannot write %s\n", mtl_path.string().c_str());
    return false;
  }
  for (size_t bi = 0; bi < model.triangle_batches.size(); ++bi) {
    const auto& b = model.triangle_batches[bi];
    std::fprintf(mtl, "newmtl mat%zu\n", bi);
    std::fprintf(mtl, "Ka 1 1 1\nKd 1 1 1\nKs 0 0 0\nd 1\nillum 1\n");
    int ti = batch_texture_index(b);
    if (ti >= 0 && ti < (int)texture_png_names.size() &&
        !texture_png_names[ti].empty()) {
      std::fprintf(mtl, "map_Kd %s\n", texture_png_names[ti].c_str());
    }
    std::fprintf(mtl, "\n");
  }
  std::fclose(mtl);

  // ---- <prefix>.obj ----
  FILE* obj = std::fopen(obj_path.string().c_str(), "wb");
  if (!obj) {
    std::printf("ERROR: cannot write %s\n", obj_path.string().c_str());
    return false;
  }
  std::fprintf(obj, "# avatarextract OBJ export\n");
  std::fprintf(obj, "mtllib %s.mtl\n\n", prefix.c_str());

  uint32_t vbase = 1;  // OBJ indices are 1-based and global.
  for (size_t bi = 0; bi < model.triangle_batches.size(); ++bi) {
    const auto& b = model.triangle_batches[bi];
    std::fprintf(obj, "o batch%zu\n", bi);
    for (const auto& v : b.vertices) {
      std::fprintf(obj, "v %.6f %.6f %.6f\n", v.position.x, v.position.y,
                   v.position.z);
    }
    for (const auto& v : b.vertices) {
      float u = 0.f, vv = 0.f;
      if (!v.uvs.empty()) {
        u = UvHalf(v.uvs[0].x);
        vv = UvHalf(v.uvs[0].y);
      }
      // Flip V for OBJ (origin bottom-left) vs. texture (origin top-left).
      std::fprintf(obj, "vt %.6f %.6f\n", u, 1.0f - vv);
    }
    // Per-vertex normals computed from triangle geometry (area-weighted face
    // normals, normalized per vertex). The Xbox vertex packs a normal at offset
    // 0x0C (FMT_10_11_11) but the decode pipeline never unpacks it to a float
    // direction, so we derive usable normals from the mesh.
    std::vector<rex::avatars::Vector3<float>> normals(b.vertices.size(),
                                                      {0.f, 0.f, 0.f});
    for (size_t i = 0; i + 2 < b.indices.size(); i += 3) {
      uint16_t i0 = b.indices[i + 0];
      uint16_t i1 = b.indices[i + 1];
      uint16_t i2 = b.indices[i + 2];
      if (i0 >= b.vertices.size() || i1 >= b.vertices.size() ||
          i2 >= b.vertices.size()) {
        continue;
      }
      const auto& p0 = b.vertices[i0].position;
      const auto& p1 = b.vertices[i1].position;
      const auto& p2 = b.vertices[i2].position;
      float ex1 = p1.x - p0.x, ey1 = p1.y - p0.y, ez1 = p1.z - p0.z;
      float ex2 = p2.x - p0.x, ey2 = p2.y - p0.y, ez2 = p2.z - p0.z;
      float nx = ey1 * ez2 - ez1 * ey2;
      float ny = ez1 * ex2 - ex1 * ez2;
      float nz = ex1 * ey2 - ey1 * ex2;  // magnitude == 2 * triangle area
      normals[i0].x += nx; normals[i0].y += ny; normals[i0].z += nz;
      normals[i1].x += nx; normals[i1].y += ny; normals[i1].z += nz;
      normals[i2].x += nx; normals[i2].y += ny; normals[i2].z += nz;
    }
    for (auto& n : normals) {
      float len = std::sqrt(n.x * n.x + n.y * n.y + n.z * n.z);
      if (len > 1e-12f) {
        n.x /= len; n.y /= len; n.z /= len;
      } else {
        n.x = 0.f; n.y = 0.f; n.z = 1.f;
      }
      std::fprintf(obj, "vn %.6f %.6f %.6f\n", n.x, n.y, n.z);
    }
    std::fprintf(obj, "usemtl mat%zu\n", bi);
    for (size_t i = 0; i + 2 < b.indices.size(); i += 3) {
      uint32_t a = vbase + b.indices[i + 0];
      uint32_t c = vbase + b.indices[i + 1];
      uint32_t d = vbase + b.indices[i + 2];
      std::fprintf(obj, "f %u/%u/%u %u/%u/%u %u/%u/%u\n", a, a, a, c, c, c, d, d, d);
    }
    vbase += (uint32_t)b.vertices.size();
    std::fprintf(obj, "\n");
  }
  std::fclose(obj);

  // ---- bounding box ----
  bool have_bbox = false;
  float minx = 0, miny = 0, minz = 0, maxx = 0, maxy = 0, maxz = 0;
  for (const auto& b : model.triangle_batches) {
    for (const auto& v : b.vertices) {
      if (!have_bbox) {
        minx = maxx = v.position.x;
        miny = maxy = v.position.y;
        minz = maxz = v.position.z;
        have_bbox = true;
      } else {
        minx = std::min(minx, v.position.x); maxx = std::max(maxx, v.position.x);
        miny = std::min(miny, v.position.y); maxy = std::max(maxy, v.position.y);
        minz = std::min(minz, v.position.z); maxz = std::max(maxz, v.position.z);
      }
    }
  }
  if (have_bbox) {
    std::printf("  bbox: X [% .4f .. % .4f] Y [% .4f .. % .4f] Z [% .4f .. % .4f]\n",
                minx, maxx, miny, maxy, minz, maxz);
  }
  std::printf("  -> %s\n  -> %s\n", obj_path.string().c_str(),
              mtl_path.string().c_str());
  return true;
}

// Decode a u16 (UTF-16) string to UTF-8 for printing (asset names are ASCII in
// practice; non-ASCII code units fall back to '?').
std::string U16ToUtf8(const std::u16string& s) {
  std::string out;
  for (char16_t c : s) {
    if (c < 0x80) {
      out.push_back((char)c);
    } else {
      out.push_back('?');
    }
  }
  return out;
}

// Human-readable list of the category bit flags set in a categories mask.
std::string CategoryNames(uint32_t categories) {
  using namespace rex::avatars::ComponentCategory;
  struct {
    uint32_t bit;
    const char* name;
  } kCats[] = {
      {kHead, "Head"},   {kBody, "Body"},       {kHair, "Hair"},
      {kTop, "Top"},     {kBottom, "Bottom"},   {kShoes, "Shoes"},
      {kHat, "Hat"},     {kGloves, "Gloves"},   {kGlasses, "Glasses"},
      {kWristwear, "Wristwear"}, {kEarrings, "Earrings"}, {kRing, "Ring"},
      {kProp, "Prop"},   {kAnimation, "Animation"},
  };
  std::string out;
  for (const auto& c : kCats) {
    if (categories & c.bit) {
      if (!out.empty()) out.push_back('|');
      out += c.name;
    }
  }
  if (out.empty()) out = "(none)";
  return out;
}

}  // namespace

// ===========================================================================
// Mode 1: marketplace STFS "LIVE"/"CON "/"PIRS" package -> model.obj
// ===========================================================================
namespace {

// Generic mode: dump every file in the container to out_dir, unmodified.
// (Flat package layout; nested directories are not resolved by ListFiles.)
int RunStfsExtractAll(const fs::path& in_path, const fs::path& out_dir) {
  StfsReader stfs;
  if (!stfs.Load(in_path)) {
    std::printf("ERROR: not a valid STFS package (or unreadable): %s\n",
                in_path.string().c_str());
    return 1;
  }
  fs::create_directories(out_dir);
  int written = 0;
  for (const auto& f : stfs.ListFiles()) {
    if (f.is_dir) {
      fs::create_directories(out_dir / f.name);
      continue;
    }
    std::vector<uint8_t> bytes;
    if (!stfs.ReadFile(f, bytes)) {
      std::printf("ERROR: failed to read '%s' from STFS block chain.\n", f.name.c_str());
      return 2;
    }
    fs::path dst = out_dir / f.name;
    FILE* o = std::fopen(dst.string().c_str(), "wb");
    if (!o) {
      std::printf("ERROR: cannot write %s\n", dst.string().c_str());
      return 3;
    }
    std::fwrite(bytes.data(), 1, bytes.size(), o);
    std::fclose(o);
    std::printf("  wrote %-44s %zu bytes\n", f.name.c_str(), bytes.size());
    ++written;
  }
  std::printf("extracted %d file(s) to %s\n", written, out_dir.string().c_str());
  return 0;
}

int RunStfsMode(const fs::path& in_path, const fs::path& out_dir) {
  StfsReader stfs;
  if (!stfs.Load(in_path)) {
    std::printf("ERROR: not a valid STFS package (or unreadable): %s\n",
                in_path.string().c_str());
    return 1;
  }

  // ---- Step 1: list embedded files ----
  auto files = stfs.ListFiles();
  std::printf("=== STFS contents (%zu entries) ===\n", files.size());
  for (const auto& f : files) {
    std::printf("  %-44s %s size=%-9u start_block=%u\n", f.name.c_str(),
                f.is_dir ? "[DIR] " : "[FILE]", f.length, f.start_block);
  }
  std::printf("\n");

  // Pick the avatar asset blob: prefer asset_v2.bin, else asset.bin, else the
  // largest .bin file. Also locate icon.png if present.
  const StfsFile* asset = nullptr;
  const StfsFile* icon = nullptr;
  for (const auto& f : files) {
    if (f.is_dir) continue;
    std::string lower = f.name;
    for (auto& c : lower) c = (char)std::tolower((unsigned char)c);
    if (lower == "asset_v2.bin") asset = &f;
    else if (lower == "asset.bin" && !asset) asset = &f;
    if (lower == "icon.png") icon = &f;
  }
  if (!asset) {
    const StfsFile* biggest = nullptr;
    for (const auto& f : files) {
      if (f.is_dir) continue;
      std::string lower = f.name;
      for (auto& c : lower) c = (char)std::tolower((unsigned char)c);
      if (lower.size() >= 4 && lower.substr(lower.size() - 4) == ".bin") {
        if (!biggest || f.length > biggest->length) biggest = &f;
      }
    }
    asset = biggest;
  }
  if (!asset) {
    std::printf("ERROR: no avatar asset blob (.bin) found in package. Items that only\n"
                "reference asset ids resolve against an AvatarAssetPack.toc; use --pack.\n");
    return 2;
  }
  std::printf("Using avatar asset blob: %s (%u bytes)\n", asset->name.c_str(),
              asset->length);

  std::vector<uint8_t> strb_bytes;
  if (!stfs.ReadFile(*asset, strb_bytes)) {
    std::printf("ERROR: failed to read asset blob from STFS block chain.\n");
    return 3;
  }
  if (strb_bytes.size() < 4 ||
      (std::memcmp(strb_bytes.data(), "STRB", 4) != 0 &&
       std::memcmp(strb_bytes.data(), "YTGR", 4) != 0)) {
    std::printf("ERROR: asset blob is not an STRB/YTGR container (first4: %02X %02X %02X %02X).\n",
                strb_bytes[0], strb_bytes[1], strb_bytes[2], strb_bytes[3]);
    return 4;
  }

  // ---- Step 2: decode the model ----
  auto model = rex::avatars::Model::Load(strb_bytes.data(), strb_bytes.size(),
                                         rex::avatars::ModelLoadOption::kNone);
  if (!model) {
    std::printf("ERROR: Model::Load failed (no kModel STRB block or decompress failed).\n");
    return 5;
  }

  // ---- Step 3+4: textures + model.obj + model.mtl ----
  size_t total_verts = 0, total_tris = 0;
  if (!WriteModel(*model, out_dir, "model", total_verts, total_tris)) {
    return 6;
  }

  // ---- Also copy the marketplace icon if present ----
  if (icon) {
    std::vector<uint8_t> icon_bytes;
    if (stfs.ReadFile(*icon, icon_bytes)) {
      fs::path ip = out_dir / "icon.png";
      FILE* o = std::fopen(ip.string().c_str(), "wb");
      if (o) { std::fwrite(icon_bytes.data(), 1, icon_bytes.size(), o); std::fclose(o); }
    }
  }

  std::printf("\n=== Done ===\n");
  std::printf("Wrote model with %zu batches, %zu verts, %zu tris.\n",
              model->triangle_batches.size(), total_verts, total_tris);
  return 0;
}

// ===========================================================================
// Mode 2: avatar asset pack (AvatarAssetPack.toc) -> body.obj
// The base body lives in the asset pack, not in marketplace STFS items. The
// engine selects it as metadata.body_component.asset_id, whose canonical values
// are {a=2,b=0,c=1,...} (male) and {a=2,b=0,c=2,...} (female); see
// avatars/guest_load_asset.cpp GetBodyType(). AssetPack indexes assets by id.b,
// so both resolve to pack asset index 0 (the body). Lists every asset, then
// extracts the chosen one by index.
// ===========================================================================
int RunPackMode(const fs::path& in_path, const fs::path& out_dir,
                int want_index) {
  // Load the whole .toc into memory and hand it to the recomp's AssetPack loader.
  FILE* f = std::fopen(in_path.string().c_str(), "rb");
  if (!f) {
    std::printf("ERROR: cannot open asset pack: %s\n", in_path.string().c_str());
    return 1;
  }
  std::fseek(f, 0, SEEK_END);
  long sz = std::ftell(f);
  std::fseek(f, 0, SEEK_SET);
  if (sz <= 0) { std::fclose(f); std::printf("ERROR: empty asset pack.\n"); return 1; }
  std::vector<uint8_t> toc((size_t)sz);
  size_t got = std::fread(toc.data(), 1, toc.size(), f);
  std::fclose(f);
  if (got != toc.size()) {
    std::printf("ERROR: short read on asset pack.\n");
    return 1;
  }

  rex::avatars::AssetPack pack;
  if (!pack.Load(toc)) {  // NOTE: swaps toc into the pack; do not use toc after.
    std::printf("ERROR: AssetPack::Load failed (unknown version / corrupt header).\n");
    return 2;
  }
  const auto& infos = pack.asset_infos();
  std::printf("=== AvatarAssetPack: %zu assets ===\n", infos.size());
  std::printf("  %-5s %-28s %-10s %-6s %-38s %-12s %s\n", "idx", "category",
              "subcat", "bodies", "asset_id", "data_size", "name");
  for (size_t i = 0; i < infos.size(); ++i) {
    const auto& a = infos[i];
    std::u16string name16 = pack.GetAssetName(a.asset_ids[0]);
    if (name16.empty()) name16 = pack.GetAssetName(a.asset_ids[1]);
    std::string name = U16ToUtf8(name16);
    std::string cat = CategoryNames(a.categories);
    std::printf("  %-5zu %-28s 0x%-8X %-6u %-38s %-12zu %s\n", i, cat.c_str(),
                a.subcategory, (unsigned)a.bodies,
                a.asset_ids[0].to_string().c_str(), a.data_size, name.c_str());
  }
  std::printf("\n");

  // Collect candidate body assets (Body category set) for reporting.
  std::vector<size_t> body_assets;
  for (size_t i = 0; i < infos.size(); ++i) {
    if (infos[i].categories & rex::avatars::ComponentCategory::kBody) {
      body_assets.push_back(i);
    }
  }
  std::printf("=== Body-category assets (categories & kBody) ===\n");
  if (body_assets.empty()) {
    std::printf("  (none found; falling back to asset index 0)\n");
  }
  for (size_t idx : body_assets) {
    const auto& a = infos[idx];
    std::u16string name16 = pack.GetAssetName(a.asset_ids[0]);
    if (name16.empty()) name16 = pack.GetAssetName(a.asset_ids[1]);
    std::printf("  idx=%zu subcat=0x%X bodies=%u name=\"%s\" id=%s size=%zu\n",
                idx, a.subcategory, (unsigned)a.bodies,
                U16ToUtf8(name16).c_str(), a.asset_ids[0].to_string().c_str(),
                a.data_size);
  }
  std::printf("\n");

  // Decide which asset is the base body.
  //  - explicit --index wins
  //  - else the canonical male body resolves to pack index id.b == 0
  //  - else the first Body-category asset
  int chosen = want_index;
  const char* reason = "explicit --index";
  if (chosen < 0) {
    // Canonical male base body id (from GetBodyType): {a=2,b=0,c=1,...}. The pack
    // looks assets up by id.b only, so the body is at index 0.
    if (!infos.empty() &&
        (infos[0].categories & rex::avatars::ComponentCategory::kBody)) {
      chosen = 0;
      reason = "canonical body id.b==0 (Body category)";
    } else if (!body_assets.empty()) {
      chosen = (int)body_assets.front();
      reason = "first Body-category asset";
    } else if (!infos.empty()) {
      chosen = 0;
      reason = "fallback to asset index 0";
    }
  }
  if (chosen < 0 || chosen >= (int)infos.size()) {
    std::printf("ERROR: no asset to extract (chosen index %d out of range).\n",
                chosen);
    return 3;
  }
  std::printf("Chosen base body asset: index %d (%s)\n", chosen, reason);

  // Fetch the raw STRB/YTGR asset bytes by index, not by asset id:
  // AssetPack::GetAssetData() resolves via id.b, and every asset in this pack
  // shares a colliding/zero id.b, so it would always hand back asset index 0
  // regardless of `chosen`. GetAssetDataByIndex() slices the same internal
  // buffer using the chosen asset's own data_offset/data_size. The same
  // ModelLoadOption::kNone as the STFS path keeps the asset in the identical
  // rest-pose coordinate space.
  const uint8_t* asset_buffer = nullptr;
  size_t asset_size = 0;
  if (!pack.GetAssetDataByIndex((size_t)chosen, asset_buffer, asset_size)) {
    std::printf("ERROR: GetAssetDataByIndex failed for chosen asset.\n");
    return 4;
  }
  std::printf("Asset bytes: %zu (first4: %02X %02X %02X %02X)\n", asset_size,
              asset_buffer[0], asset_buffer[1], asset_buffer[2], asset_buffer[3]);

  auto model = rex::avatars::Model::Load(asset_buffer, asset_size,
                                         rex::avatars::ModelLoadOption::kNone);
  if (!model) {
    // Not a model; if it is a blend shape (e.g. a wearable's body-hiding
    // template), print its patch structure instead.
    auto shape = rex::avatars::BlendShape::Load(asset_buffer, asset_size,
                                                rex::avatars::BlendShapeLoadOption::kNone);
    if (shape) {
      std::printf("=== blend shape ===\n");
      std::printf("  index_patch : original_id=%s total_buffer_size=0x%X indices=%zu\n",
                  shape->index_patch.original_asset_id.to_string().c_str(),
                  shape->index_patch.total_buffer_size, shape->index_patch.indices.size());
      std::printf("  vertex_patch: original_id=%s total_buffer_size=0x%X vertices=%zu\n",
                  shape->vertex_patch.original_asset_id.to_string().c_str(),
                  shape->vertex_patch.total_buffer_size, shape->vertex_patch.vertices.size());
      return 0;
    }
    std::printf("ERROR: Model::Load failed for body asset (no kModel STRB block, or\n"
                "this asset is a skeleton/animation rather than a renderable model).\n");
    return 5;
  }

  size_t total_verts = 0, total_tris = 0;
  if (!WriteModel(*model, out_dir, "body", total_verts, total_tris)) {
    return 6;
  }

  std::printf("\n=== Done ===\n");
  std::printf("Wrote base body (asset idx %d) with %zu batches, %zu verts, %zu tris.\n",
              chosen, model->triangle_batches.size(), total_verts, total_tris);
  return 0;
}

// ===========================================================================
// Mode 5: raw STRB/YTGR blob (a marketplace item's asset_v2.bin): summarize
// the blocks and decode the model to OBJ (import validation for closet items).
// ===========================================================================
static uint32_t DetectItemBodies(const uint8_t* bytes, size_t size);

int RunRawStrbMode(const fs::path& in_path, const fs::path& out_dir) {
  FILE* f = std::fopen(in_path.string().c_str(), "rb");
  if (!f) {
    std::printf("ERROR: cannot open: %s\n", in_path.string().c_str());
    return 1;
  }
  std::fseek(f, 0, SEEK_END);
  long sz = std::ftell(f);
  std::fseek(f, 0, SEEK_SET);
  std::vector<uint8_t> bytes((size_t)std::max(0L, sz));
  size_t got = std::fread(bytes.data(), 1, bytes.size(), f);
  std::fclose(f);
  if (!sz || got != bytes.size()) {
    std::printf("ERROR: bad read.\n");
    return 1;
  }
  using rex::avatars::strb::CountSTRBBlocks;
  using rex::avatars::strb::STRBBlockId;
  std::printf("=== raw STRB/YTGR blob: %ld bytes ===\n", sz);
  std::printf("  closet-import bodies verdict: %u (0=malformed 1=male 2=female 3=both)\n",
              DetectItemBodies(bytes.data(), bytes.size()));
  std::printf("  blocks: model=%zu texture=%zu skeleton=%zu animation=%zu shape=%zu\n",
              CountSTRBBlocks(bytes.data(), bytes.size(), STRBBlockId::kModel),
              CountSTRBBlocks(bytes.data(), bytes.size(), STRBBlockId::kTexture),
              CountSTRBBlocks(bytes.data(), bytes.size(), STRBBlockId::kSkeleton),
              CountSTRBBlocks(bytes.data(), bytes.size(), STRBBlockId::kAnimation),
              CountSTRBBlocks(bytes.data(), bytes.size(), STRBBlockId::kShapeOverrides));
  auto shape = rex::avatars::BlendShape::Load(bytes.data(), bytes.size(),
                                              rex::avatars::BlendShapeLoadOption::kNone);
  if (shape) {
    std::printf("  shape: index target=%s (%zu idx, tbs=%u), vertex target=%s (%zu verts, tbs=%u)\n",
                shape->index_patch.original_asset_id.to_string().c_str(),
                shape->index_patch.indices.size(), shape->index_patch.total_buffer_size,
                shape->vertex_patch.original_asset_id.to_string().c_str(),
                shape->vertex_patch.vertices.size(), shape->vertex_patch.total_buffer_size);
  }
  auto model = rex::avatars::Model::Load(bytes.data(), bytes.size(),
                                         rex::avatars::ModelLoadOption::kNone);
  if (!model) {
    std::printf("  (no decodable model)\n");
    return 0;
  }
  std::printf("  model: %zu batches, textures=%zu\n", model->triangle_batches.size(),
              model->textures.size());
  size_t total_verts = 0, total_tris = 0;
  if (WriteModel(*model, out_dir, "item", total_verts, total_tris)) {
    std::printf("  wrote OBJ: %zu verts, %zu tris -> %s\n", total_verts, total_tris,
                out_dir.string().c_str());
  }
  return 0;
}

// ===========================================================================
// Mode 6: closet import (--closet-import <raw_archive_root> <closet_dir>
// [--icon-cat=<hex>]): walk a raw marketplace archive (either layout known
// to WalkArchiveDir below), dedupe by GUID, copy each blob to
// <closet_dir>/<guid>.bin and write closet_index.tsv with
//   guid TAB categories-hex TAB bodies TAB name
// Category comes from the GUID's first dword; bodies from the embedded
// body-hiding shape's target id (c=1 male, c=2 female; no shape = both);
// name from the item folder.
// ===========================================================================
// Validates a marketplace item's STRB blocks and detects the gender of its
// embedded body-hiding shape WITHOUT parsing the full (data-driven,
// crash-prone on malformed items) patch structures: the shape block starts
// with count(u32) + total_buffer_size(u32) + original_asset_id(16B), and the
// target id's .c is 1=male / 2=female.
// Returns 1/2 for gendered, 3 for unisex/no shape, 0 for malformed (skip).
// Only kModel and kTexture STRB blocks are chunk-compressed
// (BlockHeader{compressed_size, uncompressed_offset, uncompressed_size} + LZX
// per chunk); kShapeOverrides, animation and skeleton blocks are stored raw,
// so the chunk parser must never run over those.
//
// Allocation-free (the SEH guard would leak on fault otherwise).
static uint32_t DetectItemBodiesImpl(const uint8_t* bytes, size_t size) {
  using rex::avatars::strb::GetSTRBBlock;
  using rex::avatars::strb::STRBBlockId;
  // Sanity for the chunk-compressed blocks: claimed uncompressed size must
  // be plausible (the runtime allocates that much when the item is worn),
  // and the first chunk must fit inside the block.
  static const STRBBlockId kCompressedBlocks[] = {STRBBlockId::kTexture, STRBBlockId::kModel};
  for (STRBBlockId bid : kCompressedBlocks) {
    const uint8_t* blk = nullptr;
    size_t blk_size = 0;
    if (!GetSTRBBlock(bytes, size, bid, blk, blk_size)) {
      continue;
    }
    if (blk_size < 12) {
      return 0;
    }
    const uint32_t first_chunk_compressed =
        (uint32_t)blk[0] | ((uint32_t)blk[1] << 8) | ((uint32_t)blk[2] << 16) |
        ((uint32_t)blk[3] << 24);
    if (first_chunk_compressed + 12u > blk_size) {
      return 0;
    }
    size_t unc = 0;
    // GetUncompressedSize validates every chunk header against the block
    // bounds; a false return means a truncated or corrupt block that would
    // fault in lzxd at load time.
    if (!rex::avatars::compression::GetUncompressedSize(blk, blk_size, unc) ||
        unc > (64u << 20)) {
      return 0;
    }
  }
  // Gender from the RAW (uncompressed) hiding-shape block headers:
  // count(u32) + total_buffer_size(u32) + original_asset_id{a u32, b u16,
  // c u16, d 8B}. Items can carry several shapes (head-targeting ones for
  // masks, body-targeting ones for clothing); gender comes from the shape
  // whose target category is the BODY (a == 2), .c 1=male / 2=female.
  for (size_t occurrence = 0;; ++occurrence) {
    const uint8_t* block = nullptr;
    size_t block_size = 0;
    if (!rex::avatars::strb::GetSTRBBlockN(bytes, size, STRBBlockId::kShapeOverrides, occurrence,
                                           block, block_size)) {
      break;
    }
    if (block_size < 24) {
      continue;
    }
    rex::avatars::BitStream stream(block, block_size * 8);
    const uint32_t index_count = stream.Read<uint32_t>();
    if (index_count > 8192) {
      return 0;  // the runtime's parser trusts this count as an allocation size
    }
    (void)stream.Read<uint32_t>();  // total buffer size
    const uint32_t a = stream.Read<uint32_t>();
    (void)stream.Read<uint16_t>();  // id.b
    const uint16_t c = stream.Read<uint16_t>();
    if (a != 2) {
      continue;  // head/hair-targeting shape; says nothing about gender
    }
    if (c == 1) return 1;
    if (c == 2) return 2;
  }
  return 3;
}

// SEH wrapper: a malformed item can make the data-driven STRB block walk
// read out of bounds; treat any SEH fault as "malformed, skip", logging the
// exception code. (No C++ objects in this frame, as SEH requires; the impl
// is allocation-free so skipping unwind leaks nothing.)
static uint32_t DetectItemBodies(const uint8_t* bytes, size_t size) {
  DWORD code = 0;
  __try {
    return DetectItemBodiesImpl(bytes, size);
  } __except (code = GetExceptionCode(), EXCEPTION_EXECUTE_HANDLER) {
    std::printf("  (item parse fault, SEH code %08lX)\n", code);
    return 0;
  }
}

// Paths as UTF-8 and files opened by their wide path: marketplace folders are
// named in every script (Japanese, Cyrillic, Korean, Turkish), and
// path::string() throws std::system_error for characters outside the local
// codepage.
static std::string PathU8(const fs::path& p) {
  const auto s = p.u8string();
  return std::string(reinterpret_cast<const char*>(s.data()), s.size());
}

static FILE* OpenFile(const fs::path& p, const wchar_t* mode) {
  return _wfopen(p.c_str(), mode);
}

static std::string ReadTextFile(const fs::path& path) {
  FILE* f = OpenFile(path, L"rb");
  if (!f) return std::string();
  std::fseek(f, 0, SEEK_END);
  long sz = std::ftell(f);
  std::fseek(f, 0, SEEK_SET);
  std::string text((size_t)std::max(0L, sz), '\0');
  size_t got = std::fread(text.data(), 1, text.size(), f);
  std::fclose(f);
  text.resize(got);
  return text;
}

// ---------------------------------------------------------------------------
// Raw archive enumeration shared by --closet-import / --closet-icons. Two
// per-item-folder layouts are recognised:
//   (a) avataritems_raw: asset_v2[_N].bin paired with ID[_N].TXT (the product
//       guid), or, single-variant, the 36-char <guid>.xml, plus the
//       package's own ICON[_N].PNG.
//   (b) Avatar_Items marketplace dump (2021): <guid>.bin (+ <guid>.xml) and
//       the store's thumbsm/thumbm/thumblg.png art. A folder holding several
//       products (male/female variants, colourways) carries the N-th
//       product's art as *_N.png (N >= 2). Nothing in the files ties a suffix
//       to a guid except the downloader's write order: each product's bin,
//       xml and art were written back-to-back, so the bins sorted by mtime
//       give the suffixes, and an icon is accepted only when its own mtime
//       falls between its bin's and the next bin's. A folder that fails
//       that check gets no icon rather than a guessed one.
// Icons: thumbm.png (128 px) preferred, thumbsm.png (64 px) fallback, never
// over kMaxIconBytes: the editor calls XamAvatarGetAssetIcon with a
// 0x10000-byte buffer and larger PNGs are refused at serve time.
// ---------------------------------------------------------------------------
struct ArchiveItem {
  fs::path bin;
  fs::path item_dir;
  std::string guid;       // lowercase dashed; empty = blob without a product id
  fs::path icon;          // empty = no usable art
  std::string icon_note;  // why icon is empty (for the log)
};

static constexpr uintmax_t kMaxIconBytes = 0x10000;

static bool IsDashedGuid(const std::string& s) {
  if (s.size() != 36) return false;
  for (size_t i = 0; i < s.size(); ++i) {
    const unsigned char c = (unsigned char)s[i];
    if (i == 8 || i == 13 || i == 18 || i == 23) {
      if (c != '-') return false;
    } else if (!std::isxdigit(c)) {
      return false;
    }
  }
  return true;
}

static std::string LowerAscii(std::string s) {
  for (auto& c : s) c = (char)std::tolower((unsigned char)c);
  return s;
}

// Layout (a): the guid from ID<suffix>.TXT, else (single variant) the
// 36-char <guid>.xml stem.
static std::string LegacyItemGuid(const fs::path& item_dir, const std::string& suffix) {
  std::error_code ec;
  const std::string id_text = ReadTextFile(item_dir / ("ID" + suffix + ".TXT"));
  const size_t eol = id_text.find_first_of("\r\n");
  const std::string guid =
      LowerAscii(id_text.substr(0, eol == std::string::npos ? id_text.size() : eol));
  if (IsDashedGuid(guid)) return guid;
  if (!suffix.empty()) return std::string();
  for (const auto& entry : fs::directory_iterator(item_dir, ec)) {
    if (!entry.is_regular_file(ec) || entry.path().extension() != ".xml") continue;
    const std::string stem = LowerAscii(PathU8(entry.path().stem()));
    if (IsDashedGuid(stem)) return stem;
  }
  return std::string();
}

template <typename Fn>
static void WalkArchiveDir(const fs::path& dir, const Fn& fn) {
  std::error_code ec;
  std::vector<fs::path> subdirs;
  std::vector<fs::path> legacy_bins;  // layout (a): asset_v2*.bin
  struct GuidBin {
    fs::path path;
    fs::file_time_type mtime;
  };
  std::vector<GuidBin> guid_bins;  // layout (b): <guid>.bin
  for (const auto& entry : fs::directory_iterator(dir, ec)) {
    if (entry.is_directory(ec)) {
      subdirs.push_back(entry.path());
      continue;
    }
    if (!entry.is_regular_file(ec) || entry.path().extension() != ".bin") continue;
    const std::string fname = PathU8(entry.path().filename());
    if (fname.rfind("asset_v2", 0) == 0) {
      legacy_bins.push_back(entry.path());
    } else if (IsDashedGuid(LowerAscii(PathU8(entry.path().stem())))) {
      guid_bins.push_back({entry.path(), fs::last_write_time(entry.path(), ec)});
    }
  }
  for (const auto& bin : legacy_bins) {
    const std::string fname = PathU8(bin.filename());
    const std::string suffix = fname.substr(8, fname.size() - 8 - 4);  // "" or "_1", ...
    ArchiveItem item;
    item.bin = bin;
    item.item_dir = dir;
    item.guid = LegacyItemGuid(dir, suffix);
    // ICON<suffix>.PNG pairs with the variant like ID<suffix>.TXT does; a
    // folder-level ICON.PNG is shared across variants.
    fs::path icon = dir / ("ICON" + suffix + ".PNG");
    if (suffix.empty() || !fs::exists(icon, ec)) icon = dir / "ICON.PNG";
    if (fs::exists(icon, ec)) {
      item.icon = icon;
    } else {
      item.icon_note = "no ICON.PNG";
    }
    fn(item);
  }
  std::sort(guid_bins.begin(), guid_bins.end(), [](const GuidBin& a, const GuidBin& b) {
    return a.mtime != b.mtime ? a.mtime < b.mtime : a.path < b.path;
  });
  for (size_t i = 0; i < guid_bins.size(); ++i) {
    ArchiveItem item;
    item.bin = guid_bins[i].path;
    item.item_dir = dir;
    item.guid = LowerAscii(PathU8(item.bin.stem()));
    const std::string suffix = i == 0 ? std::string() : "_" + std::to_string(i + 1);
    item.icon_note = "no thumbm" + suffix + ".png / thumbsm" + suffix + ".png";
    for (const char* base : {"thumbm", "thumbsm"}) {
      const fs::path cand = dir / (std::string(base) + suffix + ".png");
      if (!fs::exists(cand, ec)) continue;
      const auto art_time = fs::last_write_time(cand, ec);
      if (art_time < guid_bins[i].mtime ||
          (i + 1 < guid_bins.size() && art_time > guid_bins[i + 1].mtime)) {
        item.icon_note = PathU8(cand.filename()) +
                         " was not written together with this bin (pairing ambiguous), skipped";
        break;
      }
      if (fs::file_size(cand, ec) > kMaxIconBytes) {
        item.icon_note = PathU8(cand.filename()) + " over 64KB (editor buffer), skipped";
        continue;
      }
      item.icon = cand;
      item.icon_note.clear();
      break;
    }
    fn(item);
  }
  for (const auto& sd : subdirs) WalkArchiveDir(sd, fn);
}

// closet_index.tsv -> guid -> rest-of-line. Tolerates a UTF-8 BOM (a Notepad
// edit adds one, after which the runtime's parser silently drops the first
// item; the rewrite below always emits the file without it) and CRLF.
static std::map<std::string, std::string> ReadClosetIndex(const fs::path& closet_dir) {
  std::map<std::string, std::string> lines;
  const std::string text = ReadTextFile(closet_dir / "closet_index.tsv");
  size_t pos = text.rfind("\xEF\xBB\xBF", 0) == 0 ? 3 : 0;
  while (pos < text.size()) {
    size_t eol = text.find('\n', pos);
    if (eol == std::string::npos) eol = text.size();
    std::string line = text.substr(pos, eol - pos);
    pos = eol + 1;
    if (!line.empty() && line.back() == '\r') line.pop_back();
    const size_t tab = line.find('\t');
    if (tab == 36 && IsDashedGuid(line.substr(0, 36))) {
      lines[LowerAscii(line.substr(0, 36))] = line.substr(tab + 1);
    }
  }
  return lines;
}

static uint32_t GuidCategories(const std::string& guid) {
  return (uint32_t)std::strtoul(guid.substr(0, 8).c_str(), nullptr, 16);
}

int RunClosetImport(const fs::path& raw_root, const fs::path& closet_dir, uint32_t icon_cat_mask) {
  std::error_code ec;
  fs::create_directories(closet_dir, ec);
  std::map<std::string, std::string> index_lines = ReadClosetIndex(closet_dir);
  std::map<std::string, int> attempted;  // guid -> 1 (crash journal)
  size_t scanned = 0, imported = 0, dupes = 0, failed = 0, poisoned = 0, icons = 0, no_icon = 0,
         unpaired = 0;
  // Resume support: seed from the existing (incrementally appended) index,
  // and from the attempt journal: a guid that was attempted but never made
  // the index crashed the process; skip it on this run.
  {
    const std::string journal = ReadTextFile(closet_dir / "closet_attempted.log");
    size_t pos = 0;
    while (pos < journal.size()) {
      size_t eol = journal.find('\n', pos);
      if (eol == std::string::npos) eol = journal.size();
      std::string guid = journal.substr(pos, eol - pos);
      pos = eol + 1;
      if (!guid.empty() && guid.back() == '\r') guid.pop_back();
      if (guid.size() == 36 && !index_lines.count(guid)) {
        attempted[guid] = 1;  // crashed last time -> poison, skip
      }
    }
    if (!index_lines.empty() || !attempted.empty()) {
      std::printf("  resuming: %zu indexed, %zu poison-skipped\n", index_lines.size(),
                  attempted.size());
    }
  }
  const fs::path index_path = closet_dir / "closet_index.tsv";
  if (fs::exists(index_path, ec)) {
    // The walk appends and the end rewrites: keep the pre-run index.
    fs::copy_file(index_path, closet_dir / "closet_index.tsv.bak",
                  fs::copy_options::overwrite_existing, ec);
  }
  FILE* tsv_append = OpenFile(index_path, L"ab");
  FILE* journal_append = OpenFile(closet_dir / "closet_attempted.log", L"ab");
  if (!tsv_append || !journal_append) {
    std::printf("ERROR: cannot open index/journal for append.\n");
    return 1;
  }
  {
    const std::string existing = ReadTextFile(index_path);
    if (!existing.empty() && existing.back() != '\n') std::fputc('\n', tsv_append);
  }
  WalkArchiveDir(raw_root, [&](const ArchiveItem& item) {
    ++scanned;
    try {
      if (item.guid.empty()) {
        ++unpaired;
        ++failed;
        return;
      }
      if (index_lines.count(item.guid)) {
        ++dupes;
        return;
      }
      if (attempted.count(item.guid)) {
        ++poisoned;
        return;
      }
      // Journal the attempt before parsing: if this item crashes the process,
      // the next run skips it.
      std::fprintf(journal_append, "%s\n", item.guid.c_str());
      std::fflush(journal_append);
      FILE* f = OpenFile(item.bin, L"rb");
      if (!f) {
        ++failed;
        return;
      }
      std::fseek(f, 0, SEEK_END);
      long sz = std::ftell(f);
      std::fseek(f, 0, SEEK_SET);
      std::vector<uint8_t> bytes((size_t)std::max(0L, sz));
      size_t got = std::fread(bytes.data(), 1, bytes.size(), f);
      std::fclose(f);
      if (!sz || got != bytes.size()) {
        ++failed;
        return;
      }
      const uint32_t bodies = DetectItemBodies(bytes.data(), bytes.size());
      if (bodies == 0) {
        std::printf("  skipping malformed item: %s\n", PathU8(item.item_dir).c_str());
        ++failed;
        return;
      }
      const uint32_t categories = GuidCategories(item.guid);
      std::string name = PathU8(item.item_dir.filename());
      for (auto& c : name) {
        if (c == '\t' || c == '\n' || c == '\r') c = ' ';
      }
      {
        // Always overwrite: a multi-variant folder can pair a bin with the
        // wrong product guid, so a re-import must be able to correct it.
        FILE* of = OpenFile(closet_dir / (item.guid + ".bin"), L"wb");
        if (!of) {
          ++failed;
          return;
        }
        std::fwrite(bytes.data(), 1, bytes.size(), of);
        std::fclose(of);
      }
      if (categories & icon_cat_mask) {
        if (item.icon.empty()) {
          ++no_icon;
        } else {
          fs::create_directories(closet_dir / "icons", ec);
          std::error_code copy_ec;
          fs::copy_file(item.icon, closet_dir / "icons" / (item.guid + ".png"),
                        fs::copy_options::overwrite_existing, copy_ec);
          if (copy_ec) {
            ++no_icon;
          } else {
            ++icons;
          }
        }
      }
      char meta[64];
      std::snprintf(meta, sizeof(meta), "%08X\t%u\t", categories, bodies);
      index_lines[item.guid] = std::string(meta) + name;
      std::fprintf(tsv_append, "%s\t%s%s\n", item.guid.c_str(), meta, name.c_str());
      std::fflush(tsv_append);
      ++imported;
      if (imported % 1000 == 0) {
        std::printf("  ... %zu imported\n", imported);
        std::fflush(stdout);
      }
    } catch (const std::exception& e) {
      std::printf("  item threw (%s), skipping\n", e.what());
      std::fflush(stdout);
      ++failed;
    }
  });
  std::fclose(journal_append);
  std::fclose(tsv_append);
  // Rewrite the index cleanly (sorted, deduped, no BOM, LF) now that the
  // walk finished.
  FILE* idx = OpenFile(index_path, L"wb");
  if (!idx) {
    std::printf("ERROR: cannot write %s\n", PathU8(index_path).c_str());
    return 1;
  }
  for (const auto& [guid, rest] : index_lines) {
    std::fprintf(idx, "%s\t%s\n", guid.c_str(), rest.c_str());
  }
  std::fclose(idx);
  std::printf("closet import: scanned=%zu imported=%zu (icons=%zu, no-icon=%zu) dupes=%zu "
              "failed=%zu (unpaired=%zu) poisoned=%zu -> %s\n",
              scanned, imported, icons, no_icon, dupes, failed, unpaired, poisoned,
              PathU8(closet_dir).c_str());
  return 0;
}

// ===========================================================================
// Mode 6b: closet icon backfill (--closet-icons <raw_archive_root>
// <closet_dir> [--icon-cat=<hex>]): walk the raw archive and copy each
// item's art to <closet_dir>/icons/<guid>.png for items ALREADY in
// closet_index.tsv, without touching the bins or the index. Existing icon
// files are kept.
// ===========================================================================
int RunClosetIcons(const fs::path& raw_root, const fs::path& closet_dir, uint32_t icon_cat_mask) {
  std::error_code ec;
  const std::map<std::string, std::string> indexed = ReadClosetIndex(closet_dir);
  if (indexed.empty()) {
    std::printf("ERROR: no closet_index.tsv (or empty) at %s; run --closet-import first.\n",
                PathU8(closet_dir).c_str());
    return 1;
  }
  fs::create_directories(closet_dir / "icons", ec);
  size_t scanned = 0, copied = 0, present = 0, no_icon = 0, unknown = 0, filtered = 0,
         noted = 0, replaced = 0;
  // Stand-ins written by --gen-icons (icons_generated.tsv) yield to real art.
  std::set<std::string> generated;
  const fs::path gen_tsv = closet_dir / "icons_generated.tsv";
  if (FILE* gf = _wfopen(gen_tsv.wstring().c_str(), L"rb")) {
    char line[1024];
    while (std::fgets(line, sizeof(line), gf)) {
      std::string s(line);
      if (s.size() > 36 && s[36] == '\t') generated.insert(s.substr(0, 36));
    }
    std::fclose(gf);
  }
  std::set<std::string> replaced_guids;
  WalkArchiveDir(raw_root, [&](const ArchiveItem& item) {
    ++scanned;
    try {
      if (item.guid.empty() || !indexed.count(item.guid)) {
        ++unknown;
        return;
      }
      if (!(GuidCategories(item.guid) & icon_cat_mask)) {
        ++filtered;
        return;
      }
      const fs::path dst = closet_dir / "icons" / (item.guid + ".png");
      const bool stand_in = generated.count(item.guid) != 0;
      if (fs::exists(dst, ec) && !(stand_in && !item.icon.empty())) {
        ++present;
        return;
      }
      if (stand_in && !item.icon.empty()) replaced_guids.insert(item.guid);
      if (item.icon.empty()) {
        ++no_icon;
        // Plain "no art" is the common case; only the pairing and size
        // refusals are worth a line.
        if (item.icon_note.rfind("no ", 0) != 0 && noted++ < 200) {
          std::printf("  %s  %s: %s\n", item.guid.c_str(),
                      PathU8(item.item_dir.filename()).c_str(), item.icon_note.c_str());
        }
        return;
      }
      std::error_code copy_ec;
      fs::copy_file(item.icon, dst, copy_ec);
      if (copy_ec) {
        ++no_icon;
      } else {
        ++copied;
      }
      if (copied && copied % 1000 == 0) {
        std::printf("  ... %zu icons copied\n", copied);
        std::fflush(stdout);
      }
    } catch (const std::exception& e) {
      std::printf("  item threw (%s), skipping\n", e.what());
      ++no_icon;
    }
  });
  if (!replaced_guids.empty()) {
    // drop the replaced stand-ins from icons_generated.tsv
    std::vector<std::string> keep;
    if (FILE* gf = _wfopen(gen_tsv.wstring().c_str(), L"rb")) {
      char line[1024];
      while (std::fgets(line, sizeof(line), gf)) {
        std::string s(line);
        if (s.size() > 36 && s[36] == '\t' && replaced_guids.count(s.substr(0, 36))) { ++replaced; continue; }
        keep.push_back(s);
      }
      std::fclose(gf);
    }
    if (FILE* gf = _wfopen(gen_tsv.wstring().c_str(), L"wb")) {
      for (const auto& k : keep) std::fputs(k.c_str(), gf);
      std::fclose(gf);
    }
    std::printf("  generated stand-ins replaced by store art: %zu\n", replaced);
  }
  std::printf("closet icons: scanned=%zu copied=%zu already-present=%zu no-icon=%zu "
              "not-in-index=%zu category-filtered=%zu -> %s\n",
              scanned, copied, present, no_icon, unknown, filtered,
              PathU8(closet_dir / "icons").c_str());
  return 0;
}

// ===========================================================================
// Mode 7: fix saved outfits (--fix-outfits <toc> <outfits_dir>): normalize
// the asset ids inside saved X_AVATAR_METADATA outfit files (the editor's
// OUTFIT00_* savegame packages). The Avatar Editor's "My Outfits" display
// gate requires an exact id match against the enumerated catalog, and the
// GUID's .c field is the asset's gender mask (1=male/2=female/3=both), so:
//  - stock-tail ids get .c rewritten to the pack asset's bodies mask
//  - references to companion entries (filtered from enumeration) are
//    remapped to their catalog partner
// ===========================================================================
int RunFixOutfits(const fs::path& toc_path, const fs::path& outfits_dir) {
  FILE* f = std::fopen(toc_path.string().c_str(), "rb");
  if (!f) {
    std::printf("ERROR: cannot open asset pack: %s\n", toc_path.string().c_str());
    return 1;
  }
  std::fseek(f, 0, SEEK_END);
  long sz = std::ftell(f);
  std::fseek(f, 0, SEEK_SET);
  std::vector<uint8_t> toc((size_t)std::max(0L, sz));
  size_t got = std::fread(toc.data(), 1, toc.size(), f);
  std::fclose(f);
  if (!sz || got != toc.size()) {
    std::printf("ERROR: bad read on asset pack.\n");
    return 1;
  }
  rex::avatars::AssetPack pack;
  if (!pack.Load(toc)) {
    std::printf("ERROR: AssetPack::Load failed.\n");
    return 2;
  }
  const auto& infos = pack.asset_infos();
  // catalog[j].asset_ids[0].b -> j maps a companion index to its catalog.
  std::vector<int32_t> companion_to_catalog(infos.size(), -1);
  for (size_t j = 0; j < infos.size(); ++j) {
    const auto& id0 = infos[j].asset_ids[0];
    const size_t partner = id0.b;
    if (!id0.is_zero() && partner != j && partner < infos.size()) {
      companion_to_catalog[partner] = (int32_t)j;
    }
  }
  static const uint8_t kTail[8] = {0xC1, 0xC8, 0xF1, 0x09, 0xA1, 0x9C, 0xB2, 0xE0};
  size_t files = 0, patched_files = 0, patched_ids = 0;
  std::error_code ec;
  for (fs::recursive_directory_iterator it(outfits_dir, ec), end; it != end; it.increment(ec)) {
    if (ec) break;
    if (!it->is_regular_file()) continue;
    const std::string name = it->path().filename().string();
    if (name.rfind("OUTFIT00_", 0) != 0 || it->path().extension() == ".header") continue;
    FILE* of = std::fopen(it->path().string().c_str(), "r+b");
    if (!of) continue;
    std::fseek(of, 0, SEEK_END);
    long osz = std::ftell(of);
    std::fseek(of, 0, SEEK_SET);
    if (osz < 1000) { std::fclose(of); continue; }
    std::vector<uint8_t> bytes((size_t)osz);
    if (std::fread(bytes.data(), 1, bytes.size(), of) != bytes.size()) { std::fclose(of); continue; }
    ++files;
    bool changed = false;
    // The editor's outfit-display gate (sub_920B9898 -> sub_9213BF30) only
    // walks components[13] @0x160 (stride 32, X_AVATAR_COMPONENT_INFO);
    // fallback_components[4] @0x300 get the same normalization for
    // consistency. Body @0x120 / head @0x140 / textures / blend shapes keep
    // their canonical encodings untouched.
    static const size_t kIdOffsets[] = {
        0x160, 0x180, 0x1A0, 0x1C0, 0x1E0, 0x200, 0x220, 0x240,
        0x260, 0x280, 0x2A0, 0x2C0, 0x2E0,                           // components
        0x300, 0x320, 0x340, 0x360,                                  // fallbacks
    };
    for (size_t off : kIdOffsets) {
      if (off + 16 > bytes.size()) continue;
      uint8_t* id = &bytes[off];
      if (std::memcmp(id + 8, kTail, 8) != 0) continue;  // not a stock pack id
      uint32_t a = (id[0] << 24) | (id[1] << 16) | (id[2] << 8) | id[3];
      uint32_t b = (id[4] << 8) | id[5];
      uint32_t c = (id[6] << 8) | id[7];
      if (a == 0 && b == 0 && c == 0) continue;  // empty slot
      if (b >= infos.size()) continue;
      uint32_t nb = b;
      uint32_t na = a;
      if (companion_to_catalog[b] >= 0) {
        nb = (uint32_t)companion_to_catalog[b];
        na = infos[nb].categories;
      }
      uint32_t nc = infos[nb].bodies ? infos[nb].bodies : 3;
      if (na != a || nb != b || nc != c) {
        id[0] = (uint8_t)(na >> 24); id[1] = (uint8_t)(na >> 16);
        id[2] = (uint8_t)(na >> 8);  id[3] = (uint8_t)na;
        id[4] = (uint8_t)(nb >> 8);  id[5] = (uint8_t)nb;
        id[6] = (uint8_t)(nc >> 8);  id[7] = (uint8_t)nc;
        changed = true;
        ++patched_ids;
        std::printf("  %s @%#zx: %08X-%04X-%04X -> %08X-%04X-%04X\n", name.c_str(), off, a, b, c,
                    na, nb, nc);
      }
    }
    if (changed) {
      std::fseek(of, 0, SEEK_SET);
      std::fwrite(bytes.data(), 1, bytes.size(), of);
      ++patched_files;
    }
    std::fclose(of);
  }
  std::printf("fix-outfits: %zu files scanned, %zu patched (%zu ids)\n", files, patched_files,
              patched_ids);
  return 0;
}

}  // namespace

// ===========================================================================
// Mode 8: closet scan (--scan-closet <closet_dir>): run every <guid>.bin
// through the same validation the importer uses (DetectItemBodies: STRB
// block walk + per-chunk header bounds via GetUncompressedSize). An item
// that fails would fault in lzxd when a title loads it. Prints one line
// per bad item so it can be removed from the closet + closet_index.tsv.
// ===========================================================================
static int RunClosetScan(const fs::path& closet_dir) {
  size_t scanned = 0, bad = 0;
  std::error_code ec;
  for (fs::directory_iterator it(closet_dir, ec), end; it != end; it.increment(ec)) {
    if (ec) break;
    if (!it->is_regular_file() || it->path().extension() != ".bin") {
      continue;
    }
    ++scanned;
    FILE* f = std::fopen(it->path().string().c_str(), "rb");
    if (!f) continue;
    std::fseek(f, 0, SEEK_END);
    long sz = std::ftell(f);
    std::fseek(f, 0, SEEK_SET);
    std::vector<uint8_t> bytes((size_t)std::max(0L, sz));
    size_t got = std::fread(bytes.data(), 1, bytes.size(), f);
    std::fclose(f);
    if (!sz || got != bytes.size() || DetectItemBodies(bytes.data(), bytes.size()) == 0) {
      std::printf("BAD %s\n", it->path().filename().string().c_str());
      std::fflush(stdout);
      ++bad;
    }
    if (scanned % 5000 == 0) {
      std::printf("  ... %zu scanned\n", scanned);
      std::fflush(stdout);
    }
  }
  std::printf("closet scan: scanned=%zu bad=%zu\n", scanned, bad);
  return bad ? 2 : 0;
}

// ---------------------------------------------------------------------------
// Main: dispatch to STFS or asset-pack mode based on flags / file magic.
// Usage:
//   avatarextract <input> [<out_dir>]            auto-detect input type
//   avatarextract --pack <toc> [<out_dir>] [--index N]
//   avatarextract <asset_v2.bin> [<out_dir>]     inspect a raw STRB/YTGR blob
//   avatarextract --closet-import <raw_root> <closet_dir> [--icon-cat=<hex>]
//   avatarextract --closet-icons <raw_root> <closet_dir> [--icon-cat=<hex>]
//       (raw_root: an avataritems_raw or Avatar_Items marketplace archive;
//        --icon-cat limits icon import to a category mask, e.g. 1000 = props)
//   avatarextract --scan-closet <closet_dir>
//   avatarextract --gen-icons <closet_dir> [--toc <pack.toc>] [--filter s] [--limit N] [--force]
//   avatarextract --fix-outfits <toc> <outfits_dir>
//   avatarextract --avatar <avatar_manifest.bin> <out_dir> [...]   saved-avatar export
//   avatarextract --avatar-info <avatar_manifest.bin>              summary JSON
// ---------------------------------------------------------------------------
// --avatar / --avatar-info: saved-avatar export mode (avatar_export.cpp).
int RunAvatarExport(const std::vector<std::string>& argv);
// --refit-female: female hiding shape for a male-only wearable (refit.cpp).
int RunRefitFemale(const std::vector<std::string>& argv);
// --prop-icons: render closet carryables into icons/<guid>.png (+ survey CSV).
int RunPropIcons(const std::vector<std::string>& argv);
// --gen-icons: fill-in icons (props posed at rest, wearables on the mannequin) for closet items without store art.
int RunGenIcons(const std::vector<std::string>& argv);

int main(int argc, char** argv) {
  for (int i = 1; i < argc; ++i) {
    std::string a = argv[i];
    if (a == "--prop-icons") {
      std::vector<std::string> rest;
      for (int k = 1; k < argc; ++k) {
        if (k != i) rest.push_back(argv[k]);
      }
      return RunPropIcons(rest);
    }
    if (a == "--gen-icons") {
      std::vector<std::string> rest;
      for (int k = 1; k < argc; ++k) {
        if (k != i) rest.push_back(argv[k]);
      }
      return RunGenIcons(rest);
    }
    if (a == "--refit-female") {
      std::vector<std::string> rest;
      for (int k = 1; k < argc; ++k) {
        if (k != i) rest.push_back(argv[k]);
      }
      return RunRefitFemale(rest);
    }
    if (a == "--avatar" || a == "--avatar-info") {
      std::vector<std::string> rest;
      if (a == "--avatar-info") rest.push_back("--avatar-info");
      for (int k = 1; k < argc; ++k) {
        if (k != i) rest.push_back(argv[k]);
      }
      return RunAvatarExport(rest);
    }
  }
  bool force_pack = false;
  bool extract_all = false;  // --extract: dump every STFS file raw (generic
                             // container extraction, e.g. DLC packages)
  bool closet_import = false;  // --closet-import: build a closet dir from raw archive
  bool closet_icons = false;   // --closet-icons: backfill icons/<guid>.png for an existing closet
  bool closet_scan = false;    // --scan-closet: validate every closet .bin, list corrupt items
  bool fix_outfits = false;    // --fix-outfits: normalize saved outfit manifests
  uint32_t icon_cat_mask = 0xFFFFFFFFu;  // --icon-cat=<hex>: closet icons only for these categories
  int want_index = -1;  // -1 = auto-pick the base body
  std::vector<std::string> positionals;
  for (int i = 1; i < argc; ++i) {
    std::string arg = argv[i];
    if (arg == "--pack") {
      force_pack = true;
    } else if (arg == "--extract") {
      extract_all = true;
    } else if (arg == "--closet-import") {
      closet_import = true;
    } else if (arg == "--closet-icons") {
      closet_icons = true;
    } else if (arg == "--scan-closet") {
      closet_scan = true;
    } else if (arg == "--fix-outfits") {
      fix_outfits = true;
    } else if (arg.rfind("--icon-cat=", 0) == 0) {
      icon_cat_mask = (uint32_t)std::strtoul(arg.c_str() + 11, nullptr, 16);
    } else if (arg == "--index" && i + 1 < argc) {
      want_index = std::atoi(argv[++i]);
    } else if (arg.rfind("--index=", 0) == 0) {
      want_index = std::atoi(arg.c_str() + 8);
    } else {
      positionals.push_back(arg);
    }
  }
  if (positionals.empty()) {
    std::printf("usage: avatarextract <input> [<out_dir>] [options]\n"
                "see the mode list at the top of main().\n");
    return 1;
  }
  fs::path in_path = positionals[0];
  fs::path out_dir = positionals.size() >= 2 ? fs::path(positionals[1])
                                             : in_path.parent_path() / "extracted";

  std::printf("avatarextract\n  input : %s\n  output: %s\n\n",
              in_path.string().c_str(), out_dir.string().c_str());

  // Auto-detect: read the first 16 bytes. STFS containers start with a known
  // 4CC; asset packs start with an avatar-TOC version GUID. The .toc extension
  // or --pack flag also forces pack mode.
  bool is_pack = force_pack;
  if (!is_pack) {
    std::string ext = in_path.extension().string();
    for (auto& c : ext) c = (char)std::tolower((unsigned char)c);
    if (ext == ".toc") is_pack = true;
  }
  if (!is_pack) {
    FILE* f = std::fopen(in_path.string().c_str(), "rb");
    if (f) {
      uint8_t magic[16] = {0};
      size_t n = std::fread(magic, 1, sizeof(magic), f);
      std::fclose(f);
      bool is_stfs = n >= 4 && (std::memcmp(magic, "LIVE", 4) == 0 ||
                                std::memcmp(magic, "CON ", 4) == 0 ||
                                std::memcmp(magic, "PIRS", 4) == 0);
      if (!is_stfs && n == 16) {
        // Compare against the three known avatar-TOC version GUIDs (mirrors the
        // private kAvatarTOCv1/v2/v3 in avatars/asset_pack.cpp; duplicated here
        // since those have internal linkage and are not exported in the header).
        static const uint8_t kTocV1[16] = {0x9A, 0xD6, 0xEB, 0xCE, 0x62, 0x62,
                                           0x4E, 0xEB, 0x8A, 0x82, 0xA3, 0xF7,
                                           0x0B, 0x81, 0x73, 0x69};
        static const uint8_t kTocV2[16] = {0x8A, 0x76, 0x2D, 0xF4, 0xC0, 0x67,
                                           0x4B, 0x66, 0xB4, 0xEE, 0x56, 0x46,
                                           0x6B, 0x1B, 0x82, 0x80};
        static const uint8_t kTocV3[16] = {0x58, 0x0A, 0x07, 0xD6, 0x4B, 0xCD,
                                           0x40, 0xBE, 0xBF, 0x37, 0x25, 0x4B,
                                           0xE8, 0x26, 0xFB, 0x0B};
        if (!std::memcmp(magic, kTocV1, 16) ||
            !std::memcmp(magic, kTocV2, 16) ||
            !std::memcmp(magic, kTocV3, 16)) {
          is_pack = true;
        }
      }
    }
  }

  if (closet_scan) {
    std::printf("(mode: closet scan)\n\n");
    return RunClosetScan(in_path);
  }
  if (closet_import) {
    std::printf("(mode: closet import)\n\n");
    return RunClosetImport(in_path, out_dir, icon_cat_mask);
  }
  if (closet_icons) {
    std::printf("(mode: closet icon backfill)\n\n");
    return RunClosetIcons(in_path, out_dir, icon_cat_mask);
  }
  if (fix_outfits) {
    std::printf("(mode: fix saved outfits)\n\n");
    return RunFixOutfits(in_path, out_dir);
  }
  if (extract_all) {
    std::printf("(mode: generic STFS extraction)\n\n");
    return RunStfsExtractAll(in_path, out_dir);
  }
  if (is_pack) {
    std::printf("(mode: avatar asset pack)\n\n");
    return RunPackMode(in_path, out_dir, want_index);
  }
  {
    // Raw STRB/YTGR blob (e.g. a marketplace item's asset_v2.bin).
    FILE* f = std::fopen(in_path.string().c_str(), "rb");
    if (f) {
      uint8_t magic[4] = {0};
      size_t n = std::fread(magic, 1, sizeof(magic), f);
      std::fclose(f);
      if (n == 4 && (!std::memcmp(magic, "YTGR", 4) || !std::memcmp(magic, "STRB", 4))) {
        std::printf("(mode: raw STRB/YTGR blob)\n\n");
        return RunRawStrbMode(in_path, out_dir);
      }
    }
  }
  std::printf("(mode: marketplace STFS package)\n\n");
  return RunStfsMode(in_path, out_dir);
}
