/**
 * refit.cpp: `avatarextract --refit-female <item.bin> <out.bin> [--toc <pack>]`
 *
 * Gives a male-only wearable (one body-hiding shape targeting the male stock
 * body) a female hiding shape, so the same mesh can be worn on the female
 * body without the skin clipping through. The stock skeleton is shared by
 * both genders, so the male-cut garment sits on the female body with its
 * joints aligned; what it lacks is the kShapeOverrides block that collapses
 * the covered body triangles and tucks the rim vertices.
 *
 * The two stock bodies do not share topology (male 1137 verts / 2264 tris,
 * female 1232 / 2454), so the male shape is mapped geometrically:
 *   - hidden triangles: a female triangle is hidden when its centroid's
 *     nearest male triangle centroid is hidden;
 *   - tucks: a female vertex whose nearest male vertex is tucked gets the
 *     same tuck delta (skipped when implausibly large, like the runtime).
 * The new block is appended to the STRB container; the male shape stays.
 */

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <string>
#include <vector>

#include "asset_pack.h"
#include "bit_stream.h"
#include "blend_shape.h"
#include "blend_shape_apply.h"
#include "common.h"
#include "model.h"
#include "strb.h"

namespace fs = std::filesystem;
using namespace rex::avatars;

namespace refit {

// ---------------------------------------------------------------------------
// helpers
// ---------------------------------------------------------------------------

static std::string PathStr(const fs::path& p) {
  auto u8 = p.u8string();
  return std::string(u8.begin(), u8.end());
}

static bool ReadFileBytes(const fs::path& path, std::vector<uint8_t>& out) {
  FILE* f = _wfopen(path.wstring().c_str(), L"rb");
  if (!f) return false;
  std::fseek(f, 0, SEEK_END);
  long sz = std::ftell(f);
  std::fseek(f, 0, SEEK_SET);
  if (sz < 0) {
    std::fclose(f);
    return false;
  }
  out.resize((size_t)sz);
  size_t got = sz ? std::fread(out.data(), 1, out.size(), f) : 0;
  std::fclose(f);
  return got == out.size();
}

static bool WriteFileBytes(const fs::path& path, const std::vector<uint8_t>& data) {
  FILE* f = _wfopen(path.wstring().c_str(), L"wb");
  if (!f) return false;
  size_t n = std::fwrite(data.data(), 1, data.size(), f);
  std::fclose(f);
  return n == data.size();
}

// LSB-first bit writer (the mirror of BitStream::Read).
class BitWriter {
 public:
  void Write(uint64_t value, size_t bits) {
    for (size_t i = 0; i < bits; ++i) {
      if (bit_ == 0) bytes_.push_back(0);
      if ((value >> i) & 1u) bytes_.back() |= (uint8_t)(1u << bit_);
      bit_ = (bit_ + 1) & 7;
    }
  }
  void U32(uint32_t v) { Write(v, 32); }
  void U16(uint16_t v) { Write(v, 16); }
  void U8(uint8_t v) { Write(v, 8); }
  void F32(float f) {
    uint32_t u;
    std::memcpy(&u, &f, 4);
    Write(u, 32);
  }
  void Align() { bit_ = 0; }
  const std::vector<uint8_t>& bytes() const { return bytes_; }

 private:
  std::vector<uint8_t> bytes_;
  int bit_ = 0;
};

static void WriteAssetId(BitWriter& w, const AssetId& id) {
  w.U32(id.a.get());
  w.U16(id.b.get());
  w.U16(id.c.get());
  for (int i = 0; i < 8; ++i) w.U8(id.d[i]);
}

static uint32_t BitsFor(uint64_t range) {
  uint32_t n = 1;
  while (n < 63 && (range >> n) != 0) ++n;
  return n;
}

// ValueSerializer<T> encoder: base = min, bit_count = bits for (max - min).
struct ValueEnc {
  uint32_t base = 0;
  uint32_t bits = 1;
  template <typename It>
  void Fit(It begin, It end) {
    uint32_t lo = 0xFFFFFFFFu, hi = 0;
    bool any = false;
    for (It it = begin; it != end; ++it) {
      lo = std::min(lo, (uint32_t)*it);
      hi = std::max(hi, (uint32_t)*it);
      any = true;
    }
    if (!any) lo = hi = 0;
    base = lo;
    bits = BitsFor((uint64_t)hi - lo);
  }
  void Header(BitWriter& w) const {
    w.U32(base);
    w.U32(bits);
  }
  void Put(BitWriter& w, uint32_t v) const { w.Write((uint64_t)(v - base), bits); }
};

// VectorSerializer encoder (the parity-lattice quantizer of serializers.h).
struct VectorEnc {
  float q = 0.0002f;  // quant_radius -> ~0.4 mm steps
  float bx = 0, by = 0, bz = 0;
  float dx = 0, dy = 0, dz = 0;
  uint32_t nx = 16, ny = 16, nz = 16;

  void Fit(const std::vector<Vector3<float>>& pts) {
    dx = q * 2.f;
    dy = q * (2.f / 3.f) * std::sqrt(6.f);
    dz = q * std::sqrt(3.f);
    float lox = 1e9f, loy = 1e9f, loz = 1e9f, hix = -1e9f, hiy = -1e9f, hiz = -1e9f;
    for (const auto& p : pts) {
      lox = std::min(lox, p.x); hix = std::max(hix, p.x);
      loy = std::min(loy, p.y); hiy = std::max(hiy, p.y);
      loz = std::min(loz, p.z); hiz = std::max(hiz, p.z);
    }
    if (pts.empty()) lox = loy = loz = hix = hiy = hiz = 0.f;
    // a margin of a few steps below the minimum keeps every index >= 0
    bx = lox - 4 * dx;
    by = loy - 4 * dy;
    bz = loz - 4 * dz;
    nx = std::min(30u, BitsFor((uint64_t)((hix - bx) / dx) + 8));
    ny = std::min(30u, BitsFor((uint64_t)((hiy - by) / dy) + 8));
    nz = std::min(30u, BitsFor((uint64_t)((hiz - bz) / dz) + 8));
  }
  void Header(BitWriter& w) const {
    w.F32(q);
    w.F32(bx);
    w.F32(by);
    w.F32(bz);
    w.Write(nx, 6);
    w.Write(ny, 6);
    w.Write(nz, 6);
  }
  // Mirrors VectorSerializer::Read: y first, z depends on y parity, x on both.
  void Put(BitWriter& w, const Vector3<float>& p) const {
    int64_t iy = (int64_t)std::llround((p.y - by) / dy);
    iy = std::max<int64_t>(0, std::min<int64_t>(iy, (1ll << ny) - 1));
    float zb = bz + ((iy & 1) ? (1.f / 3.f) * dz : 0.f);
    int64_t iz = (int64_t)std::llround((p.z - zb) / dz);
    iz = std::max<int64_t>(0, std::min<int64_t>(iz, (1ll << nz) - 1));
    float xb = bx + (((iy & 1) != (iz & 1)) ? 0.5f * dx : 0.f);
    int64_t ix = (int64_t)std::llround((p.x - xb) / dx);
    ix = std::max<int64_t>(0, std::min<int64_t>(ix, (1ll << nx) - 1));
    w.Write((uint64_t)ix, nx);
    w.Write((uint64_t)iy, ny);
    w.Write((uint64_t)iz, nz);
  }
};

struct FlatBody {
  std::vector<Vector3<float>> pos;       // all vertices, batch order
  std::vector<uint32_t> normal, weight, bindex, color;
  std::vector<size_t> vert_byte_offset;  // byte offset of each vertex
  std::vector<Vector3<float>> tri_centroid;  // all triangles, batch order
  size_t index_buffer_size = 0;
  size_t vertex_buffer_size = 0;  // (sum + 127) & ~127, ApplyBlendShape's gate
};

static FlatBody Flatten(const Model& m) {
  FlatBody f;
  size_t byte_off = 0;
  size_t ibytes = 0;
  for (const auto& b : m.triangle_batches) {
    const size_t base = f.pos.size();
    for (const auto& v : b.vertices) {
      f.pos.push_back(v.position);
      f.normal.push_back(v.normal);
      f.weight.push_back(v.blend_weight);
      f.bindex.push_back(v.blend_indices);
      f.color.push_back(v.color);
      f.vert_byte_offset.push_back(byte_off);
      byte_off += b.vertex_size;
    }
    for (size_t t = 0; t + 2 < b.indices.size(); t += 3) {
      const auto& a = b.vertices[b.indices[t]].position;
      const auto& c = b.vertices[b.indices[t + 1]].position;
      const auto& d = b.vertices[b.indices[t + 2]].position;
      f.tri_centroid.push_back({(a.x + c.x + d.x) / 3.f, (a.y + c.y + d.y) / 3.f,
                                (a.z + c.z + d.z) / 3.f});
    }
    ibytes += b.indices.size() * sizeof(uint16_t);
    (void)base;
  }
  f.index_buffer_size = ibytes;
  f.vertex_buffer_size = (byte_off + 127) & ~(size_t)127;
  return f;
}

static float Dist2(const Vector3<float>& a, const Vector3<float>& b) {
  const float dx = a.x - b.x, dy = a.y - b.y, dz = a.z - b.z;
  return dx * dx + dy * dy + dz * dz;
}

static size_t Nearest(const std::vector<Vector3<float>>& set, const Vector3<float>& p) {
  size_t best = 0;
  float bd = 1e30f;
  for (size_t i = 0; i < set.size(); ++i) {
    float d = Dist2(set[i], p);
    if (d < bd) {
      bd = d;
      best = i;
    }
  }
  return best;
}

// Serialize a shape (index patch + vertex patch) as a raw kShapeOverrides
// payload, in the exact bit layout BlendShape::Read expects.
static std::vector<uint8_t> EncodeShape(const std::vector<int32_t>& hidden_tris,
                                        size_t index_buffer_size, const AssetId& target,
                                        const std::vector<BlendShapeVertex>& tucks,
                                        size_t vertex_buffer_size) {
  BitWriter w;
  // ---- index patch ----
  w.U32((uint32_t)hidden_tris.size());
  w.U32((uint32_t)index_buffer_size);
  WriteAssetId(w, target);
  ValueEnc ienc;
  ienc.Fit(hidden_tris.begin(), hidden_tris.end());
  ienc.Header(w);
  for (int32_t t : hidden_tris) ienc.Put(w, (uint32_t)t);
  w.Align();
  // ---- vertex patch ----
  w.U32((uint32_t)tucks.size());
  w.U32((uint32_t)vertex_buffer_size);
  WriteAssetId(w, target);
  std::vector<uint32_t> offs, nrm, wgt, idx, col;
  std::vector<Vector3<float>> pts;
  for (const auto& t : tucks) {
    offs.push_back((uint32_t)t.original_offset);
    pts.push_back(t.position);
    nrm.push_back(t.normal);
    wgt.push_back(t.blend_weight);
    idx.push_back(t.blend_indices);
    col.push_back(t.color);
  }
  ValueEnc oenc, nenc, wenc, xenc, cenc;
  oenc.Fit(offs.begin(), offs.end());
  nenc.Fit(nrm.begin(), nrm.end());
  wenc.Fit(wgt.begin(), wgt.end());
  xenc.Fit(idx.begin(), idx.end());
  cenc.Fit(col.begin(), col.end());
  VectorEnc venc;
  venc.Fit(pts);
  oenc.Header(w);
  venc.Header(w);
  nenc.Header(w);
  wenc.Header(w);
  xenc.Header(w);
  cenc.Header(w);
  for (size_t i = 0; i < tucks.size(); ++i) {
    oenc.Put(w, offs[i]);
    venc.Put(w, pts[i]);
    nenc.Put(w, nrm[i]);
    wenc.Put(w, wgt[i]);
    xenc.Put(w, idx[i]);
    cenc.Put(w, col[i]);
  }
  w.Align();
  return w.bytes();
}

// Append one block to an STRB container (no YTGR wrapper expected).
static bool AppendBlock(std::vector<uint8_t>& strb, strb::STRBBlockId id,
                        const std::vector<uint8_t>& payload) {
  if (strb.size() < 26 || std::memcmp(strb.data(), "STRB", 4) != 0) return false;
  const bool has_align = strb[4] != 0;
  const bool le = strb[5] != 0;
  const size_t id_size = strb[22], size_size = strb[23];
  const size_t alignment = has_align ? strb[26] : 1;
  if (alignment == 0) return false;
  auto align_up = [&](size_t v) { return (v + alignment - 1) / alignment * alignment; };
  auto put = [&](uint64_t v, size_t n) {
    for (size_t i = 0; i < n; ++i) {
      size_t k = le ? i : (n - 1 - i);
      strb.push_back((uint8_t)(v >> (8 * k)));
    }
  };
  // Re-walk the existing blocks to find the true end (ignore trailing junk).
  size_t header = align_up(id_size + size_size + size_size);
  size_t off = align_up(has_align ? 30 : 26);
  auto rd = [&](size_t o, size_t n) -> uint64_t {
    uint64_t v = 0;
    for (size_t i = 0; i < n; ++i) {
      size_t k = le ? i : (n - 1 - i);
      v |= (uint64_t)strb[o + i] << (8 * k);
    }
    return v;
  };
  size_t end = off;
  while (off + header <= strb.size()) {
    uint64_t data_size = rd(off + id_size, size_size);
    uint64_t entry_size = rd(off + id_size + size_size, size_size);
    (void)entry_size;
    off += header;
    off += align_up((size_t)data_size);
    if (off > strb.size()) break;
    end = off;
  }
  strb.resize(end);
  put((uint64_t)id, id_size);
  put(payload.size(), size_size);
  put(1, size_size);
  while (strb.size() % alignment) strb.push_back(0);  // header padding (align_up)
  strb.insert(strb.end(), payload.begin(), payload.end());
  while (strb.size() % alignment) strb.push_back(0);
  return true;
}

// ---------------------------------------------------------------------------
// the mode
// ---------------------------------------------------------------------------

int Run(const fs::path& in_path, const fs::path& out_path, const fs::path& toc_path) {
  std::printf("(mode: refit female)\n\n");
  std::vector<uint8_t> toc;
  if (!ReadFileBytes(toc_path, toc) || toc.empty()) {
    std::printf("ERROR: cannot read asset pack %s\n", PathStr(toc_path).c_str());
    return 1;
  }
  AssetPack pack;
  if (!pack.Load(toc)) {
    std::printf("ERROR: AssetPack::Load failed\n");
    return 1;
  }
  auto load_body = [&](size_t index) -> std::shared_ptr<Model> {
    const uint8_t* buf = nullptr;
    size_t size = 0;
    if (!pack.GetAssetDataByIndex(index, buf, size)) return nullptr;
    return Model::Load(buf, size, ModelLoadOption::kNone);
  };
  auto male = load_body(0);
  auto female = load_body(1);
  if (!male || !female) {
    std::printf("ERROR: cannot decode the stock bodies (pack indices 0/1)\n");
    return 1;
  }
  FlatBody fm = Flatten(*male), ff = Flatten(*female);
  std::printf("male body:   %zu verts / %zu tris (vb %zu, ib %zu)\n", fm.pos.size(),
              fm.tri_centroid.size(), fm.vertex_buffer_size, fm.index_buffer_size);
  std::printf("female body: %zu verts / %zu tris (vb %zu, ib %zu)\n", ff.pos.size(),
              ff.tri_centroid.size(), ff.vertex_buffer_size, ff.index_buffer_size);

  std::vector<uint8_t> item;
  if (!ReadFileBytes(in_path, item) || item.size() < 30) {
    std::printf("ERROR: cannot read %s\n", PathStr(in_path).c_str());
    return 1;
  }
  if (std::memcmp(item.data(), "YTGR", 4) == 0) {
    item.erase(item.begin(), item.begin() + 0x140);  // drop the signature wrapper
    std::printf("note: YTGR wrapper dropped\n");
  }
  const size_t shape_count =
      strb::CountSTRBBlocks(item.data(), item.size(), strb::STRBBlockId::kShapeOverrides);
  std::shared_ptr<BlendShape> male_shape;
  bool has_female = false;
  for (size_t i = 0; i < shape_count; ++i) {
    const uint8_t* blk = nullptr;
    size_t blk_size = 0;
    if (!strb::GetSTRBBlockN(item.data(), item.size(), strb::STRBBlockId::kShapeOverrides, i, blk,
                             blk_size)) {
      break;
    }
    auto shape = BlendShape::Read(blk, blk_size, BlendShapeLoadOption::kNone);
    if (!shape) continue;
    const auto& t = shape->index_patch.original_asset_id;
    std::printf("shape %zu: target %s  hidden tris %zu  tucks %zu\n", i, t.to_string().c_str(),
                shape->index_patch.indices.size(), shape->vertex_patch.vertices.size());
    if (t.a.get() == 2u && t.c.get() == 1u) male_shape = shape;
    if (t.a.get() == 2u && t.c.get() == 2u) has_female = true;
  }
  if (has_female) {
    std::printf("item already carries a female body shape; nothing to do\n");
    return 0;
  }
  if (!male_shape) {
    std::printf("ERROR: no male body-hiding shape in this item\n");
    return 2;
  }
  if (male_shape->index_patch.total_buffer_size != fm.index_buffer_size ||
      male_shape->vertex_patch.total_buffer_size != fm.vertex_buffer_size) {
    std::printf("WARN: male shape buffer sizes (%u/%u) differ from the current male body "
                "(%zu/%zu); mapping by geometry anyway\n",
                male_shape->index_patch.total_buffer_size,
                male_shape->vertex_patch.total_buffer_size, fm.index_buffer_size,
                fm.vertex_buffer_size);
  }

  // ---- hidden triangles: nearest male centroid decides ----
  std::vector<uint8_t> male_hidden(fm.tri_centroid.size(), 0);
  for (int32_t t : male_shape->index_patch.indices) {
    if (t >= 0 && (size_t)t < male_hidden.size()) male_hidden[t] = 1;
  }
  std::vector<int32_t> female_hidden;
  for (size_t t = 0; t < ff.tri_centroid.size(); ++t) {
    const size_t m = Nearest(fm.tri_centroid, ff.tri_centroid[t]);
    if (male_hidden[m]) female_hidden.push_back((int32_t)t);
  }

  // ---- tucks: nearest male vertex's delta ----
  std::vector<Vector3<float>> male_tuck_delta(fm.pos.size(), {0.f, 0.f, 0.f});
  std::vector<uint8_t> male_tucked(fm.pos.size(), 0);
  {
    const size_t vsize = male->triangle_batches[0].vertex_size;
    for (const auto& pv : male_shape->vertex_patch.vertices) {
      const size_t vi = (size_t)pv.original_offset / vsize;
      if (vi >= fm.pos.size()) continue;
      Vector3<float> d = {pv.position.x - fm.pos[vi].x, pv.position.y - fm.pos[vi].y,
                          pv.position.z - fm.pos[vi].z};
      if (std::sqrt(Dist2(pv.position, fm.pos[vi])) > 0.04f) continue;  // implausible
      male_tuck_delta[vi] = d;
      male_tucked[vi] = 1;
    }
  }
  std::vector<BlendShapeVertex> tucks;
  for (size_t v = 0; v < ff.pos.size(); ++v) {
    const size_t m = Nearest(fm.pos, ff.pos[v]);
    if (!male_tucked[m]) continue;
    if (std::sqrt(Dist2(fm.pos[m], ff.pos[v])) > 0.06f) continue;  // no counterpart nearby
    BlendShapeVertex bv{};
    bv.original_offset = (int32_t)ff.vert_byte_offset[v];
    bv.position = {ff.pos[v].x + male_tuck_delta[m].x, ff.pos[v].y + male_tuck_delta[m].y,
                   ff.pos[v].z + male_tuck_delta[m].z};
    bv.normal = ff.normal[v];
    bv.blend_weight = ff.weight[v];
    bv.blend_indices = ff.bindex[v];
    bv.color = ff.color[v];
    tucks.push_back(bv);
  }
  std::printf("female shape: %zu hidden tris (male had %zu), %zu tucks (male had %zu)\n",
              female_hidden.size(), male_shape->index_patch.indices.size(), tucks.size(),
              male_shape->vertex_patch.vertices.size());

  // target id: the canonical female body {a=2, b=0, c=2, stock tail}
  AssetId target = male_shape->index_patch.original_asset_id;
  target.b = 0;
  target.c = 2;
  std::vector<uint8_t> payload =
      EncodeShape(female_hidden, ff.index_buffer_size, target, tucks, ff.vertex_buffer_size);

  // ---- verify the encoding round-trips and applies ----
  auto check = BlendShape::Read(payload.data(), payload.size(), BlendShapeLoadOption::kNone);
  if (!check || check->index_patch.indices.size() != female_hidden.size() ||
      check->vertex_patch.vertices.size() != tucks.size()) {
    std::printf("ERROR: encoded shape does not round-trip\n");
    return 3;
  }
  float max_err = 0.f;
  for (size_t i = 0; i < tucks.size(); ++i) {
    max_err = std::max(max_err, std::sqrt(Dist2(check->vertex_patch.vertices[i].position,
                                                tucks[i].position)));
    if (check->vertex_patch.vertices[i].original_offset != tucks[i].original_offset) {
      std::printf("ERROR: tuck %zu offset mismatch\n", i);
      return 3;
    }
  }
  AssetId worn_female = target;
  worn_female.b = 1;  // how manifests name the worn female body
  auto female_copy = std::make_shared<Model>(*female);
  const bool applies = check->matches(worn_female) &&
                       ApplyBlendShape(check, worn_female, female_copy);
  size_t degenerate = 0;
  for (const auto& b : female_copy->triangle_batches) {
    for (size_t t = 0; t + 2 < b.indices.size(); t += 3) {
      if (b.indices[t] == b.indices[t + 1] && b.indices[t + 1] == b.indices[t + 2]) ++degenerate;
    }
  }
  std::printf("round-trip OK: max tuck quantization error %.4f mm; applies to the female body: "
              "%s (%zu triangles collapsed)\n",
              max_err * 1000.f, applies ? "yes" : "no", degenerate);
  if (!applies) return 4;

  std::vector<uint8_t> out = item;
  if (!AppendBlock(out, strb::STRBBlockId::kShapeOverrides, payload)) {
    std::printf("ERROR: cannot append the block\n");
    return 5;
  }
  // the appended block must parse back from the container
  const size_t n_after =
      strb::CountSTRBBlocks(out.data(), out.size(), strb::STRBBlockId::kShapeOverrides);
  auto model_after = Model::Load(out.data(), out.size(), ModelLoadOption::kNone);
  if (n_after != shape_count + 1 || !model_after) {
    std::printf("ERROR: container check failed (shapes %zu -> %zu, model %s)\n", shape_count,
                n_after, model_after ? "ok" : "LOST");
    return 6;
  }
  if (!WriteFileBytes(out_path, out)) {
    std::printf("ERROR: cannot write %s\n", PathStr(out_path).c_str());
    return 7;
  }
  std::printf("wrote %s (%zu -> %zu bytes, %zu shape blocks)\n", PathStr(out_path).c_str(),
              item.size(), out.size(), n_after);
  return 0;
}

}  // namespace refit

int RunRefitFemale(const std::vector<std::string>& argv) {
  fs::path in, out, toc;
  std::vector<std::string> pos;
  for (size_t i = 0; i < argv.size(); ++i) {
    if (argv[i] == "--toc" && i + 1 < argv.size()) toc = fs::u8path(argv[++i]);
    else pos.push_back(argv[i]);
  }
  if (pos.size() < 2) {
    std::printf("usage: avatarextract --refit-female <item.bin> <out.bin> [--toc <AvatarAssetPack.toc>]\n");
    return 1;
  }
  in = fs::u8path(pos[0]);
  out = fs::u8path(pos[1]);
  if (toc.empty()) {
    std::printf("ERROR: --toc <AvatarAssetPack.toc> is required (the stock bodies come from the pack).\n");
    return 1;
  }
  return refit::Run(in, out, toc);
}
