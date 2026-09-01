/**
 * avatar_export.cpp: `avatarextract --avatar` exports the saved avatar
 * (a 1000-byte X_AVATAR_METADATA manifest) as a fully resolved, baked model.
 *
 * Mirrors the runtime loader (guest_load_asset.cpp LoadAssetsToGuest) step
 * for step: components from the pack or the closet, the "(Hat)" hair swap,
 * body-hiding shapes, face morphs, replacement-texture slotting and healing,
 * the eye-shadow auto-pair, the old-body rescue, the blend-shape apply loop.
 * Then does the draw-time work offline: the scaled skeleton skinned into the
 * vertices, every batch's shader evaluated into one diffuse texture (the
 * BODY_* recolor and decal; HEAD_OPAQUE's face composite as a UV-to-UV
 * bake), the face-feature frame stacks as tinted layers, and animations
 * decoded to per-joint local TRS tracks plus the face-channel and motion
 * tracks.
 *
 * Output: <out_dir>/avatar.json + <Material>_Diffuse.png (+ face/). The
 * Avatar Export app (tools/avatar_export) turns that into DAE, glTF, OBJ,
 * SMD and the Tower Unite character rig.
 *
 * Data frame (ModelLoadOption::kNone): right-handed, Y up, the avatar faces
 * +Z, its left is +X, metres.
 */

#include <algorithm>
#include <array>
#include <cctype>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <map>
#include <memory>
#include <set>
#include <string>
#include <vector>

#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/quaternion.hpp>

#include <rex/math.h>

#include "animation.h"
#include "asset_pack.h"
#include "bit_stream.h"
#include "blend_shape.h"
#include "blend_shape_apply.h"
#include "closet.h"
#include "compression.h"
#include "guest_asset.h"
#include "model.h"
#include "prop.h"
#include "skeleton.h"
#include "skeleton_data.h"
#include "skeleton_scaling.h"
#include "strb.h"
#include "texture.h"

#include "stb_image_write.h"  // implementation lives in main.cpp
#include "texdecode.h"

namespace fs = std::filesystem;
using namespace rex::avatars;

namespace avexp {

// ---------------------------------------------------------------------------
// Small helpers
// ---------------------------------------------------------------------------

struct Float4 {
  float x = 1.f, y = 1.f, z = 1.f, w = 1.f;
};

static Float4 ColorToFloat4(uint32_t argb) {
  Float4 c;
  c.x = ((argb >> 16) & 0xFF) / 255.f;
  c.y = ((argb >> 8) & 0xFF) / 255.f;
  c.z = ((argb >> 0) & 0xFF) / 255.f;
  c.w = ((argb >> 24) & 0xFF) / 255.f;
  return c;
}

static std::string PathStr(const fs::path& p) {
  auto u8 = p.u8string();
  return std::string(u8.begin(), u8.end());
}

static fs::path PathFromUtf8(const std::string& s) {
  // This toolchain builds with char8_t disabled, so u8path() is the portable
  // UTF-8 constructor here (and on a char8_t build it is merely deprecated).
  return fs::u8path(s);
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

static std::string U16ToUtf8(const std::u16string& s) {
  std::string out;
  for (size_t i = 0; i < s.size(); ++i) {
    uint32_t c = s[i];
    if (c >= 0xD800 && c <= 0xDBFF && i + 1 < s.size()) {
      uint32_t lo = s[i + 1];
      if (lo >= 0xDC00 && lo <= 0xDFFF) {
        c = 0x10000 + ((c - 0xD800) << 10) + (lo - 0xDC00);
        ++i;
      }
    }
    if (c < 0x80) {
      out.push_back((char)c);
    } else if (c < 0x800) {
      out.push_back((char)(0xC0 | (c >> 6)));
      out.push_back((char)(0x80 | (c & 0x3F)));
    } else if (c < 0x10000) {
      out.push_back((char)(0xE0 | (c >> 12)));
      out.push_back((char)(0x80 | ((c >> 6) & 0x3F)));
      out.push_back((char)(0x80 | (c & 0x3F)));
    } else {
      out.push_back((char)(0xF0 | (c >> 18)));
      out.push_back((char)(0x80 | ((c >> 12) & 0x3F)));
      out.push_back((char)(0x80 | ((c >> 6) & 0x3F)));
      out.push_back((char)(0x80 | (c & 0x3F)));
    }
  }
  return out;
}

static std::string JsonEscape(const std::string& s) {
  std::string out;
  for (unsigned char c : s) {
    switch (c) {
      case '"': out += "\\\""; break;
      case '\\': out += "\\\\"; break;
      case '\n': out += "\\n"; break;
      case '\r': out += "\\r"; break;
      case '\t': out += "\\t"; break;
      default:
        if (c < 0x20) {
          char buf[8];
          std::snprintf(buf, sizeof(buf), "\\u%04x", c);
          out += buf;
        } else {
          out.push_back((char)c);
        }
    }
  }
  return out;
}

struct CategoryName {
  uint32_t bit;
  const char* name;
};
static const CategoryName kCategoryNames[] = {
    {ComponentCategory::kHead, "Head"},         {ComponentCategory::kBody, "Body"},
    {ComponentCategory::kHair, "Hair"},         {ComponentCategory::kTop, "Top"},
    {ComponentCategory::kBottom, "Bottom"},     {ComponentCategory::kShoes, "Shoes"},
    {ComponentCategory::kHat, "Hat"},           {ComponentCategory::kGloves, "Gloves"},
    {ComponentCategory::kGlasses, "Glasses"},   {ComponentCategory::kWristwear, "Wristwear"},
    {ComponentCategory::kEarrings, "Earrings"}, {ComponentCategory::kRing, "Ring"},
    {ComponentCategory::kProp, "Prop"},         {ComponentCategory::kAnimation, "Animation"},
};

static std::string CategoryNames(uint32_t categories) {
  std::string out;
  for (const auto& c : kCategoryNames) {
    if (categories & c.bit) {
      if (!out.empty()) out.push_back('|');
      out += c.name;
    }
  }
  return out.empty() ? "(none)" : out;
}

// Short slot label used for material/mesh names.
static std::string SlotLabel(uint32_t categories) {
  const uint32_t garments = ComponentCategory::kTop | ComponentCategory::kBottom |
                            ComponentCategory::kShoes | ComponentCategory::kGloves;
  int garment_bits = 0;
  for (uint32_t b = 0; b < 13; ++b) {
    if ((categories & (1u << b)) && ((1u << b) & garments)) ++garment_bits;
  }
  if (categories & ComponentCategory::kBody) return "Body";
  if (categories & ComponentCategory::kHead) return "Head";
  if (garment_bits >= 3) return "Costume";
  static const uint32_t order[] = {
      ComponentCategory::kTop,     ComponentCategory::kBottom,    ComponentCategory::kShoes,
      ComponentCategory::kHair,    ComponentCategory::kHat,       ComponentCategory::kGloves,
      ComponentCategory::kGlasses, ComponentCategory::kWristwear, ComponentCategory::kEarrings,
      ComponentCategory::kRing,    ComponentCategory::kProp};
  for (uint32_t bit : order) {
    if (categories & bit) {
      for (const auto& c : kCategoryNames) {
        if (c.bit == bit) return c.name;
      }
    }
  }
  return "Item";
}

static std::string SanitizeName(const std::string& s) {
  std::string out;
  for (unsigned char c : s) {
    if (std::isalnum(c) || c == '_' || c == '-') {
      out.push_back((char)c);
    } else if (c == ' ') {
      out.push_back('_');
    }
  }
  if (out.empty()) out = "item";
  return out;
}

// ---------------------------------------------------------------------------
// Arguments
// ---------------------------------------------------------------------------

struct Args {
  fs::path manifest;
  fs::path out_dir;
  fs::path toc;         // AvatarAssetPack.toc
  fs::path legacy_toc;  // AvatarAssetPackLegacyV1.toc (optional, rescue)
  fs::path closet;      // closet dir (optional)
  int skeleton_version = 2;
  bool no_scale = false;
  bool want_prop = true;
  bool info_only = false;   // --avatar-info: summary JSON to stdout, no bake
  int bake_size = 1024;     // head composite resolution
  bool face_frames = false; // whole-head composites per feature frame
  int face_frame_size = 512;
  bool eye_whites = false;  // the avatar_eye_whites preference mod
  std::vector<fs::path> anims;
  fs::path anim_dir;
  bool pack_anims = false;
  std::vector<std::string> pack_anim_filters;  // --pack-anim <substring> (repeatable)
  bool list_anims = false;
  fs::path preview_dir;             // --preview-dir: render preview PNGs (T-pose + every clip)
  int preview_size = 384;           // --preview-size N
  std::string preview_frame = "mid";  // --preview-frame mid|peak|<0..1>|<N>
  // --gen-icons: when icon_out is set, Run renders the closet item icon_target
  // worn on the manifest's (blank mannequin) body into icon_out and skips
  // every export; the remaining fields frame the shot.
  fs::path icon_out;
  AssetId icon_target{};
  uint32_t icon_categories = 0;
  int icon_size = 128;
  float icon_yaw = 25.f;
  float icon_pitch = 10.f;
};

static void PrintUsage() {
  std::printf(
      "avatarextract --avatar <avatar_manifest.bin> <out_dir> [options]\n"
      "  --toc <AvatarAssetPack.toc>          (default: next to the manifest's\n"
      "                                        userdata: ..\\avatarpack\\AvatarAssetPack.toc)\n"
      "  --legacy-toc <AvatarAssetPackLegacyV1.toc>\n"
      "  --closet <dir>                       (default: <pack_dir>\\closet)\n"
      "  --skeleton-version 1|2               (default 2 = Kinect-era titles)\n"
      "  --no-scale                           ignore height/weight factors\n"
      "  --no-prop                            skip the carryable\n"
      "  --bake-size N                        head composite size (default 1024)\n"
      "  --face-frames [N]                    whole-head composite per feature frame\n"
      "  --eye-whites                         paint a sclera mask into eye art lacking one\n"
      "  --anim <file.AvatarAnimation>        export this animation (repeatable)\n"
      "  --anim-dir <dir>                     export every *.AvatarAnimation in dir\n"
      "  --pack-anims                         export every animation asset in the pack\n"
      "  --pack-anim <name>[@<frame>]         export pack animations whose name contains\n"
      "                                       <name> (case-insensitive, repeatable); @<frame>\n"
      "                                       overrides --preview-frame for that clip\n"
      "  --list-anims                         list pack animations and exit\n"
      "  --preview-dir <dir>                  render preview_T-Pose.png + preview_<clip>.png\n"
      "  --preview-size N                     preview image size (default 384)\n"
      "  --preview-frame mid|peak|<0..1>|<N>  which clip frame to pose (default mid)\n"
      "  --avatar-info                        print an avatar summary JSON, no export\n");
}

// ---------------------------------------------------------------------------
// Asset access (pack + closet), mirroring guest_load_asset.cpp LoadAsset
// ---------------------------------------------------------------------------

struct Assets {
  AssetPack pack;
  AssetPack legacy_pack;
  bool has_legacy = false;

  bool LoadBytes(const AssetId& id, const uint8_t*& buffer, size_t& size,
                 std::vector<uint8_t>& temp) const {
    if (!IsStockPackId(id)) {
      if (GetCloset().ReadItemBytes(id, temp) && !temp.empty()) {
        buffer = temp.data();
        size = temp.size();
        return true;
      }
    }
    return pack.GetAssetData(id, buffer, size);
  }

  std::shared_ptr<Model> LoadModel(const AssetId& id) const {
    const uint8_t* buf = nullptr;
    size_t size = 0;
    std::vector<uint8_t> temp;
    if (!LoadBytes(id, buf, size, temp)) return nullptr;
    return Model::Load(buf, size, ModelLoadOption::kNone);
  }

  std::shared_ptr<Texture> LoadTexture(const AssetId& id) const {
    const uint8_t* buf = nullptr;
    size_t size = 0;
    std::vector<uint8_t> temp;
    if (!LoadBytes(id, buf, size, temp)) return nullptr;
    return Texture::Load(buf, size);
  }

  std::shared_ptr<BlendShape> LoadBlendShape(const AssetId& id) const {
    const uint8_t* buf = nullptr;
    size_t size = 0;
    std::vector<uint8_t> temp;
    if (!LoadBytes(id, buf, size, temp)) return nullptr;
    return BlendShape::Load(buf, size, BlendShapeLoadOption::kNone);
  }

  std::string Name(const AssetId& id) const {
    if (!IsStockPackId(id)) {
      const ClosetItem* item = GetCloset().Find(id);
      return item ? item->name : std::string();
    }
    const size_t index = id.b.get();
    if (index < pack.asset_infos().size()) {
      return U16ToUtf8(pack.GetAssetNameByIndex(index));
    }
    return std::string();
  }

  std::shared_ptr<Model> LoadLegacyBody(uint16_t gender_c) const {
    if (!has_legacy || (gender_c != 1 && gender_c != 2)) return nullptr;
    const auto& infos = legacy_pack.asset_infos();
    for (size_t i = 0; i < infos.size(); ++i) {
      if (!(infos[i].categories & ComponentCategory::kBody) ||
          infos[i].bodies != (uint32_t)gender_c) {
        continue;
      }
      const uint8_t* buffer = nullptr;
      size_t size = 0;
      if (legacy_pack.GetAssetDataByIndex(i, buffer, size)) {
        return Model::Load(buffer, size, ModelLoadOption::kNone);
      }
      break;
    }
    return nullptr;
  }
};

static bool LoadPack(const fs::path& toc, AssetPack& pack) {
  std::vector<uint8_t> bytes;
  if (!ReadFileBytes(toc, bytes) || bytes.empty()) return false;
  return pack.Load(bytes);
}

// ---------------------------------------------------------------------------
// Resolution (the LoadAssetsToGuest mirror)
// ---------------------------------------------------------------------------

struct ResolvedComponent {
  X_AVATAR_COMPONENT_INFO info{};
  std::shared_ptr<Model> model;
  bool from_closet = false;
  std::string name;
  std::string slot;
  std::map<uint32_t, Float4> overrides;  // shader usage -> constant
  std::vector<std::string> notes;
};

struct Resolved {
  BodyType body_type = BodyType::kUnknown;
  std::shared_ptr<Skeleton> skeleton;
  std::vector<ResolvedComponent> components;
  std::shared_ptr<Texture> replacement_textures[6];
  AssetId replacement_ids[6];
  std::shared_ptr<Prop> prop;
  X_AVATAR_COMPONENT_INFO prop_info{};
  std::string prop_name;
  std::vector<std::string> log;
};

static BodyType GetBodyType(const X_AVATAR_METADATA& metadata) {
  const AssetId male = {2, 0, 1, {0xC1, 0xC8, 0xF1, 0x09, 0xA1, 0x9C, 0xB2, 0xE0}};
  const AssetId female = {2, 0, 2, {0xC1, 0xC8, 0xF1, 0x09, 0xA1, 0x9C, 0xB2, 0xE0}};
  if (metadata.body_component.asset_id == male) return BodyType::kMale;
  if (metadata.body_component.asset_id == female) return BodyType::kFemale;
  // Closet/marketplace bodies are unknown; guess from the gender nibble.
  if (metadata.body_component.asset_id.c.get() == 1) return BodyType::kMale;
  if (metadata.body_component.asset_id.c.get() == 2) return BodyType::kFemale;
  return BodyType::kUnknown;
}

// GetShaderOverrides + ApplyItemColorTable, as a usage -> constant map.
static void ComputeOverrides(const X_AVATAR_METADATA& metadata, ResolvedComponent& comp,
                             const Assets& assets) {
  const uint32_t categories = comp.info.categories.get();
  auto& ov = comp.overrides;
  if (categories & ComponentCategory::kBody) {
    ov[22] = ColorToFloat4(metadata.colors[0].get());
  } else if ((categories & ComponentCategory::kHair) &&
             !(categories & (ComponentCategory::kTop | ComponentCategory::kBottom |
                             ComponentCategory::kShoes | ComponentCategory::kGloves))) {
    ov[22] = ColorToFloat4(metadata.colors[1].get());
  } else {
    ov[22] = Float4{1.f, 1.f, 1.f, 1.f};
  }
  if (categories & ComponentCategory::kHead) {
    for (uint32_t i = 0, usage = 13; i < 9; ++i, ++usage) {
      ov[usage] = ColorToFloat4(metadata.colors[i].get());
    }
  }
  // Closet items: block 7 custom color table wins for CUSTOM_0..2.
  if (comp.from_closet) {
    std::vector<uint8_t> item_bytes;
    if (GetCloset().ReadItemBytes(comp.info.asset_id, item_bytes) && !item_bytes.empty()) {
      const uint8_t* table = nullptr;
      size_t table_size = 0;
      if (strb::GetSTRBBlock(item_bytes.data(), item_bytes.size(),
                             strb::STRBBlockId::kCustomColorTable, table, table_size) &&
          table_size >= 4 + 3 * 8) {
        auto read_u32 = [](const uint8_t* p, bool le) -> uint32_t {
          return le ? (uint32_t(p[0]) | (uint32_t(p[1]) << 8) | (uint32_t(p[2]) << 16) |
                       (uint32_t(p[3]) << 24))
                    : ((uint32_t(p[0]) << 24) | (uint32_t(p[1]) << 16) |
                       (uint32_t(p[2]) << 8) | uint32_t(p[3]));
        };
        const bool le = read_u32(table, true) <= 0xFFFFu;
        uint32_t colors[3];
        bool any = false;
        for (size_t i = 0; i < 3; ++i) {
          colors[i] = read_u32(&table[4 + i * 8], le);
          any |= colors[i] != 0;
        }
        if (any) {
          for (size_t i = 0; i < 3; ++i) ov[22 + (uint32_t)i] = ColorToFloat4(colors[i]);
          char buf[96];
          std::snprintf(buf, sizeof(buf), "custom color table %08X %08X %08X", colors[0],
                        colors[1], colors[2]);
          comp.notes.push_back(buf);
        }
      }
    }
  }
}

static bool Resolve(const X_AVATAR_METADATA& metadata, const Args& args, const Assets& assets,
                    Resolved& out) {
  auto log = [&](const std::string& s) {
    out.log.push_back(s);
    std::printf("  %s\n", s.c_str());
  };

  uint32_t category_mask = 0x1FFF;  // all 13 slots
  const bool want_prop = args.want_prop;
  category_mask &= ~ComponentCategory::kProp;

  out.body_type = GetBodyType(metadata);

  out.skeleton = LoadSkeleton((uint32_t)args.skeleton_version, SkeletonLoadOption::kNone);
  if (!out.skeleton) {
    std::printf("ERROR: cannot load skeleton version %d\n", args.skeleton_version);
    return false;
  }
  if (!args.no_scale) {
    if (args.skeleton_version == 1) {
      ApplyScalesToSkeletonV1(out.body_type, metadata.weight_factor.get(),
                              metadata.height_factor.get(), out.skeleton);
    } else {
      ApplyScalesToSkeletonV2(out.body_type, metadata.weight_factor.get(),
                              metadata.height_factor.get(), out.skeleton);
    }
  }

  // Manifest blend shapes (face morphs picked in the editor).
  std::vector<std::pair<AssetId, std::shared_ptr<BlendShape>>> blend_shapes;
  for (size_t i = 0; i < 3; ++i) {
    const auto& id = metadata.blend_shapes[i].asset_id;
    if (id.is_zero()) continue;
    auto shape = assets.LoadBlendShape(id);
    if (!shape) {
      log("WARN: failed to load blend shape " + id.to_string());
      continue;
    }
    blend_shapes.push_back({id, shape});
  }

  std::vector<X_AVATAR_COMPONENT_INFO> infos;
  if (metadata.body_component.matches(category_mask)) infos.push_back(metadata.body_component);
  if (metadata.head_component.matches(category_mask)) infos.push_back(metadata.head_component);
  for (const auto& c : metadata.components) {
    if (c.matches(category_mask)) infos.push_back(c);
  }

  // Hat worn -> hair swaps to its "(Hat)" companion pack entry.
  {
    bool hat_worn = false;
    for (const auto& info : infos) {
      if (info.matches(ComponentCategory::kHat)) hat_worn = true;
    }
    if (hat_worn) {
      const auto& pack_infos = assets.pack.asset_infos();
      for (auto& info : infos) {
        if (!info.matches(ComponentCategory::kHair) || !IsStockPackId(info.asset_id)) continue;
        const size_t hair_index = info.asset_id.b.get();
        if (hair_index >= pack_infos.size()) continue;
        const auto& id0 = pack_infos[hair_index].asset_ids[0];
        const size_t partner = id0.b.get();
        if (id0.is_zero() || partner == hair_index || partner >= pack_infos.size()) continue;
        log("hat worn: hair " + info.asset_id.to_string() + " -> companion pack index " +
            std::to_string(partner));
        info.asset_id.b = (uint16_t)partner;
      }
    }
  }

  // Body-hiding shapes.
  {
    const auto& pack_infos = assets.pack.asset_infos();
    for (const auto& info : infos) {
      if (!IsStockPackId(info.asset_id)) {
        std::vector<uint8_t> item_bytes;
        if (!GetCloset().ReadItemBytes(info.asset_id, item_bytes) || item_bytes.empty()) {
          continue;
        }
        const size_t count = strb::CountSTRBBlocks(item_bytes.data(), item_bytes.size(),
                                                   strb::STRBBlockId::kShapeOverrides);
        for (size_t occ = 0; occ < count; ++occ) {
          const uint8_t* block = nullptr;
          size_t block_size = 0;
          if (!strb::GetSTRBBlockN(item_bytes.data(), item_bytes.size(),
                                   strb::STRBBlockId::kShapeOverrides, occ, block,
                                   block_size)) {
            break;
          }
          auto shape = BlendShape::Read(block, block_size, BlendShapeLoadOption::kNone);
          if (!shape) {
            log("WARN: bad embedded hiding shape in closet item " + info.asset_id.to_string());
            continue;
          }
          if (shape->index_patch.original_asset_id.a.get() == 2) {
            shape->index_patch.original_asset_id.b = 0;
          }
          if (shape->vertex_patch.original_asset_id.a.get() == 2) {
            shape->vertex_patch.original_asset_id.b = 0;
          }
          log("closet item " + info.asset_id.to_string() + " -> embedded hiding shape " +
              std::to_string(occ + 1) + "/" + std::to_string(count) + " (target " +
              shape->index_patch.original_asset_id.to_string() + ")");
          blend_shapes.push_back({info.asset_id, shape});
        }
        continue;
      }
      const size_t index = info.asset_id.b.get();
      if (index >= pack_infos.size()) continue;
      const auto& id0 = pack_infos[index].asset_ids[0];
      if (id0.is_zero() || id0.a.get() != 0x01000000u || id0.b.get() >= pack_infos.size()) {
        continue;
      }
      auto shape = assets.LoadBlendShape(id0);
      if (!shape) {
        log("WARN: failed to load hiding shape " + id0.to_string());
        continue;
      }
      log("component " + info.asset_id.to_string() + " -> body-hiding shape " +
          id0.to_string());
      blend_shapes.push_back({id0, shape});
    }
  }

  // Components (models), routing shape-only assets into the blend-shape set.
  for (const auto& info : infos) {
    {
      const uint8_t* raw = nullptr;
      size_t raw_size = 0;
      std::vector<uint8_t> temp;
      if (assets.LoadBytes(info.asset_id, raw, raw_size, temp) &&
          strb::CountSTRBBlocks(raw, raw_size, strb::STRBBlockId::kShapeOverrides) > 0 &&
          strb::CountSTRBBlocks(raw, raw_size, strb::STRBBlockId::kModel) == 0) {
        auto shape = BlendShape::Load(raw, raw_size, BlendShapeLoadOption::kNone);
        if (shape) {
          log("component " + info.asset_id.to_string() + " is a shape asset (face morph)");
          blend_shapes.push_back({info.asset_id, shape});
        } else {
          log("WARN: failed to load component blend shape " + info.asset_id.to_string());
        }
        continue;
      }
    }
    ResolvedComponent comp;
    comp.info = info;
    comp.model = assets.LoadModel(info.asset_id);
    if (!comp.model) {
      log("WARN: failed to load " + info.asset_id.to_string() + " (" +
          CategoryNames(info.categories.get()) + "), trying fallback");
      for (const auto& cand : metadata.fallback_components) {
        if (cand.categories.get() == info.categories.get() && !cand.asset_id.is_zero()) {
          comp.model = assets.LoadModel(cand.asset_id);
          comp.info = cand;
          break;
        }
      }
      if (!comp.model) {
        log("ERROR: no model for component " + info.asset_id.to_string());
        continue;
      }
    }
    comp.from_closet = !IsStockPackId(comp.info.asset_id);
    comp.name = assets.Name(comp.info.asset_id);
    comp.slot = SlotLabel(comp.info.categories.get());
    out.components.push_back(std::move(comp));
  }

  // Replacement (face) textures: fixed slot meanings + broken-era healing.
  static const uint32_t kSlotAccepted[6][2] = {
      {0x8000, 0},         // 0 mouth
      {0x2000, 0},         // 1 eyes
      {0x4000, 0},         // 2 brows
      {0x40000, 0x10000},  // 3 face paint
      {0x40000, 0x10000},  // 4 eye shadow
      {0x20000, 0x10000},  // 5 face texture
  };
  auto slot_accepts = [&](size_t slot, uint32_t cat) {
    return kSlotAccepted[slot][0] == cat || kSlotAccepted[slot][1] == cat;
  };
  std::shared_ptr<Texture> loaded[6];
  size_t slot_source[6] = {SIZE_MAX, SIZE_MAX, SIZE_MAX, SIZE_MAX, SIZE_MAX, SIZE_MAX};
  for (size_t i = 0; i < 6; ++i) {
    const auto& ti = metadata.textures[i];
    if (ti.asset_id.is_zero()) continue;
    loaded[i] = assets.LoadTexture(ti.asset_id);
    if (!loaded[i]) {
      log("WARN: failed to load replacement texture " + ti.asset_id.to_string());
      continue;
    }
    if (slot_accepts(i, ti.asset_id.a.get())) {
      out.replacement_textures[i] = loaded[i];
      out.replacement_ids[i] = ti.asset_id;
      slot_source[i] = i;
      loaded[i] = nullptr;
    }
  }
  for (size_t i = 0; i < 6; ++i) {
    if (!loaded[i]) continue;
    const uint32_t cat = metadata.textures[i].asset_id.a.get();
    for (size_t s = 0; s < 6; ++s) {
      if (!out.replacement_textures[s] && slot_accepts(s, cat)) {
        log("replacement texture " + metadata.textures[i].asset_id.to_string() +
            " reslotted " + std::to_string(i) + " -> " + std::to_string(s));
        out.replacement_textures[s] = loaded[i];
        out.replacement_ids[s] = metadata.textures[i].asset_id;
        slot_source[s] = i;
        loaded[i] = nullptr;
        break;
      }
    }
  }
  // Eye-shadow overlay auto-pair (skipped for the opaque-black "none" tint).
  const bool eyeshadow_none = (metadata.colors[5].get() & 0xFFFFFFu) == 0;
  if (!eyeshadow_none && !out.replacement_textures[4] && out.replacement_textures[1] &&
      slot_source[1] != SIZE_MAX &&
      IsStockPackId(metadata.textures[slot_source[1]].asset_id)) {
    const auto& eye_id = metadata.textures[slot_source[1]].asset_id;
    const auto& pack_infos = assets.pack.asset_infos();
    const size_t eye_index = eye_id.b.get();
    if (eye_index < pack_infos.size()) {
      const auto& id0 = pack_infos[eye_index].asset_ids[0];
      if (!id0.is_zero() && id0.a.get() == 0x40000u && id0.b.get() < pack_infos.size()) {
        auto shadow = assets.LoadTexture(id0);
        if (shadow) {
          log("eyes " + eye_id.to_string() + " -> paired eye-shadow overlay " + id0.to_string());
          out.replacement_textures[4] = shadow;
          out.replacement_ids[4] = id0;
        }
      }
    }
  }

  // Old-body item rescue, then the blend-shape apply loop.
  for (auto& bs : blend_shapes) {
    if (bs.second->vertex_patch.original_asset_id.a.get() != 2u) continue;
    for (auto& body : out.components) {
      if (!bs.second->matches(body.info.asset_id)) continue;
      std::shared_ptr<Model> owner;
      for (auto& cand : out.components) {
        if (cand.info.asset_id == bs.first) owner = cand.model;
      }
      RescueOldBodyItem(bs.second, body.model, owner,
                        assets.LoadLegacyBody(bs.second->vertex_patch.original_asset_id.c.get()));
      break;
    }
  }
  for (auto& comp : out.components) {
    for (const auto& bs : blend_shapes) {
      if (!bs.second->matches(comp.info.asset_id)) continue;
      if (!ApplyBlendShape(bs.second, comp.info.asset_id, comp.model)) {
        log("WARN: failed to apply blend shape " + bs.first.to_string() + " to " +
            comp.info.asset_id.to_string());
      } else {
        log("applied blend shape " + bs.first.to_string() + " to " +
            comp.info.asset_id.to_string() + " (" + comp.slot + ")");
      }
    }
  }

  for (auto& comp : out.components) {
    ComputeOverrides(metadata, comp, assets);
  }

  // Carryable.
  if (want_prop) {
    for (const auto& c : metadata.components) {
      if (c.matches(ComponentCategory::kProp)) {
        out.prop_info = c;
        break;
      }
    }
    if (!out.prop_info.asset_id.is_zero()) {
      const uint8_t* raw = nullptr;
      size_t raw_size = 0;
      std::vector<uint8_t> temp;
      if (assets.LoadBytes(out.prop_info.asset_id, raw, raw_size, temp)) {
        PropLoadOptions opts{};
        opts.model = ModelLoadOption::kNone;
        opts.skeleton = SkeletonLoadOption::kNone;
        opts.animation = AnimationLoadOption::kElements;
        opts.blend_shape = BlendShapeLoadOption::kNone;
        out.prop = Prop::Load(raw, raw_size, opts);
        out.prop_name = assets.Name(out.prop_info.asset_id);
        if (out.prop) {
          log("carryable " + out.prop_info.asset_id.to_string() + " '" + out.prop_name + "'");
        } else {
          log("WARN: failed to load carryable " + out.prop_info.asset_id.to_string());
        }
      }
    }
  }
  return true;
}

// ---------------------------------------------------------------------------
// Vertex decode
// ---------------------------------------------------------------------------

// XMHENDN3: 11-bit signed x (bits 0-10), 11-bit signed y (11-21), 10-bit z.
static glm::vec3 DecodeNormal(uint32_t packed) {
  auto sext = [](uint32_t v, int bits) -> int32_t {
    const uint32_t sign = 1u << (bits - 1);
    return (int32_t)((v ^ sign) - sign);
  };
  int32_t x = sext(packed & 0x7FF, 11);
  int32_t y = sext((packed >> 11) & 0x7FF, 11);
  int32_t z = sext((packed >> 22) & 0x3FF, 10);
  glm::vec3 n((float)x / 1023.f, (float)y / 1023.f, (float)z / 511.f);
  n = glm::clamp(n, -1.f, 1.f);
  float len = glm::length(n);
  return len > 1e-6f ? n / len : glm::vec3(0.f, 1.f, 0.f);
}

struct SkinVertex {
  glm::vec3 position;       // rest-posed (scaled skeleton)
  glm::vec3 normal;
  glm::vec3 orig_position;  // as authored (bind pose), for re-posing
  glm::vec3 orig_normal;
  std::array<float, 4> weights;
  std::array<uint8_t, 4> joints;
  uint32_t color;  // ARGB
  std::array<glm::vec2, 6> uvs;
};

static void UnpackSkin(const Vertex& v, SkinVertex& out, uint32_t uv_count) {
  out.position = glm::vec3(v.position.x, v.position.y, v.position.z);
  out.normal = DecodeNormal(v.normal);
  out.orig_position = out.position;
  out.orig_normal = out.normal;
  for (int k = 0; k < 4; ++k) {
    out.weights[k] = ((v.blend_weight >> (8 * k)) & 0xFF) / 255.f;
    out.joints[k] = (uint8_t)((v.blend_indices >> (8 * k)) & 0xFF);
  }
  out.color = v.color;
  for (uint32_t j = 0; j < 6; ++j) {
    if (j < v.uvs.size() && j < uv_count) {
      out.uvs[j] = glm::vec2(rex::xenos_half_to_float(v.uvs[j].x),
                             rex::xenos_half_to_float(v.uvs[j].y));
    } else {
      out.uvs[j] = j > 0 ? out.uvs[0] : glm::vec2(0.f);
    }
  }
}

// ---------------------------------------------------------------------------
// Skeleton math
// ---------------------------------------------------------------------------

struct JointXform {
  glm::mat4 bind_world{1.f};  // stored bind pose (world)
  glm::mat4 rest_world{1.f};  // scaled rest pose (world) = new bind pose
  glm::mat4 skin{1.f};        // rest_world * inverse(bind_world)
  glm::vec3 rest_local_t{0.f};
  glm::quat rest_local_r{1.f, 0.f, 0.f, 0.f};
  glm::vec3 scale{1.f};
  glm::vec3 parent_world_scale{1.f};
};

static const char* kJointNames[71] = {
    "BASE", "BACKA", "LF_H", "RT_H", "SC_BASE", "BACKB", "LF_K", "LF_SC_H", "RT_K", "RT_SC_H",
    "SC_BACKA", "LF_A", "LF_C", "LF_SC_K", "NECK", "RT_A", "RT_C", "RT_SC_K", "SC_BACKB", "HEAD",
    "LF_S", "LF_T", "RT_S", "RT_T", "SC_NECK", "LF_E", "LF_SC_S", "LF_SC_TWIST_S", "RT_E",
    "RT_SC_S", "RT_SC_TWIST_S", "LF_E_TWIST", "LF_SC_E", "LF_W", "RT_E_TWIST", "RT_SC_E", "RT_W",
    "LF_FINGA", "LF_FINGB", "LF_FINGC", "LF_FINGD", "LF_PROP", "LF_SPECIAL", "LF_THUMB",
    "RT_FINGA", "RT_FINGB", "RT_FINGC", "RT_FINGD", "RT_PROP", "RT_SPECIAL", "RT_THUMB",
    "LF_FINGA1", "LF_FINGB1", "LF_FINGC1", "LF_FINGD1", "LF_THUMB1", "RT_FINGA1", "RT_FINGB1",
    "RT_FINGC1", "RT_FINGD1", "RT_THUMB1", "LF_FINGA2", "LF_FINGB2", "LF_FINGC2", "LF_FINGD2",
    "LF_THUMB2", "RT_FINGA2", "RT_FINGB2", "RT_FINGC2", "RT_FINGD2", "RT_THUMB2"};

static std::string JointName(size_t i, size_t count) {
  if (count == 71 && i < 71) return kJointNames[i];
  return "joint_" + std::to_string(i);
}

static std::vector<JointXform> BuildJointXforms(const Skeleton& skel, bool apply_scale) {
  std::vector<JointXform> out(skel.joints.size());
  for (size_t i = 0; i < skel.joints.size(); ++i) {
    const Joint& j = skel.joints[i];
    glm::quat bq(j.bindpose.rotation.w, j.bindpose.rotation.x, j.bindpose.rotation.y,
                 j.bindpose.rotation.z);
    glm::mat4 bw = glm::translate(glm::mat4(1.f), glm::vec3(j.bindpose.position.x,
                                                            j.bindpose.position.y,
                                                            j.bindpose.position.z)) *
                   glm::mat4_cast(glm::normalize(bq));
    out[i].bind_world = bw;
    out[i].scale = apply_scale ? glm::vec3(j.pose.scale.x, j.pose.scale.y, j.pose.scale.z)
                               : glm::vec3(1.f);
  }
  // Joints are breadth-first sorted: parents precede children.
  for (size_t i = 0; i < skel.joints.size(); ++i) {
    const Joint& j = skel.joints[i];
    const bool has_parent = j.parent_index != 255 && j.parent_index < i;
    glm::mat4 local = has_parent
                          ? glm::inverse(out[j.parent_index].bind_world) * out[i].bind_world
                          : out[i].bind_world;
    out[i].rest_local_t = glm::vec3(local[3]);
    out[i].rest_local_r = glm::normalize(glm::quat_cast(glm::mat3(local)));
    glm::mat4 posed_local = local * glm::scale(glm::mat4(1.f), out[i].scale);
    if (has_parent) {
      out[i].rest_world = out[j.parent_index].rest_world * posed_local;
      const glm::mat4& pw = out[j.parent_index].rest_world;
      out[i].parent_world_scale =
          glm::vec3(glm::length(glm::vec3(pw[0])), glm::length(glm::vec3(pw[1])),
                    glm::length(glm::vec3(pw[2])));
    } else {
      out[i].rest_world = posed_local;
    }
    out[i].skin = out[i].rest_world * glm::inverse(out[i].bind_world);
  }
  return out;
}

// ---------------------------------------------------------------------------
// Baked materials
// ---------------------------------------------------------------------------

struct Sampler {
  const texdec::Image* img = nullptr;
  int uv_layer = 0;
  uint32_t flags = 0;               // XAVATAR_TEXTURE_FLAGS (bit0 WRAP_U, bit1 WRAP_V)
  bool transparent_border = false;  // decal layers: outside [0,1] contributes nothing

  bool wrap_u() const { return (flags & 1u) != 0; }
  bool wrap_v() const { return (flags & 2u) != 0; }

  glm::vec4 Sample(glm::vec2 uv) const {
    if (!img || !img->ok || img->width == 0 || img->height == 0) return glm::vec4(0.f);
    if (transparent_border &&
        (uv.x < 0.f || uv.x > 1.f || uv.y < 0.f || uv.y > 1.f)) {
      return glm::vec4(0.f);
    }
    if (!wrap_u()) uv.x = glm::clamp(uv.x, 0.f, 1.f);
    if (!wrap_v()) uv.y = glm::clamp(uv.y, 0.f, 1.f);
    float fx = uv.x * (float)img->width - 0.5f;
    float fy = uv.y * (float)img->height - 0.5f;
    int x0 = (int)std::floor(fx), y0 = (int)std::floor(fy);
    float tx = fx - (float)x0, ty = fy - (float)y0;
    auto wrap = [](int v, int n) {
      v %= n;
      return v < 0 ? v + n : v;
    };
    auto clampi = [](int v, int n) { return v < 0 ? 0 : (v >= n ? n - 1 : v); };
    int xa = wrap_u() ? wrap(x0, (int)img->width) : clampi(x0, (int)img->width);
    int xb = wrap_u() ? wrap(x0 + 1, (int)img->width) : clampi(x0 + 1, (int)img->width);
    int ya = wrap_v() ? wrap(y0, (int)img->height) : clampi(y0, (int)img->height);
    int yb = wrap_v() ? wrap(y0 + 1, (int)img->height) : clampi(y0 + 1, (int)img->height);
    auto px = [&](int x, int y) {
      const uint8_t* p = img->at((uint32_t)x, (uint32_t)y);
      return glm::vec4(p[0], p[1], p[2], p[3]) / 255.f;
    };
    glm::vec4 c00 = px(xa, ya), c10 = px(xb, ya), c01 = px(xa, yb), c11 = px(xb, yb);
    return glm::mix(glm::mix(c00, c10, tx), glm::mix(c01, c11, tx), ty);
  }
};

enum class ShaderKind { kBody, kHead };

struct MaterialRecipe {
  ShaderKind kind = ShaderKind::kBody;
  uint32_t shader_id = 0;
  int out_uv_layer = 0;
  // body
  Sampler color, intensity, decal;
  Float4 custom[3];
  // head (by usage)
  Sampler base, eyeshadow, mouth, eye, facial_hair, brow;
  Float4 tint[9];  // colors[0..8]
  bool eye_white_in_b = true;

  glm::vec4 Evaluate(const std::array<glm::vec2, 6>& uv) const {
    auto f4 = [](const Float4& c) { return glm::vec3(c.x, c.y, c.z); };
    if (kind == ShaderKind::kBody) {
      glm::vec4 c = color.img ? color.Sample(uv[color.uv_layer]) : glm::vec4(1.f);
      glm::vec3 rgb(c);
      if (intensity.img) {
        glm::vec4 i = intensity.Sample(uv[intensity.uv_layer]);
        glm::vec3 recolor = i.r * f4(custom[0]) + i.g * f4(custom[1]) + i.b * f4(custom[2]);
        rgb = glm::mix(rgb, recolor, i.a);
      }
      if (decal.img) {
        glm::vec4 d = decal.Sample(uv[decal.uv_layer]);
        rgb = glm::mix(rgb, glm::vec3(d), d.a);
      }
      return glm::vec4(glm::clamp(rgb, 0.f, 1.f), c.a);
    }
    const glm::vec3 skin = f4(tint[0]);
    glm::vec3 rgb = skin;
    if (base.img) {
      glm::vec4 b = base.Sample(uv[base.uv_layer]);
      rgb = glm::mix(skin, b.r * f4(tint[7]) + b.g * f4(tint[8]) + b.b * skin, b.a);
    }
    if (eyeshadow.img) {
      glm::vec4 s = eyeshadow.Sample(uv[eyeshadow.uv_layer]);
      rgb = glm::mix(rgb, s.r * f4(tint[5]), s.a);
    }
    if (mouth.img) {
      glm::vec4 m = mouth.Sample(uv[mouth.uv_layer]);
      rgb = glm::mix(rgb, m.r * f4(tint[2]) + glm::vec3(m.g) + m.b * skin, m.a);
    }
    if (eye.img) {
      glm::vec4 e = eye.Sample(uv[eye.uv_layer]);
      float white = eye_white_in_b ? e.b : e.g;
      float blend = eye_white_in_b ? e.g : e.b;
      rgb = glm::mix(rgb, e.r * f4(tint[3]) + glm::vec3(white) + blend * skin, e.a);
    }
    if (facial_hair.img) {
      glm::vec4 f = facial_hair.Sample(uv[facial_hair.uv_layer]);
      rgb = glm::mix(rgb, f.r * f4(tint[6]) + glm::vec3(f.g) + f.b * skin, f.a);
    }
    if (brow.img) {
      glm::vec4 b = brow.Sample(uv[brow.uv_layer]);
      rgb = glm::mix(rgb, b.r * f4(tint[4]) + glm::vec3(b.g) + b.b * skin, b.a);
    }
    return glm::vec4(glm::clamp(rgb, 0.f, 1.f), 1.f);
  }

  // All bound layers read the same UV set as the output -> texel-wise path.
  bool SingleUvSet() const {
    const Sampler* all[] = {&color, &intensity, &decal, &base, &eyeshadow,
                            &mouth, &eye,       &facial_hair, &brow};
    for (const Sampler* s : all) {
      if (s->img && s->uv_layer != out_uv_layer) return false;
    }
    return true;
  }

  glm::ivec2 NativeSize() const {
    int w = 0, h = 0;
    const Sampler* all[] = {&color, &intensity, &decal, &base, &eyeshadow,
                            &mouth, &eye,       &facial_hair, &brow};
    for (const Sampler* s : all) {
      if (s->img && s->img->ok) {
        w = std::max(w, (int)s->img->width);
        h = std::max(h, (int)s->img->height);
      }
    }
    return glm::ivec2(std::max(w, 4), std::max(h, 4));
  }
};

// Rasterize the batch's triangles in the output UV space and evaluate the
// recipe per texel; uncovered texels are dilated from covered neighbours and
// finally filled with the identity-UV evaluation.
static texdec::Image BakeMaterial(const MaterialRecipe& recipe,
                                  const std::vector<SkinVertex>& verts,
                                  const std::vector<uint16_t>& indices, int size_hint) {
  texdec::Image out;
  glm::ivec2 native = recipe.NativeSize();
  int W, H;
  if (recipe.SingleUvSet()) {
    W = native.x;
    H = native.y;
  } else {
    // The output is the [0,1]^2 UV space of the output layer: square, unless
    // the output layer's own texture (color/base) is not.
    const Sampler& primary = recipe.kind == ShaderKind::kHead ? recipe.base : recipe.color;
    int target = size_hint;
    if (recipe.kind == ShaderKind::kBody) {
      target = std::min(size_hint, std::max(256, 4 * std::max(native.x, native.y)));
    }
    W = H = target;
    if (primary.img && primary.img->ok && primary.img->width != primary.img->height) {
      if (primary.img->width > primary.img->height) {
        H = std::max(4, target * (int)primary.img->height / (int)primary.img->width);
      } else {
        W = std::max(4, target * (int)primary.img->width / (int)primary.img->height);
      }
    }
  }
  out.width = (uint32_t)W;
  out.height = (uint32_t)H;
  out.rgba.assign((size_t)W * H * 4, 0);
  std::vector<uint8_t> covered((size_t)W * H, 0);

  auto store = [&](int x, int y, const glm::vec4& c) {
    uint8_t* p = out.at((uint32_t)x, (uint32_t)y);
    p[0] = (uint8_t)std::lround(glm::clamp(c.r, 0.f, 1.f) * 255.f);
    p[1] = (uint8_t)std::lround(glm::clamp(c.g, 0.f, 1.f) * 255.f);
    p[2] = (uint8_t)std::lround(glm::clamp(c.b, 0.f, 1.f) * 255.f);
    p[3] = (uint8_t)std::lround(glm::clamp(c.a, 0.f, 1.f) * 255.f);
  };

  if (recipe.SingleUvSet()) {
    std::array<glm::vec2, 6> uv;
    for (int y = 0; y < H; ++y) {
      for (int x = 0; x < W; ++x) {
        glm::vec2 t(((float)x + 0.5f) / W, ((float)y + 0.5f) / H);
        uv.fill(t);
        store(x, y, recipe.Evaluate(uv));
      }
    }
    out.ok = true;
    return out;
  }

  const int L = recipe.out_uv_layer;
  auto wrapi = [](int v, int n) {
    v %= n;
    return v < 0 ? v + n : v;
  };
  for (size_t t = 0; t + 2 < indices.size(); t += 3) {
    const SkinVertex* v[3] = {&verts[indices[t]], &verts[indices[t + 1]], &verts[indices[t + 2]]};
    glm::vec2 p[3];
    for (int k = 0; k < 3; ++k) p[k] = glm::vec2(v[k]->uvs[L].x * W, v[k]->uvs[L].y * H);
    float area = (p[1].x - p[0].x) * (p[2].y - p[0].y) - (p[2].x - p[0].x) * (p[1].y - p[0].y);
    if (std::fabs(area) < 1e-8f) continue;
    float minx = std::min({p[0].x, p[1].x, p[2].x}), maxx = std::max({p[0].x, p[1].x, p[2].x});
    float miny = std::min({p[0].y, p[1].y, p[2].y}), maxy = std::max({p[0].y, p[1].y, p[2].y});
    if (maxx - minx > 4.f * W || maxy - miny > 4.f * H) continue;  // degenerate wrap span
    int x0 = (int)std::floor(minx) - 1, x1 = (int)std::ceil(maxx) + 1;
    int y0 = (int)std::floor(miny) - 1, y1 = (int)std::ceil(maxy) + 1;
    const float inv_area = 1.f / area;
    for (int y = y0; y <= y1; ++y) {
      for (int x = x0; x <= x1; ++x) {
        glm::vec2 c((float)x + 0.5f, (float)y + 0.5f);
        float w0 = ((p[1].x - c.x) * (p[2].y - c.y) - (p[2].x - c.x) * (p[1].y - c.y)) * inv_area;
        float w1 = ((p[2].x - c.x) * (p[0].y - c.y) - (p[0].x - c.x) * (p[2].y - c.y)) * inv_area;
        float w2 = 1.f - w0 - w1;
        // distance-based tolerance (~0.75 texel) so seams get covered
        const float eps = 0.75f / std::sqrt(std::fabs(area)) * 2.0f;
        if (w0 < -eps || w1 < -eps || w2 < -eps) continue;
        int ox = wrapi(x, W), oy = wrapi(y, H);
        const bool inside = w0 >= 0.f && w1 >= 0.f && w2 >= 0.f;
        if (!inside && covered[(size_t)oy * W + ox]) continue;  // don't overwrite true hits
        glm::vec3 b(glm::clamp(w0, 0.f, 1.f), glm::clamp(w1, 0.f, 1.f), glm::clamp(w2, 0.f, 1.f));
        b /= (b.x + b.y + b.z);
        std::array<glm::vec2, 6> uv;
        for (int l = 0; l < 6; ++l) {
          uv[l] = b.x * v[0]->uvs[l] + b.y * v[1]->uvs[l] + b.z * v[2]->uvs[l];
        }
        store(ox, oy, recipe.Evaluate(uv));
        covered[(size_t)oy * W + ox] = inside ? 2 : 1;
      }
    }
  }
  // Dilate covered texels outward (a few passes) to hide filtering seams.
  for (int pass = 0; pass < 6; ++pass) {
    std::vector<uint8_t> next = covered;
    for (int y = 0; y < H; ++y) {
      for (int x = 0; x < W; ++x) {
        if (covered[(size_t)y * W + x]) continue;
        int n = 0;
        glm::ivec4 acc(0);
        for (int dy = -1; dy <= 1; ++dy) {
          for (int dx = -1; dx <= 1; ++dx) {
            if (!dx && !dy) continue;
            int sx = wrapi(x + dx, W), sy = wrapi(y + dy, H);
            if (!covered[(size_t)sy * W + sx]) continue;
            const uint8_t* p = out.at((uint32_t)sx, (uint32_t)sy);
            acc += glm::ivec4(p[0], p[1], p[2], p[3]);
            ++n;
          }
        }
        if (n) {
          uint8_t* p = out.at((uint32_t)x, (uint32_t)y);
          p[0] = (uint8_t)(acc.x / n);
          p[1] = (uint8_t)(acc.y / n);
          p[2] = (uint8_t)(acc.z / n);
          p[3] = (uint8_t)(acc.w / n);
          next[(size_t)y * W + x] = 1;
        }
      }
    }
    covered.swap(next);
  }
  // Whatever is still empty (UV space this batch does not own): evaluate
  // only the layers that live in the output UV set at identity UV, never
  // the decal/feature layers, which would otherwise paint stretched eyes over
  // the unused half of a head texture.
  MaterialRecipe background = recipe;
  {
    Sampler* layers[] = {&background.color,     &background.intensity, &background.decal,
                         &background.base,      &background.eyeshadow, &background.mouth,
                         &background.eye,       &background.facial_hair, &background.brow};
    for (Sampler* l : layers) {
      if (l->uv_layer != recipe.out_uv_layer || l->transparent_border) l->img = nullptr;
    }
    background.decal.img = nullptr;
    background.eyeshadow.img = nullptr;
    background.mouth.img = nullptr;
    background.eye.img = nullptr;
    background.facial_hair.img = nullptr;
    background.brow.img = nullptr;
  }
  std::array<glm::vec2, 6> uv;
  for (int y = 0; y < H; ++y) {
    for (int x = 0; x < W; ++x) {
      if (covered[(size_t)y * W + x]) continue;
      glm::vec2 t(((float)x + 0.5f) / W, ((float)y + 0.5f) / H);
      uv.fill(t);
      store(x, y, background.Evaluate(uv));
    }
  }
  out.ok = true;
  return out;
}

static bool WritePng(const fs::path& path, const texdec::Image& img) {
  if (!img.ok) return false;
  std::string p = PathStr(path);
  return stbi_write_png(p.c_str(), (int)img.width, (int)img.height, 4, img.rgba.data(),
                        (int)img.width * 4) != 0;
}

// ---------------------------------------------------------------------------
// JSON writer (minimal)
// ---------------------------------------------------------------------------

class Json {
 public:
  explicit Json(FILE* f) : f_(f) {}
  void Raw(const char* s) { std::fputs(s, f_); }
  void Str(const std::string& s) {
    std::fputc('"', f_);
    std::fputs(JsonEscape(s).c_str(), f_);
    std::fputc('"', f_);
  }
  void Key(const char* k) {
    Comma();
    Str(k);
    std::fputc(':', f_);
    first_ = true;
  }
  void BeginObj() {
    std::fputc('{', f_);
    stack_.push_back(first_);
    first_ = true;
  }
  void EndObj() {
    std::fputc('}', f_);
    first_ = false;
    stack_.pop_back();
  }
  void BeginArr() {
    std::fputc('[', f_);
    stack_.push_back(first_);
    first_ = true;
  }
  void EndArr() {
    std::fputc(']', f_);
    first_ = false;
    stack_.pop_back();
  }
  void Comma() {
    if (!first_) std::fputc(',', f_);
    first_ = false;
  }
  void Int(long long v) {
    Comma();
    std::fprintf(f_, "%lld", v);
  }
  void Bool(bool v) {
    Comma();
    std::fputs(v ? "true" : "false", f_);
  }
  void Null() {
    Comma();
    std::fputs("null", f_);
  }
  void Num(double v, int decimals = 6) {
    Comma();
    if (!std::isfinite(v)) v = 0.0;
    std::fprintf(f_, "%.*f", decimals, v);
  }
  void S(const std::string& s) {
    Comma();
    Str(s);
  }
  void KeyStr(const char* k, const std::string& v) {
    Key(k);
    Comma();
    first_ = false;
    Str(v);
  }
  void KeyInt(const char* k, long long v) {
    Key(k);
    first_ = true;
    Int(v);
  }
  void KeyBool(const char* k, bool v) {
    Key(k);
    first_ = true;
    Bool(v);
  }
  void KeyNum(const char* k, double v, int decimals = 6) {
    Key(k);
    first_ = true;
    Num(v, decimals);
  }
  void Vec(const float* v, int n, int decimals = 6) {
    Comma();
    std::fputc('[', f_);
    for (int i = 0; i < n; ++i) {
      if (i) std::fputc(',', f_);
      double d = v[i];
      if (!std::isfinite(d)) d = 0.0;
      std::fprintf(f_, "%.*f", decimals, d);
    }
    std::fputc(']', f_);
  }

 private:
  FILE* f_;
  bool first_ = true;
  std::vector<bool> stack_;
};

// ---------------------------------------------------------------------------
// Animations
// ---------------------------------------------------------------------------

struct AnimExport {
  std::string name;
  std::shared_ptr<Animation> anim;
  std::string preview_frame;  // per-clip override of --preview-frame ("" = global)
};

static std::shared_ptr<Animation> LoadAnimationFile(const fs::path& path) {
  std::vector<uint8_t> bytes;
  if (!ReadFileBytes(path, bytes) || bytes.size() < 4) return nullptr;
  return Animation::Load(bytes.data(), bytes.size(), AnimationLoadOption::kElements);
}

static void WriteAnimation(Json& j, const AnimExport& ae, const std::vector<JointXform>& xf,
                           size_t joint_count) {
  const Animation& a = *ae.anim;
  j.Comma();
  j.BeginObj();
  j.KeyStr("name", ae.name);
  j.KeyNum("fps", a.frames_per_second, 3);
  j.KeyInt("frame_count", a.frame_count);
  j.KeyInt("joint_count", a.pose_counts[0]);
  j.KeyInt("carryable_joint_count", a.pose_counts[1]);
  j.KeyInt("motion_count", a.motion_count);
  j.KeyInt("texture_count", a.texture_count);
  j.KeyNum("duration", a.frames_per_second > 0 ? (a.frame_count > 0 ? (a.frame_count - 1) / a.frames_per_second : 0) : 0, 4);
  j.KeyStr("translation_mode", "absolute_local");

  // Joint tracks: local TRS per frame. Translations are scaled by the parent's
  // rest world scale so the tracks drive the scaled (exported) skeleton.
  j.Key("tracks");
  j.BeginArr();
  const auto& ps = a.pose_frame_sets[0];
  for (size_t ji = 0; ji < a.pose_counts[0]; ++ji) {
    j.Comma();
    j.BeginObj();
    j.KeyInt("joint", (long long)ji);
    glm::vec3 pscale = ji < xf.size() ? xf[ji].parent_world_scale : glm::vec3(1.f);
    glm::vec3 jscale = ji < xf.size() ? xf[ji].scale : glm::vec3(1.f);
    glm::vec3 rest = ji < xf.size() ? xf[ji].rest_local_t : glm::vec3(0.f);
    j.Key("t");
    j.BeginArr();
    for (const auto& frame : ps.frames) {
      if (ji >= frame.elements.size()) break;
      const auto& e = frame.elements[ji];
      // Pose translations are DELTAS from the rest local offset (frame 0 of
      // every stock clip is ~0 for all joints); emit absolute local values on
      // the scaled skeleton.
      float v[3] = {(rest.x + e.position.x) * pscale.x, (rest.y + e.position.y) * pscale.y,
                    (rest.z + e.position.z) * pscale.z};
      j.Vec(v, 3, 5);
    }
    j.EndArr();
    j.Key("r");
    j.BeginArr();
    for (const auto& frame : ps.frames) {
      if (ji >= frame.elements.size()) break;
      const auto& e = frame.elements[ji];
      float v[4] = {e.rotation.x, e.rotation.y, e.rotation.z, e.rotation.w};
      j.Vec(v, 4, 6);
    }
    j.EndArr();
    j.Key("s");
    j.BeginArr();
    for (const auto& frame : ps.frames) {
      if (ji >= frame.elements.size()) break;
      const auto& e = frame.elements[ji];
      float v[3] = {e.scale.x * jscale.x, e.scale.y * jscale.y, e.scale.z * jscale.z};
      j.Vec(v, 3, 5);
    }
    j.EndArr();
    j.EndObj();
  }
  j.EndArr();

  // Carryable tracks (raw local TRS).
  j.Key("carryable_tracks");
  j.BeginArr();
  const auto& cs = a.pose_frame_sets[1];
  for (size_t ji = 0; ji < a.pose_counts[1]; ++ji) {
    j.Comma();
    j.BeginObj();
    j.KeyInt("joint", (long long)ji);
    j.Key("t");
    j.BeginArr();
    for (const auto& frame : cs.frames) {
      if (ji >= frame.elements.size()) break;
      const auto& e = frame.elements[ji];
      float v[3] = {e.position.x, e.position.y, e.position.z};
      j.Vec(v, 3, 5);
    }
    j.EndArr();
    j.Key("r");
    j.BeginArr();
    for (const auto& frame : cs.frames) {
      if (ji >= frame.elements.size()) break;
      const auto& e = frame.elements[ji];
      float v[4] = {e.rotation.x, e.rotation.y, e.rotation.z, e.rotation.w};
      j.Vec(v, 4, 6);
    }
    j.EndArr();
    j.Key("s");
    j.BeginArr();
    for (const auto& frame : cs.frames) {
      if (ji >= frame.elements.size()) break;
      const auto& e = frame.elements[ji];
      float v[3] = {e.scale.x, e.scale.y, e.scale.z};
      j.Vec(v, 3, 5);
    }
    j.EndArr();
    j.EndObj();
  }
  j.EndArr();

  // Motion contexts.
  j.Key("motion");
  j.BeginArr();
  const auto& ms = a.motion_frame_set;
  for (size_t mi = 0; mi < a.motion_count; ++mi) {
    j.Comma();
    j.BeginObj();
    j.Key("t");
    j.BeginArr();
    for (const auto& frame : ms.frames) {
      if (mi >= frame.elements.size()) break;
      const auto& e = frame.elements[mi];
      float v[3] = {e.position.x, e.position.y, e.position.z};
      j.Vec(v, 3, 5);
    }
    j.EndArr();
    j.Key("r");
    j.BeginArr();
    for (const auto& frame : ms.frames) {
      if (mi >= frame.elements.size()) break;
      const auto& e = frame.elements[mi];
      float v[4] = {e.rotation.x, e.rotation.y, e.rotation.z, e.rotation.w};
      j.Vec(v, 4, 6);
    }
    j.EndArr();
    j.EndObj();
  }
  j.EndArr();

  // Face channels: XAVATAR_ANIMATED_TEXTURE order.
  static const char* kChannels[5] = {"mouth", "brow_left", "brow_right", "eye_left", "eye_right"};
  j.Key("face");
  j.BeginObj();
  const auto& ts = a.texture_frame_set;
  for (size_t ci = 0; ci < a.texture_count && ci < 5; ++ci) {
    j.Key(kChannels[ci]);
    j.BeginArr();
    for (const auto& frame : ts.frames) {
      if (ci >= frame.elements.size()) break;
      j.Int(frame.elements[ci].layer_index);
    }
    j.EndArr();
  }
  j.EndObj();
  j.EndObj();
}


// ---------------------------------------------------------------------------
// The export
// ---------------------------------------------------------------------------

struct BakedMesh {
  std::string name;
  int component = -1;
  int material = -1;
  uint32_t uv_count = 1;
  std::vector<SkinVertex> verts;  // posed
  std::vector<uint16_t> indices;
  bool is_prop = false;
};

struct BakedMaterial {
  std::string name;
  std::string file;
  texdec::Image image;  // the baked diffuse, kept for the preview renderer
  uint32_t shader_id = 0;
  bool has_alpha = false;
  int uv_layer = 0;
  std::string component_guid;
};

static const char* ShaderName(uint32_t id) {
  switch (id) {
    case 0: return "BODY_OPAQUE";
    case 1: return "BODY_TRANSPARENT";
    case 2: return "BODY_SHINY_OPAQUE";
    case 3: return "BODY_SHINY_TRANSPARENT";
    case 4: return "HEAD_OPAQUE";
    default: return "UNKNOWN";
  }
}

// usage -> replacement slot (ModelToGuest's usage_to_replacement_texture_indices)
static int ReplacementSlotForUsage(uint32_t usage) {
  static const int table[] = {-1, -1, -1, -1, -1, 5, 3, 2, 2, 1, 1, 4, 0};
  return usage < 13 ? table[usage] : -1;
}

struct HeadTextureSet {
  // decoded model textures by index, with replacements substituted per usage
  std::map<int, texdec::Image> model_images;             // texture index -> layer 0
  std::map<int, std::vector<texdec::Image>> rep_layers;  // slot -> all layers
};

// ---------------------------------------------------------------------------
// Preview renderer: orthographic front view (camera at +Z), z-buffer, smooth
// three-light shading, bilinear textures, 2x supersampling, the Avatar
// Editor's green radial backdrop and a contact shadow. Poses come from the
// decoded clips (local TRS re-composed through the scaled skeleton).
// ---------------------------------------------------------------------------

struct PosedMesh {
  std::vector<glm::vec3> pos;
  std::vector<glm::vec3> nrm;
};

static std::vector<glm::mat4> PoseSkinMatrices(const std::vector<JointXform>& xf,
                                               const Skeleton& skel, const Animation* anim,
                                               size_t frame, int pose_set = 0) {
  std::vector<glm::mat4> world(xf.size()), skin(xf.size());
  const auto* frames = anim ? &anim->pose_frame_sets[pose_set].frames : nullptr;
  for (size_t j = 0; j < xf.size(); ++j) {
    glm::vec3 t = xf[j].rest_local_t;
    glm::quat r = xf[j].rest_local_r;
    glm::vec3 sc = xf[j].scale;
    if (frames && frame < frames->size() && j < (*frames)[frame].elements.size()) {
      const auto& e = (*frames)[frame].elements[j];
      t += glm::vec3(e.position.x, e.position.y, e.position.z);
      r = glm::normalize(glm::quat(e.rotation.w, e.rotation.x, e.rotation.y, e.rotation.z));
      sc *= glm::vec3(e.scale.x, e.scale.y, e.scale.z);
    }
    glm::mat4 local = glm::translate(glm::mat4(1.f), t) * glm::mat4_cast(r) *
                      glm::scale(glm::mat4(1.f), sc);
    const uint8_t parent = skel.joints[j].parent_index;
    world[j] = (parent != 255 && parent < j) ? world[parent] * local : local;
    skin[j] = world[j] * glm::inverse(xf[j].bind_world);
  }
  return skin;
}

static std::vector<PosedMesh> PoseMeshes(const std::vector<BakedMesh>& meshes,
                                         const std::vector<glm::mat4>& skin) {
  std::vector<PosedMesh> out(meshes.size());
  for (size_t mi = 0; mi < meshes.size(); ++mi) {
    const BakedMesh& m = meshes[mi];
    out[mi].pos.resize(m.verts.size());
    out[mi].nrm.resize(m.verts.size());
    for (size_t vi = 0; vi < m.verts.size(); ++vi) {
      const SkinVertex& v = m.verts[vi];
      if (m.is_prop) {  // carryables keep their rest pose in the preview
        out[mi].pos[vi] = v.position;
        out[mi].nrm[vi] = v.normal;
        continue;
      }
      glm::vec3 p(0.f), n(0.f);
      float wsum = 0.f;
      for (int k = 0; k < 4; ++k) {
        float w = v.weights[k];
        if (w <= 0.f || v.joints[k] >= skin.size()) continue;
        const glm::mat4& M = skin[v.joints[k]];
        p += w * glm::vec3(M * glm::vec4(v.orig_position, 1.f));
        n += w * (glm::transpose(glm::inverse(glm::mat3(M))) * v.orig_normal);
        wsum += w;
      }
      if (wsum > 0.f) {
        out[mi].pos[vi] = p / wsum;
        float len = glm::length(n);
        out[mi].nrm[vi] = len > 1e-6f ? n / len : v.normal;
      } else {
        out[mi].pos[vi] = v.orig_position;
        out[mi].nrm[vi] = v.orig_normal;
      }
    }
  }
  return out;
}

struct PreviewCamera {
  glm::vec2 center;  // world x/y at the image centre
  float scale;       // pixels per metre (at the supersampled size)
  float feet_y;      // lowest world y (shadow)
  float width;       // world width of the avatar
};

static PreviewCamera FrameCamera(const std::vector<std::vector<PosedMesh>>& all_poses, int px) {
  glm::vec2 lo(1e9f), hi(-1e9f);
  for (const auto& poses : all_poses) {
    for (const auto& pm : poses) {
      for (const auto& p : pm.pos) {
        lo = glm::min(lo, glm::vec2(p.x, p.y));
        hi = glm::max(hi, glm::vec2(p.x, p.y));
      }
    }
  }
  PreviewCamera cam;
  const float margin = 0.08f;
  float span = std::max(hi.x - lo.x, hi.y - lo.y) * (1.f + 2.f * margin);
  if (span < 1e-3f) span = 1.f;
  cam.center = (lo + hi) * 0.5f;
  cam.scale = (float)px / span;
  cam.feet_y = lo.y;
  cam.width = hi.x - lo.x;
  return cam;
}

static float PreviewShade(const glm::vec3& n) {
  static const glm::vec3 key = glm::normalize(glm::vec3(-0.45f, 0.60f, 0.70f));
  static const glm::vec3 fill = glm::normalize(glm::vec3(0.70f, 0.15f, 0.55f));
  static const glm::vec3 rim = glm::normalize(glm::vec3(0.30f, 0.50f, -0.80f));
  float k = std::max(0.f, glm::dot(n, key)) * 0.5f + 0.5f;  // half-lambert
  float f = std::max(0.f, glm::dot(n, fill));
  float r = std::max(0.f, glm::dot(n, rim));
  return std::min(1.25f, 0.22f + 0.72f * k * k + 0.20f * f + 0.18f * r * r);
}

static bool RenderPreviewImage(const std::vector<BakedMesh>& meshes,
                               const std::vector<BakedMaterial>& materials,
                               const std::vector<PosedMesh>& posed, const PreviewCamera& cam,
                               int out_size, const fs::path& out_png,
                               const std::map<int, const texdec::Image*>& overrides,
                               bool transparent = false, size_t* covered_px = nullptr) {
  const int SS = 2;
  const int W = out_size * SS, H = out_size * SS;
  std::vector<float> rgb((size_t)W * H * 3, 0.f);
  std::vector<uint8_t> cover((size_t)W * H, transparent ? 0 : 1);
  std::vector<float> zbuf((size_t)W * H, -1e30f);
  // backdrop: the editor's green radial gradient
  const glm::vec3 inner(146.f, 204.f, 112.f), outer(34.f, 78.f, 44.f);
  const float cx = W * 0.5f, cy = H * 0.40f, rmax = std::hypot(W * 0.5f, H * 0.6f);
  if (!transparent) {
    for (int y = 0; y < H; ++y) {
      for (int x = 0; x < W; ++x) {
        float t = std::hypot(x - cx, y - cy) / rmax;
        t = std::min(1.f, t * t * 0.9f + t * 0.1f);
        glm::vec3 c = inner + (outer - inner) * t;
        float* p = &rgb[((size_t)y * W + x) * 3];
        p[0] = c.r; p[1] = c.g; p[2] = c.b;
      }
    }
  }
  auto sx_of = [&](float wx) { return (wx - cam.center.x) * cam.scale + W * 0.5f; };
  auto sy_of = [&](float wy) { return H * 0.5f - (wy - cam.center.y) * cam.scale; };
  // contact shadow
  if (!transparent) {
    float scx = sx_of(cam.center.x), scy = std::min((float)H - 2.f, sy_of(cam.feet_y) + 2.f * SS);
    float rx = std::max(12.f, cam.width * cam.scale * 0.28f), ry = std::max(4.f, cam.width * cam.scale * 0.05f);
    for (int y = std::max(0, (int)(scy - ry)); y <= std::min(H - 1, (int)(scy + ry)); ++y) {
      for (int x = std::max(0, (int)(scx - rx)); x <= std::min(W - 1, (int)(scx + rx)); ++x) {
        float d = ((x - scx) / rx) * ((x - scx) / rx) + ((y - scy) / ry) * ((y - scy) / ry);
        if (d >= 1.f) continue;
        float a = 0.55f * std::pow(1.f - d, 1.5f);
        float* p = &rgb[((size_t)y * W + x) * 3];
        p[0] *= 1.f - a; p[1] *= 1.f - a; p[2] *= 1.f - a;
      }
    }
  }
  // triangles
  for (size_t mi = 0; mi < meshes.size(); ++mi) {
    const BakedMesh& m = meshes[mi];
    const BakedMaterial& mat = materials[m.material];
    Sampler tex;
    auto ov = overrides.find(m.material);
    tex.img = ov != overrides.end() ? ov->second : (mat.image.ok ? &mat.image : nullptr);
    tex.flags = 3;  // wrap
    const bool mask = mat.has_alpha;
    const auto& P = posed[mi].pos;
    const auto& N = posed[mi].nrm;
    std::vector<float> shade(P.size());
    for (size_t i = 0; i < P.size(); ++i) shade[i] = PreviewShade(N[i]);
    for (size_t t = 0; t + 2 < m.indices.size(); t += 3) {
      const uint16_t ia = m.indices[t], ib = m.indices[t + 1], ic = m.indices[t + 2];
      glm::vec2 a(sx_of(P[ia].x), sy_of(P[ia].y)), b(sx_of(P[ib].x), sy_of(P[ib].y)),
          c(sx_of(P[ic].x), sy_of(P[ic].y));
      float area = (b.x - a.x) * (c.y - a.y) - (c.x - a.x) * (b.y - a.y);
      if (std::fabs(area) < 1e-6f) continue;
      int x0 = std::max(0, (int)std::floor(std::min({a.x, b.x, c.x})));
      int x1 = std::min(W - 1, (int)std::ceil(std::max({a.x, b.x, c.x})));
      int y0 = std::max(0, (int)std::floor(std::min({a.y, b.y, c.y})));
      int y1 = std::min(H - 1, (int)std::ceil(std::max({a.y, b.y, c.y})));
      const float inv = 1.f / area;
      const glm::vec3 vc[3] = {
          glm::vec3(((m.verts[ia].color >> 16) & 0xFF), ((m.verts[ia].color >> 8) & 0xFF), (m.verts[ia].color & 0xFF)) / 255.f,
          glm::vec3(((m.verts[ib].color >> 16) & 0xFF), ((m.verts[ib].color >> 8) & 0xFF), (m.verts[ib].color & 0xFF)) / 255.f,
          glm::vec3(((m.verts[ic].color >> 16) & 0xFF), ((m.verts[ic].color >> 8) & 0xFF), (m.verts[ic].color & 0xFF)) / 255.f};
      const glm::vec2 uva = m.verts[ia].uvs[mat.uv_layer], uvb = m.verts[ib].uvs[mat.uv_layer],
                      uvc = m.verts[ic].uvs[mat.uv_layer];
      for (int y = y0; y <= y1; ++y) {
        const float py = y + 0.5f;
        for (int x = x0; x <= x1; ++x) {
          const float px = x + 0.5f;
          float w0 = ((b.x - px) * (c.y - py) - (c.x - px) * (b.y - py)) * inv;
          float w1 = ((c.x - px) * (a.y - py) - (a.x - px) * (c.y - py)) * inv;
          float w2 = 1.f - w0 - w1;
          if (w0 < -1e-4f || w1 < -1e-4f || w2 < -1e-4f) continue;
          float z = w0 * P[ia].z + w1 * P[ib].z + w2 * P[ic].z;
          float& zb = zbuf[(size_t)y * W + x];
          if (z <= zb) continue;
          glm::vec4 texel(0.8f, 0.8f, 0.8f, 1.f);
          if (tex.img) {
            texel = tex.Sample(w0 * uva + w1 * uvb + w2 * uvc);
            if (mask && texel.a < 0.5f) continue;
          }
          zb = z;
          cover[(size_t)y * W + x] = 1;
          float sh = w0 * shade[ia] + w1 * shade[ib] + w2 * shade[ic];
          glm::vec3 col = w0 * vc[0] + w1 * vc[1] + w2 * vc[2];
          float* p = &rgb[((size_t)y * W + x) * 3];
          p[0] = std::min(255.f, texel.r * col.r * sh * 255.f);
          p[1] = std::min(255.f, texel.g * col.g * sh * 255.f);
          p[2] = std::min(255.f, texel.b * col.b * sh * 255.f);
        }
      }
    }
  }
  // downsample
  texdec::Image out;
  out.width = out_size;
  out.height = out_size;
  out.rgba.resize((size_t)out_size * out_size * 4);
  for (int y = 0; y < out_size; ++y) {
    for (int x = 0; x < out_size; ++x) {
      glm::vec3 acc(0.f);
      int n = 0;
      for (int dy = 0; dy < SS; ++dy) {
        for (int dx = 0; dx < SS; ++dx) {
          const size_t i = (size_t)(y * SS + dy) * W + (x * SS + dx);
          if (!cover[i]) continue;
          const float* p = &rgb[i * 3];
          acc += glm::vec3(p[0], p[1], p[2]);
          ++n;
        }
      }
      uint8_t* o = out.at((uint32_t)x, (uint32_t)y);
      if (n) acc /= (float)n;
      o[0] = (uint8_t)std::lround(acc.r);
      o[1] = (uint8_t)std::lround(acc.g);
      o[2] = (uint8_t)std::lround(acc.b);
      o[3] = transparent ? (uint8_t)(255 * n / (SS * SS)) : 255;
    }
  }
  if (covered_px) {
    size_t n = 0;
    for (uint8_t c : cover) n += c ? 1 : 0;
    *covered_px = n;  // at the supersampled size (W*H)
  }
  out.ok = true;
  return WritePng(out_png, out);
}

// Which frame of a clip to pose: "mid" (50%), "peak" (largest total joint
// rotation), a fraction 0..1, or an absolute frame number.
static size_t PickPreviewFrame(const Animation& a, const std::string& rule) {
  const size_t n = a.frame_count;
  if (n == 0) return 0;
  if (rule == "mid") return n / 2;
  if (rule == "peak") {
    size_t best = 0;
    double best_score = -1.0;
    const auto& frames = a.pose_frame_sets[0].frames;
    for (size_t f = 0; f < frames.size(); ++f) {
      double score = 0.0;
      for (const auto& e : frames[f].elements) {
        score += 2.0 * std::acos(std::min(1.f, std::fabs(e.rotation.w)));
      }
      if (score > best_score) {
        best_score = score;
        best = f;
      }
    }
    return best;
  }
  char* end = nullptr;
  double v = std::strtod(rule.c_str(), &end);
  if (end && *end == '\0') {
    if (v >= 0.0 && v <= 1.0) return std::min(n - 1, (size_t)std::lround(v * (n - 1)));
    if (v > 1.0) return std::min(n - 1, (size_t)v);
  }
  return n / 2;
}


// ---------------------------------------------------------------------------
// Icon framing helpers (--gen-icons). A 3/4 view is a rotation of the posed
// vertices about the framed box's centre (the renderer stays an orthographic
// front view), and the camera is set straight from a world box.
// ---------------------------------------------------------------------------
static void RotatePosed(std::vector<PosedMesh>& posed, const glm::vec3& pivot, float yaw_deg,
                        float pitch_deg) {
  const glm::mat3 R = glm::mat3(
      glm::rotate(glm::mat4(1.f), glm::radians(pitch_deg), glm::vec3(1.f, 0.f, 0.f)) *
      glm::rotate(glm::mat4(1.f), glm::radians(yaw_deg), glm::vec3(0.f, 1.f, 0.f)));
  for (auto& pm : posed) {
    for (auto& p : pm.pos) p = R * (p - pivot) + pivot;
    for (auto& n : pm.nrm) {
      const glm::vec3 r = R * n;
      const float len = glm::length(r);
      n = len > 1e-8f ? r / len : n;
    }
  }
}

static void RotateBox(glm::vec3& lo, glm::vec3& hi, const glm::vec3& pivot, float yaw_deg,
                      float pitch_deg) {
  const glm::mat3 R = glm::mat3(
      glm::rotate(glm::mat4(1.f), glm::radians(pitch_deg), glm::vec3(1.f, 0.f, 0.f)) *
      glm::rotate(glm::mat4(1.f), glm::radians(yaw_deg), glm::vec3(0.f, 1.f, 0.f)));
  glm::vec3 nlo(1e9f), nhi(-1e9f);
  for (int i = 0; i < 8; ++i) {
    const glm::vec3 c((i & 1) ? hi.x : lo.x, (i & 2) ? hi.y : lo.y, (i & 4) ? hi.z : lo.z);
    const glm::vec3 r = R * (c - pivot) + pivot;
    nlo = glm::min(nlo, r);
    nhi = glm::max(nhi, r);
  }
  lo = nlo;
  hi = nhi;
}

static PreviewCamera CameraForBox(const glm::vec3& lo, const glm::vec3& hi, int size, float margin) {
  PreviewCamera cam;
  const glm::vec3 ext = hi - lo;
  float span = std::max(ext.x, ext.y) * margin;
  if (!(span > 1e-4f)) span = 0.1f;
  cam.center = glm::vec2((lo.x + hi.x) * 0.5f, (lo.y + hi.y) * 0.5f);
  cam.scale = (float)(size * 2) / span;
  cam.feet_y = lo.y;
  cam.width = ext.x;
  return cam;
}

// The worn-item icon: the closet item `args.icon_target` on the blank
// mannequin, framed the way the marketplace art frames it (head for
// hair/hats/glasses/earrings, the item plus lower-leg context for shoes, the
// item plus hand/arm context for rings/wristwear/gloves, the item itself for
// tops/bottoms, the whole figure otherwise), turned to a 3/4 view and drawn on
// a transparent background.
static bool RenderWornItemIcon(const std::vector<BakedMesh>& meshes,
                               const std::vector<BakedMaterial>& materials,
                               std::vector<PosedMesh> posed, const Resolved& res,
                               const Args& args) {
  int target = -1, head = -1;
  for (size_t i = 0; i < res.components.size(); ++i) {
    if (res.components[i].info.asset_id == args.icon_target) target = (int)i;
    if (res.components[i].info.categories & ComponentCategory::kHead) head = (int)i;
  }
  auto box_of = [&](int component, glm::vec3& lo, glm::vec3& hi) {
    lo = glm::vec3(1e9f);
    hi = glm::vec3(-1e9f);
    bool any = false;
    for (size_t mi = 0; mi < meshes.size() && mi < posed.size(); ++mi) {
      if (component >= 0 && meshes[mi].component != component) continue;
      for (const auto& p : posed[mi].pos) {
        lo = glm::min(lo, p);
        hi = glm::max(hi, p);
        any = true;
      }
    }
    return any;
  };
  glm::vec3 ilo, ihi, hlo, hhi, alo, ahi;
  const bool has_item = target >= 0 && box_of(target, ilo, ihi);
  const bool has_head = head >= 0 && box_of(head, hlo, hhi);
  if (!box_of(-1, alo, ahi)) return false;
  const uint32_t cat = args.icon_categories;
  glm::vec3 lo = alo, hi = ahi;
  float margin = 1.08f;
  if (has_item) {
    lo = ilo;
    hi = ihi;
    margin = 1.18f;
    if (cat & (ComponentCategory::kHair | ComponentCategory::kHat | ComponentCategory::kGlasses |
               ComponentCategory::kEarrings)) {
      if (has_head) {
        lo = glm::min(lo, hlo);
        hi = glm::max(hi, hhi);
      }
      margin = 1.14f;
    } else if (cat & ComponentCategory::kShoes) {
      hi.y += (hi.y - lo.y) * 0.45f;
    } else if (cat & (ComponentCategory::kRing | ComponentCategory::kWristwear |
                      ComponentCategory::kGloves)) {
      const glm::vec3 e = hi - lo;
      lo -= e * 0.35f;
      hi += e * 0.35f;
    }
  }
  const glm::vec3 pivot = (lo + hi) * 0.5f;
  RotatePosed(posed, pivot, args.icon_yaw, args.icon_pitch);
  RotateBox(lo, hi, pivot, args.icon_yaw, args.icon_pitch);
  const PreviewCamera cam = CameraForBox(lo, hi, args.icon_size, margin);
  std::map<int, const texdec::Image*> no_overrides;
  return RenderPreviewImage(meshes, materials, posed, cam, args.icon_size, args.icon_out,
                            no_overrides, true);
}

int Run(const Args& args) {
  std::printf("(mode: avatar export)\n\n");
  // ---- manifest ----
  std::vector<uint8_t> manifest_bytes;
  if (!ReadFileBytes(args.manifest, manifest_bytes) ||
      manifest_bytes.size() != sizeof(X_AVATAR_METADATA)) {
    std::printf("ERROR: manifest %s is not a 1000-byte X_AVATAR_METADATA\n",
                PathStr(args.manifest).c_str());
    return 1;
  }
  X_AVATAR_METADATA metadata;
  std::memcpy(&metadata, manifest_bytes.data(), sizeof(metadata));

  // ---- packs + closet ----
  Assets assets;
  if (!LoadPack(args.toc, assets.pack)) {
    std::printf("ERROR: cannot load asset pack %s\n", PathStr(args.toc).c_str());
    return 2;
  }
  std::printf("asset pack: %s (%zu assets)\n", PathStr(args.toc).c_str(),
              assets.pack.asset_infos().size());
  if (!args.legacy_toc.empty() && fs::exists(args.legacy_toc)) {
    assets.has_legacy = LoadPack(args.legacy_toc, assets.legacy_pack);
    std::printf("legacy pack: %s (%s)\n", PathStr(args.legacy_toc).c_str(),
                assets.has_legacy ? "loaded" : "FAILED");
  }
  if (!args.closet.empty()) {
    GetCloset().Load(args.closet);
    std::printf("closet: %s (%zu items)\n", PathStr(args.closet).c_str(),
                GetCloset().items().size());
  }

  if (args.list_anims) {
    const auto& infos = assets.pack.asset_infos();
    std::printf("=== pack animations ===\n");
    for (size_t i = 0; i < infos.size(); ++i) {
      if (infos[i].categories & ComponentCategory::kAnimation) {
        std::printf("  %-5zu %s\n", i, U16ToUtf8(assets.pack.GetAssetNameByIndex(i)).c_str());
      }
    }
    return 0;
  }

  // ---- resolve ----
  std::printf("\n=== resolving ===\n");
  Resolved res;
  if (!Resolve(metadata, args, assets, res)) return 3;
  std::printf("  body type: %s  height %.3f  weight %.3f\n",
              res.body_type == BodyType::kMale     ? "male"
              : res.body_type == BodyType::kFemale ? "female"
                                                   : "unknown",
              metadata.height_factor.get(), metadata.weight_factor.get());
  for (const auto& c : res.components) {
    std::printf("  [%s] %s '%s' (%s, %s) batches=%zu textures=%zu\n", c.slot.c_str(),
                c.info.asset_id.to_string().c_str(), c.name.c_str(),
                CategoryNames(c.info.categories.get()).c_str(),
                c.from_closet ? "closet" : "pack", c.model->triangle_batches.size(),
                c.model->textures.size());
  }

  if (args.info_only) {
    Json j(stdout);
    j.BeginObj();
    j.KeyStr("manifest", PathStr(args.manifest));
    j.KeyStr("body_type", res.body_type == BodyType::kMale     ? "male"
                          : res.body_type == BodyType::kFemale ? "female"
                                                               : "unknown");
    j.KeyNum("height_factor", metadata.height_factor.get(), 4);
    j.KeyNum("weight_factor", metadata.weight_factor.get(), 4);
    j.Key("colors");
    j.BeginArr();
    for (int i = 0; i < 9; ++i) {
      char buf[16];
      std::snprintf(buf, sizeof(buf), "%08X", metadata.colors[i].get());
      j.S(buf);
    }
    j.EndArr();
    j.Key("components");
    j.BeginArr();
    for (const auto& c : res.components) {
      j.Comma();
      j.BeginObj();
      j.KeyStr("slot", c.slot);
      j.KeyStr("guid", c.info.asset_id.to_string());
      j.KeyStr("name", c.name);
      j.KeyInt("categories", c.info.categories.get());
      j.KeyStr("category_names", CategoryNames(c.info.categories.get()));
      j.KeyStr("source", c.from_closet ? "closet" : "pack");
      j.KeyInt("batches", (long long)c.model->triangle_batches.size());
      j.EndObj();
    }
    j.EndArr();
    j.Key("face_textures");
    j.BeginArr();
    static const char* kSlotNames[6] = {"mouth", "eyes", "brows", "face_paint", "eye_shadow", "face"};
    for (int s = 0; s < 6; ++s) {
      if (!res.replacement_textures[s]) continue;
      j.Comma();
      j.BeginObj();
      j.KeyStr("slot", kSlotNames[s]);
      j.KeyStr("guid", res.replacement_ids[s].to_string());
      j.KeyStr("name", assets.Name(res.replacement_ids[s]));
      j.KeyInt("layers", res.replacement_textures[s]->layer_count);
      j.EndObj();
    }
    j.EndArr();
    j.Key("prop");
    if (res.prop) {
      j.BeginObj();
      j.KeyStr("guid", res.prop_info.asset_id.to_string());
      j.KeyStr("name", res.prop_name);
      j.EndObj();
    } else {
      j.Null();
    }
    j.EndObj();
    std::printf("\n");
    return 0;
  }

  fs::create_directories(args.out_dir);

  // ---- skeleton ----
  const auto xforms = BuildJointXforms(*res.skeleton, !args.no_scale);
  const size_t joint_count = res.skeleton->joints.size();
  std::printf("\n=== skeleton v%d: %zu joints, scale %s ===\n", args.skeleton_version,
              joint_count, args.no_scale ? "off" : "on");

  // ---- bake ----
  std::printf("\n=== baking ===\n");
  std::vector<BakedMaterial> materials;
  std::vector<BakedMesh> meshes;
  std::map<std::string, int> material_by_key;
  std::map<std::string, int> name_counts;
  // head bookkeeping for face frames
  struct HeadBatchInfo {
    int mesh_index;
    MaterialRecipe recipe;
    std::map<uint32_t, int> usage_to_texture;  // usage -> model texture index
    std::map<uint32_t, int> usage_to_uv;
    int material_index;
    uint32_t eye_usage = 9;   // 9 = left eye, 10 = right eye
    uint32_t brow_usage = 7;  // 7 = left brow, 8 = right brow
  };
  std::vector<HeadBatchInfo> head_batches;
  // decoded replacement stacks (all layers) per slot
  std::vector<texdec::Image> rep_layers[6];
  for (int s = 0; s < 6; ++s) {
    if (!res.replacement_textures[s]) continue;
    const Texture& t = *res.replacement_textures[s];
    for (uint32_t l = 0; l < t.layer_count; ++l) {
      rep_layers[s].push_back(texdec::DecodeTextureLayer(t, l));
    }
    std::printf("  face slot %d: %ux%u %s x%u layers\n", s, t.width, t.height,
                texdec::KindName(texdec::ClassifyFormat(t.format)), t.layer_count);
  }

  size_t total_verts = 0;
  // Decoded model textures per component (+1 for the carryable). Kept alive
  // for the whole export: head MaterialRecipes hold pointers into them.
  std::vector<std::vector<texdec::Image>> component_images(res.components.size() + 1);

  auto bake_component = [&](const ResolvedComponent& comp, int comp_index, bool is_prop,
                            const std::vector<JointXform>* skin_xf) {
    const Model& model = *comp.model;
    // decode the model's own textures (layer 0)
    std::vector<texdec::Image>& images = component_images[(size_t)comp_index];
    images.resize(model.textures.size());
    for (size_t i = 0; i < model.textures.size(); ++i) {
      images[i] = texdec::DecodeTexture(model.textures[i].texture);
    }
    const bool is_head = !is_prop && (comp.info.categories.get() & ComponentCategory::kHead) != 0;

    for (size_t bi = 0; bi < model.triangle_batches.size(); ++bi) {
      const TriangleBatch& b = model.triangle_batches[bi];
      if (b.triangle_count == 0 || b.vertices.empty()) continue;

      // shader parameter table
      std::map<uint32_t, int> usage_tex, usage_uv;
      std::map<uint32_t, uint32_t> usage_flags;
      std::map<uint32_t, Float4> usage_const;
      for (const auto& sp : b.shader_parameters) {
        if (sp.type == ShaderParameterType::kTexture) {
          usage_tex[sp.usage] = sp.texture.index;
          usage_uv[sp.usage] = sp.texture.uv_layer;
          usage_flags[sp.usage] = sp.texture.flags;
        } else if (sp.type == ShaderParameterType::kPixelConstant ||
                   sp.type == ShaderParameterType::kVertexConstant) {
          usage_const[sp.usage] = Float4{sp.constant_values[0], sp.constant_values[1],
                                         sp.constant_values[2], sp.constant_values[3]};
        }
      }
      for (const auto& ov : comp.overrides) usage_const[ov.first] = ov.second;

      MaterialRecipe recipe;
      recipe.shader_id = b.shader_id;
      recipe.kind = (b.shader_id == 4 || is_head) && usage_tex.count(5) ? ShaderKind::kHead
                                                                          : ShaderKind::kBody;
      auto tex_for = [&](uint32_t usage, Sampler& s) {
        auto it = usage_tex.find(usage);
        if (it == usage_tex.end()) return;
        int ti = it->second;
        s.uv_layer = std::min(5, std::max(0, usage_uv[usage]));
        s.flags = usage_flags[usage];
        // Clamp-addressed face feature decals (6..12, flags 0) must not repeat
        // across the head: outside [0,1] they contribute nothing.
        s.transparent_border = usage >= 6 && usage <= 12 && s.flags == 0;
        int slot = is_head ? ReplacementSlotForUsage(usage) : -1;
        if (slot >= 0 && !rep_layers[slot].empty() && rep_layers[slot][0].ok) {
          s.img = &rep_layers[slot][0];
        } else if (ti >= 0 && ti < (int)images.size() && images[ti].ok) {
          s.img = &images[ti];
        }
      };
      if (recipe.kind == ShaderKind::kBody) {
        tex_for(1, recipe.color);
        tex_for(2, recipe.intensity);
        tex_for(3, recipe.decal);
        for (int k = 0; k < 3; ++k) {
          auto it = usage_const.find(22 + k);
          recipe.custom[k] = it != usage_const.end() ? it->second : Float4{1, 1, 1, 1};
        }
        recipe.out_uv_layer = recipe.color.img ? recipe.color.uv_layer : 0;
      } else {
        tex_for(5, recipe.base);
        tex_for(11, recipe.eyeshadow);
        tex_for(12, recipe.mouth);
        if (usage_tex.count(9)) tex_for(9, recipe.eye);
        else tex_for(10, recipe.eye);
        tex_for(6, recipe.facial_hair);
        if (usage_tex.count(7)) tex_for(7, recipe.brow);
        else tex_for(8, recipe.brow);
        for (int k = 0; k < 9; ++k) {
          auto it = usage_const.find(13 + k);
          recipe.tint[k] = it != usage_const.end() ? it->second : Float4{1, 1, 1, 1};
        }
        recipe.out_uv_layer = recipe.base.img ? recipe.base.uv_layer : 0;
      }
      if (!is_head && is_prop) {
        // carryables render with their own (white) palette
      }

      // vertices
      BakedMesh mesh;
      mesh.component = comp_index;
      mesh.is_prop = is_prop;
      mesh.uv_count = b.uv_count;
      mesh.verts.resize(b.vertices.size());
      for (size_t vi = 0; vi < b.vertices.size(); ++vi) {
        UnpackSkin(b.vertices[vi], mesh.verts[vi], b.uv_count);
      }
      mesh.indices = b.indices;

      // material (dedup by textures + constants within the component)
      std::string key = comp.info.asset_id.to_string() + "|" + std::to_string(b.shader_id);
      for (const auto& ut : usage_tex) key += "|t" + std::to_string(ut.first) + "=" + std::to_string(ut.second) + "/" + std::to_string(usage_uv[ut.first]);
      for (const auto& uc : usage_const) {
        char buf[80];
        std::snprintf(buf, sizeof(buf), "|c%u=%.3f,%.3f,%.3f,%.3f", uc.first, uc.second.x, uc.second.y, uc.second.z, uc.second.w);
        key += buf;
      }
      int mat_index;
      auto mit = material_by_key.find(key);
      if (mit != material_by_key.end()) {
        mat_index = mit->second;
      } else {
        texdec::Image baked = BakeMaterial(recipe, mesh.verts, mesh.indices, args.bake_size);
        BakedMaterial mat;
        std::string base_name = is_prop ? "Prop" : comp.slot;
        if (comp.from_closet && !comp.name.empty() && !is_prop) {
          base_name = comp.slot + "_" + SanitizeName(comp.name);
        }
        int n = name_counts[base_name]++;
        mat.name = n == 0 ? base_name : base_name + "_" + std::to_string(n);
        mat.file = mat.name + "_Diffuse.png";
        mat.shader_id = b.shader_id;
        mat.uv_layer = recipe.out_uv_layer;
        mat.component_guid = comp.info.asset_id.to_string();
        for (size_t i = 3; i < baked.rgba.size(); i += 4) {
          if (baked.rgba[i] < 128) {
            mat.has_alpha = true;
            break;
          }
        }
        if (!WritePng(args.out_dir / mat.file, baked)) {
          std::printf("  WARN: failed to write %s\n", mat.file.c_str());
        }
        mat.image = std::move(baked);
        std::string flag_desc;
        for (const auto& uf : usage_flags) {
          char fb[40];
          std::snprintf(fb, sizeof(fb), " u%u:t%d/uv%d/f%u", uf.first, usage_tex[uf.first],
                        usage_uv[uf.first], uf.second);
          flag_desc += fb;
        }
        char cdesc[160] = "";
        if (recipe.kind == ShaderKind::kBody) {
          std::snprintf(cdesc, sizeof(cdesc), " C0=%.2f,%.2f,%.2f C1=%.2f,%.2f,%.2f C2=%.2f,%.2f,%.2f",
                        recipe.custom[0].x, recipe.custom[0].y, recipe.custom[0].z,
                        recipe.custom[1].x, recipe.custom[1].y, recipe.custom[1].z,
                        recipe.custom[2].x, recipe.custom[2].y, recipe.custom[2].z);
        }
        std::printf("  material %-28s %s %ux%u uv%d%s  [%s ]%s\n", mat.name.c_str(),
                    ShaderName(b.shader_id), mat.image.width, mat.image.height, recipe.out_uv_layer,
                    mat.has_alpha ? " alpha" : "", flag_desc.c_str(), cdesc);
        mat_index = (int)materials.size();
        materials.push_back(mat);
        material_by_key[key] = mat_index;
      }
      mesh.material = mat_index;
      mesh.name = materials[mat_index].name + "_b" + std::to_string(bi);

      // skinning (posed)
      if (skin_xf) {
        for (auto& v : mesh.verts) {
          glm::vec3 p(0.f), n(0.f);
          float wsum = 0.f;
          for (int k = 0; k < 4; ++k) {
            float w = v.weights[k];
            if (w <= 0.f) continue;
            uint32_t ji = v.joints[k];
            if (ji >= skin_xf->size()) continue;
            const glm::mat4& m = (*skin_xf)[ji].skin;
            p += w * glm::vec3(m * glm::vec4(v.position, 1.f));
            glm::mat3 nm = glm::transpose(glm::inverse(glm::mat3(m)));
            n += w * (nm * v.normal);
            wsum += w;
          }
          if (wsum > 0.f) {
            v.position = p / wsum;
            float len = glm::length(n);
            v.normal = len > 1e-6f ? n / len : v.normal;
          }
          ++total_verts;
        }
      }
      if (is_head && recipe.kind == ShaderKind::kHead) {
        HeadBatchInfo hb;
        hb.mesh_index = (int)meshes.size();
        hb.recipe = recipe;
        hb.usage_to_texture = usage_tex;
        hb.usage_to_uv = usage_uv;
        hb.material_index = mat_index;
        hb.eye_usage = usage_tex.count(9) ? 9u : 10u;
        hb.brow_usage = usage_tex.count(7) ? 7u : 8u;
        head_batches.push_back(hb);
      }
      meshes.push_back(std::move(mesh));
    }
  };

  for (size_t ci = 0; ci < res.components.size(); ++ci) {
    bake_component(res.components[ci], (int)ci, false, &xforms);
  }

  // carryable (its own skeleton)
  std::vector<JointXform> prop_xforms;
  std::vector<BakedMesh> prop_meshes;
  if (res.prop && res.prop->model) {
    if (res.prop->skeleton) prop_xforms = BuildJointXforms(*res.prop->skeleton, false);
    ResolvedComponent pc;
    pc.info = res.prop_info;
    pc.model = res.prop->model;
    pc.name = res.prop_name;
    pc.slot = "Prop";
    pc.from_closet = !IsStockPackId(res.prop_info.asset_id);
    pc.info.categories = (uint16_t)ComponentCategory::kProp;
    ComputeOverrides(metadata, pc, assets);  // white CUSTOM_0; closet colour table wins
    size_t before = meshes.size();
    bake_component(pc, (int)res.components.size(), true, prop_xforms.empty() ? nullptr : &prop_xforms);
    (void)before;
  }

  std::printf("  %zu vertices skinned\n", total_verts);

  // ---- face frames ----
  struct FrameFile {
    std::string channel;
    int frame;
    std::string file;
  };
  std::vector<FrameFile> layer_files, composite_files;
  {
    fs::path face_dir = args.out_dir / "face";
    fs::create_directories(face_dir);
    struct Chan {
      const char* name;
      int slot;
      uint32_t usage_a, usage_b;
      int tint;
      bool white_in_b;
    };
    // tint index: colors[] index; eyes use the B-white convention
    const Chan chans[] = {{"mouth", 0, 12, 12, 2, false},
                          {"brows", 2, 7, 8, 4, false},
                          {"eyes", 1, 9, 10, 3, true}};
    const glm::vec3 skin = glm::vec3(ColorToFloat4(metadata.colors[0].get()).x,
                                     ColorToFloat4(metadata.colors[0].get()).y,
                                     ColorToFloat4(metadata.colors[0].get()).z);
    for (const Chan& ch : chans) {
      const auto& layers = rep_layers[ch.slot];
      Float4 tc = ColorToFloat4(metadata.colors[ch.tint].get());
      glm::vec3 tint(tc.x, tc.y, tc.z);
      for (size_t f = 0; f < layers.size(); ++f) {
        const texdec::Image& L = layers[f];
        if (!L.ok) continue;
        texdec::Image tinted;
        tinted.width = L.width;
        tinted.height = L.height;
        tinted.rgba.resize(L.rgba.size());
        for (size_t i = 0; i < L.rgba.size(); i += 4) {
          glm::vec4 t(L.rgba[i], L.rgba[i + 1], L.rgba[i + 2], L.rgba[i + 3]);
          t /= 255.f;
          float white = ch.white_in_b ? t.b : t.g;
          float blend = ch.white_in_b ? t.g : t.b;
          glm::vec3 rgb = glm::clamp(t.r * tint + glm::vec3(white) + blend * skin, 0.f, 1.f);
          tinted.rgba[i] = (uint8_t)std::lround(rgb.r * 255.f);
          tinted.rgba[i + 1] = (uint8_t)std::lround(rgb.g * 255.f);
          tinted.rgba[i + 2] = (uint8_t)std::lround(rgb.b * 255.f);
          tinted.rgba[i + 3] = L.rgba[i + 3];
        }
        tinted.ok = true;
        char name[64];
        std::snprintf(name, sizeof(name), "%s_%02zu.png", ch.name, f);
        std::string rel = std::string("face/") + name;
        if (WritePng(args.out_dir / rel, tinted)) {
          layer_files.push_back({ch.name, (int)f, rel});
        }
        // whole-head composite with this channel at frame f
        if (args.face_frames && !head_batches.empty()) {
          for (size_t hbi = 0; hbi < head_batches.size(); ++hbi) {
            HeadBatchInfo hb = head_batches[hbi];
            MaterialRecipe r = hb.recipe;
            Sampler* target = ch.slot == 0 ? &r.mouth : ch.slot == 2 ? &r.brow : &r.eye;
            if (!target->img) continue;
            target->img = &layers[f];
            const BakedMesh& m = meshes[hb.mesh_index];
            texdec::Image baked = BakeMaterial(r, m.verts, m.indices, args.face_frame_size);
            char cname[96];
            std::snprintf(cname, sizeof(cname), "%s_%s_%02zu.png",
                          materials[hb.material_index].name.c_str(), ch.name, f);
            std::string crel = std::string("face/") + cname;
            if (WritePng(args.out_dir / crel, baked)) {
              composite_files.push_back({ch.name, (int)f, crel});
            }
          }
        }
      }
    }
    std::printf("  face layers written: %zu, composites: %zu\n", layer_files.size(),
                composite_files.size());
  }

  // ---- animations ----
  std::vector<AnimExport> anims;
  for (const auto& p : args.anims) {
    auto a = LoadAnimationFile(p);
    if (a) anims.push_back({PathStr(p.stem()), a});
    else std::printf("  WARN: animation %s failed to load\n", PathStr(p).c_str());
  }
  if (!args.anim_dir.empty() && fs::is_directory(args.anim_dir)) {
    std::vector<fs::path> files;
    for (const auto& e : fs::directory_iterator(args.anim_dir)) {
      std::string ext = PathStr(e.path().extension());
      for (auto& c : ext) c = (char)std::tolower((unsigned char)c);
      if (ext == ".avataranimation" || ext == ".anim" || ext == ".bin") files.push_back(e.path());
    }
    std::sort(files.begin(), files.end());
    for (const auto& p : files) {
      auto a = LoadAnimationFile(p);
      if (a) anims.push_back({PathStr(p.stem()), a});
      else std::printf("  WARN: animation %s failed to load\n", PathStr(p).c_str());
    }
  }
  if (args.pack_anims || !args.pack_anim_filters.empty()) {
    auto lower = [](std::string v) {
      for (auto& c : v) c = (char)std::tolower((unsigned char)c);
      return v;
    };
    // "<name>@<rule>": the rule (mid|peak|0..1|N) picks that clip's preview frame.
    std::vector<std::pair<std::string, std::string>> filters;
    for (const auto& f : args.pack_anim_filters) {
      const size_t at = f.find('@');
      if (at == std::string::npos) filters.push_back({lower(f), ""});
      else filters.push_back({lower(f.substr(0, at)), f.substr(at + 1)});
    }
    const auto& infos = assets.pack.asset_infos();
    for (size_t i = 0; i < infos.size(); ++i) {
      if (!(infos[i].categories & ComponentCategory::kAnimation)) continue;
      std::string name = U16ToUtf8(assets.pack.GetAssetNameByIndex(i));
      if (name.empty()) name = "pack_anim_" + std::to_string(i);
      std::string frame_rule;
      if (!args.pack_anims) {
        const std::string lname = lower(name);
        bool hit = false;
        for (const auto& f : filters) {
          if (lname.find(f.first) != std::string::npos) {
            hit = true;
            frame_rule = f.second;
            break;
          }
        }
        if (!hit) continue;
      }
      const uint8_t* buf = nullptr;
      size_t size = 0;
      if (!assets.pack.GetAssetDataByIndex(i, buf, size)) continue;
      auto a = Animation::Load(buf, size, AnimationLoadOption::kElements);
      if (!a) continue;
      anims.push_back({name, a, frame_rule});
    }
  }
  if (res.prop && res.prop->animation) {
    anims.push_back({"Carryable_" + SanitizeName(res.prop_name), res.prop->animation});
  }
  if (!anims.empty()) {
    std::printf("\n=== animations: %zu ===\n", anims.size());
  }

  // ---- icon mode (--gen-icons): one worn-item icon, no exports ----
  if (!args.icon_out.empty()) {
    const AnimExport* clip = nullptr;
    for (const auto& a : anims) {
      if (a.anim && a.anim->pose_counts[0] > 0 && !a.anim->pose_frame_sets[0].frames.empty()) {
        clip = &a;
        break;
      }
    }
    std::vector<PosedMesh> posed;
    const Animation* perform = (res.prop && res.prop->animation && res.prop->animation->pose_counts[0] > 0 &&
                                !res.prop->animation->pose_frame_sets[0].frames.empty())
                                   ? res.prop->animation.get()
                                   : nullptr;
    if (perform) {
      // the avatar performing the prop's own clip at its most expressive frame
      const size_t f = PickPreviewFrame(*perform, "peak");
      posed = PoseMeshes(meshes, PoseSkinMatrices(xforms, *res.skeleton, perform, f));
    } else if (clip) {
      const size_t f = PickPreviewFrame(
          *clip->anim, clip->preview_frame.empty() ? args.preview_frame : clip->preview_frame);
      posed = PoseMeshes(meshes, PoseSkinMatrices(xforms, *res.skeleton, clip->anim.get(), f));
    } else {
      posed = PoseMeshes(meshes, PoseSkinMatrices(xforms, *res.skeleton, nullptr, 0));
    }
    return RenderWornItemIcon(meshes, materials, std::move(posed), res, args) ? 0 : 5;
  }
  // ---- preview renders ----
  if (!args.preview_dir.empty()) {
    fs::create_directories(args.preview_dir);
    const auto t0 = std::chrono::steady_clock::now();
    std::vector<std::vector<PosedMesh>> all_poses;
    std::vector<std::string> names;
    std::vector<std::map<int, const texdec::Image*>> faces;
    // Face animation: the clip's texture track names the eye/brow/mouth layer
    // per frame (XAVATAR_ANIMATED_TEXTURE order: mouth, brow L, brow R, eye L,
    // eye R). Re-composite each head material with that frame's layers, once
    // per distinct combination.
    std::map<std::string, texdec::Image> face_cache;
    size_t face_bakes = 0;
    auto face_for = [&](const Animation* anim, size_t f) {
      std::map<int, const texdec::Image*> ov;
      if (!anim || anim->texture_count < 5 || f >= anim->texture_frame_set.frames.size()) return ov;
      const auto& el = anim->texture_frame_set.frames[f].elements;
      if (el.size() < 5) return ov;
      auto layer = [&](int slot, uint32_t idx) -> const texdec::Image* {
        const auto& L = rep_layers[slot];
        if (L.empty()) return nullptr;
        if (idx >= L.size()) idx = 0;  // runtime clamp: out-of-range -> neutral
        return L[idx].ok ? &L[idx] : nullptr;
      };
      const uint32_t mouth = el[0].layer_index, brow_l = el[1].layer_index,
                     brow_r = el[2].layer_index, eye_l = el[3].layer_index,
                     eye_r = el[4].layer_index;
      for (const auto& hb : head_batches) {
        const uint32_t brow = hb.brow_usage == 8 ? brow_r : brow_l;
        const uint32_t eye = hb.eye_usage == 10 ? eye_r : eye_l;
        if (mouth == 0 && brow == 0 && eye == 0) continue;  // rest face = the baked material
        char key[96];
        std::snprintf(key, sizeof(key), "%d|%u|%u|%u", hb.material_index, mouth, brow, eye);
        auto it = face_cache.find(key);
        if (it == face_cache.end()) {
          MaterialRecipe r = hb.recipe;
          if (r.mouth.img) { if (const auto* L = layer(0, mouth)) r.mouth.img = L; }
          if (r.brow.img) { if (const auto* L = layer(2, brow)) r.brow.img = L; }
          if (r.eye.img) { if (const auto* L = layer(1, eye)) r.eye.img = L; }
          const BakedMesh& hm = meshes[hb.mesh_index];
          it = face_cache.emplace(key, BakeMaterial(r, hm.verts, hm.indices, 256)).first;
          ++face_bakes;
        }
        ov[hb.material_index] = &it->second;
      }
      return ov;
    };
    all_poses.push_back(PoseMeshes(meshes, PoseSkinMatrices(xforms, *res.skeleton, nullptr, 0)));
    names.push_back("T-Pose");
    faces.push_back({});
    for (const auto& a : anims) {
      if (a.anim->pose_counts[0] == 0 || a.anim->pose_frame_sets[0].frames.empty()) continue;
      size_t f = PickPreviewFrame(*a.anim, a.preview_frame.empty() ? args.preview_frame : a.preview_frame);
      all_poses.push_back(PoseMeshes(meshes, PoseSkinMatrices(xforms, *res.skeleton, a.anim.get(), f)));
      names.push_back(SanitizeName(a.name));
      faces.push_back(face_for(a.anim.get(), f));
    }
    // One camera for every pose so switching poses does not re-frame.
    PreviewCamera cam = FrameCamera(all_poses, args.preview_size * 2);
    for (size_t i = 0; i < all_poses.size(); ++i) {
      RenderPreviewImage(meshes, materials, all_poses[i], cam, args.preview_size,
                         args.preview_dir / ("preview_" + names[i] + ".png"), faces[i]);
    }
    const auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(
                        std::chrono::steady_clock::now() - t0).count();
    std::printf("\n=== previews: %zu images (%dpx, %zu face composites) in %lld ms -> %s ===\n",
                all_poses.size(), args.preview_size, face_bakes, (long long)ms,
                PathStr(args.preview_dir).c_str());
  }

  // ---- avatar.json ----
  fs::path json_path = args.out_dir / "avatar.json";
  FILE* jf = _wfopen(json_path.wstring().c_str(), L"wb");
  if (!jf) {
    std::printf("ERROR: cannot write %s\n", PathStr(json_path).c_str());
    return 4;
  }
  Json j(jf);
  j.BeginObj();
  j.KeyStr("format", "rexglue-avatar-export");
  j.KeyInt("version", 1);
  j.Key("source");
  j.BeginObj();
  j.KeyStr("manifest", PathStr(args.manifest));
  j.KeyStr("pack", PathStr(args.toc));
  j.KeyStr("closet", PathStr(args.closet));
  j.EndObj();
  j.Key("axes");
  j.BeginObj();
  j.KeyStr("units", "meters");
  j.KeyStr("up", "+Y");
  j.KeyStr("forward", "+Z");
  j.KeyStr("left", "+X");
  j.KeyStr("handedness", "right");
  j.KeyStr("uv_origin", "top-left");
  j.EndObj();
  j.Key("avatar");
  j.BeginObj();
  j.KeyStr("body_type", res.body_type == BodyType::kMale     ? "male"
                        : res.body_type == BodyType::kFemale ? "female"
                                                             : "unknown");
  j.KeyNum("height_factor", metadata.height_factor.get(), 4);
  j.KeyNum("weight_factor", metadata.weight_factor.get(), 4);
  j.KeyBool("scale_applied", !args.no_scale);
  j.Key("colors");
  j.BeginArr();
  for (int i = 0; i < 9; ++i) {
    char buf[16];
    std::snprintf(buf, sizeof(buf), "%08X", metadata.colors[i].get());
    j.S(buf);
  }
  j.EndArr();
  j.Key("color_names");
  j.BeginArr();
  static const char* kColorNames[9] = {"skin", "hair", "mouth", "iris", "eyebrow", "eye_shadow", "facial_hair", "skin_feature_1", "skin_feature_2"};
  for (int i = 0; i < 9; ++i) j.S(kColorNames[i]);
  j.EndArr();
  j.EndObj();

  // skeleton
  j.Key("skeleton");
  j.BeginObj();
  j.KeyInt("version", args.skeleton_version);
  j.Key("joints");
  j.BeginArr();
  for (size_t i = 0; i < joint_count; ++i) {
    const Joint& jt = res.skeleton->joints[i];
    j.Comma();
    j.BeginObj();
    j.KeyStr("name", JointName(i, joint_count));
    j.KeyInt("parent", jt.parent_index == 255 ? -1 : (long long)jt.parent_index);
    glm::vec3 bw = glm::vec3(xforms[i].bind_world[3]);
    glm::vec3 rw = glm::vec3(xforms[i].rest_world[3]);
    glm::quat rq = glm::normalize(glm::quat_cast(glm::mat3(xforms[i].rest_world)));
    // orthonormalize (scale lives in the matrix)
    glm::mat3 m3(xforms[i].rest_world);
    for (int c = 0; c < 3; ++c) m3[c] = glm::normalize(m3[c]);
    rq = glm::normalize(glm::quat_cast(m3));
    float bwv[3] = {bw.x, bw.y, bw.z};
    float rwv[3] = {rw.x, rw.y, rw.z};
    float rqv[4] = {rq.x, rq.y, rq.z, rq.w};
    float rlt[3] = {xforms[i].rest_local_t.x * xforms[i].parent_world_scale.x,
                    xforms[i].rest_local_t.y * xforms[i].parent_world_scale.y,
                    xforms[i].rest_local_t.z * xforms[i].parent_world_scale.z};
    float sc[3] = {xforms[i].scale.x, xforms[i].scale.y, xforms[i].scale.z};
    j.Key("bind_world");
    j.Vec(bwv, 3, 5);
    j.Key("rest_world");
    j.Vec(rwv, 3, 5);
    j.Key("rest_world_rot");
    j.Vec(rqv, 4, 6);
    j.Key("rest_local");
    j.Vec(rlt, 3, 5);
    j.Key("scale");
    j.Vec(sc, 3, 4);
    j.EndObj();
  }
  j.EndArr();
  j.EndObj();

  // components
  j.Key("components");
  j.BeginArr();
  for (size_t ci = 0; ci < res.components.size(); ++ci) {
    const auto& c = res.components[ci];
    j.Comma();
    j.BeginObj();
    j.KeyInt("index", (long long)ci);
    j.KeyStr("slot", c.slot);
    j.KeyStr("guid", c.info.asset_id.to_string());
    j.KeyStr("name", c.name);
    j.KeyInt("categories", c.info.categories.get());
    j.KeyStr("category_names", CategoryNames(c.info.categories.get()));
    j.KeyStr("source", c.from_closet ? "closet" : "pack");
    j.Key("notes");
    j.BeginArr();
    for (const auto& n : c.notes) j.S(n);
    j.EndArr();
    j.EndObj();
  }
  if (res.prop && res.prop->model) {
    j.Comma();
    j.BeginObj();
    j.KeyInt("index", (long long)res.components.size());
    j.KeyStr("slot", "Prop");
    j.KeyStr("guid", res.prop_info.asset_id.to_string());
    j.KeyStr("name", res.prop_name);
    j.KeyInt("categories", res.prop_info.categories.get());
    j.KeyStr("category_names", "Prop");
    j.KeyStr("source", IsStockPackId(res.prop_info.asset_id) ? "pack" : "closet");
    j.Key("notes");
    j.BeginArr();
    j.EndArr();
    j.EndObj();
  }
  j.EndArr();

  // materials
  j.Key("materials");
  j.BeginArr();
  for (const auto& m : materials) {
    j.Comma();
    j.BeginObj();
    j.KeyStr("name", m.name);
    j.KeyStr("diffuse", m.file);
    j.KeyInt("shader", m.shader_id);
    j.KeyStr("shader_name", ShaderName(m.shader_id));
    j.KeyBool("has_alpha", m.has_alpha);
    j.KeyBool("alpha_mask", m.has_alpha);
    j.KeyBool("double_sided", false);
    j.KeyInt("uv_layer", m.uv_layer);
    j.KeyStr("component_guid", m.component_guid);
    j.EndObj();
  }
  j.EndArr();

  // meshes
  j.Key("meshes");
  j.BeginArr();
  for (const auto& m : meshes) {
    j.Comma();
    j.BeginObj();
    j.KeyStr("name", m.name);
    j.KeyInt("component", m.component);
    j.KeyInt("material", m.material);
    j.KeyBool("is_prop", m.is_prop);
    j.KeyInt("vertex_count", (long long)m.verts.size());
    j.KeyInt("triangle_count", (long long)(m.indices.size() / 3));
    j.KeyInt("uv_count", m.uv_count);
    const int out_uv = materials[m.material].uv_layer;
    j.Key("positions");
    j.BeginArr();
    for (const auto& v : m.verts) {
      j.Num(v.position.x, 5);
      j.Num(v.position.y, 5);
      j.Num(v.position.z, 5);
    }
    j.EndArr();
    j.Key("normals");
    j.BeginArr();
    for (const auto& v : m.verts) {
      j.Num(v.normal.x, 4);
      j.Num(v.normal.y, 4);
      j.Num(v.normal.z, 4);
    }
    j.EndArr();
    j.Key("uv");
    j.BeginArr();
    for (const auto& v : m.verts) {
      j.Num(v.uvs[out_uv].x, 5);
      j.Num(v.uvs[out_uv].y, 5);
    }
    j.EndArr();
    if (m.uv_count > 1) {
      j.Key("uv_layers");
      j.BeginArr();
      for (uint32_t l = 0; l < m.uv_count && l < 6; ++l) {
        j.Comma();
        j.BeginArr();
        for (const auto& v : m.verts) {
          j.Num(v.uvs[l].x, 5);
          j.Num(v.uvs[l].y, 5);
        }
        j.EndArr();
      }
      j.EndArr();
    }
    j.Key("colors");
    j.BeginArr();
    for (const auto& v : m.verts) {
      j.Int((v.color >> 16) & 0xFF);
      j.Int((v.color >> 8) & 0xFF);
      j.Int(v.color & 0xFF);
      j.Int((v.color >> 24) & 0xFF);
    }
    j.EndArr();
    j.Key("joints");
    j.BeginArr();
    for (const auto& v : m.verts) {
      for (int k = 0; k < 4; ++k) j.Int(v.joints[k]);
    }
    j.EndArr();
    j.Key("weights");
    j.BeginArr();
    for (const auto& v : m.verts) {
      for (int k = 0; k < 4; ++k) j.Num(v.weights[k], 4);
    }
    j.EndArr();
    j.Key("indices");
    j.BeginArr();
    for (uint16_t idx : m.indices) j.Int(idx);
    j.EndArr();
    j.EndObj();
  }
  j.EndArr();

  // prop skeleton
  j.Key("prop_skeleton");
  if (!prop_xforms.empty() && res.prop && res.prop->skeleton) {
    j.BeginObj();
    j.Key("joints");
    j.BeginArr();
    for (size_t i = 0; i < prop_xforms.size(); ++i) {
      const Joint& jt = res.prop->skeleton->joints[i];
      j.Comma();
      j.BeginObj();
      j.KeyStr("name", "prop_joint_" + std::to_string(i));
      j.KeyInt("parent", jt.parent_index == 255 ? -1 : (long long)jt.parent_index);
      glm::vec3 rw = glm::vec3(prop_xforms[i].rest_world[3]);
      glm::mat3 m3(prop_xforms[i].rest_world);
      for (int c = 0; c < 3; ++c) m3[c] = glm::normalize(m3[c]);
      glm::quat rq = glm::normalize(glm::quat_cast(m3));
      float rwv[3] = {rw.x, rw.y, rw.z};
      float rqv[4] = {rq.x, rq.y, rq.z, rq.w};
      float rlt[3] = {prop_xforms[i].rest_local_t.x, prop_xforms[i].rest_local_t.y,
                      prop_xforms[i].rest_local_t.z};
      j.Key("rest_world");
      j.Vec(rwv, 3, 5);
      j.Key("rest_world_rot");
      j.Vec(rqv, 4, 6);
      j.Key("rest_local");
      j.Vec(rlt, 3, 5);
      j.EndObj();
    }
    j.EndArr();
    j.EndObj();
  } else {
    j.Null();
  }

  // face
  j.Key("face");
  j.BeginObj();
  j.Key("head_materials");
  j.BeginArr();
  {
    std::set<int> seen;
    for (const auto& hb : head_batches) {
      if (seen.insert(hb.material_index).second) j.S(materials[hb.material_index].name);
    }
  }
  j.EndArr();
  j.Key("slots");
  j.BeginObj();
  static const char* kSlotNames[6] = {"mouth", "eyes", "brows", "face_paint", "eye_shadow", "face"};
  for (int s = 0; s < 6; ++s) {
    if (!res.replacement_textures[s]) continue;
    j.Key(kSlotNames[s]);
    j.BeginObj();
    j.KeyStr("guid", res.replacement_ids[s].to_string());
    j.KeyStr("name", assets.Name(res.replacement_ids[s]));
    j.KeyInt("layers", res.replacement_textures[s]->layer_count);
    j.KeyInt("width", res.replacement_textures[s]->width);
    j.KeyInt("height", res.replacement_textures[s]->height);
    j.EndObj();
  }
  j.EndObj();
  j.Key("layer_files");
  j.BeginArr();
  for (const auto& f : layer_files) {
    j.Comma();
    j.BeginObj();
    j.KeyStr("channel", f.channel);
    j.KeyInt("frame", f.frame);
    j.KeyStr("file", f.file);
    j.EndObj();
  }
  j.EndArr();
  j.Key("composite_files");
  j.BeginArr();
  for (const auto& f : composite_files) {
    j.Comma();
    j.BeginObj();
    j.KeyStr("channel", f.channel);
    j.KeyInt("frame", f.frame);
    j.KeyStr("file", f.file);
    j.EndObj();
  }
  j.EndArr();
  j.Key("layer_names");
  j.BeginObj();
  j.Key("eyes");
  j.BeginArr();
  for (const char* n : {"NEUTRAL", "SAD", "ANGRY", "CONFUSED", "LAUGHING", "SHOCKED", "HAPPY", "YAWNING", "SLEEPING", "LOOK_UP", "LOOK_DOWN", "LOOK_OUTER", "LOOK_INNER", "BLINK"}) j.S(n);
  j.EndArr();
  j.Key("mouth");
  j.BeginArr();
  for (const char* n : {"NEUTRAL", "SAD", "ANGRY", "CONFUSED", "LAUGHING", "SHOCKED", "HAPPY", "PHONETIC_O", "PHONETIC_AI", "PHONETIC_EE", "PHONETIC_FV", "PHONETIC_W", "PHONETIC_L", "PHONETIC_DTH"}) j.S(n);
  j.EndArr();
  j.Key("brows");
  j.BeginArr();
  for (const char* n : {"NEUTRAL", "SAD", "ANGRY", "CONFUSED", "RAISED"}) j.S(n);
  j.EndArr();
  j.EndObj();
  j.EndObj();

  // animations
  j.Key("animations");
  j.BeginArr();
  for (const auto& a : anims) WriteAnimation(j, a, xforms, joint_count);
  j.EndArr();

  // log
  j.Key("log");
  j.BeginArr();
  for (const auto& l : res.log) j.S(l);
  j.EndArr();

  j.EndObj();
  std::fputc('\n', jf);
  std::fclose(jf);

  std::printf("\n=== Done ===\n  %s\n  %zu meshes, %zu materials, %zu animations\n",
              PathStr(json_path).c_str(), meshes.size(), materials.size(), anims.size());
  return 0;
}

int RunFromArgs(const std::vector<std::string>& argv) {
  Args args;
  std::vector<std::string> positionals;
  for (size_t i = 0; i < argv.size(); ++i) {
    const std::string& a = argv[i];
    auto next = [&](fs::path& dst) {
      if (i + 1 < argv.size()) dst = PathFromUtf8(argv[++i]);
    };
    if (a == "--toc") next(args.toc);
    else if (a == "--legacy-toc") next(args.legacy_toc);
    else if (a == "--closet") next(args.closet);
    else if (a == "--skeleton-version" && i + 1 < argv.size()) args.skeleton_version = std::atoi(argv[++i].c_str());
    else if (a == "--no-scale") args.no_scale = true;
    else if (a == "--no-prop") args.want_prop = false;
    else if (a == "--avatar-info") args.info_only = true;
    else if (a == "--bake-size" && i + 1 < argv.size()) args.bake_size = std::max(64, std::atoi(argv[++i].c_str()));
    else if (a == "--face-frames") {
      args.face_frames = true;
      if (i + 1 < argv.size() && std::isdigit((unsigned char)argv[i + 1][0])) {
        args.face_frame_size = std::max(64, std::atoi(argv[++i].c_str()));
      }
    } else if (a == "--eye-whites") args.eye_whites = true;
    else if (a == "--anim" && i + 1 < argv.size()) args.anims.push_back(PathFromUtf8(argv[++i]));
    else if (a == "--anim-dir") next(args.anim_dir);
    else if (a == "--pack-anims") args.pack_anims = true;
    else if (a == "--pack-anim" && i + 1 < argv.size()) args.pack_anim_filters.push_back(argv[++i]);
    else if (a == "--preview-dir") next(args.preview_dir);
    else if (a == "--preview-size" && i + 1 < argv.size()) args.preview_size = std::max(64, std::atoi(argv[++i].c_str()));
    else if (a == "--preview-frame" && i + 1 < argv.size()) args.preview_frame = argv[++i];
    else if (a == "--list-anims") args.list_anims = true;
    else if (a == "--help" || a == "-h") {
      PrintUsage();
      return 0;
    } else positionals.push_back(a);
  }
  if (positionals.empty()) {
    PrintUsage();
    return 1;
  }
  args.manifest = PathFromUtf8(positionals[0]);
  if (positionals.size() >= 2) args.out_dir = PathFromUtf8(positionals[1]);
  else args.out_dir = args.manifest.parent_path() / "export";

  // Defaults: the shared userdata layout (…\userdata\avatars\avatar_manifest.bin
  // next to …\userdata\avatarpack\AvatarAssetPack.toc).
  if (args.toc.empty()) {
    fs::path userdata = args.manifest.parent_path().parent_path();
    fs::path cand = userdata / "avatarpack" / "AvatarAssetPack.toc";
    if (fs::exists(cand)) args.toc = cand;
  }
  if (args.toc.empty() || !fs::exists(args.toc)) {
    std::printf("ERROR: asset pack not found; pass --toc <AvatarAssetPack.toc>\n");
    return 1;
  }
  if (args.legacy_toc.empty()) {
    fs::path cand = args.toc.parent_path() / "AvatarAssetPackLegacyV1.toc";
    if (fs::exists(cand)) args.legacy_toc = cand;
  }
  if (args.closet.empty()) {
    fs::path cand = args.toc.parent_path() / "closet";
    if (fs::is_directory(cand)) args.closet = cand;
  }
  return Run(args);
}


// ---------------------------------------------------------------------------
// --prop-icons <closet_dir> [--icons-out <dir>] [--size N] [--report <csv>]
//              [--filter <substr>] [--limit N]
// Renders every closet carryable (category kProp) alone, posed at the middle
// of its own clip (props spawn in: frame 0 is often scale 0), with its colour
// table applied, onto a transparent plate -> icons/<guid>.png for the editor's
// XamAvatarGetAssetIcon; writes a CSV survey (size, darkness, decode status).
// ---------------------------------------------------------------------------

// ---------------------------------------------------------------------------
// Prop scene: the closet prop's textures baked and its batches unpacked as
// BakedMesh/BakedMaterial (shared by --prop-icons and --gen-icons).
// ---------------------------------------------------------------------------
struct PropScene {
  std::vector<BakedMaterial> materials;
  std::vector<BakedMesh> meshes;
  bool has_table = false;
  bool empty_intensity = false;
  std::string shaders;
  size_t verts = 0, tris = 0;
};

static void BuildPropScene(const std::vector<uint8_t>& bytes, const Model& model, PropScene& s) {
  // colour table (block 7)
  std::map<uint32_t, Float4> overrides;
  overrides[22] = Float4{1, 1, 1, 1};
  {
    const uint8_t* table = nullptr;
    size_t table_size = 0;
    if (strb::GetSTRBBlock(bytes.data(), bytes.size(), strb::STRBBlockId::kCustomColorTable, table, table_size) &&
        table_size >= 4 + 3 * 8) {
      auto read_u32 = [](const uint8_t* q, bool le) -> uint32_t {
        return le ? (uint32_t(q[0]) | (uint32_t(q[1]) << 8) | (uint32_t(q[2]) << 16) | (uint32_t(q[3]) << 24))
                  : ((uint32_t(q[0]) << 24) | (uint32_t(q[1]) << 16) | (uint32_t(q[2]) << 8) | uint32_t(q[3]));
      };
      const bool le = read_u32(table, true) <= 0xFFFFu;
      uint32_t colors[3];
      bool any = false;
      for (size_t i = 0; i < 3; ++i) { colors[i] = read_u32(&table[4 + i * 8], le); any |= colors[i] != 0; }
      if (any) {
        for (size_t i = 0; i < 3; ++i) overrides[22 + (uint32_t)i] = ColorToFloat4(colors[i]);
        s.has_table = true;
      }
    }
  }
  // decode textures, bake materials, build meshes
  std::vector<texdec::Image> images(model.textures.size());
  for (size_t i = 0; i < images.size(); ++i) images[i] = texdec::DecodeTexture(model.textures[i].texture);
  for (size_t bi = 0; bi < model.triangle_batches.size(); ++bi) {
    const TriangleBatch& b = model.triangle_batches[bi];
    if (b.triangle_count == 0 || b.vertices.empty()) continue;
    std::map<uint32_t, int> usage_tex, usage_uv;
    std::map<uint32_t, uint32_t> usage_flags;
    std::map<uint32_t, Float4> usage_const;
    for (const auto& sp : b.shader_parameters) {
      if (sp.type == ShaderParameterType::kTexture) {
        usage_tex[sp.usage] = sp.texture.index; usage_uv[sp.usage] = sp.texture.uv_layer; usage_flags[sp.usage] = sp.texture.flags;
      } else if (sp.type != ShaderParameterType::kInvalid) {
        usage_const[sp.usage] = Float4{sp.constant_values[0], sp.constant_values[1], sp.constant_values[2], sp.constant_values[3]};
      }
    }
    for (const auto& ov : overrides) usage_const[ov.first] = ov.second;
    MaterialRecipe recipe;
    recipe.kind = ShaderKind::kBody;
    recipe.shader_id = b.shader_id;
    auto tex_for = [&](uint32_t usage, Sampler& smp) {
      auto it = usage_tex.find(usage);
      if (it == usage_tex.end()) return;
      smp.uv_layer = std::min(5, std::max(0, usage_uv[usage]));
      smp.flags = usage_flags[usage];
      if (it->second >= 0 && it->second < (int)images.size() && images[it->second].ok) smp.img = &images[it->second];
      else if (usage == 2) s.empty_intensity = true;
    };
    tex_for(1, recipe.color); tex_for(2, recipe.intensity); tex_for(3, recipe.decal);
    for (int k = 0; k < 3; ++k) { auto it = usage_const.find(22 + k); recipe.custom[k] = it != usage_const.end() ? it->second : Float4{1, 1, 1, 1}; }
    recipe.out_uv_layer = recipe.color.img ? recipe.color.uv_layer : 0;
    s.shaders += (s.shaders.empty() ? "" : "|") + std::string(ShaderName(b.shader_id));
    BakedMesh mesh;
    mesh.component = 0; mesh.is_prop = true; mesh.uv_count = b.uv_count;
    mesh.verts.resize(b.vertices.size());
    for (size_t vi = 0; vi < b.vertices.size(); ++vi) UnpackSkin(b.vertices[vi], mesh.verts[vi], b.uv_count);
    mesh.indices = b.indices;
    BakedMaterial mat;
    mat.name = "Prop_" + std::to_string(bi);
    mat.shader_id = b.shader_id;
    mat.uv_layer = recipe.out_uv_layer;
    mat.image = BakeMaterial(recipe, mesh.verts, mesh.indices, 256);
    for (size_t i = 3; i < mat.image.rgba.size(); i += 4) if (mat.image.rgba[i] < 128) { mat.has_alpha = true; break; }
    mesh.material = (int)s.materials.size();
    s.materials.push_back(std::move(mat));
    s.verts += mesh.verts.size(); s.tris += mesh.indices.size() / 3;
    s.meshes.push_back(std::move(mesh));
  }
}

int RunPropIcons(const std::vector<std::string>& argv) {
  fs::path closet_dir, icons_out, report;
  int size = 128, limit = -1;
  std::string filter;
  std::vector<std::string> pos;
  for (size_t i = 0; i < argv.size(); ++i) {
    if (argv[i] == "--icons-out" && i + 1 < argv.size()) icons_out = PathFromUtf8(argv[++i]);
    else if (argv[i] == "--report" && i + 1 < argv.size()) report = PathFromUtf8(argv[++i]);
    else if (argv[i] == "--size" && i + 1 < argv.size()) size = std::max(16, std::atoi(argv[++i].c_str()));
    else if (argv[i] == "--limit" && i + 1 < argv.size()) limit = std::atoi(argv[++i].c_str());
    else if (argv[i] == "--filter" && i + 1 < argv.size()) filter = argv[++i];
    else pos.push_back(argv[i]);
  }
  if (pos.empty()) {
    std::printf("usage: avatarextract --prop-icons <closet_dir> [--icons-out <dir>] [--size N] "
                "[--report <csv>] [--filter <substr>] [--limit N]\n");
    return 1;
  }
  closet_dir = PathFromUtf8(pos[0]);
  if (icons_out.empty()) icons_out = closet_dir / "icons";
  GetCloset().Load(closet_dir);
  std::printf("(mode: prop icons)\ncloset: %s (%zu items)\n", PathStr(closet_dir).c_str(),
              GetCloset().items().size());
  fs::create_directories(icons_out);
  FILE* csv = nullptr;
  if (!report.empty()) {
    csv = _wfopen(report.wstring().c_str(), L"wb");
    if (csv) std::fprintf(csv, "guid\tname\tstatus\tverts\ttris\tbatches\tframes\tbbox_x\tbbox_y\tbbox_z\tdark_frac\tcolor_table\tempty_intensity\tshaders\n");
  }
  auto lower = [](std::string v) { for (auto& c : v) c = (char)std::tolower((unsigned char)c); return v; };
  const std::string lfilter = lower(filter);
  size_t done = 0, ok = 0, failed = 0, dark = 0;
  const auto t0 = std::chrono::steady_clock::now();
  for (const ClosetItem& item : GetCloset().items()) {
    if (!(item.categories & ComponentCategory::kProp)) continue;
    if (!lfilter.empty() && lower(item.name).find(lfilter) == std::string::npos) continue;
    if (limit >= 0 && (int)done >= limit) break;
    ++done;
    const std::string guid = item.id.to_string();
    std::string status = "ok";
    std::vector<uint8_t> bytes;
    std::shared_ptr<Prop> prop;
    std::shared_ptr<Model> model;
    if (!GetCloset().ReadItemBytes(item.id, bytes) || bytes.empty()) {
      status = "unreadable";
    } else {
      PropLoadOptions opts{};
      opts.model = ModelLoadOption::kNone;
      opts.skeleton = SkeletonLoadOption::kNone;
      opts.animation = AnimationLoadOption::kElements;
      opts.blend_shape = BlendShapeLoadOption::kNone;
      prop = Prop::Load(bytes.data(), bytes.size(), opts);
      model = prop ? prop->model : Model::Load(bytes.data(), bytes.size(), ModelLoadOption::kNone);
      if (!model) status = "no-model";
      else if (!prop) status = "no-skeleton";
      else if (!prop->animation) status = "no-animation";
    }
    if (!model) {
      ++failed;
      if (csv) std::fprintf(csv, "%s\t%s\t%s\t0\t0\t0\t0\t0\t0\t0\t0\t0\t0\t\n", guid.c_str(), item.name.c_str(), status.c_str());
      continue;
    }
    PropScene scene;
    BuildPropScene(bytes, *model, scene);
    std::vector<BakedMaterial>& materials = scene.materials;
    std::vector<BakedMesh>& meshes = scene.meshes;
    const bool has_table = scene.has_table;
    const bool empty_intensity = scene.empty_intensity;
    const std::string& shaders = scene.shaders;
    const size_t verts = scene.verts, tris = scene.tris;
    // pose at the middle of the carryable clip (spawn-in props are at scale 0 at frame 0)
    std::vector<PosedMesh> posed;
    size_t frames = 0;
    if (prop && prop->skeleton) {
      auto xf = BuildJointXforms(*prop->skeleton, false);
      const Animation* a = prop->animation.get();
      size_t f = 0;
      if (a && a->pose_counts[1] > 0 && !a->pose_frame_sets[1].frames.empty()) {
        frames = a->frame_count;
        f = a->frame_count / 2;
      } else {
        a = nullptr;
      }
      // carryable meshes are flagged is_prop (PoseMeshes keeps them at rest): pose explicitly
      std::vector<glm::mat4> skin = PoseSkinMatrices(xf, *prop->skeleton, a, f, 1);
      for (auto& m : meshes) m.is_prop = false;
      posed = PoseMeshes(meshes, skin);
      for (auto& m : meshes) m.is_prop = true;
    } else {
      posed.resize(meshes.size());
      for (size_t mi = 0; mi < meshes.size(); ++mi) {
        for (const auto& v : meshes[mi].verts) { posed[mi].pos.push_back(v.position); posed[mi].nrm.push_back(v.normal); }
      }
    }
    // bbox of the posed prop
    glm::vec3 lo(1e9f), hi(-1e9f);
    for (const auto& pm : posed) for (const auto& q : pm.pos) { lo = glm::min(lo, q); hi = glm::max(hi, q); }
    glm::vec3 ext = hi - lo;
    PreviewCamera cam;
    {
      // frame on x/y extents with a square margin
      float span = std::max(ext.x, ext.y) * 1.15f;
      if (!(span > 1e-4f)) span = 0.1f;
      cam.center = glm::vec2((lo.x + hi.x) * 0.5f, (lo.y + hi.y) * 0.5f);
      cam.scale = (float)(size * 2) / span;
      cam.feet_y = lo.y;
      cam.width = ext.x;
    }
    const fs::path png = icons_out / (guid + ".png");
    std::map<int, const texdec::Image*> no_overrides;
    RenderPreviewImage(meshes, materials, posed, cam, size, png, no_overrides, true);
    // darkness of the rendered prop (covered pixels only)
    float dark_frac = 0.f;
    {
      texdec::Image img;  // re-read is wasteful; measure on the baked textures instead
      size_t dark_px = 0, all_px = 0;
      for (const auto& m : materials) {
        for (size_t i = 0; i + 3 < m.image.rgba.size(); i += 4) {
          if (m.image.rgba[i + 3] < 128) continue;
          ++all_px;
          if ((int)m.image.rgba[i] + m.image.rgba[i + 1] + m.image.rgba[i + 2] < 60) ++dark_px;
        }
      }
      dark_frac = all_px ? (float)dark_px / all_px : 0.f;
    }
    if (dark_frac > 0.6f) ++dark;
    ++ok;
    if (csv) {
      std::fprintf(csv, "%s\t%s\t%s\t%zu\t%zu\t%zu\t%zu\t%.3f\t%.3f\t%.3f\t%.3f\t%d\t%d\t%s\n", guid.c_str(),
                   item.name.c_str(), status.c_str(), verts, tris, meshes.size(), frames, ext.x, ext.y, ext.z,
                   dark_frac, has_table ? 1 : 0, empty_intensity ? 1 : 0, shaders.c_str());
    }
    if (done % 200 == 0) std::printf("  %zu props...\n", done);
  }
  if (csv) std::fclose(csv);
  const auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - t0).count();
  std::printf("props: %zu processed, %zu rendered, %zu failed, %zu mostly-dark, in %lld ms -> %s\n", done, ok,
              failed, dark, (long long)ms, PathStr(icons_out).c_str());
  return 0;
}

// ---------------------------------------------------------------------------
// --gen-icons <closet_dir> [--toc <AvatarAssetPack.toc>] [--mannequins <dir>]
//             [--icons-out <dir>] [--size N] [--filter <substr>] [--limit N]
//             [--force] [--report <tsv>] [--yaw DEG] [--pitch DEG]
//
// Fill-in icons for closet items that have no store art. Items
// that already have an icon are never touched; every icon written here is
// listed in <closet>/icons_generated.tsv so --closet-icons can replace a
// stand-in with real art later (and --force regenerates only stand-ins).
//   props (kProp): the prop alone, posed at its rest frame (ChoosePropFrame),
//                  3/4 view, transparent background: the marketplace style
//                  for pets and carryables.
//   everything else: worn by the blank white mannequin (Stand pose, as the
//                  Dry Cleaner previews), framed on the item region with
//                  category context (RenderWornItemIcon in Run icon mode).
// ---------------------------------------------------------------------------
struct PropFrameChoice {
  size_t frame = 0;
  size_t frames = 0;
  float motion = 0.f;   // smoothed joint motion at the chosen frame
  float extent = 0.f;   // joint-cloud extent at the chosen frame
  const char* why = "bind";
};

// The frame that reads as the prop: fully spawned (root scale), fully
// extended (joint-cloud extent near the clip maximum), and as still as the
// clip gets there (smoothed joint motion), with the intro (first fifth)
// discouraged so a run-in or unfold settles first. The pick is then centred
// in its low-motion run so a hold, not its edge, is shown.
static PropFrameChoice ChoosePropFrame(const Skeleton& skel, const std::vector<JointXform>& xf,
                                       const Animation* a, const std::vector<BakedMesh>& meshes) {
  PropFrameChoice c;
  if (!a || a->pose_counts[1] == 0 || a->pose_frame_sets[1].frames.empty()) return c;
  const size_t n = std::min<size_t>(a->frame_count, a->pose_frame_sets[1].frames.size());
  if (n == 0) return c;
  c.frames = n;
  // vertex mass per joint (dominant weight): what a frame still shows when
  // some joints are scaled to nothing
  std::vector<float> joint_mass(xf.size(), 0.f);
  for (const auto& m : meshes) {
    for (const auto& v : m.verts) {
      int best = -1;
      float bw = 0.f;
      for (int k = 0; k < 4; ++k) {
        if (v.weights[k] > bw && v.joints[k] < joint_mass.size()) {
          bw = v.weights[k];
          best = (int)v.joints[k];
        }
      }
      if (best >= 0) joint_mass[(size_t)best] += 1.f;
    }
  }
  std::vector<std::vector<glm::vec3>> pos(n);
  std::vector<float> extent(n, 0.f), mass(n, 0.f), motion(n, 0.f);
  for (size_t f = 0; f < n; ++f) {
    const auto skin = PoseSkinMatrices(xf, skel, a, f, 1);
    pos[f].resize(skin.size());
    glm::vec3 lo(1e9f), hi(-1e9f);
    for (size_t j = 0; j < skin.size(); ++j) {
      const glm::mat4 world = skin[j] * xf[j].bind_world;
      pos[f][j] = glm::vec3(world[3]);
      lo = glm::min(lo, pos[f][j]);
      hi = glm::max(hi, pos[f][j]);
      const float sc = std::cbrt(std::fabs(glm::determinant(glm::mat3(world))));
      if (sc >= 0.1f && j < joint_mass.size()) mass[f] += joint_mass[j];
    }
    extent[f] = skin.empty() ? 0.f : glm::length(hi - lo);
  }
  float max_extent = 0.f, max_mass = 0.f;
  for (size_t f = 0; f < n; ++f) {
    max_extent = std::max(max_extent, extent[f]);
    max_mass = std::max(max_mass, mass[f]);
  }
  for (size_t f = 1; f < n; ++f) {
    float m = 0.f;
    for (size_t j = 0; j < pos[f].size() && j < pos[f - 1].size(); ++j) {
      m += glm::length(pos[f][j] - pos[f - 1][j]);
    }
    motion[f] = m;
  }
  if (n > 1) motion[0] = motion[1];
  std::vector<float> smooth(n, 0.f);
  float max_smooth = 0.f;
  for (size_t f = 0; f < n; ++f) {
    float sum = 0.f;
    int cnt = 0;
    for (int d = -2; d <= 2; ++d) {
      const long long g = (long long)f + d;
      if (g < 0 || g >= (long long)n) continue;
      sum += motion[(size_t)g];
      ++cnt;
    }
    smooth[f] = cnt ? sum / cnt : 0.f;
    max_smooth = std::max(max_smooth, smooth[f]);
  }
  // Candidates: spawned (no joint collapsed to scale 0). The clip's spread
  // is deliberately not a criterion: multi-part props (spawn crates, flying
  // parts, unfolding wings) are most spread out exactly when they read worst.
  std::vector<char> ok(n, 0);
  size_t candidates = 0;
  (void)max_extent;
  for (size_t f = 0; f < n; ++f) {
    ok[f] = max_mass <= 0.f || mass[f] >= 0.6f * max_mass;  // not (mostly) despawned
    candidates += ok[f];
  }
  if (candidates == 0) {
    for (size_t f = 0; f < n; ++f) ok[f] = 1;
  }
  size_t best = n / 2;
  float best_score = 1e30f;
  for (size_t f = 0; f < n; ++f) {
    if (!ok[f]) continue;
    float score = max_smooth > 1e-9f ? smooth[f] / max_smooth : 0.f;
    score += 0.15f * (1.f - (float)f / (float)n);  // the settled state: later holds win
    if (f < n / 5) score += 0.10f;                  // intro: settle first
    if (score < best_score) {
      best_score = score;
      best = f;
    }
  }
  // centre the pick in its low-motion run
  const float band = smooth[best] * 1.25f + 1e-6f;
  size_t lo = best, hi = best;
  while (lo > 0 && ok[lo - 1] && smooth[lo - 1] <= band) --lo;
  while (hi + 1 < n && ok[hi + 1] && smooth[hi + 1] <= band) ++hi;
  c.frame = (lo + hi) / 2;
  c.motion = smooth[c.frame];
  c.extent = extent[c.frame];
  c.why = candidates ? "rest" : "rest-unfiltered";
  return c;
}

static bool RenderPropIcon(const std::vector<uint8_t>& bytes, int size, float yaw, float pitch,
                           const fs::path& png, std::string& status, PropFrameChoice& choice) {
  PropLoadOptions opts{};
  opts.model = ModelLoadOption::kNone;
  opts.skeleton = SkeletonLoadOption::kNone;
  opts.animation = AnimationLoadOption::kElements;
  opts.blend_shape = BlendShapeLoadOption::kNone;
  std::shared_ptr<Prop> prop = Prop::Load(bytes.data(), bytes.size(), opts);
  std::shared_ptr<Model> model =
      prop ? prop->model : Model::Load(bytes.data(), bytes.size(), ModelLoadOption::kNone);
  if (!model) {
    status = "no-model";
    return false;
  }
  PropScene scene;
  BuildPropScene(bytes, *model, scene);
  if (scene.meshes.empty()) {
    status = "no-geometry";
    return false;
  }
  std::vector<PosedMesh> posed;
  // Framing ignores geometry the pose scales away (a spawn crate shrunk to a
  // point, parts hidden by scale 0): a vertex counts when its blended joint
  // scale is still material.
  std::vector<std::vector<char>> visible(scene.meshes.size());
  if (prop && prop->skeleton) {
    const auto xf = BuildJointXforms(*prop->skeleton, false);
    const Animation* a = prop->animation.get();
    choice = ChoosePropFrame(*prop->skeleton, xf, a, scene.meshes);
    if (choice.frames == 0) a = nullptr;
    const auto skin = PoseSkinMatrices(xf, *prop->skeleton, a, choice.frame, 1);
    std::vector<float> jscale(skin.size(), 1.f);
    for (size_t j = 0; j < skin.size() && j < xf.size(); ++j) {
      const glm::mat4 world = skin[j] * xf[j].bind_world;
      jscale[j] = std::cbrt(std::fabs(glm::determinant(glm::mat3(world))));
    }
    for (auto& m : scene.meshes) m.is_prop = false;
    posed = PoseMeshes(scene.meshes, skin);
    for (auto& m : scene.meshes) m.is_prop = true;
    for (size_t mi = 0; mi < scene.meshes.size(); ++mi) {
      const auto& verts = scene.meshes[mi].verts;
      visible[mi].assign(verts.size(), 1);
      for (size_t vi = 0; vi < verts.size(); ++vi) {
        float sc = 0.f, wsum = 0.f;
        for (int k = 0; k < 4; ++k) {
          const float w = verts[vi].weights[k];
          if (w <= 0.f || verts[vi].joints[k] >= jscale.size()) continue;
          sc += w * jscale[verts[vi].joints[k]];
          wsum += w;
        }
        if (wsum > 0.f && sc / wsum < 0.05f) visible[mi][vi] = 0;
      }
    }
  } else {
    posed.resize(scene.meshes.size());
    for (size_t mi = 0; mi < scene.meshes.size(); ++mi) {
      for (const auto& v : scene.meshes[mi].verts) {
        posed[mi].pos.push_back(v.position);
        posed[mi].nrm.push_back(v.normal);
      }
    }
  }
  auto frame_box = [&](glm::vec3& lo, glm::vec3& hi) {
    lo = glm::vec3(1e9f);
    hi = glm::vec3(-1e9f);
    bool any = false;
    for (size_t mi = 0; mi < posed.size(); ++mi) {
      for (size_t vi = 0; vi < posed[mi].pos.size(); ++vi) {
        if (mi < visible.size() && vi < visible[mi].size() && !visible[mi][vi]) continue;
        lo = glm::min(lo, posed[mi].pos[vi]);
        hi = glm::max(hi, posed[mi].pos[vi]);
        any = true;
      }
    }
    if (!any) {  // everything scaled away: fall back to all geometry
      for (const auto& pm : posed) for (const auto& q : pm.pos) { lo = glm::min(lo, q); hi = glm::max(hi, q); any = true; }
    }
    return any;
  };
  glm::vec3 lo, hi;
  if (!frame_box(lo, hi)) {
    status = "no-geometry";
    return false;
  }
  // Flat props (mats, rugs, boards lying down) are edge-on at the default
  // pitch: look down on them instead.
  {
    const glm::vec3 e = hi - lo;
    const float footprint = std::max(e.x, e.z);
    const float flatness = footprint > 1e-6f ? e.y / footprint : 1.f;
    if (flatness < 0.25f) pitch = std::max(pitch, 55.f);
    else if (flatness < 0.5f) pitch = std::max(pitch, 32.f);
  }
  const glm::vec3 pivot = (lo + hi) * 0.5f;
  RotatePosed(posed, pivot, yaw, pitch);
  frame_box(lo, hi);
  const PreviewCamera cam = CameraForBox(lo, hi, size, 1.14f);
  std::map<int, const texdec::Image*> no_overrides;
  size_t covered = 0;
  if (!RenderPreviewImage(scene.meshes, scene.materials, posed, cam, size, png, no_overrides, true,
                          &covered)) {
    status = "render-failed";
    return false;
  }
  // Nothing visible (animation-only props, fully transparent art): not an
  // icon. The driver renders the avatar performing the prop instead.
  if (covered < (size_t)(size * 2) * (size_t)(size * 2) / 250) {
    status = "empty-render";
    return false;
  }
  status = "ok";
  return true;
}

extern "C" __declspec(dllimport) unsigned long __stdcall GetModuleFileNameW(void* module,
                                                                              wchar_t* buffer,
                                                                              unsigned long size);

int RunGenIcons(const std::vector<std::string>& argv) {
  fs::path closet_dir, icons_out, report, toc, mannequins, existing;
  int size = 128, limit = -1, shard = 0, shards = 1;
  float yaw = 25.f, pitch = 10.f;
  bool force = false;
  std::string filter;
  std::vector<std::string> pos;
  for (size_t i = 0; i < argv.size(); ++i) {
    if (argv[i] == "--icons-out" && i + 1 < argv.size()) icons_out = PathFromUtf8(argv[++i]);
    else if (argv[i] == "--report" && i + 1 < argv.size()) report = PathFromUtf8(argv[++i]);
    else if (argv[i] == "--toc" && i + 1 < argv.size()) toc = PathFromUtf8(argv[++i]);
    else if (argv[i] == "--mannequins" && i + 1 < argv.size()) mannequins = PathFromUtf8(argv[++i]);
    else if (argv[i] == "--existing" && i + 1 < argv.size()) existing = PathFromUtf8(argv[++i]);
    else if (argv[i] == "--size" && i + 1 < argv.size()) size = std::max(16, std::atoi(argv[++i].c_str()));
    else if (argv[i] == "--limit" && i + 1 < argv.size()) limit = std::atoi(argv[++i].c_str());
    else if (argv[i] == "--yaw" && i + 1 < argv.size()) yaw = (float)std::atof(argv[++i].c_str());
    else if (argv[i] == "--pitch" && i + 1 < argv.size()) pitch = (float)std::atof(argv[++i].c_str());
    else if (argv[i] == "--filter" && i + 1 < argv.size()) filter = argv[++i];
    else if (argv[i] == "--force") force = true;
    else if (argv[i] == "--shard" && i + 1 < argv.size()) {  // k/n: this process takes items with index % n == k
      const std::string v = argv[++i];
      const size_t slash = v.find('/');
      if (slash != std::string::npos) { shard = std::atoi(v.substr(0, slash).c_str()); shards = std::max(1, std::atoi(v.substr(slash + 1).c_str())); }
    }
    else pos.push_back(argv[i]);
  }
  if (pos.empty()) {
    std::printf("usage: avatarextract --gen-icons <closet_dir> [--toc <AvatarAssetPack.toc>] "
                "[--mannequins <dir>] [--icons-out <dir>] [--existing <dir>] [--size N] [--filter <substr>] "
                "[--limit N] [--force] [--report <tsv>] [--yaw DEG] [--pitch DEG]\n");
    return 1;
  }
  closet_dir = PathFromUtf8(pos[0]);
  if (icons_out.empty()) icons_out = closet_dir / "icons";
  // --existing <dir>: the icons that count as "already there" (default: icons_out);
  // a review run can render into a scratch dir while skipping the closet's real art.
  if (existing.empty()) existing = icons_out;
  if (toc.empty()) {
    const fs::path cand = closet_dir.parent_path() / "AvatarAssetPack.toc";
    if (fs::exists(cand)) toc = cand;
  }
  if (mannequins.empty()) {
    wchar_t exe[1024] = {};
    GetModuleFileNameW(nullptr, exe, 1024);
    mannequins = fs::path(exe).parent_path();
  }
  if (toc.empty() || !fs::exists(toc)) {
    std::printf("ERROR: asset pack not found; pass --toc <AvatarAssetPack.toc>\n");
    return 1;
  }
  const fs::path mq_male = mannequins / "mannequin_male.amd";
  const fs::path mq_female = mannequins / "mannequin_female.amd";
  if (!fs::exists(mq_male)) {
    std::printf("ERROR: mannequin manifests not found in %s (pass --mannequins <dir>)\n",
                PathStr(mannequins).c_str());
    return 1;
  }
  std::error_code ec;
  fs::create_directories(icons_out, ec);
  GetCloset().Load(closet_dir);
  // stand-ins already generated
  const fs::path gen_tsv = icons_out.parent_path() / (shards > 1 ? "icons_generated_shard" + std::to_string(shard) + ".tsv" : "icons_generated.tsv");
  std::set<std::string> generated;
  if (FILE* gf = _wfopen(gen_tsv.wstring().c_str(), L"rb")) {
    char line[1024];
    while (std::fgets(line, sizeof(line), gf)) {
      std::string s(line);
      const size_t tab = s.find('\t');
      if (tab != std::string::npos && tab == 36) generated.insert(s.substr(0, 36));
    }
    std::fclose(gf);
  }
  FILE* gen = _wfopen(gen_tsv.wstring().c_str(), L"ab");
  FILE* csv = report.empty() ? nullptr : _wfopen(report.wstring().c_str(), L"wb");
  if (csv) std::fprintf(csv, "guid\tname\tkind\tstatus\tframe\tframes\tmotion\textent\tms\n");
  // per-shard scratch: parallel shards must not share the manifest/bake dir
  const fs::path work = fs::temp_directory_path() / ("avatarextract_genicons_" + std::to_string(shard));
  fs::create_directories(work, ec);
  auto read_file = [](const fs::path& p, std::vector<uint8_t>& out) {
    FILE* f = _wfopen(p.wstring().c_str(), L"rb");
    if (!f) return false;
    std::fseek(f, 0, SEEK_END);
    const long sz = std::ftell(f);
    std::fseek(f, 0, SEEK_SET);
    out.resize((size_t)std::max(0L, sz));
    const size_t got = std::fread(out.data(), 1, out.size(), f);
    std::fclose(f);
    return got == out.size();
  };
  std::vector<uint8_t> mq_bytes[2];
  read_file(mq_male, mq_bytes[0]);
  if (fs::exists(mq_female)) read_file(mq_female, mq_bytes[1]);
  auto lower = [](std::string v) {
    for (auto& c : v) c = (char)std::tolower((unsigned char)c);
    return v;
  };
  const std::string lfilter = lower(filter);
  std::printf("(mode: generate fill-in icons)\ncloset: %s (%zu items, %zu stand-ins on record)\n",
              PathStr(closet_dir).c_str(), GetCloset().items().size(), generated.size());
  size_t done = 0, ok = 0, failed = 0, present = 0, props = 0, worn = 0;
  const auto t0 = std::chrono::steady_clock::now();
  size_t index = 0;
  for (const ClosetItem& item : GetCloset().items()) {
    if ((index++ % (size_t)shards) != (size_t)shard) continue;
    if (!lfilter.empty() && lower(item.name).find(lfilter) == std::string::npos) continue;
    const std::string guid = item.id.to_string();
    const fs::path png = icons_out / (guid + ".png");
    // present in the reference set, or already rendered into icons_out (resume)
    if ((fs::exists(existing / (guid + ".png"), ec) || fs::exists(png, ec)) &&
        !(force && generated.count(guid))) {
      ++present;
      continue;
    }
    if (limit >= 0 && (int)done >= limit) break;
    ++done;
    const auto t1 = std::chrono::steady_clock::now();
    std::vector<uint8_t> bytes;
    std::string status = "ok";
    const char* kind = "worn";
    PropFrameChoice choice;
    bool good = false;
    // The blank white mannequin wearing (or performing) the item: see
    // render_worn below. Props come here only when they render empty alone
    // (animation-only props): the avatar then performs the prop's clip.
    auto render_worn = [&](bool as_prop) {
      const std::vector<uint8_t>& base =
          (item.bodies == 2 && !mq_bytes[1].empty()) ? mq_bytes[1] : mq_bytes[0];
      if (base.size() != 1000) {
        status = "mannequin-manifest";
        return false;
      }
      (void)as_prop;
        // The blank white mannequin wearing the item (the Dry Cleaner's
        // wear_on_manifest): default height/weight, no blend shapes, no face
        // features, every avatar colour white, no hair/clothes/props; the
        // item takes component slot 0 with its category mask.
        std::vector<uint8_t> m = base;
        std::memset(&m[4], 0, 8);
        std::memset(&m[0x0C], 0, 0x30);
        std::memset(&m[0x3C], 0, 0xC0);
        std::memset(&m[0xFC], 0xFF, 9 * 4);
        std::memset(&m[0x160], 0, 0x1A0);
        std::memset(&m[0x300], 0, 0x80);
        std::memcpy(&m[0x160], &item.id, sizeof(AssetId));
        const uint16_t cats = (uint16_t)(item.categories & 0x1FFFu);
        m[0x160 + 16] = (uint8_t)(cats >> 8);
        m[0x160 + 17] = (uint8_t)(cats & 0xFF);
        const fs::path manifest = work / "wear.bin";
        if (FILE* mf = _wfopen(manifest.wstring().c_str(), L"wb")) {
          std::fwrite(m.data(), 1, m.size(), mf);
          std::fclose(mf);
          Args a;
          a.manifest = manifest;
          a.out_dir = work / "out";
          a.toc = toc;
          const fs::path legacy = toc.parent_path() / "AvatarAssetPackLegacyV1.toc";
          if (fs::exists(legacy)) a.legacy_toc = legacy;
          a.closet = closet_dir;
          a.no_scale = true;
          a.bake_size = 256;
          a.pack_anim_filters = {"Animation Generic Stand 0@0.5"};
          a.icon_out = png;
          a.icon_target = item.id;
          a.icon_categories = item.categories;
          a.icon_size = size;
          a.icon_yaw = yaw;
          a.icon_pitch = pitch;
          fs::create_directories(a.out_dir, ec);
          const int rc = Run(a);
          fs::remove_all(a.out_dir, ec);
          const bool done_ok = rc == 0 && fs::exists(png, ec);
          if (!done_ok) status = "render-failed(" + std::to_string(rc) + ")";
          return done_ok;
        }
        status = "work-dir";
        return false;
    };
    if (!GetCloset().ReadItemBytes(item.id, bytes) || bytes.empty()) {
      status = "unreadable";
    } else if (item.categories & ComponentCategory::kProp) {
      kind = "prop";
      ++props;
      good = RenderPropIcon(bytes, size, yaw, pitch, png, status, choice);
      if (!good && status == "empty-render") {
        kind = "performed";
        good = render_worn(true);
        if (good) status = "ok(avatar performing)";
      }
    } else {
      ++worn;
      good = render_worn(false);
    }
    const auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(
                        std::chrono::steady_clock::now() - t1).count();
    if (good) {
      ++ok;
      if (gen && !generated.count(guid)) {
        std::fprintf(gen, "%s\t%s\t%s\t%zu\t%zu\n", guid.c_str(), item.name.c_str(), kind,
                     choice.frame, choice.frames);
        std::fflush(gen);
        generated.insert(guid);
      }
    } else {
      ++failed;
      fs::remove(png, ec);
    }
    if (csv) {
      std::fprintf(csv, "%s\t%s\t%s\t%s\t%zu\t%zu\t%.4f\t%.4f\t%lld\n", guid.c_str(),
                   item.name.c_str(), kind, status.c_str(), choice.frame, choice.frames,
                   choice.motion, choice.extent, (long long)ms);
    }
    if (done % 100 == 0) {
      std::printf("  %zu done (%zu ok, %zu failed) ...\n", done, ok, failed);
      std::fflush(stdout);
    }
  }
  if (csv) std::fclose(csv);
  if (gen) std::fclose(gen);
  fs::remove_all(work, ec);
  const auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(
                      std::chrono::steady_clock::now() - t0).count();
  std::printf("gen-icons: %zu rendered (%zu props, %zu worn), %zu failed, %zu already had icons, "
              "in %lld ms -> %s\n",
              ok, props, worn, failed, present, (long long)ms, PathStr(icons_out).c_str());
  return 0;
}

}  // namespace avexp

int RunAvatarExport(const std::vector<std::string>& argv) { return avexp::RunFromArgs(argv); }
int RunPropIcons(const std::vector<std::string>& argv) { return avexp::RunPropIcons(argv); }
int RunGenIcons(const std::vector<std::string>& argv) { return avexp::RunGenIcons(argv); }
