/**
 ******************************************************************************
 * Xenia : Xbox 360 Emulator Research Project                                 *
 ******************************************************************************
 * Copyright 2026 Ben Vanik. All rights reserved.                             *
 * Released under the BSD license - see LICENSE in the root for more details. *
 ******************************************************************************
 */

#include <array>
#include <cmath>
#include <filesystem>
#include <mutex>
#include <map>
#include <stack>
#include <unordered_map>

#include "guest_load_asset.h"

#include "asset_pack.h"
#include "blend_shape.h"
#include "blend_shape_apply.h"
#include "closet.h"
#include "compression.h"
#include "guest_animation.h"
#include "guest_asset.h"
#include "guest_load_animation.h"
#include "memory_block.h"
#include "model.h"
#include "prop.h"
#include "skeleton.h"
#include "skeleton_data.h"
#include "skeleton_scaling.h"
#include "strb.h"
#include "bit_stream.h"
#include <rex/cvar.h>
#include <rex/filesystem.h>
#include <rex/logging.h>
#include <rex/system/kernel_state.h>

#include "xe_compat.h"

namespace rex {
namespace avatars {

// Legacy (pre-Fall-2010) asset pack, registered by xam_avatar.cpp: source
// of the old-revision body meshes for the old-body item rescue. Bodies are
// cached per gender after first decode (invalidate on re-registration).
static AssetPack* g_legacy_asset_pack = nullptr;
static std::shared_ptr<Model> g_legacy_bodies[2];  // [0] male c=1, [1] female c=2
static ModelLoadOptions g_legacy_body_options[2];

void SetLegacyAssetPack(AssetPack* pack) {
  g_legacy_asset_pack = pack;
  g_legacy_bodies[0].reset();
  g_legacy_bodies[1].reset();
}

// load_options must match the options the worn components were decoded with
// (kInvert per the title's coordinate system); an inverted patch compared
// against a non-inverted legacy body reads as a large Z mismatch and drops the
// rescue back to the affine fallback.
static std::shared_ptr<Model> LoadLegacyBody(uint16_t gender_c,
                                             ModelLoadOptions load_options) {
  if (!g_legacy_asset_pack || (gender_c != 1 && gender_c != 2)) {
    return nullptr;
  }
  auto& slot = g_legacy_bodies[gender_c - 1];
  if (slot && g_legacy_body_options[gender_c - 1] == load_options) {
    return slot;
  }
  slot.reset();
  const auto& infos = g_legacy_asset_pack->asset_infos();
  for (size_t i = 0; i < infos.size(); ++i) {
    if (!(infos[i].categories & ComponentCategory::kBody) ||
        infos[i].bodies != (uint32_t)gender_c) {
      continue;
    }
    const uint8_t* buffer = nullptr;
    size_t size = 0;
    if (g_legacy_asset_pack->GetAssetDataByIndex(i, buffer, size)) {
      slot = Model::Load(buffer, size, load_options);
      if (slot) {
        g_legacy_body_options[gender_c - 1] = load_options;
      }
    }
    break;
  }
  return slot;
}

// Mirrored halves and UV seams carry twin vertices one quantization step
// apart (about 10 um), and without the console's 4x MSAA the rasterizer turns
// that sliver into pinholes down the face's centre line. Snap near-coincident
// vertices to one position after the shapes have moved them.
static void WeldSeams(Model& model) {
  constexpr float kCell = 0.0002f;       // bin size, metres
  constexpr float kTolerance = 0.0001f;  // legit vertices are never this close
  std::unordered_map<uint64_t, Vector3<float>> anchors;
  const auto cell_key = [](int32_t x, int32_t y, int32_t z) {
    const uint64_t bias = 1u << 20;
    return ((uint64_t(x) + bias) << 42) | ((uint64_t(y) + bias) << 21) | (uint64_t(z) + bias);
  };
  for (auto& batch : model.triangle_batches) {
    for (auto& v : batch.vertices) {
      const int32_t cx = int32_t(std::floor(v.position.x / kCell));
      const int32_t cy = int32_t(std::floor(v.position.y / kCell));
      const int32_t cz = int32_t(std::floor(v.position.z / kCell));
      bool snapped = false;
      for (int32_t dx = -1; dx <= 1 && !snapped; ++dx) {
        for (int32_t dy = -1; dy <= 1 && !snapped; ++dy) {
          for (int32_t dz = -1; dz <= 1 && !snapped; ++dz) {
            auto it = anchors.find(cell_key(cx + dx, cy + dy, cz + dz));
            if (it == anchors.end()) continue;
            const float ex = v.position.x - it->second.x, ey = v.position.y - it->second.y,
                        ez = v.position.z - it->second.z;
            if (ex * ex + ey * ey + ez * ez < kTolerance * kTolerance) {
              v.position = it->second;
              snapped = true;
            }
          }
        }
      }
      if (!snapped) {
        anchors.emplace(cell_key(cx, cy, cz), v.position);
      }
    }
  }
}

static BodyType GetBodyType(const X_AVATAR_METADATA& metadata) {
  const AssetId male_body_asset_id = {
      2, 0, 1, {0xC1, 0xC8, 0xF1, 0x09, 0xA1, 0x9C, 0xB2, 0xE0}};
  const AssetId female_body_asset_id = {
      2, 0, 2, {0xC1, 0xC8, 0xF1, 0x09, 0xA1, 0x9C, 0xB2, 0xE0}};
  if (metadata.body_component.asset_id == male_body_asset_id) {
    return BodyType::kMale;
  }
  if (metadata.body_component.asset_id == female_body_asset_id) {
    return BodyType::kFemale;
  }
  return BodyType::kUnknown;
}

static bool LoadFile(std::filesystem::path path, std::vector<uint8_t>& buffer) {
  bool was_loaded = false;
  auto handle = rex::filesystem::OpenFile(path, "rb");
  if (handle != nullptr) {
    fseek(handle, 0, SEEK_END);
    buffer.resize(ftell(handle));
    fseek(handle, 0, SEEK_SET);
    size_t bytes_read = fread(buffer.data(), 1, buffer.size(), handle);
    if (bytes_read == buffer.size()) {
      was_loaded = true;
    }
    fclose(handle);
  }
  return was_loaded;
}

static bool LoadAsset(AssetPack* asset_pack, const AssetId& asset_id,
                      const uint8_t*& buffer, size_t& size,
                      std::vector<uint8_t>& temp) {
  // Imported marketplace items (full product GUIDs) resolve through the
  // closet; only stock-tail ids are pack indexes (their .b would otherwise
  // alias an unrelated pack entry).
  if (!IsStockPackId(asset_id)) {
    if (GetCloset().ReadItemBytes(asset_id, temp) && !temp.empty()) {
      buffer = temp.data();
      size = temp.size();
      return true;
    }
  }
  if (!asset_pack->GetAssetData(asset_id, buffer, size)) {
    // TODO(gibbed): load from user data path
    std::filesystem::path bin_path =
        fmt::format("avatar_blobs\\{}.bin", asset_id.to_string());
    if (!LoadFile(bin_path, temp)) {
      return false;
    }
    buffer = temp.data();
    size = temp.size();
  }
  return true;
}

template <typename Type>
static std::shared_ptr<Type> LoadAsset(AssetPack* asset_pack,
                                       const AssetId& asset_id) {
  const uint8_t* strb_buffer;
  size_t strb_size;
  std::vector<uint8_t> strb_bytes;
  if (!LoadAsset(asset_pack, asset_id, strb_buffer, strb_size, strb_bytes)) {
    return nullptr;
  }
  return Type::Load(strb_buffer, strb_size);
}

template <typename Type, typename Options>
static std::shared_ptr<Type> LoadAsset(AssetPack* asset_pack,
                                       const AssetId& asset_id,
                                       Options load_options) {
  const uint8_t* strb_buffer;
  size_t strb_size;
  std::vector<uint8_t> strb_bytes;
  if (!LoadAsset(asset_pack, asset_id, strb_buffer, strb_size, strb_bytes)) {
    return nullptr;
  }
  return Type::Load(strb_buffer, strb_size, load_options);
}

// --- Decoded-model cache ------------------------------------------------
// Grid tile builds re-decode the same base body/head/worn components on every
// GetAssets call, which is a lot of vertex de-quantization per page flip.
// Decoded models are mutated downstream (ApplyBlendShape patches vertices in
// place; the custom-avatar substitution zeroes triangle counts), so the cache
// stores pristine decodes and hands out deep copies; Model is value-copyable
// and a copy is far cheaper than a decode. Keyed by asset id + load options;
// small LRU.
namespace {
std::mutex g_model_cache_mx;
struct CachedModel {
  std::shared_ptr<Model> pristine;
  uint64_t stamp;
};
std::map<std::pair<std::array<uint8_t, 16>, uint32_t>, CachedModel>
    g_model_cache;
uint64_t g_model_cache_stamp = 0;

std::array<uint8_t, 16> ModelCacheKeyBytes(const AssetId& id) {
  std::array<uint8_t, 16> k{};
  std::memcpy(k.data(), &id, sizeof(k));
  return k;
}
}  // namespace

static std::shared_ptr<Model> LoadModelCached(AssetPack* asset_pack,
                                              const AssetId& asset_id,
                                              ModelLoadOptions load_options) {
  const auto key =
      std::make_pair(ModelCacheKeyBytes(asset_id), uint32_t(load_options));
  {
    std::lock_guard<std::mutex> lk(g_model_cache_mx);
    auto it = g_model_cache.find(key);
    if (it != g_model_cache.end()) {
      it->second.stamp = ++g_model_cache_stamp;
      // Deep copy: downstream mutates (blend shapes, custom substitution).
      return std::make_shared<Model>(*it->second.pristine);
    }
  }
  auto model =
      LoadAsset<Model, ModelLoadOptions>(asset_pack, asset_id, load_options);
  if (model == nullptr) {
    return nullptr;
  }
  {
    std::lock_guard<std::mutex> lk(g_model_cache_mx);
    if (g_model_cache.size() >= 48) {
      auto oldest = g_model_cache.begin();
      for (auto it = g_model_cache.begin(); it != g_model_cache.end(); ++it) {
        if (it->second.stamp < oldest->second.stamp) oldest = it;
      }
      g_model_cache.erase(oldest);
    }
    g_model_cache[key] = {model, ++g_model_cache_stamp};
  }
  // The caller gets its own mutable copy; the pristine stays cached.
  return std::make_shared<Model>(*model);
}

REXCVAR_DEFINE_BOOL(avatar_eye_whites, false, "Kernel",
                    "Paint white sclera into eye styles whose art has none (preference "
                    "mod): the eye textures are channel-coded masks (R=iris tint, "
                    "G=white-add, B=skin-blend per the FB79 head-composite ucode); "
                    "styles without whites fill the eye region via B. This converts "
                    "strong-B endpoints to G in the DXT5 color endpoints in place.");

// Eye-whites: endpoint-level DXT5 channel rewrite. The wire data is BE
// 8-in-16 (every 16-bit word byte-swapped), so '>H'-style reads of the
// color endpoint words at block+8/+10 yield canonical RGB565. Idempotent:
// a converted endpoint has B=0 and is skipped on re-application. Styles
// that already carry whites (enough G-dominant endpoints) are left alone.
static void WhitenEyeStack(Texture& tex) {
  if ((tex.format & 0x3F) != 20) {  // k_DXT4_5 only (the eye stacks)
    return;
  }
  auto& d = tex.data_bytes;
  auto rd16 = [&](size_t off) -> uint32_t {
    return (uint32_t(d[off]) << 8) | d[off + 1];
  };
  auto wr16 = [&](size_t off, uint32_t v) {
    d[off] = uint8_t(v >> 8);
    d[off + 1] = uint8_t(v);
  };
  // Detection pass: count G-dominant endpoints (whites-having art).
  uint32_t g_dominant = 0;
  for (size_t b = 0; b + 16 <= d.size(); b += 16) {
    for (size_t e = 0; e < 2; ++e) {
      const uint32_t c = rd16(b + 8 + e * 2);
      const uint32_t g6 = (c >> 5) & 0x3F, b5 = c & 0x1F;
      if (g6 >= 24 && b5 < 8) g_dominant++;
    }
  }
  if (g_dominant >= 24) {
    return;  // art already has a whites mask
  }
  // Convert: strong skin-blend (B) endpoints become white-add (G).
  for (size_t b = 0; b + 16 <= d.size(); b += 16) {
    for (size_t e = 0; e < 2; ++e) {
      const size_t off = b + 8 + e * 2;
      uint32_t c = rd16(off);
      const uint32_t b5 = c & 0x1F;
      if (b5 < 8) continue;  // faint B = outline shading, keep
      const uint32_t g6 = (c >> 5) & 0x3F;
      const uint32_t g6n = std::max<uint32_t>(g6, (b5 << 1) | (b5 >> 4));
      c = (c & ~0x7E0u & ~0x1Fu) | (g6n << 5);
      wr16(off, c);
    }
  }
}

struct ShaderParameterOverride {
  uint32_t usage;
  float x;
  float y;
  float z;
  float w;
};

static bool VertexToGuest(X_AVATAR_VERTEX* guest, const Vertex& host) {
  guest->position.x = host.position.x;
  guest->position.y = host.position.y;
  guest->position.z = host.position.z;
  guest->normal = host.normal;
  guest->blend_weight = host.blend_weight;
  guest->blend_indices = host.blend_indices;
  guest->color = host.color;
  for (size_t i = 0; i < host.uvs.size(); ++i) {
    auto& guest_uv = guest->uvs[i];
    const auto& host_uv = host.uvs[i];
    guest_uv.u = host_uv.x;
    guest_uv.v = host_uv.y;
  }
  return true;
}

static void OverrideShaderParameter(
    X_AVATAR_SHADER_PARAM* parameters,
    const ShaderParameterOverride& parameter_override) {
  for (size_t i = 0, o = 19; i < 20; ++i, --o) {
    auto& parameter = parameters[o];
    if (parameter.usage != parameter_override.usage) {
      continue;
    }
    assert_true(
        parameter.type == uint32_t(ShaderParameterType::kPixelConstant) ||
        parameter.type == uint32_t(ShaderParameterType::kVertexConstant));
    parameter.constant_values[0] = parameter_override.x;
    parameter.constant_values[1] = parameter_override.y;
    parameter.constant_values[2] = parameter_override.z;
    parameter.constant_values[3] = parameter_override.w;
  }
}

static bool TriangleBatchToGuest(
    X_AVATAR_TRIANGLE_BATCH& guest, const TriangleBatch& host,
    MemoryBlock* cpu_memory, uint8_t* cpu_buffer, MemoryBlock* gpu_memory,
    uint8_t* gpu_buffer, uint32_t gpu_buffer_base_ptr,
    const std::vector<ShaderParameterOverride>& shader_parameter_overrides) {
  guest.shader_id = host.shader_id;
  std::memset(guest.shader_parameters, 0, sizeof(guest.shader_parameters));
  for (size_t i = 0; i < host.shader_parameters.size(); ++i) {
    auto& guest_parameter = guest.shader_parameters[i];
    const auto& host_parameter = host.shader_parameters[i];
    guest_parameter.type = static_cast<uint32_t>(host_parameter.type);
    guest_parameter.usage = host_parameter.usage;
    if (host_parameter.type == ShaderParameterType::kTexture) {
      guest_parameter.texture.index = host_parameter.texture.index;
      guest_parameter.texture.uv_index = host_parameter.texture.uv_layer;
      guest_parameter.texture.flags = host_parameter.texture.flags;
    } else {
      for (size_t j = 0; j < 4; ++j) {
        guest_parameter.constant_values[j] = host_parameter.constant_values[j];
      }
    }
  }

  for (const auto& shader_parameter_override : shader_parameter_overrides) {
    OverrideShaderParameter(guest.shader_parameters, shader_parameter_override);
  }

  guest.triangle_count = host.triangle_count;
  guest.vertex_count = static_cast<uint32_t>(host.vertices.size());
  guest.uv_count = host.uv_count;
  guest.vertex_size = host.vertex_size;
  guest.index_size = host.index_size;
  gpu_memory->SetPointer(&guest.vertices_ptr,
                         gpu_buffer_base_ptr + host.vertex_array_offset);
  gpu_memory->SetPointer(&guest.indices_ptr,
                         gpu_buffer_base_ptr + host.index_array_offset);
  uint32_t vertex_offset = host.vertex_array_offset;
  for (size_t i = 0; i < host.vertices.size(); ++i) {
    if (!VertexToGuest(
            reinterpret_cast<X_AVATAR_VERTEX*>(&gpu_buffer[vertex_offset]),
            host.vertices[i])) {
      return false;
    }
    vertex_offset += guest.vertex_size;
  }
  uint32_t index_offset = host.index_array_offset;
  for (size_t i = 0; i < host.indices.size(); ++i) {
    auto guest_index =
        reinterpret_cast<be<uint16_t>*>(&gpu_buffer[index_offset]);
    *guest_index = host.indices[i];
    index_offset += guest.index_size;
  }
  return true;
}

static bool TextureToGuest(X_AVATAR_TEXTURE& guest, const ModelTexture& host,
                           MemoryBlock* cpu_memory, uint8_t* cpu_buffer,
                           MemoryBlock* gpu_memory, uint8_t* gpu_buffer,
                           uint32_t gpu_buffer_base_ptr) {
  auto format = host.texture.format;
  format &= ~0x00000100u;
  guest.format = format;
  guest.width = host.texture.width;
  guest.height = host.texture.height;
  guest.total_base_size = host.texture.total_data_size;
  guest.total_mip_size = 0;
  guest.base_size = host.texture.data_size;
  guest.mip_size = 0;
  guest.mip_levels = 1;
  guest.layer_count = host.texture.layer_count;
  gpu_memory->SetPointer(&guest.base_data_ptr,
                         gpu_buffer_base_ptr + host.gpu_offset);
  guest.mip_data_ptr = 0;

  auto data_buffer = &gpu_buffer[host.gpu_offset];
  // Zero the whole slot first: empty placeholders (unworn decal slots) and the
  // tail beyond a short replacement stack must decode as transparent DXT
  // blocks rather than sampling the never-written guest buffer.
  std::memset(data_buffer, 0, host.gpu_size);
  if (!host.texture.is_empty) {
    uint32_t output_align = (host.texture.format == 0x1A200152u ||
                             host.texture.format == 0x1A200052u)
                                ? 256u
                                : 512u;
    size_t input_stride = host.texture.data_stride;
    size_t output_stride = align(host.texture.data_stride, output_align);
    size_t input_offset = 0;
    size_t output_offset = 0;
    assert_true(host.texture.data_rows * output_stride <= host.gpu_size);
    // Never write past the model's texture slot: a replacement stack larger
    // than the placeholder slot (mismatched category slotting) would clobber
    // the neighbouring textures/vertex data in the shared GPU buffer.
    uint32_t layer_count = host.texture.layer_count;
    const size_t layer_bytes = host.texture.data_rows * output_stride;
    if (layer_bytes && layer_count * layer_bytes > host.gpu_size) {
      const uint32_t max_layers = static_cast<uint32_t>(host.gpu_size / layer_bytes);
      REXKRNL_WARN("[avatar] texture stack ({} layers x {:#x}) exceeds its slot ({:#x}); "
                   "clamping to {} layers",
                   layer_count, layer_bytes, host.gpu_size, max_layers);
      layer_count = max_layers;
    }
    for (size_t layer = 0; layer < layer_count; ++layer) {
      for (size_t y = 0; y < host.texture.data_rows; ++y) {
        std::memcpy(&data_buffer[output_offset],
                    &host.texture.data_bytes.data()[input_offset],
                    input_stride);
        input_offset += input_stride;
        output_offset += output_stride;
      }
    }
  }
  return true;
}

static bool ModelToGuest(
    std::shared_ptr<Model> host_model, X_AVATAR_MODEL* guest_model,
    MemoryBlock* cpu_memory, MemoryBlock* gpu_memory, uint32_t category_mask,
    std::shared_ptr<Texture> replacement_textures[6],
    const std::vector<ShaderParameterOverride>& shader_parameter_overrides) {
  if (host_model == nullptr) {
    return false;
  }

  uint32_t cpu_buffer_ptr, gpu_buffer_ptr;
  uint8_t* cpu_buffer =
      cpu_memory->ClaimBytes(&cpu_buffer_ptr, host_model->cpu_size);
  uint8_t* gpu_buffer =
      gpu_memory->ClaimBytes(&gpu_buffer_ptr, host_model->gpu_size);

  *guest_model = {};
  guest_model->cpu_size = host_model->cpu_size;
  guest_model->gpu_size = host_model->gpu_size;
  guest_model->texture_size = host_model->texture_buffer_size;
  guest_model->vertex_size = host_model->vertex_buffer_size;
  guest_model->index_size = host_model->index_buffer_size;
  guest_model->triangle_batch_count =
      static_cast<uint32_t>(host_model->triangle_batches.size());
  guest_model->texture_count =
      static_cast<uint32_t>(host_model->textures.size());
  cpu_memory->SetPointer(&guest_model->cpu_buffer_ptr, cpu_buffer_ptr);
  gpu_memory->SetPointer(&guest_model->gpu_buffer_ptr, gpu_buffer_ptr);
  gpu_memory->SetPointer(&guest_model->vertex_buffer_ptr,
                         gpu_buffer_ptr + host_model->vertex_buffer_offset);
  gpu_memory->SetPointer(&guest_model->index_buffer_ptr,
                         gpu_buffer_ptr + host_model->index_buffer_offset);
  cpu_memory->SetPointer(
      &guest_model->triangle_batches_ptr,
      cpu_buffer_ptr + host_model->triangle_batch_array_offset);
  cpu_memory->SetPointer(&guest_model->textures_ptr,
                         cpu_buffer_ptr + host_model->texture_array_offset);

  auto guest_triangle_batches = reinterpret_cast<X_AVATAR_TRIANGLE_BATCH*>(
      &cpu_buffer[host_model->triangle_batch_array_offset]);
  for (size_t i = 0; i < guest_model->triangle_batch_count; ++i) {
    if (!TriangleBatchToGuest(guest_triangle_batches[i],
                              host_model->triangle_batches[i], cpu_memory,
                              cpu_buffer, gpu_memory, gpu_buffer,
                              gpu_buffer_ptr, shader_parameter_overrides)) {
      return false;
    }
  }

  // override head textures with those from the metadata
  if (category_mask == ComponentCategory::kHead) {
    int usage_indices[20];
    for (size_t i = 0; i < 20; ++i) {
      usage_indices[i] = -1;
    }
    const int usage_to_replacement_texture_indices[] = {
        -1, -1, -1, -1, -1, 5, 3, 2, 2, 1, 1, 4, 0,
    };
    std::vector<int> replacement_texture_indices(guest_model->texture_count);
    for (size_t i = 0; i < guest_model->triangle_batch_count; ++i) {
      const auto& triangle_batch = host_model->triangle_batches[i];
      for (const auto& shader_parameter : triangle_batch.shader_parameters) {
        if (shader_parameter.usage >= 20 ||
            shader_parameter.type != ShaderParameterType::kTexture) {
          continue;
        }
        const auto& usage = shader_parameter.usage;
        const auto& texture_index = shader_parameter.texture.index;
        assert_true(usage_indices[usage] == -1 ||
                    usage_indices[usage] == texture_index);
        usage_indices[usage] = texture_index;
        replacement_texture_indices[texture_index] =
            usage < countof(usage_to_replacement_texture_indices)
                ? usage_to_replacement_texture_indices[usage]
                : -1;
      }
    }

    auto guest_textures = reinterpret_cast<X_AVATAR_TEXTURE*>(
        &cpu_buffer[host_model->texture_array_offset]);
    for (size_t i = 0; i < guest_model->texture_count; ++i) {
      ModelTexture model_texture;
      auto replacement_texture_index = replacement_texture_indices[i];
      if (replacement_texture_index >= 0 &&
          replacement_textures[replacement_texture_index] != nullptr) {
        model_texture.gpu_offset = host_model->textures[i].gpu_offset;
        model_texture.gpu_size = host_model->textures[i].gpu_size;
        model_texture.texture =
            *replacement_textures[replacement_texture_index];
      } else {
        model_texture = host_model->textures[i];
      }
      if (!TextureToGuest(guest_textures[i], model_texture, cpu_memory,
                          cpu_buffer, gpu_memory, gpu_buffer, gpu_buffer_ptr)) {
        return false;
      }
    }
  } else {
    auto guest_textures = reinterpret_cast<X_AVATAR_TEXTURE*>(
        &cpu_buffer[host_model->texture_array_offset]);
    for (size_t i = 0; i < guest_model->texture_count; ++i) {
      if (!TextureToGuest(guest_textures[i], host_model->textures[i],
                          cpu_memory, cpu_buffer, gpu_memory, gpu_buffer,
                          gpu_buffer_ptr)) {
        return false;
      }
    }
  }

  return true;
}

static bool SkeletonToGuest(X_AVATAR_SKELETON* guest,
                            std::shared_ptr<Skeleton> host,
                            MemoryBlock* cpu_memory) {
  assert_true(host->joints.size() <= 72);
  uint8_t joint_count = static_cast<uint8_t>(host->joints.size());

  *guest = {};
  guest->joint_count = joint_count;

  const uint8_t invalid_index = 255;

  uint32_t guest_joints_ptr;
  auto guest_joints = cpu_memory->Claim<X_AVATAR_SKELETON_JOINT>(
      &guest_joints_ptr, joint_count);
  cpu_memory->SetPointer(&guest->joints_ptr, guest_joints_ptr);

  for (uint8_t i = 0; i < joint_count; ++i) {
    auto& guest_joint = guest_joints[i];
    const auto& host_joint = host->joints[i];
    guest_joint.parent_index = host_joint.parent_index;
    guest_joint.first_child_index = host_joint.first_child_index;
    guest_joint.next_index = host_joint.next_index;
    guest_joint.bindpose.position.x = host_joint.bindpose.position.x;
    guest_joint.bindpose.position.y = host_joint.bindpose.position.y;
    guest_joint.bindpose.position.z = host_joint.bindpose.position.z;
    guest_joint.bindpose.position.w = 1.f;
    guest_joint.bindpose.rotation.x = host_joint.bindpose.rotation.x;
    guest_joint.bindpose.rotation.y = host_joint.bindpose.rotation.y;
    guest_joint.bindpose.rotation.z = host_joint.bindpose.rotation.z;
    guest_joint.bindpose.rotation.w = host_joint.bindpose.rotation.w;
    guest_joint.pose.position.x = host_joint.pose.position.x;
    guest_joint.pose.position.y = host_joint.pose.position.y;
    guest_joint.pose.position.z = host_joint.pose.position.z;
    guest_joint.pose.position.w = 1.f;
    guest_joint.pose.rotation.x = host_joint.pose.rotation.x;
    guest_joint.pose.rotation.y = host_joint.pose.rotation.y;
    guest_joint.pose.rotation.z = host_joint.pose.rotation.z;
    guest_joint.pose.rotation.w = host_joint.pose.rotation.w;
    guest_joint.pose.scale.x = host_joint.pose.scale.x;
    guest_joint.pose.scale.y = host_joint.pose.scale.y;
    guest_joint.pose.scale.z = host_joint.pose.scale.z;
    guest_joint.pose.scale.w = 1.f;
  }

  return true;
}

static void GetShaderOverrides(
    const X_AVATAR_METADATA& metadata, uint32_t category_mask,
    std::vector<ShaderParameterOverride>& shader_parameter_overrides) {
  // skin color
  if (category_mask & ComponentCategory::kBody) {
    auto color = metadata.colors[0];
    ShaderParameterOverride shader_parameter_override;
    shader_parameter_override.usage = 22;
    shader_parameter_override.x = ((color >> 16) & 0xFF) / 255.f;
    shader_parameter_override.y = ((color >> 8) & 0xFF) / 255.f;
    shader_parameter_override.z = ((color >> 0) & 0xFF) / 255.f;
    shader_parameter_override.w = ((color >> 24) & 0xFF) / 255.f;
    shader_parameter_overrides.push_back(shader_parameter_override);
  }
  // hair color, actual hair components only. Full costumes carry the kHair bit
  // too (they occupy/hide the hair slot; e.g. category mask 0xFFC), and
  // tinting a whole costume with the avatar's hair color crushes dark-art
  // outfits to near-black. Garment bits veto the tint; hair-with-hat items
  // (kHair|kHat, no garment bits) still tint.
  else if ((category_mask & ComponentCategory::kHair) &&
           !(category_mask &
             (ComponentCategory::kTop | ComponentCategory::kBottom |
              ComponentCategory::kShoes | ComponentCategory::kGloves))) {
    auto color = metadata.colors[1];
    ShaderParameterOverride shader_parameter_override;
    shader_parameter_override.usage = 22;
    shader_parameter_override.x = ((color >> 16) & 0xFF) / 255.f;
    shader_parameter_override.y = ((color >> 8) & 0xFF) / 255.f;
    shader_parameter_override.z = ((color >> 0) & 0xFF) / 255.f;
    shader_parameter_override.w = ((color >> 24) & 0xFF) / 255.f;
    shader_parameter_overrides.push_back(shader_parameter_override);
  } else {
    ShaderParameterOverride shader_parameter_override;
    shader_parameter_override.usage = 22;
    shader_parameter_override.x = 1.f;
    shader_parameter_override.y = 1.f;
    shader_parameter_override.z = 1.f;
    shader_parameter_override.w = 1.f;
    shader_parameter_overrides.push_back(shader_parameter_override);
  }

  if (category_mask & ComponentCategory::kHead) {
    for (uint32_t i = 0, usage = 13; i < 9; ++i, ++usage) {
      auto color = metadata.colors[i];
      ShaderParameterOverride shader_parameter_override;
      shader_parameter_override.usage = usage;
      shader_parameter_override.x = ((color >> 16) & 0xFF) / 255.f;
      shader_parameter_override.y = ((color >> 8) & 0xFF) / 255.f;
      shader_parameter_override.z = ((color >> 0) & 0xFF) / 255.f;
      shader_parameter_override.w = ((color >> 24) & 0xFF) / 255.f;
      // Eyeshadow "none" sentinel = opaque black (see the slot-4 auto-bind
      // gate in LoadAssetsToGuest): defensively zero the tint opacity too,
      // in case a cached build still carries the overlay.
      if (i == 5 && (color & 0xFFFFFFu) == 0) {
        shader_parameter_override.w = 0.f;
      }
      shader_parameter_overrides.push_back(shader_parameter_override);
    }
  }

  // rim light color
  {
    auto color = metadata.colors[0];
    ShaderParameterOverride shader_parameter_override;
    shader_parameter_override.usage = 27;
    // TODO(gibbed): calculate rim light color properly
    shader_parameter_override.x = 0.54901963f;
    shader_parameter_override.y = 0.5019608f;
    shader_parameter_override.z = 0.40392157f;
    shader_parameter_override.w = 1.6f;
    shader_parameter_overrides.push_back(shader_parameter_override);
  }
}

// The XDK body shaders recolor via TEXTURE_INTENSITY (usage 2):
//   albedo = lerp(TEXTURE_COLOR.rgb,
//                 i.r*CUSTOM_0 + i.g*CUSTOM_1 + i.b*CUSTOM_2,
//                 TEXTURE_INTENSITY.a)
// with CUSTOM_0..2 = shader usages 22/23/24, supplied by the item's own
// kCustomColorTable (STRB block 7). Marketplace items ship pre-colored atlases
// plus the identity palette (pure R/G/B), which makes the recolor sum
// reproduce the texel so the atlas passes through verbatim. Feeding synthetic
// colors instead (skin/hair/white in 22, the item's black 23/24 placeholders
// untouched) collapses the sum to texel.r * tint, which crushes cool-toned art
// to black.
// Layout (28 bytes): u32 version(=1), then 3 x { u32 ARGB color, u32
// reserved } in CUSTOM_0..2 order.
static void ApplyItemColorTable(
    const AssetId& asset_id,
    std::vector<ShaderParameterOverride>& shader_parameter_overrides) {
  std::vector<uint8_t> item_bytes;
  if (!GetCloset().ReadItemBytes(asset_id, item_bytes) || item_bytes.empty()) {
    return;
  }
  const uint8_t* table = nullptr;
  size_t table_size = 0;
  if (!strb::GetSTRBBlock(item_bytes.data(), item_bytes.size(),
                          strb::STRBBlockId::kCustomColorTable, table,
                          table_size) ||
      table_size < 4 + 3 * 8) {
    return;
  }
  auto read_u32 = [](const uint8_t* p, bool le) -> uint32_t {
    return le ? (uint32_t(p[0]) | (uint32_t(p[1]) << 8) |
                 (uint32_t(p[2]) << 16) | (uint32_t(p[3]) << 24))
              : ((uint32_t(p[0]) << 24) | (uint32_t(p[1]) << 16) |
                 (uint32_t(p[2]) << 8) | uint32_t(p[3]));
  };
  // Version word doubles as the endianness probe (LE 1 vs BE 0x01000000).
  const bool le = read_u32(table, true) <= 0xFFFFu;
  uint32_t colors[3];
  bool any_nonzero = false;
  for (size_t i = 0; i < 3; ++i) {
    colors[i] = read_u32(&table[4 + i * 8], le);
    any_nonzero |= colors[i] != 0;
  }
  if (!any_nonzero) {
    return;
  }
  for (size_t i = 0; i < 3; ++i) {
    ShaderParameterOverride shader_parameter_override;
    shader_parameter_override.usage = uint32_t(22 + i);
    shader_parameter_override.x = ((colors[i] >> 16) & 0xFF) / 255.f;
    shader_parameter_override.y = ((colors[i] >> 8) & 0xFF) / 255.f;
    shader_parameter_override.z = ((colors[i] >> 0) & 0xFF) / 255.f;
    shader_parameter_override.w = ((colors[i] >> 24) & 0xFF) / 255.f;
    shader_parameter_overrides.push_back(shader_parameter_override);
  }
}

bool LoadAssetsToGuest(const X_AVATAR_METADATA& metadata,
                       uint32_t category_mask, uint32_t flags,
                       AssetPack* asset_pack, MemoryBlock* cpu_memory,
                       MemoryBlock* gpu_memory, uint32_t skeleton_version,
                       uint32_t coordinate_system) {
  bool want_prop = !!(category_mask & ComponentCategory::kProp);

  category_mask &= ~ComponentCategory::kProp;

  BlendShapeLoadOptions blend_shape_load_options = BlendShapeLoadOption::kNone;
  SkeletonLoadOptions skeleton_load_options = SkeletonLoadOption::kNone;
  ModelLoadOptions model_load_options = ModelLoadOption::kNone;
  if (coordinate_system == 0) {
    blend_shape_load_options |= BlendShapeLoadOption::kInvert;
    skeleton_load_options |= SkeletonLoadOption::kInvert;
    model_load_options |= ModelLoadOption::kInvert;
  }

  BodyType body_type = GetBodyType(metadata);

  auto skeleton = LoadSkeleton(skeleton_version, skeleton_load_options);
  if (skeleton == nullptr) {
    REXKRNL_WARN("Failed to load avatar skeleton version {}!", skeleton_version);
    return false;
  }

  if (skeleton_version == 1) {
    ApplyScalesToSkeletonV1(body_type, metadata.weight_factor,
                            metadata.height_factor, skeleton);
  } else if (skeleton_version == 2) {
    ApplyScalesToSkeletonV2(body_type, metadata.weight_factor,
                            metadata.height_factor, skeleton);
  } else {
    REXKRNL_WARN("Unknown avatar skeleton version {}!", skeleton_version);
    return false;
  }

  std::vector<std::pair<AssetId, std::shared_ptr<BlendShape>>> blend_shapes;
  // Bounded by countof: reading six entries here overruns blend_shapes into
  // textures[0..1], whose ids are texture assets.
  for (size_t i = 0; i < countof(metadata.blend_shapes); ++i) {
    const auto& asset_id = metadata.blend_shapes[i].asset_id;
    if (asset_id.is_zero()) {
      continue;
    }
    auto blend_shape = LoadAsset<BlendShape, BlendShapeLoadOptions>(
        asset_pack, asset_id, blend_shape_load_options);
    if (blend_shape == nullptr) {
      REXKRNL_ERROR("Failed to load avatar blend shape {}!", asset_id.to_string());
      continue;
    }
    blend_shapes.push_back({asset_id, blend_shape});
  }

  std::vector<X_AVATAR_COMPONENT_INFO> source_component_infos;
  if (metadata.body_component.matches(category_mask)) {
    source_component_infos.push_back(metadata.body_component);
  }
  if (metadata.head_component.matches(category_mask)) {
    source_component_infos.push_back(metadata.head_component);
  }
  for (const auto& component : metadata.components) {
    if (component.matches(category_mask)) {
      source_component_infos.push_back(component);
    }
  }

  // When a hat is worn, swap the hair to its companion variant: catalog hair
  // entries carry asset_ids[0] whose .b names the partner pack entry: the
  // flattened "(Hat)" mesh authored to sit under headwear. Rendering the
  // full-volume catalog hair with a hat makes the hair clip through it.
  {
    bool hat_worn = false;
    for (const auto& info : source_component_infos) {
      if (info.matches(ComponentCategory::kHat)) {
        hat_worn = true;
        break;
      }
    }
    if (hat_worn && asset_pack != nullptr) {
      const auto& pack_infos = asset_pack->asset_infos();
      for (auto& info : source_component_infos) {
        if (!info.matches(ComponentCategory::kHair) || !IsStockPackId(info.asset_id)) {
          continue;
        }
        const size_t hair_index = info.asset_id.b;
        if (hair_index >= pack_infos.size()) {
          continue;
        }
        const auto& id0 = pack_infos[hair_index].asset_ids[0];
        const size_t partner_index = id0.b;
        if (id0.is_zero() || partner_index == hair_index ||
            partner_index >= pack_infos.size()) {
          continue;
        }
        info.asset_id.b = static_cast<uint16_t>(partner_index);
      }
    }
  }

  // Wearables link a body-hiding blend shape through asset_ids[0]: entries
  // like "Track Jacket Hiding Template" carry an id with a = 0x01000000 (the
  // shape population) and b = the pack index of a small kShapeOverrides STRB.
  // Applying it collapses the body vertices the garment covers, so the skin
  // does not clip through the clothing. The shape's own original_asset_id
  // names its target (the stock body), so pushing it into blend_shapes lets
  // the existing apply loop route it.
  // Imported marketplace items are self-contained: their hiding shape lives
  // as a kShapeOverrides block inside the item's own YTGR/STRB blob, and its
  // target body id uses b = the body's pack index (male 0 / female 1) where
  // manifests use the canonical {a=2, b=0, c=gender}; normalize b so the
  // apply loop's exact-id match works.
  if (asset_pack != nullptr) {
    const auto& pack_infos = asset_pack->asset_infos();
    for (const auto& info : source_component_infos) {
      if (!IsStockPackId(info.asset_id)) {
        // Closet item: embedded hiding shape(s). Unisex marketplace items
        // carry two kShapeOverrides blocks, one per gender body, and
        // BlendShape::Load only reads the first, so load every block. The
        // apply loop matches the right one to the worn body by id, and the
        // buffer-size checks reject the rest.
        std::vector<uint8_t> item_bytes;
        if (!GetCloset().ReadItemBytes(info.asset_id, item_bytes) || item_bytes.empty()) {
          continue;
        }
        const size_t shape_count = strb::CountSTRBBlocks(item_bytes.data(), item_bytes.size(),
                                                         strb::STRBBlockId::kShapeOverrides);
        for (size_t occurrence = 0; occurrence < shape_count; ++occurrence) {
          const uint8_t* shape_block = nullptr;
          size_t shape_block_size = 0;
          if (!strb::GetSTRBBlockN(item_bytes.data(), item_bytes.size(),
                                   strb::STRBBlockId::kShapeOverrides, occurrence, shape_block,
                                   shape_block_size)) {
            break;
          }
          auto hiding_shape =
              BlendShape::Read(shape_block, shape_block_size, blend_shape_load_options);
          if (hiding_shape == nullptr) {
            REXKRNL_WARN("[avatar] failed to load embedded hiding shape {} for closet item {}",
                         occurrence, info.asset_id.to_string());
            continue;
          }
          // Marketplace shapes reference the body by pack index (male 0 /
          // female 1) where manifests use the canonical {a=2, b=0, c=gender}.
          if (hiding_shape->index_patch.original_asset_id.a.get() == 2) {
            hiding_shape->index_patch.original_asset_id.b = 0;
          }
          if (hiding_shape->vertex_patch.original_asset_id.a.get() == 2) {
            hiding_shape->vertex_patch.original_asset_id.b = 0;
          }
          blend_shapes.push_back({info.asset_id, hiding_shape});
        }
        continue;
      }
      const size_t pack_index = info.asset_id.b;
      if (pack_index >= pack_infos.size()) {
        continue;
      }
      const auto& id0 = pack_infos[pack_index].asset_ids[0];
      if (id0.is_zero() || id0.a.get() != 0x01000000u || id0.b.get() >= pack_infos.size()) {
        continue;
      }
      auto hiding_shape = LoadAsset<BlendShape, BlendShapeLoadOptions>(
          asset_pack, id0, blend_shape_load_options);
      if (hiding_shape == nullptr) {
        REXKRNL_WARN("[avatar] failed to load hiding shape {} for component {}",
                     id0.to_string(), info.asset_id.to_string());
        continue;
      }
      blend_shapes.push_back({id0, hiding_shape});
    }
  }

  std::vector<std::pair<X_AVATAR_COMPONENT_INFO, std::shared_ptr<Model>>>
      source_components;
  for (const auto& source_info : source_component_infos) {
    // The Avatar Editor stores face-morph selections as manifest components,
    // but their pack assets are shape-override STRBs (the 0x01000000-category
    // population carries no model blocks). Route those into the blend-shape
    // set (each blend shape names its target component internally) instead
    // of failing Model::Load and dropping the component entirely.
    {
      const uint8_t* raw_buffer = nullptr;
      size_t raw_size = 0;
      std::vector<uint8_t> raw_temp;
      if (LoadAsset(asset_pack, source_info.asset_id, raw_buffer, raw_size, raw_temp) &&
          strb::CountSTRBBlocks(raw_buffer, raw_size,
                                strb::STRBBlockId::kShapeOverrides) > 0 &&
          strb::CountSTRBBlocks(raw_buffer, raw_size, strb::STRBBlockId::kModel) == 0) {
        auto component_shape = BlendShape::Load(raw_buffer, raw_size,
                                                blend_shape_load_options);
        if (component_shape != nullptr) {
          blend_shapes.push_back({source_info.asset_id, component_shape});
        } else {
          REXKRNL_ERROR("Failed to load component blend shape {}!",
                        source_info.asset_id.to_string());
        }
        continue;
      }
    }
    auto model =
        LoadModelCached(asset_pack, source_info.asset_id, model_load_options);
    if (model != nullptr) {
      source_components.push_back({source_info, model});
      continue;
    }
    REXKRNL_ERROR("Failed to load avatar asset {}, looking for fallback...",
           source_info.asset_id.to_string());
    X_AVATAR_COMPONENT_INFO fallback_info{};
    for (const auto& candidate_info : metadata.fallback_components) {
      if (candidate_info.categories == source_info.categories) {
        // TODO(gibbed): if this fails... fall back even further?
        model = LoadModelCached(asset_pack, candidate_info.asset_id,
                                model_load_options);
        fallback_info = candidate_info;
        break;
      }
    }
    if (model == nullptr) {
      if (!fallback_info.asset_id.is_zero()) {
        REXKRNL_ERROR("Failed to load fallback avatar asset {}!",
               fallback_info.asset_id.to_string());
      }
      continue;
    }
    source_components.push_back({fallback_info, model});
  }

  std::shared_ptr<Texture> replacement_textures[6];
  // metadata.textures[] slots have fixed meanings, established by the head
  // model's own shader-parameter table (usage -> head texture index) plus the
  // usage_to_replacement_texture_indices map in ModelToGuest:
  //   slot 0 -> usage 12    = mouth      (0x8000, 14-frame stacks)
  //   slot 1 -> usages 9/10 = eyes       (0x2000, 14-frame stacks, L/R)
  //   slot 2 -> usages 7/8  = eyebrows   (0x4000, 5-frame stacks, L/R)
  //   slot 3 -> usage 6     = face paint (0x40000/0x10000 single-layer RGBA
  //                           decal)
  //   slot 4 -> usage 11    = eye shadow (0x40000 single-layer red-channel
  //                           mask, tinted by metadata.colors[5])
  //   slot 5 -> usage 5     = face tex   (0x20000, whole 128x128 face)
  // Positions are authoritative: that is how real XAM binds them, and eye
  // shadow and face paint share category 0x40000 so category alone cannot
  // disambiguate slots 3/4. Manifests with mis-slotted entries exist in the
  // wild, so keep every entry whose category is valid for its slot and
  // relocate mismatched entries to the first free slot that accepts their
  // category.
  static const uint32_t kSlotAcceptedCategories[6][2] = {
      {0x8000, 0},        // mouth
      {0x2000, 0},        // eyes
      {0x4000, 0},        // brows
      {0x40000, 0x10000}, // face paint
      {0x40000, 0x10000}, // eye shadow
      {0x20000, 0x10000}, // face texture
  };
  auto slot_accepts = [&](size_t slot, uint32_t category) {
    return kSlotAcceptedCategories[slot][0] == category ||
           kSlotAcceptedCategories[slot][1] == category;
  };
  std::shared_ptr<Texture> loaded_textures[6];
  size_t slot_source[6] = {SIZE_MAX, SIZE_MAX, SIZE_MAX, SIZE_MAX, SIZE_MAX, SIZE_MAX};
  for (size_t i = 0; i < 6; ++i) {
    const auto& texture_info = metadata.textures[i];
    if (texture_info.asset_id.is_zero()) {
      continue;
    }
    loaded_textures[i] = LoadAsset<Texture>(asset_pack, texture_info.asset_id);
    if (loaded_textures[i] == nullptr) {
      REXKRNL_ERROR("Failed to load avatar replacement texture {}!",
             texture_info.asset_id.to_string());
      continue;
    }
    // Pass 1: entries already in a slot that accepts their category.
    if (slot_accepts(i, texture_info.asset_id.a.get())) {
      replacement_textures[i] = loaded_textures[i];
      slot_source[i] = i;
      loaded_textures[i] = nullptr;
    }
  }
  for (size_t i = 0; i < 6; ++i) {
    // Pass 2: relocate category-mismatched leftovers.
    if (loaded_textures[i] == nullptr) {
      continue;
    }
    const uint32_t id_category = metadata.textures[i].asset_id.a.get();
    for (size_t s = 0; s < 6; ++s) {
      if (replacement_textures[s] == nullptr && slot_accepts(s, id_category)) {
        replacement_textures[s] = loaded_textures[i];
        slot_source[s] = i;
        break;
      }
    }
  }
  // Eye-whites preference mod (avatar_eye_whites in the toml): paint a whites
  // mask into eye styles whose art lacks one.
  if (REXCVAR_GET(avatar_eye_whites) && replacement_textures[1] != nullptr) {
    WhitenEyeStack(*replacement_textures[1]);
  }
  // The eye-shadow overlay (slot 4 / usage 11) is not stored in the manifest:
  // every eyes stack (0x2000) pairs 1:1 with its shadow overlay through
  // asset_ids[0] (a = 0x40000, and it is the shadow entry that carries the
  // "Eye shadow NNN" name). XAM binds the worn eyes' partner automatically;
  // the "Eye Shadow Color" UI only edits its tint (metadata.colors[5]).
  //
  // "None" gate: the editor's "no eyeshadow" swatch writes colors[5] =
  // FF000000, so opaque black is the sentinel rather than alpha 0, and the
  // manifest texture slots are untouched either way. Skip the auto-bind
  // entirely when the tint is the sentinel, or the overlay paints black lids.
  const bool eyeshadow_none = (metadata.colors[5].get() & 0xFFFFFFu) == 0;
  if (!eyeshadow_none &&
      replacement_textures[4] == nullptr && replacement_textures[1] != nullptr &&
      slot_source[1] != SIZE_MAX && asset_pack != nullptr &&
      IsStockPackId(metadata.textures[slot_source[1]].asset_id)) {
    const auto& eye_id = metadata.textures[slot_source[1]].asset_id;
    const auto& pack_infos = asset_pack->asset_infos();
    const size_t eye_index = eye_id.b;
    if (eye_index < pack_infos.size()) {
      const auto& id0 = pack_infos[eye_index].asset_ids[0];
      if (!id0.is_zero() && id0.a.get() == 0x40000u && id0.b.get() < pack_infos.size()) {
        auto shadow = LoadAsset<Texture>(asset_pack, id0);
        if (shadow != nullptr) {
          replacement_textures[4] = shadow;
        }
      }
    }
  }

  // Old-body item rescue (pre-Fall-2010 authored items): before applying any
  // hiding shape, pair each body-targeting shape with the body model it
  // matches, detect stale absolute tuck positions, and remap both the shape
  // and the owning garment's mesh into current body space (details in
  // RescueOldBodyItem). Must run before the apply loop so the tucks land
  // where the (now-rescaled) garment covers.
  for (auto& blend_shape : blend_shapes) {
    if (blend_shape.second->vertex_patch.original_asset_id.a.get() != 2u) {
      continue;
    }
    for (auto& body_candidate : source_components) {
      if (!blend_shape.second->matches(body_candidate.first.asset_id)) {
        continue;
      }
      std::shared_ptr<Model> owner_model;
      for (auto& owner_candidate : source_components) {
        if (owner_candidate.first.asset_id == blend_shape.first) {
          owner_model = owner_candidate.second;
          break;
        }
      }
      RescueOldBodyItem(blend_shape.second, body_candidate.second, owner_model,
                        LoadLegacyBody(blend_shape.second->vertex_patch.original_asset_id.c,
                                       model_load_options));
      break;
    }
  }

  for (const auto& source_component : source_components) {
    for (const auto& blend_shape : blend_shapes) {
      if (!blend_shape.second->matches(source_component.first.asset_id)) {
        continue;
      }
      if (!ApplyBlendShape(blend_shape.second, source_component.first.asset_id,
                           source_component.second)) {
        REXKRNL_ERROR("Failed to apply blend shape {} to asset {}!",
               blend_shape.first.to_string(),
               source_component.first.asset_id.to_string());
      }
      REXKRNL_ERROR("Applied blend shape {} to asset {}.",
             blend_shape.first.to_string(),
             source_component.first.asset_id.to_string());
    }
  }
  for (const auto& source_component : source_components) {
    WeldSeams(*source_component.second);
  }

  std::shared_ptr<Prop> prop = nullptr;
  X_AVATAR_COMPONENT_INFO prop_info{};
  if (want_prop) {
    for (const auto& component : metadata.components) {
      if (component.matches(ComponentCategory::kProp)) {
        prop_info = component;
        break;
      }
    }
    if (!prop_info.asset_id.is_zero()) {
      PropLoadOptions prop_load_options{};
      prop_load_options.model = model_load_options;
      prop_load_options.skeleton = skeleton_load_options;
      prop_load_options.animation = AnimationLoadOption::kGuest;
      prop_load_options.blend_shape = blend_shape_load_options;
      if (coordinate_system == 0) {
        prop_load_options.animation |= AnimationLoadOption::kInvert;
      }
      prop_load_options.model = model_load_options;
      prop = LoadAsset<Prop, PropLoadOptions>(asset_pack, prop_info.asset_id,
                                              prop_load_options);
    }
  }

  auto assets = cpu_memory->Claim<X_AVATAR_ASSETS>();
  *assets = {};

  uint32_t guest_skeleton_ptr;
  auto guest_skeleton =
      cpu_memory->Claim<X_AVATAR_SKELETON>(&guest_skeleton_ptr);

  if (!SkeletonToGuest(guest_skeleton, skeleton, cpu_memory)) {
    return false;
  }

  cpu_memory->SetPointer(&assets->skeleton_ptr, guest_skeleton_ptr);

  if (source_components.size()) {
    uint32_t component_infos_ptr;
    auto component_infos = cpu_memory->Claim<X_AVATAR_COMPONENT_INFO>(
        &component_infos_ptr, source_components.size());
    uint32_t component_models_ptr;
    auto component_models = cpu_memory->Claim<X_AVATAR_MODEL>(
        &component_models_ptr, source_components.size());

    assets->component_count = uint32_t(source_components.size());

    cpu_memory->SetPointer(&assets->component_infos_ptr, component_infos_ptr);
    cpu_memory->SetPointer(&assets->component_models_ptr, component_models_ptr);

    auto component_info = component_infos;
    auto component_model = component_models;
    int component_index = 0;
    for (const auto& source_component : source_components) {
      *component_info = source_component.first;

      const uint32_t effective_categories =
          uint32_t(source_component.first.categories);

      std::vector<ShaderParameterOverride> shader_parameter_overrides;
      GetShaderOverrides(metadata, effective_categories,
                         shader_parameter_overrides);

      // Closet (marketplace) items: the item's own custom color table wins
      // over the synthetic skin/hair/white tint for CUSTOM_0..2 (usages
      // 22/23/24). Later entries win inside OverrideShaderParameter's apply
      // loop, so pushing these after GetShaderOverrides overrides usage 22.
      if (!IsStockPackId(source_component.first.asset_id)) {
        ApplyItemColorTable(source_component.first.asset_id,
                            shader_parameter_overrides);
      }

      ModelToGuest(source_component.second, component_model, cpu_memory,
                   gpu_memory, effective_categories,
                   replacement_textures, shader_parameter_overrides);
      component_info++;
      component_model++;
      component_index++;
    }
  } else {
    assets->component_count = 0;
  }

  if (prop != nullptr) {
    uint32_t guest_prop_ptr;
    auto guest_prop = cpu_memory->Claim<X_AVATAR_PROP>(&guest_prop_ptr);

    *guest_prop = {};
    guest_prop->component_info = prop_info;

    uint32_t guest_prop_skeleton_ptr;
    auto guest_prop_skeleton =
        cpu_memory->Claim<X_AVATAR_SKELETON>(&guest_prop_skeleton_ptr);

    if (!SkeletonToGuest(guest_prop_skeleton, prop->skeleton, cpu_memory)) {
      return false;
    }

    cpu_memory->SetPointer(&guest_prop->skeleton_ptr, guest_prop_skeleton_ptr);

    // Carryables take the same palette rules as worn components: usage 22
    // (CUSTOM_0) white for stock props, and a closet prop's own
    // kCustomColorTable for CUSTOM_0..2. With no overrides the item's material
    // table supplies black placeholders in 22/23/24 and the BODY_* recolor
    // collapses to lerp(color, texel.r * 0, i.a), rendering the prop black.
    std::vector<ShaderParameterOverride> prop_overrides;
    GetShaderOverrides(metadata, ComponentCategory::kProp, prop_overrides);
    if (!IsStockPackId(prop_info.asset_id)) {
      ApplyItemColorTable(prop_info.asset_id, prop_overrides);
    }
    if (!ModelToGuest(prop->model, &guest_prop->component_model, cpu_memory,
                      gpu_memory, prop_info.categories, {}, prop_overrides)) {
      return false;
    }

    if (prop->animation != nullptr) {
      uint32_t guest_animation_ptr;
      auto guest_animation =
          cpu_memory->Claim<X_AVATAR_ANIMATION>(&guest_animation_ptr);

      guest_animation->compressed_data_size =
          static_cast<uint32_t>(prop->animation->compressed_data_bytes.size());

      uint32_t guest_compressed_data_buffer_ptr;
      auto guest_compressed_data_buffer =
          cpu_memory->ClaimBytes(&guest_compressed_data_buffer_ptr,
                                 guest_animation->compressed_data_size);

      if (!LoadAnimationToGuest(prop_info.asset_id, prop->animation,
                                guest_animation,
                                guest_compressed_data_buffer)) {
        return false;
      }

      cpu_memory->SetPointer(&guest_animation->compressed_data_buffer_ptr,
                             guest_compressed_data_buffer_ptr);

      cpu_memory->SetPointer(&guest_prop->animation_ptr, guest_animation_ptr);
    }

    cpu_memory->SetPointer(&assets->prop_ptr, guest_prop_ptr);
  }

  return true;
}

}  // namespace avatars
}  // namespace rex
