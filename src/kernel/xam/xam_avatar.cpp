/**
 ******************************************************************************
 * Xenia : Xbox 360 Emulator Research Project                                 *
 ******************************************************************************
 * Copyright 2025 Ben Vanik. All rights reserved.                             *
 * Released under the BSD license - see LICENSE in the root for more details. *
 ******************************************************************************
 *
 * @modified    Tom Clay, 2026 - Ported xenia-canary xam_avatar into ReXGlue.
 *
 * Port of xenia-canary's xam_avatar.cc. The avatar APIs complete their
 * overlapped requests gracefully (success, or clean failure for asset/manifest
 * queries) so titles that show the player's Xbox avatar can initialise the
 * avatar pipeline and handle "no avatar configured" offline. rexglue has no
 * per-profile avatar setting store, so the manifest get/set paths report
 * "no avatar" rather than reading/writing XPROFILE_GAMERCARD_AVATAR_INFO_1.
 */

#include <atomic>
#include <memory>
#include <thread>
#include <chrono>
#include <algorithm>
#include <functional>
#include <string_view>
#include <cstdio>
#include <cstring>
#include <cctype>
#include <cstdlib>
#include <filesystem>
#include <map>
#include <mutex>
#include <random>
#include <string>
#include <unordered_set>
#include <vector>

#include <rex/cvar.h>
#include <rex/hook.h>
#include <rex/kernel/title_id_utils.h>
#include <rex/kernel/xam/avatar_search.h>
#include <rex/kernel/xam/private.h>
#include <rex/kernel/xam/xam_avatar.h>
#include <rex/logging.h>
#include <rex/math.h>
#include <rex/runtime.h>
#include <rex/system.h>
#include <rex/system/kernel_state.h>
#include <rex/system/xtypes.h>
#include <rex/types.h>

#include "avatars/asset_pack.h"
#include "avatars/closet.h"
#include "marketplace.h"
#include "avatars/guest_animation.h"
#include "avatars/guest_asset.h"
#include "avatars/guest_load_animation.h"
#include "avatars/guest_load_asset.h"
#include "avatars/memory_block.h"

// game_data_root is defined in the runtime (runtime.cpp, global namespace);
// read here to locate the title's AI avatar looks under data/art/_avatar/.
REXCVAR_DECLARE(std::string, game_data_root);
REXCVAR_DECLARE(bool, avatar_marketplace);

// Native-renderer guest-texture notifications, called on the rexvideonative
// renderer queues directly. The exe links the videonative static lib; both
// functions only append to queues and are safe with no active renderer.
namespace rex::videonative::renderer {
void QueueGuestTextureFreeze(uint32_t guest_address, uint32_t size);
void QueueGuestTextureInvalidate(uint32_t guest_address, uint32_t size);
}  // namespace rex::videonative::renderer

// Enable Avatar Initialization. Some games require a full avatar implementation
// and may crash; enabled by default in this build so avatar-using titles boot.
REXCVAR_DEFINE_BOOL(allow_avatar_initialization, true, "Kernel",
                    "Enable Avatar Initialization (some games need a full avatar impl)");

// Host directory containing the extracted avatar asset-pack TOCs
// (AvatarAssetPack.toc + AvatarAssetPackLegacyV1.toc), extracted from the
// FFFE07DF00000002 PIRS package. Empty = game_data_root, where the documented
// asset layout puts them; set it only for packs kept outside the game data.
REXCVAR_DEFINE_STRING(avatar_asset_pack_dir, "", "Kernel",
                      "Full path of the directory with the extracted "
                      "AvatarAssetPack*.toc files. Empty = game_data_root.");

// The closet (imported marketplace/award avatar items) can live apart from
// the pack so it ships with a title's assets folder: saved outfits reference
// closet items by GUID, so a machine without the same closet hides those
// outfits from the editor's grids. A relative path resolves against
// game_data_root (same convention as content_root).
REXCVAR_DEFINE_STRING(avatar_closet_dir, "", "Kernel",
                      "Directory with imported closet items (<guid>.bin + closet_index.tsv, "
                      "built by avatarextract --closet-import). Empty = "
                      "<avatar_asset_pack_dir>\\closet. Relative paths resolve against "
                      "game_data_root.");

namespace rex {
namespace kernel {
namespace xam {

// --- Avatar asset pipeline state --------------------------------------------
static avatars::AssetPack g_avatar_asset_pack;
static avatars::AssetPack g_legacy_avatar_asset_pack;
static uint32_t g_title_version = 0;
static uint32_t g_coordinate_system = 0;

// Captured from the first successful GetAssets call (avatar metadata as
// big-endian guest bytes) and replayed from GetManifestLocalUser.
static uint8_t g_captured_metadata[1000];
static bool g_have_metadata = false;
// Guest address of the metadata buffer the game last asked GetManifestLocalUser
// to fill. GetAssets pushes the real manifest into it once available, so the
// avatar model master's "metadata not set" retry succeeds even when the
// manifest read happened before GetAssets ran.
static uint32_t g_last_manifest_dest = 0;

// Legacy per-CWD manifest cache, still read as a fallback.
static const char* kLegacyMetaCachePath = "legacy_avatar_meta.bin";

REXCVAR_DEFINE_STRING(avatar_data_dir, "", "Kernel",
                      "Shared cross-title folder for persisted avatar data: the manifest the "
                      "Avatar Editor saves and every game then loads as the local user's "
                      "avatar. Empty = <user_data_root>\\avatars (the shared "
                      "Documents\\ReXGlue\\userdata\\avatars by default).");

static std::string AvatarManifestPath() {
  std::string dir = REXCVAR_GET(avatar_data_dir);
  if (dir.empty()) {
    // Live inside the shared user-data tree so saves + avatar travel as one
    // folder (and any user_data_root override carries the avatar with it).
    const auto& root = REX_KERNEL_STATE()->emulator()->user_data_root();
    if (!root.empty()) {
      dir = (root / "avatars").string();
    }
  }
  if (dir.empty()) {
    const char* profile = std::getenv("USERPROFILE");
    dir = profile ? std::string(profile) + "\\Documents\\ReXGlue\\userdata\\avatars"
                  : std::string("avatars");
  }
  std::error_code ec;
  std::filesystem::create_directories(dir, ec);
  return dir + "\\avatar_manifest.bin";
}

std::string AvatarDataDir() {
  const std::string path = AvatarManifestPath();
  const size_t cut = path.find_last_of("\\/");
  return cut == std::string::npos ? std::string(".") : path.substr(0, cut);
}

static void WriteHostFile(const std::string& path, const uint8_t* data, size_t size) {
  FILE* f = std::fopen(path.c_str(), "wb");
  if (!f) {
    return;
  }
  std::fwrite(data, 1, size, f);
  std::fclose(f);
}

static std::vector<uint8_t> ReadHostFile(const std::string& path) {
  std::vector<uint8_t> data;
  FILE* f = std::fopen(path.c_str(), "rb");
  if (!f) {
    return data;
  }
  std::fseek(f, 0, SEEK_END);
  long size = std::ftell(f);
  std::fseek(f, 0, SEEK_SET);
  if (size > 0) {
    data.resize(static_cast<size_t>(size));
    size_t read = std::fread(data.data(), 1, data.size(), f);
    data.resize(read);
  }
  std::fclose(f);
  return data;
}

// Read the persisted manifest: shared cross-title location first, then the
// legacy per-CWD cache file.
static std::vector<uint8_t> ReadPersistedManifest(std::string* out_path) {
  std::string path = AvatarManifestPath();
  auto data = ReadHostFile(path);
  if (data.empty()) {
    path = kLegacyMetaCachePath;
    data = ReadHostFile(path);
  }
  if (out_path) {
    *out_path = path;
  }
  return data;
}

// Some titles read the player's avatar manifest from the profile
// (XamUserReadProfileSettings, XPROFILE_GAMERCARD_AVATAR_INFO_1 = 0x63E80044)
// rather than through XamAvatarGetManifestLocalUser. Back that setting with the
// persisted manifest, refreshing from disk so a save made mid-session is picked
// up on the next read.
static bool LoadAvatarAssetPack();
static void BuildRandomAvatarMetadata(avatars::X_AVATAR_METADATA* out, uint32_t body_mask,
                                      uint32_t salt);

void EnsureAvatarProfileSetting(system::xam::UserProfile* profile) {
  if (!profile) {
    return;
  }
  constexpr uint32_t kAvatarInfo1SettingId = 0x63E80044u;
  auto manifest = ReadPersistedManifest(nullptr);
  if (manifest.size() != sizeof(avatars::X_AVATAR_METADATA) && LoadAvatarAssetPack()) {
    // Fresh user: no persisted avatar yet. Consoles hand new profiles a
    // randomly generated avatar, so synthesize one (title preset manifests
    // preferred) and persist it; it sticks across boots and can be
    // customized in the Avatar Editor later.
    avatars::X_AVATAR_METADATA random_meta;
    BuildRandomAvatarMetadata(&random_meta, 3, 1);
    random_meta.owner_xuid = profile->xuid();  // player-owned, not an NPC build
    const auto* bytes = reinterpret_cast<const uint8_t*>(&random_meta);
    manifest.assign(bytes, bytes + sizeof(random_meta));
    WriteHostFile(AvatarManifestPath(), manifest.data(), manifest.size());
  }
  if (manifest.size() != sizeof(avatars::X_AVATAR_METADATA)) {
    return;
  }
  auto* setting = profile->GetSetting(kAvatarInfo1SettingId);
  if (setting) {
    setting->Deserialize(std::move(manifest));
  } else {
    profile->AddSetting(std::make_unique<system::xam::UserProfile::BinarySetting>(
        kAvatarInfo1SettingId, manifest));
  }
}

// A pack that cannot be read surfaces as an error instead of an editor with
// empty catalogs, which is what the guest sees when the enumeration comes back
// with nothing. Reported once: the loader is reached from several lazy sites.
static void ReportAssetPackMissing(const std::string& detail) {
  static bool reported = false;
  if (reported) {
    return;
  }
  reported = true;
  const auto msg =
      detail +
      "\n\nPut AvatarAssetPack.toc and AvatarAssetPackLegacyV1.toc in the game data "
      "folder, or set avatar_asset_pack_dir in avatareditor.toml to the folder "
      "holding them.";
  REXKRNL_ERROR("[avatar] {}", msg);
  rex::ShowSimpleMessageBox(rex::SimpleMessageBoxType::Error, msg);
}

static bool LoadAvatarAssetPack() {
  if (g_avatar_asset_pack.is_loaded()) {
    return true;
  }
  // The TOCs ship alongside the rest of the title's data, so game_data_root is
  // the default location; avatar_asset_pack_dir only has to name a pack kept
  // somewhere else.
  std::string dir = REXCVAR_GET(avatar_asset_pack_dir);
  if (dir.empty()) {
    dir = REXCVAR_GET(game_data_root);
  }
  if (dir.empty()) {
    ReportAssetPackMissing(
        "The avatar asset packs were not found: neither avatar_asset_pack_dir nor "
        "game_data_root is set.");
    return false;
  }
  auto toc = ReadHostFile(dir + "/AvatarAssetPack.toc");
  if (toc.empty() || !g_avatar_asset_pack.Load(toc)) {
    ReportAssetPackMissing(
        fmt::format("The avatar asset packs were not found: no readable AvatarAssetPack.toc in {}", dir));
  }
  auto legacy = ReadHostFile(dir + "/AvatarAssetPackLegacyV1.toc");
  if (!legacy.empty()) {
    g_legacy_avatar_asset_pack.Load(legacy);
    // The legacy pack doubles as the old->new body correspondence source for
    // the old-body item rescue (guest_load_asset.cpp).
    avatars::SetLegacyAssetPack(
        g_legacy_avatar_asset_pack.is_loaded() ? &g_legacy_avatar_asset_pack : nullptr);
  }
  // Imported marketplace/award items (avatarextract --closet-import): by
  // default a closet/ subdirectory next to the pack; avatar_closet_dir
  // relocates it (e.g. into the Avatar Editor's shipped assets folder).
  std::filesystem::path closet_dir = std::filesystem::path(dir) / "closet";
  const std::string closet_override = REXCVAR_GET(avatar_closet_dir);
  if (!closet_override.empty()) {
    closet_dir = closet_override;
    if (closet_dir.is_relative()) {
      const std::string game_root = REXCVAR_GET(game_data_root);
      if (!game_root.empty()) {
        closet_dir = std::filesystem::path(game_root) / closet_dir;
      }
    }
  }
  avatars::GetCloset().Load(closet_dir);
  return g_avatar_asset_pack.is_loaded();
}

// Start/End ------------------------------------------------------------------
u32 XamAvatarInitialize_entry(u32 version,            // 1, 3, 4, etc
                              u32 coordinate_system,  // 0 or 1
                              u32 processor_number,    // for thread creation?
                              mapped_u32 function_ptrs,  // 20b, 5 pointers
                              mapped_void unk5,          // data segment ptr
                              u32 unk6                   // flags - 0x00300000, 0x30
) {
  if (REX_KERNEL_STATE()->title_id() == kAvatarEditorID) {
    // The editor is the native avatar UI: skip the game-oriented init
    // (function-pointer capture), but record the version and load the asset
    // pack so GetAssets can serve it. Also load the persisted manifest so the
    // editor opens on the last saved avatar rather than a fresh random one.
    g_title_version = version;
    g_coordinate_system = coordinate_system;
    LoadAvatarAssetPack();
    if (!g_have_metadata) {
      std::string manifest_path;
      auto cached = ReadPersistedManifest(&manifest_path);
      if (cached.size() == sizeof(g_captured_metadata)) {
        std::memcpy(g_captured_metadata, cached.data(), sizeof(g_captured_metadata));
        g_have_metadata = true;
      }
    }
    // The editor's save flow drives an async validate-manifest + write-gamerpic
    // operation whose object lives at app+0x224A1B0 (static guest 0x94530040).
    // Its creator never runs in this environment, so provide a zeroed
    // placeholder: every field the op's pump touches (status @+12, XOVERLAPPED
    // @+36, flags @+64/+72, buffer slots @dwords 21..33) is plain data, and
    // every terminal path on a zeroed op exits the save screen to the menu.
    {
      auto* mem = REX_KERNEL_MEMORY();
      auto* op_field = mem->TranslateVirtual<rex::be<uint32_t>*>(0x94530040u);
      if (!*op_field) {
        const uint32_t op = mem->SystemHeapAlloc(160);
        if (op) {
          std::memset(mem->TranslateVirtual<uint8_t*>(op), 0, 160);
          *op_field = op;
        }
      }
    }
    return X_STATUS_SUCCESS;
  }
  if (!REXCVAR_GET(allow_avatar_initialization)) {
    return ~0u;  // game calls XamAvatarShutdown and runs avatar-less.
  }
  g_title_version = version;
  g_coordinate_system = coordinate_system;
  LoadAvatarAssetPack();  // best-effort; GetAssets fails gracefully if absent.
  // Load the cached avatar manifest from a prior launch so valid metadata is
  // available before the model master builds the avatar. The first launch has
  // no cache yet.
  if (!g_have_metadata) {
    std::string manifest_path;
    auto cached = ReadPersistedManifest(&manifest_path);
    if (cached.size() == sizeof(g_captured_metadata)) {
      std::memcpy(g_captured_metadata, cached.data(), sizeof(g_captured_metadata));
      g_have_metadata = true;
    }
  }
  return X_STATUS_SUCCESS;
}

void XamAvatarShutdown_entry() {
  // No-op. (Real: XMsgStartIORequestEx(0xf3,0x600002,...) / XamUnloadSysApp.)
}

// Get & Set ------------------------------------------------------------------
u32 XamAvatarGetManifestLocalUser_entry(u32 user_index,
                                        ppc_ptr_t<X_AVATAR_METADATA> avatar_metadata_ptr,
                                        mapped_void overlapped_ptr) {
  // Remember where the game wants the manifest written so GetAssets can fill it
  // once it has the real metadata (the read usually happens before GetAssets).
  if (avatar_metadata_ptr) {
    g_last_manifest_dest = avatar_metadata_ptr.guest_address();
    // With metadata cached from a prior launch, fill the buffer synchronously
    // so the model master reads a real manifest.
    if (g_have_metadata) {
      std::memcpy(REX_KERNEL_MEMORY()->TranslateVirtual<uint8_t*>(g_last_manifest_dest),
                  g_captured_metadata, sizeof(g_captured_metadata));
    }
  }
  auto run = [=](uint32_t& extended_error, uint32_t& length) -> X_RESULT {
    extended_error = X_ERROR_SUCCESS;
    length = 0;

    if (user_index >= 4) {
      extended_error = X_E_INVALIDARG;
      return X_ERROR_INVALID_PARAMETER;
    }
    if (!avatar_metadata_ptr) {
      extended_error = X_E_INVALIDARG;
      return X_ERROR_INVALID_PARAMETER;
    }
    // Replay the manifest captured by a prior GetAssets so the caller builds a
    // real avatar instead of falling back to "no avatar".
    if (g_have_metadata && avatar_metadata_ptr) {
      auto* out = REX_KERNEL_MEMORY()->TranslateVirtual<uint8_t*>(
          avatar_metadata_ptr.guest_address());
      std::memcpy(out, g_captured_metadata, sizeof(g_captured_metadata));
      return X_ERROR_SUCCESS;
    }
    extended_error = X_E_FAIL;
    return X_ERROR_FUNCTION_FAILED;
  };

  if (!overlapped_ptr) {
    uint32_t extended_error = 0, length = 0;
    X_RESULT result = run(extended_error, length);
    return result == X_ERROR_SUCCESS ? result : extended_error;
  }

  REX_KERNEL_STATE()->CompleteOverlappedDeferredEx(run, overlapped_ptr.guest_address());
  return X_ERROR_IO_PENDING;
}

u32 XamAvatarGetManifestsByXuid_entry(u32 user_index, u32 xuid_count, mapped_u64 xuid, u32 unk,
                                      u32 avatar_info_ptr, mapped_void overlapped_ptr) {
  if (overlapped_ptr) {
    REX_KERNEL_STATE()->CompleteOverlappedImmediate(overlapped_ptr.guest_address(),
                                                    X_ERROR_SUCCESS);
    return X_ERROR_IO_PENDING;
  }
  return X_STATUS_SUCCESS;
}

// Per-component CPU/GPU buffer sizes (avatars::ComponentCategory order).
namespace {
struct AvatarComponentAssetInfo {
  uint32_t cpu_size;
  uint32_t gpu_size;
};
const AvatarComponentAssetInfo kAvatarComponentAssetInfos[] = {
    {0x00000800, 0x00053800},  // kHead
    {0x00000400, 0x00012C00},  // kBody
    {0x00000800, 0x0000E000},  // kHair
    {0x00000800, 0x00014400},  // kTop
    {0x00000800, 0x0000F000},  // kBottom
    {0x00000800, 0x0000D400},  // kShoes
    {0x00000800, 0x0000A400},  // kHat
    {0x00000800, 0x00006800},  // kGloves
    {0x00000800, 0x00008400},  // kGlasses
    {0x00000800, 0x00005C00},  // kWristwear
    {0x00000800, 0x00005800},  // kEarrings
    {0x00000800, 0x00005C00},  // kRing
    {0x00000800, 0x00019000},  // kProp
};
}  // namespace

// ---------------------------------------------------------------------------
// Random avatar metadata (XamAvatarGetMetadataRandom).
//
// Synthesizes a valid manifest: stock male/female body plus random pack assets
// for the visible wardrobe categories, with version/factors/colors inherited
// from the cached player metadata when available.
//
// Synthesized metadata is stamped with kRandomAvatarXuid in owner_xuid so
// GetAssets can tell it from the player's build: those skip the local-metadata
// capture/cache and the custom-avatar substitution.
static constexpr uint64_t kRandomAvatarXuid = 0x4E50435241564154ull;  // "NPCRAVAT"

REXCVAR_DEFINE_INT32(avatar_bake_pace_ms, 16, "Kernel",
                     "Minimum gap between non-player XamAvatarGetAssets completions. A "
                     "catalog page fires 8 back-to-back mannequin builds whose tile bakes "
                     "otherwise pile into 1-2 frames (the grid-flip fps dip); pacing "
                     "completions spreads the bakes one per frame. Player builds (equips) "
                     "are never paced. 0 = off.");

// Bake pacing (see avatar_bake_pace_ms): called at the tail of a non-player
// GetAssets build, before its completion is signaled. The guest kicks each
// tile bake off that completion, so spacing completions spaces the bakes.
static void PaceBakeCompletion() {
  const int32_t pace = REXCVAR_GET(avatar_bake_pace_ms);
  if (pace <= 0) return;
  static std::mutex pace_mx;
  static std::chrono::steady_clock::time_point pace_next{};
  std::chrono::steady_clock::time_point wait_until;
  {
    std::lock_guard<std::mutex> lk(pace_mx);
    const auto now = std::chrono::steady_clock::now();
    wait_until = pace_next > now ? pace_next : now;
    pace_next = wait_until + std::chrono::milliseconds(pace);
  }
  std::this_thread::sleep_until(wait_until);
}
// The GUID `d` tail every constructed pack-asset id carries (matches the
// stock bodies and the ids inside the Avatar Editor's shipped presets).
static constexpr uint8_t kPackAssetGuidTail[8] = {0xC1, 0xC8, 0xF1, 0x09,
                                                  0xA1, 0x9C, 0xB2, 0xE0};

// Pick a random pack asset of `category` and produce a metadata AssetId that
// FindAsset (a pure index on AssetId.b) resolves back to it. Two populations
// exist in the V2 pack:
//  - legacy entries: `categories` is the raw bit and asset_ids[0] is a real,
//    self-indexed id, used verbatim;
//  - V2 entries: `categories == category << 16` and asset_ids are zero; the
//    id must be constructed as {a=category<<16, b=index, c=3, d=tail}, exactly
//    the encoding found in the editor's shipped Avatar*.Avatar manifests
//    (heads exist only in this population, so a raw-bit pick never sources a
//    head).
static bool PickRandomPackAsset(const avatars::AssetPack& pack, uint32_t category,
                                uint32_t body_bit, uint32_t& rng, avatars::AssetId* out_id) {
  auto next = [&rng]() {
    rng = rng * 1664525u + 1013904223u;
    return rng >> 8;
  };
  const auto& infos = pack.asset_infos();
  auto matches = [&](size_t i) {
    const auto& info = infos[i];
    const bool legacy_match = (info.categories & category) && !info.asset_ids[0].is_zero() &&
                              info.asset_ids[0].b.get() == i;
    const bool v2_match = info.categories == (category << 16);
    if (!legacy_match && !v2_match) return false;
    // Respect the pack's body suitability when it declares one.
    const uint8_t suit = info.random_bodies ? info.random_bodies : info.bodies;
    return !suit || (suit & body_bit) != 0;
  };
  uint32_t n_match = 0;
  for (size_t i = 0; i < infos.size(); ++i) {
    if (matches(i)) ++n_match;
  }
  if (!n_match) {
    return false;
  }
  uint32_t pick = next() % n_match;
  for (size_t i = 0; i < infos.size(); ++i) {
    if (matches(i) && !pick--) {
      const auto& info = infos[i];
      if (!info.asset_ids[0].is_zero() && info.asset_ids[0].b.get() == i) {
        *out_id = info.asset_ids[0];
      } else {
        *out_id = {};
        out_id->a = category << 16;
        out_id->b = static_cast<uint16_t>(i);
        // .c = the asset's gender/bodies mask (see the enumerator note).
        out_id->c = info.bodies ? info.bodies : 3;
        std::memcpy(out_id->d, kPackAssetGuidTail, sizeof(out_id->d));
      }
      return true;
    }
  }
  return false;
}

// Random pick from the grid-visible stock catalog: mesh population only (raw
// category bits), catalog halves of paired wearables only. The id is
// constructed exactly the way the enumerator hands ids out.
static bool PickRandomCatalogAsset(const avatars::AssetPack& pack,
                                   const std::vector<bool>& is_companion, uint32_t category,
                                   uint32_t body_bit, uint32_t& rng, avatars::AssetId* out_id,
                                   uint16_t* out_categories) {
  auto next = [&rng]() {
    rng = rng * 1664525u + 1013904223u;
    return rng >> 8;
  };
  const auto& infos = pack.asset_infos();
  auto matches = [&](size_t i) {
    const auto& info = infos[i];
    if (info.categories >= 0x10000u || !(info.categories & category)) return false;
    if (is_companion[i] || !(info.flags & 1)) return false;
    const uint8_t suit = info.random_bodies ? info.random_bodies : info.bodies;
    return !suit || (suit & body_bit) != 0;
  };
  uint32_t n_match = 0;
  for (size_t i = 0; i < infos.size(); ++i) {
    if (matches(i)) ++n_match;
  }
  if (!n_match) {
    return false;
  }
  uint32_t pick = next() % n_match;
  for (size_t i = 0; i < infos.size(); ++i) {
    if (matches(i) && !pick--) {
      const auto& info = infos[i];
      avatars::AssetId id = info.asset_ids[0];
      if (id.is_zero() || id.b.get() != static_cast<uint16_t>(i)) {
        id.a = info.categories;
        id.b = static_cast<uint16_t>(i);
        id.c = info.bodies ? info.bodies : 3;
        std::memcpy(id.d, kPackAssetGuidTail, sizeof(id.d));
      }
      *out_id = id;
      *out_categories = static_cast<uint16_t>(info.categories);
      return true;
    }
  }
  return false;
}

// Re-dress a preset-derived avatar with random stock catalog picks. The
// preset guarantees a loader-valid base (body, head, blend shapes); this
// supplies the variety a console's random lineup has.
static void RandomizeAvatarFromCatalog(avatars::X_AVATAR_METADATA* out, uint32_t body_bit,
                                       uint32_t& rng) {
  namespace cat = avatars::ComponentCategory;
  avatars::AssetPack* pack =
      (g_title_version < 3) ? &g_legacy_avatar_asset_pack : &g_avatar_asset_pack;
  if (!pack->is_loaded()) {
    pack = pack == &g_avatar_asset_pack ? &g_legacy_avatar_asset_pack : &g_avatar_asset_pack;
  }
  if (!pack->is_loaded()) {
    return;
  }
  auto next = [&rng]() {
    rng = rng * 1664525u + 1013904223u;
    return rng >> 8;
  };
  // Catalog halves of paired wearables (see the enumerator).
  const auto& infos = pack->asset_infos();
  std::vector<bool> is_companion(infos.size(), false);
  for (size_t index = 0; index < infos.size(); ++index) {
    const auto& id0 = infos[index].asset_ids[0];
    const size_t partner = id0.b;
    if (!id0.is_zero() && partner != index && partner < infos.size()) {
      is_companion[partner] = true;
    }
  }
  // Always dress the four core slots, sometimes accessorize. A pick whose
  // categories already cover a later slot (a one-piece top) skips it.
  const struct {
    uint32_t category;
    uint32_t chance_pct;
  } kSlots[] = {{cat::kHair, 100}, {cat::kTop, 100}, {cat::kBottom, 100},
                {cat::kShoes, 100}, {cat::kHat, 25},  {cat::kGlasses, 15}};
  avatars::X_AVATAR_COMPONENT_INFO core[4] = {};
  uint32_t n_core = 0;
  std::memset(out->components, 0, sizeof(out->components));
  std::memset(out->fallback_components, 0, sizeof(out->fallback_components));
  uint32_t slot = 0, covered = 0;
  for (const auto& s : kSlots) {
    if (slot >= rex::countof(out->components)) {
      break;
    }
    if (covered & s.category) {
      continue;
    }
    if ((next() % 100u) >= s.chance_pct) {
      continue;
    }
    avatars::AssetId id;
    uint16_t categories = 0;
    if (!PickRandomCatalogAsset(*pack, is_companion, s.category, body_bit, rng, &id,
                                &categories)) {
      continue;
    }
    out->components[slot].asset_id = id;
    out->components[slot].categories = categories;
    ++slot;
    covered |= categories;
    if ((s.category & (cat::kHair | cat::kTop | cat::kBottom | cat::kShoes)) &&
        n_core < rex::countof(core)) {
      core[n_core].asset_id = id;
      core[n_core].categories = categories;
      ++n_core;
    }
  }
  // Presets list their fallback wardrobe shoes-first; imitate that.
  for (uint32_t i = 0; i < n_core; ++i) {
    out->fallback_components[i] = core[n_core - 1 - i];
  }
  // Swap the brow/eye/mouth texture stacks too, keeping the preset's
  // placement floats.
  const uint32_t kFaceCats[] = {0x8000u, 0x2000u, 0x4000u};
  for (uint32_t fc : kFaceCats) {
    for (auto& tex : out->textures) {
      if (tex.asset_id.is_zero() || tex.asset_id.a.get() != fc) {
        continue;
      }
      avatars::AssetId id;
      uint16_t categories = 0;
      if (PickRandomCatalogAsset(*pack, is_companion, fc, body_bit, rng, &id, &categories)) {
        tex.asset_id = id;
      }
      break;
    }
  }
}

// Title-shipped complete avatar manifests (the Avatar Editor carries
// Avatar*.Avatar files, each exactly 1000 bytes of X_AVATAR_METADATA with a
// full body+head+wardrobe authored by the title). When present these are the
// preferred source for "random" avatars: they are guaranteed to load
// end-to-end, while pack-pick synthesis depends on loader validation rules
// that are only partially documented.
static const std::vector<std::vector<uint8_t>>& LoadPresetAvatarManifests() {
  static std::vector<std::vector<uint8_t>> presets;
  static bool attempted = false;
  if (!attempted) {
    attempted = true;
    const std::string root = REXCVAR_GET(game_data_root);
    if (!root.empty()) {
      // Recursive: titles ship preset manifests in different spots and
      // formats; the Avatar Editor carries Avatar*.Avatar at the root, and
      // other titles ship theirs as data/art/_avatar/credits/*.amd. Each is
      // exactly one X_AVATAR_METADATA.
      std::error_code ec;
      for (auto it = std::filesystem::recursive_directory_iterator(root, ec);
           it != std::filesystem::recursive_directory_iterator(); it.increment(ec)) {
        if (ec) break;
        if (!it->is_regular_file(ec)) continue;
        auto ext = it->path().extension().string();
        std::transform(ext.begin(), ext.end(), ext.begin(), ::tolower);
        if (ext != ".avatar" && ext != ".amd") continue;
        auto data = ReadHostFile(it->path().string());
        if (data.size() == sizeof(avatars::X_AVATAR_METADATA)) {
          presets.push_back(std::move(data));
        }
      }
    }
  }
  return presets;
}

static void BuildRandomAvatarMetadata(avatars::X_AVATAR_METADATA* out, uint32_t body_mask,
                                      uint32_t salt) {
  namespace cat = avatars::ComponentCategory;
  std::memset(out, 0, sizeof(*out));

  // Prefer a title-shipped preset manifest when one matches the requested
  // body type (GUID.c: 1 = male, 2 = female on the stock bodies): it
  // guarantees a loader-valid base. Few presets ship, so re-dress the copy
  // with random catalog picks for variety.
  {
    const auto& presets = LoadPresetAvatarManifests();
    if (!presets.empty()) {
      uint32_t rng = salt * 2654435761u + 0x9E3779B9u;
      auto next = [&rng]() {
        rng = rng * 1664525u + 1013904223u;
        return rng >> 8;
      };
      next();
      for (size_t k = 0; k < presets.size(); ++k) {
        const auto& raw = presets[(rng + k) % presets.size()];
        const auto* meta = reinterpret_cast<const avatars::X_AVATAR_METADATA*>(raw.data());
        const uint16_t gender = meta->body_component.asset_id.c.get();
        if (!(body_mask & (gender == 1 ? 1u : 2u))) continue;
        std::memcpy(out, raw.data(), sizeof(*out));
        out->owner_xuid = kRandomAvatarXuid;
        // Palette from a random preset (they are body-neutral) and a light
        // build shuffle, then random stock wardrobe over the preset base.
        const auto* palette = reinterpret_cast<const avatars::X_AVATAR_METADATA*>(
            presets[next() % presets.size()].data());
        std::memcpy(out->colors, palette->colors, sizeof(out->colors));
        out->weight_factor = out->weight_factor * (0.90f + 0.01f * (next() % 21u));
        out->height_factor = out->height_factor * (0.95f + 0.01f * (next() % 11u));
        // Fresh avatars never start with eye shadow: opaque black is the
        // editor's own "none" sentinel (see the loader's slot-4 gate).
        out->colors[5] = 0xFF000000u;
        RandomizeAvatarFromCatalog(out, gender == 1 ? 1u : 2u, rng);
        return;
      }
    }
  }

  // Inherit title-proven scalars from the player's captured metadata when it is
  // available; otherwise safe defaults.
  if (g_have_metadata) {
    const auto* base = reinterpret_cast<const avatars::X_AVATAR_METADATA*>(g_captured_metadata);
    out->version = base->version;
    out->weight_factor = base->weight_factor;
    out->height_factor = base->height_factor;
    std::memcpy(out->colors, base->colors, sizeof(out->colors));
  } else {
    out->version = 1;
    out->weight_factor = 1.0f;
    out->height_factor = 1.0f;
    for (auto& c : out->colors) c = 0xFF808080u;
    out->colors[0] = 0xFFE8C098u;  // skin
    out->colors[1] = 0xFF54341Cu;  // hair
  }
  // Fresh avatars never start with eye shadow (opaque black = "none").
  out->colors[5] = 0xFF000000u;

  uint32_t rng = salt * 2654435761u + 0x9E3779B9u;
  auto next = [&rng]() {
    rng = rng * 1664525u + 1013904223u;
    return rng >> 8;
  };

  // Stock body: exactly the ids GetBodyType (guest_load_asset.cpp) recognizes.
  const bool male = (body_mask & 1u) && (!(body_mask & 2u) || (next() & 1u));
  const avatars::AssetId body_id = {2, 0, static_cast<uint16_t>(male ? 1 : 2),
                                    {0xC1, 0xC8, 0xF1, 0x09, 0xA1, 0x9C, 0xB2, 0xE0}};
  const uint32_t body_bit = male ? 1u : 2u;
  out->body_component.asset_id = body_id;
  out->body_component.categories = static_cast<uint16_t>(cat::kBody);

  avatars::AssetPack* pack =
      (g_title_version < 3) ? &g_legacy_avatar_asset_pack : &g_avatar_asset_pack;
  if (!pack->is_loaded()) {
    // Fall back to whichever pack did load (the Avatar Editor initializes with
    // a low version but ships only the V2 pack, and vice versa for old dumps).
    pack = pack == &g_avatar_asset_pack ? &g_legacy_avatar_asset_pack : &g_avatar_asset_pack;
  }
  if (pack->is_loaded()) {
    avatars::AssetId picked;
    if (PickRandomPackAsset(*pack, cat::kHead, body_bit, rng, &picked)) {
      out->head_component.asset_id = picked;
      out->head_component.categories = static_cast<uint16_t>(cat::kHead);
    }
    const struct {
      uint32_t category;
      uint32_t chance_pct;
    } kSlots[] = {{cat::kHair, 100}, {cat::kTop, 100}, {cat::kBottom, 100},
                  {cat::kShoes, 100}, {cat::kHat, 35},  {cat::kGlasses, 20}};
    uint32_t slot = 0;
    for (const auto& s : kSlots) {
      if (slot >= rex::countof(out->components)) {
        break;
      }
      if ((next() % 100u) >= s.chance_pct) {
        continue;
      }
      if (PickRandomPackAsset(*pack, s.category, body_bit, rng, &picked)) {
        out->components[slot].asset_id = picked;
        out->components[slot].categories = static_cast<uint16_t>(s.category);
        ++slot;
      }
    }
  }
  out->owner_xuid = kRandomAvatarXuid;
}

// Buffer sizes a build with `mask` needs. The table budgets stock categories,
// but a worn component covering categories outside the mask (a fused costume
// is top+bottom+shoes) is still emitted whole, so its other categories join
// the budget, and marketplace items get headroom the stock figures lack. The
// title places its own data right after these sizes; overrunning them puts
// the avatar's bone palette under the next build's bytes.
static void AssetBuildBudget(uint32_t mask, uint32_t* cpu_size, uint32_t* gpu_size) {
  uint32_t budget = mask & 0x1fffu;
  if (g_have_metadata) {
    const auto* meta =
        reinterpret_cast<const avatars::X_AVATAR_METADATA*>(g_captured_metadata);
    for (const auto& comp : meta->components) {
      const uint32_t cats = comp.categories;
      if (!comp.asset_id.is_zero() && (cats & mask)) {
        budget |= cats & 0x1fffu;
      }
    }
  }
  uint32_t cpu = !(mask & avatars::ComponentCategory::kProp) ? 0x1F80u : 0x3147Cu;
  uint32_t gpu = 0;
  for (size_t i = 0; i < rex::countof(kAvatarComponentAssetInfos); ++i) {
    if (budget & (1u << i)) {
      cpu += rex::align(kAvatarComponentAssetInfos[i].cpu_size, 16u);
      gpu += rex::align(kAvatarComponentAssetInfos[i].gpu_size, 4096u);
    }
  }
  *cpu_size = cpu + 0x1000u;
  *gpu_size = gpu + 0x10000u;
}

u32 XamAvatarGetAssetsResultSize_entry(u32 avatar_component_mask, mapped_u32 result_buffer_size_ptr,
                                       mapped_u32 gpu_resource_buffer_size_ptr) {
  uint32_t cpu_size = 0, gpu_size = 0;
  AssetBuildBudget(avatar_component_mask, &cpu_size, &gpu_size);
  if (result_buffer_size_ptr) *result_buffer_size_ptr = cpu_size;
  if (gpu_resource_buffer_size_ptr) *gpu_resource_buffer_size_ptr = gpu_size;
  return X_STATUS_SUCCESS;
}

u32 XamAvatarGetAssets_entry(ppc_ptr_t<X_AVATAR_METADATA> avatar_metadata_ptr,
                             u32 avatar_component_mask, u32 flags, mapped_u32 result_buffer_ptr,
                             mapped_u32 gpu_resource_buffer_ptr, mapped_void overlapped_ptr) {
  const uint32_t meta_addr = avatar_metadata_ptr.guest_address();
  const uint32_t cpu_addr = result_buffer_ptr.guest_address();
  const uint32_t gpu_addr = gpu_resource_buffer_ptr.guest_address();

  auto run = [=]() -> X_RESULT {
    auto* mem = REX_KERNEL_MEMORY();
    if (!meta_addr || !cpu_addr || !gpu_addr) {
      return X_ERROR_FUNCTION_FAILED;
    }
    avatars::AssetPack* pack = &g_avatar_asset_pack;
    uint32_t skeleton_version = 2;
    if (g_title_version < 3) {
      pack = &g_legacy_avatar_asset_pack;
      skeleton_version = 1;
    }
    if (!pack->is_loaded() && !(pack == &g_avatar_asset_pack && LoadAvatarAssetPack())) {
      REXKRNL_WARN("[avatar] GetAssets: asset pack not loaded; cannot build avatar");
      return X_ERROR_FUNCTION_FAILED;
    }
    const auto* metadata =
        mem->TranslateVirtual<const avatars::X_AVATAR_METADATA*>(meta_addr);
    // Receipt: marketplace items in the manifest being built, so a try-on
    // that never reaches the avatar shows up as silence here. 
    for (const auto& comp : metadata->components) {
      if (!comp.asset_id.is_zero() && !avatars::IsStockPackId(comp.asset_id)) {
        REXKRNL_INFO("[avatar] build wears {} cats={:#x}", comp.asset_id.to_string(),
                     uint16_t(comp.categories));
        if (!avatars::GetCloset().HasItemBytes(comp.asset_id)) {
          REXKRNL_INFO("[avatar] worn item {} is not in the closet, fetching it: {}",
                       comp.asset_id.to_string(),
                       MarketplaceInstallItem(comp.asset_id) ? "installed" : "not available");
        }
      }
    }
    // Distinguish the local player's build from other builds. The player's
    // manifest is byte-identical to the captured one (the game is fed that
    // exact buffer via GetManifestLocalUser); synthesized metadata carries
    // kRandomAvatarXuid instead. Only the player's build refreshes the
    // manifest capture/cache or receives the custom-avatar substitution.
    const bool random_npc =
        static_cast<uint64_t>(metadata->owner_xuid) == kRandomAvatarXuid;
    const bool is_player_build =
        !random_npc &&
        (!g_have_metadata ||
         std::memcmp(metadata, g_captured_metadata, sizeof(g_captured_metadata)) == 0);
    if (is_player_build) {
      // Capture this valid manifest to replay from GetManifestLocalUser.
      std::memcpy(g_captured_metadata, metadata, sizeof(g_captured_metadata));
      if (!g_have_metadata) {
        // Persist it so the next launch has valid metadata before the model
        // master loads; this launch captures too late for that.
        WriteHostFile(AvatarManifestPath(), g_captured_metadata, sizeof(g_captured_metadata));
      }
      g_have_metadata = true;
      // Push it into the buffer the game last asked GetManifestLocalUser to
      // fill, so its "Avatar metadata not set" retry succeeds even though the
      // manifest read happened before GetAssets ran.
      if (g_last_manifest_dest) {
        std::memcpy(mem->TranslateVirtual<uint8_t*>(g_last_manifest_dest), metadata,
                    sizeof(g_captured_metadata));
      }
    }
    auto* cpu_host = mem->TranslateVirtual<uint8_t*>(cpu_addr);
    auto* gpu_host = mem->TranslateVirtual<uint8_t*>(gpu_addr);
    // The GPU resource buffer must be 4 KiB aligned: every texture inside it
    // is claimed at a 4096-aligned offset relative to this base, and Xenos
    // fetch constants address texture bases by 4 KiB page. A misaligned
    // title-side allocation shears every decal texture by (addr & 0xFFF) bytes.
    //
    // Face-feature pages build with obstructions off. The guest's own
    // Hairstyles flow hands GetAssets a pre-stripped manifest (costume slot
    // replaced by fallback top/bottom/shoes plus the trial hair) and restores
    // by rebuilding from its untouched manifest on exit. The other feature
    // pages (Ears kind 10, Nose, Mouth, Eyes) pass the manifest raw, so a fused
    // costume (categories 0xFFC or 0x278) rides its kHair/kTop bits into the
    // page's partial-mask builds (centre close-up mask 0x20f, tile bakes 0x5)
    // and draws over the feature being edited. Only partial-mask builds are
    // stripped; full 0x1fff builds happen at hub/menu level, so the persistent
    // avatar render is never built stripped.
    //
    // The transform mirrors the guest's Hairstyles one: drop components that
    // carry kHat/kGlasses, and dress the garment and kHair categories they
    // covered from the manifest's own fallback_components.
    avatars::X_AVATAR_METADATA feature_meta;
    const avatars::X_AVATAR_METADATA* build_meta = metadata;
    bool feature_partial_build = false;  // a feature sub-page's partial-mask build
    if (REX_KERNEL_STATE()->title_id() == kAvatarEditorID &&
        avatar_component_mask != 0x1fffu) {
      const auto rdnav = [&](uint32_t ea) {
        const uint32_t* q = mem->TranslateVirtual<const uint32_t*>(ea);
        return q ? __builtin_bswap32(*q) : 0u;
      };
      const uint32_t nav = 0x922E7604u;
      const int nav_top = int(rdnav(nav));
      bool in_features = false;
      uint32_t top_kind = 0;
      for (int i = 0; i <= nav_top && i < 16; ++i) {
        top_kind = rdnav(nav + 4u + 2188u * uint32_t(i));
        if (top_kind == 4u) in_features = true;
      }
      // A sub-page is open when the stack goes deeper than the hub itself.
      if (in_features && top_kind != 4u) {
        feature_partial_build = true;
        const uint32_t obstructions =
            avatars::ComponentCategory::kHat | avatars::ComponentCategory::kGlasses;
        const uint32_t dressable =
            avatars::ComponentCategory::kTop | avatars::ComponentCategory::kBottom |
            avatars::ComponentCategory::kShoes | avatars::ComponentCategory::kGloves |
            avatars::ComponentCategory::kHair;
        std::memcpy(&feature_meta, metadata, sizeof(feature_meta));
        uint32_t stripped = 0, lost = 0;
        for (auto& comp : feature_meta.components) {
          const uint32_t cats = comp.categories;
          if (cats & obstructions) {
            lost |= cats & dressable;
            comp = {};
            ++stripped;
          }
        }
        if (lost) {
          for (const auto& candidate : metadata->fallback_components) {
            const uint32_t ccats = candidate.categories;
            if (candidate.asset_id.is_zero() || !(ccats & lost) || (ccats & obstructions)) {
              continue;
            }
            for (auto& slot : feature_meta.components) {
              if (slot.asset_id.is_zero() && !uint32_t(slot.categories)) {
                slot = candidate;
                lost &= ~ccats;
                break;
              }
            }
            if (!lost) break;
          }
        }
        if (stripped) {
          build_meta = &feature_meta;
        }
      }
    }
    // The title recycles a small set of GPU resource buffers across avatar
    // builds (the editor ping-pongs two of them, one GetAssets call per
    // selection-grid tile), so the rewrite below can land while GPU-side
    // texture loads for the previous build at the same addresses are still
    // pending. The command processor lives inside the GPU plugin and exposes
    // no drain to the app, so the emulated-GPU path has no sync here; the
    // freeze and invalidate bracket below protects the native-renderer path,
    // which is the shipped configuration.
    avatars::MemoryBlock cpu_memory(cpu_host, 16);
    avatars::MemoryBlock gpu_memory(gpu_host, 4096);
    // Freeze bracket: the rewrite below takes milliseconds while the render
    // thread keeps drawing; without this, content heals can re-upload
    // half-rewritten textures (random corruption on item select). The
    // exact extent is unknown until the load finishes, so freeze a generous
    // span; the paired invalidate below thaws by overlap.
    rex::videonative::renderer::QueueGuestTextureFreeze(gpu_addr, 0x400000u);
    bool ok = avatars::LoadAssetsToGuest(*build_meta, avatar_component_mask, flags, pack,
                                         &cpu_memory, &gpu_memory, skeleton_version,
                                         g_coordinate_system);
    if (!ok) {
      REXKRNL_WARN("[avatar] GetAssets: LoadAssetsToGuest failed (mask={:#x})",
                   avatar_component_mask);
      // Thaw the freeze bracket even on failure (partial writes possible).
      rex::videonative::renderer::QueueGuestTextureInvalidate(gpu_addr, 0x400000u);
      return X_ERROR_FUNCTION_FAILED;
    }
    cpu_memory.ResolvePointers(cpu_addr);
    gpu_memory.ResolvePointers(gpu_addr);
    {
      // Past the budget means the title's neighbouring data is already gone.
      uint32_t budget_cpu = 0, budget_gpu = 0;
      AssetBuildBudget(avatar_component_mask, &budget_cpu, &budget_gpu);
      if (cpu_memory.used() > budget_cpu || gpu_memory.used() > budget_gpu) {
        REXKRNL_WARN("[avatar] GetAssets: build (mask={:#x}) used cpu {:#x} gpu {:#x}, over the "
                     "{:#x}/{:#x} the title was told to allocate",
                     avatar_component_mask, cpu_memory.used(), gpu_memory.used(), budget_cpu,
                     budget_gpu);
      }
    }
    // The title recycles a handful of GPU resource buffers across builds (each
    // grid tile is one GetAssets call into one of a few addresses). The native
    // renderer's texture cache keys on the fetch header, which is identical
    // across builds at the same address, and its polling heals are windowed, so
    // report the rewritten range explicitly to retire overlapping cache entries
    // before the next draw. No-op when no native renderer is attached.
    rex::videonative::renderer::QueueGuestTextureInvalidate(
        gpu_addr, static_cast<uint32_t>(gpu_memory.used()));
    if (!is_player_build) PaceBakeCompletion();
    return X_ERROR_SUCCESS;
  };

  if (overlapped_ptr) {
    X_RESULT result = run();
    REX_KERNEL_STATE()->CompleteOverlappedImmediate(overlapped_ptr.guest_address(), result);
    return X_ERROR_IO_PENDING;
  }
  return run();
}

// Blobs GetAssetBinary handed out lately, keyed by size and content hash.
// A raw STRB carries no asset id of its own, so when one comes back through
// SetCustomAsset this is how it gets named.
struct ServedBinary {
  avatars::AssetId id;
  size_t size;
  uint64_t hash;
};
static std::mutex g_served_mutex;
static ServedBinary g_served_binaries[64];
static size_t g_served_next = 0;

static uint64_t HashBytes(const uint8_t* data, size_t size) {
  uint64_t h = 0xcbf29ce484222325ull;
  for (size_t i = 0; i < size; ++i) {
    h = (h ^ data[i]) * 0x100000001b3ull;
  }
  return h;
}

static void RememberServedBinary(const avatars::AssetId& id, const uint8_t* data, size_t size) {
  std::lock_guard<std::mutex> lock(g_served_mutex);
  g_served_binaries[g_served_next] = {id, size, HashBytes(data, size)};
  g_served_next = (g_served_next + 1) % rex::countof(g_served_binaries);
}

// Names a blob: marketplace packages carry the id in their YTGR header, and
// anything else has to be something GetAssetBinary served a moment ago.
static bool IdentifyAssetBlob(const uint8_t* data, size_t size, avatars::AssetId* id) {
  static constexpr size_t kYtgrHeaderSize = 0x140;
  static constexpr size_t kYtgrIdOffset = 0x130;
  if (size >= kYtgrHeaderSize + 24 && std::memcmp(data, "YTGR", 4) == 0) {
    std::memcpy(id, data + kYtgrIdOffset, sizeof(*id));
    if (!id->is_zero()) {
      return true;
    }
  }
  const uint64_t hash = HashBytes(data, size);
  std::lock_guard<std::mutex> lock(g_served_mutex);
  for (const auto& served : g_served_binaries) {
    if (served.size == size && served.hash == hash && !served.id.is_zero()) {
      *id = served.id;
      return true;
    }
  }
  return false;
}

// Dresses a manifest in one item the way XAM's writer does: whatever the
// item's categories overlap comes off, garments that left a required slot
// bare come back from the manifest's own fallbacks, then the item goes in.
static bool WearComponent(avatars::X_AVATAR_METADATA& meta, const avatars::AssetId& id,
                          uint16_t categories) {
  using namespace avatars::ComponentCategory;
  avatars::X_AVATAR_COMPONENT_INFO item{};
  item.asset_id = id;
  item.categories = categories;
  if (categories & kBody) {
    meta.body_component = item;
    return true;
  }
  if (categories & kHead) {
    meta.head_component = item;
    return true;
  }
  constexpr size_t kSlots = sizeof(meta.components) / sizeof(meta.components[0]);
  avatars::X_AVATAR_COMPONENT_INFO kept[kSlots];
  size_t count = 0;
  uint32_t vacated = 0;
  for (const auto& comp : meta.components) {
    if (comp.asset_id.is_zero()) {
      continue;
    }
    if (uint32_t(comp.categories) & categories) {
      vacated |= uint32_t(comp.categories);
      continue;
    }
    kept[count++] = comp;
  }
  uint32_t bare = vacated & ~uint32_t(categories) & (kTop | kBottom | kShoes | kHair);
  for (const auto& fallback : meta.fallback_components) {
    if (!bare || count >= kSlots) {
      break;
    }
    const uint32_t cats = fallback.categories;
    if (fallback.asset_id.is_zero() || !(cats & bare) || (cats & categories)) {
      continue;
    }
    kept[count++] = fallback;
    bare &= ~cats;
  }
  if (count >= kSlots) {
    return false;
  }
  kept[count++] = item;
  for (size_t i = 0; i < kSlots; ++i) {
    meta.components[i] = i < count ? kept[i] : avatars::X_AVATAR_COMPONENT_INFO{};
  }
  return true;
}

// XamAvatarSetCustomAsset(size, data, color_count, colors, flags, metadata):
// the editor's try-on path. It hands over an item's STRB and expects the
// manifest to come back wearing it; the bytes are kept so GetAssets can load
// the id. The XAM export takes six arguments, one more than the public
// XAvatarSetCustomAsset: the title's thunk (sub_92143220) zeroes r7 and moves
// the manifest to r8.
u32 XamAvatarSetCustomAsset_entry(u32 buffer_size, mapped_void asset_data_ptr,
                                  u32 custom_color_count, mapped_void custom_colors_ptr,
                                  u32 flags, ppc_ptr_t<X_AVATAR_METADATA> avatar_metadata_ptr) {
  if (!buffer_size || !asset_data_ptr || !avatar_metadata_ptr || custom_color_count > 3 ||
      (custom_color_count && !custom_colors_ptr)) {
    REXKRNL_WARN("[avatar] SetCustomAsset: bad arguments (size={:#x} data={:#x} colors={} meta={:#x})",
                 buffer_size, asset_data_ptr.guest_address(), custom_color_count,
                 avatar_metadata_ptr.guest_address());
    return 0x80070057u;  // E_INVALIDARG
  }
  auto* mem = REX_KERNEL_MEMORY();
  const auto* data = mem->TranslateVirtual<const uint8_t*>(asset_data_ptr.guest_address());
  auto* meta =
      mem->TranslateVirtual<avatars::X_AVATAR_METADATA*>(avatar_metadata_ptr.guest_address());
  avatars::AssetId id{};
  if (!IdentifyAssetBlob(data, buffer_size, &id)) {
    REXKRNL_WARN("[avatar] SetCustomAsset: {:#x}-byte blob carries no asset id", buffer_size);
    return 0x80004005u;  // E_FAIL
  }
  // The category mask rides in the id's first dword, for stock and
  // marketplace ids alike.
  const uint16_t categories = uint16_t(id.a.get() & 0xFFFFu);
  if (!avatars::IsStockPackId(id)) {
    avatars::GetCloset().RegisterCustomItem(id, data, buffer_size);
  }
  if (!WearComponent(*meta, id, categories)) {
    REXKRNL_WARN("[avatar] SetCustomAsset: no free component slot for {}", id.to_string());
    return 0x80004005u;
  }
  REXKRNL_INFO("[avatar] SetCustomAsset: wearing {} cats={:#x} ({:#x} bytes)", id.to_string(),
               categories, buffer_size);
  return X_STATUS_SUCCESS;
}

u32 XamAvatarSetManifest_entry(u32 user_index, u32 avatar_info_ptr, mapped_void overlapped_ptr) {
  // Persist the manifest being saved into the same store
  // GetManifestLocalUser replays from, so the edited avatar becomes the local
  // user's avatar on every later launch.
  if (avatar_info_ptr) {
    const auto* meta =
        REX_KERNEL_MEMORY()->TranslateVirtual<const avatars::X_AVATAR_METADATA*>(avatar_info_ptr);
    if (!meta->body_component.asset_id.is_zero()) {
      std::memcpy(g_captured_metadata, meta, sizeof(g_captured_metadata));
      g_have_metadata = true;
      WriteHostFile(AvatarManifestPath(), g_captured_metadata, sizeof(g_captured_metadata));
    } else {
      REXKRNL_WARN(
          "[avatar] XamAvatarSetManifest: buffer at {:#x} has a zero body component; "
          "not persisting (unexpected layout?)",
          avatar_info_ptr);
    }
  }
  if (overlapped_ptr) {
    REX_KERNEL_STATE()->CompleteOverlappedImmediate(overlapped_ptr.guest_address(),
                                                    X_ERROR_SUCCESS);
    return X_ERROR_IO_PENDING;
  }
  return X_STATUS_SUCCESS;
}

u32 XamAvatarGetMetadataRandom_entry(u32 body_type, u32 avatars_count,
                                     ppc_ptr_t<X_AVATAR_METADATA> avatar_metadata_ptr,
                                     mapped_void overlapped_ptr) {
  static uint32_t call_no = 0;
  auto run = [&]() -> X_RESULT {
    if (!avatar_metadata_ptr || !avatars_count) {
      return X_ERROR_INVALID_PARAMETER;
    }
    auto* out = REX_KERNEL_MEMORY()->TranslateVirtual<avatars::X_AVATAR_METADATA*>(
        avatar_metadata_ptr.guest_address());
    const uint32_t mask = (body_type & 3u) ? (body_type & 3u) : 3u;
    for (uint32_t i = 0; i < avatars_count; ++i) {
      BuildRandomAvatarMetadata(&out[i], mask, ++call_no * 97u + i);
    }
    return X_ERROR_SUCCESS;
  };
  if (overlapped_ptr) {
    const X_RESULT result = run();
    REX_KERNEL_STATE()->CompleteOverlappedImmediate(overlapped_ptr.guest_address(), result);
    return X_ERROR_IO_PENDING;
  }
  return run();
}

u32 XamAvatarGetMetadataSignedOutProfileCount_entry(mapped_u32 profile_count_ptr,
                                                    mapped_void overlapped_ptr) {
  if (profile_count_ptr) *profile_count_ptr = 0;
  if (overlapped_ptr) {
    REX_KERNEL_STATE()->CompleteOverlappedImmediate(overlapped_ptr.guest_address(),
                                                    X_ERROR_SUCCESS);
    return X_ERROR_IO_PENDING;
  }
  return X_STATUS_SUCCESS;
}

u32 XamAvatarGetMetadataSignedOutProfile_entry(u32 profile_index,
                                               ppc_ptr_t<X_AVATAR_METADATA> avatar_metadata_ptr,
                                               mapped_void overlapped_ptr) {
  if (overlapped_ptr) {
    REX_KERNEL_STATE()->CompleteOverlappedImmediate(overlapped_ptr.guest_address(),
                                                    X_ERROR_SUCCESS);
    return X_ERROR_IO_PENDING;
  }
  return X_STATUS_SUCCESS;
}

u32 XamAvatarManifestGetBodyType_entry(ppc_ptr_t<X_AVATAR_METADATA> avatar_metadata_ptr) {
  return static_cast<u32>(X_AVATAR_BODY_TYPE::Male);
}

u32 XamAvatarGetInstrumentation_entry(u64 unk1, mapped_u32 unk2) {
  if (unk2) *unk2 = 0;
  return 1;
}

// Grid-thumbnail source. Signature (the same shape as its sibling
// GetAssetBinary, sub_920B5A28 vs sub_920B5990): (asset_id, flags, size_inout,
// buffer, overlapped). The editor allocates 0x10000 bytes (sub_920B59E8 /
// sub_920F6E50).
//
// The payload is an image file, not raw pixels: on success the icon state
// machine (sub_920F6E50) registers {buffer, size} on the load list
// unk_9422DE88; the worker thread (sub_920F7178) formats "memory://%X,%X"
// (literal at 0x920126B0) from them and hands it to the image loader
// sub_9220CE38, which dispatches on the buffer's leading 4-byte magic:
// 89 50 4E 47 (PNG -> sub_921B0280, libpng), FF D8 FF E0/E1 (JPEG),
// 20 52 47 42 (' RGB' raw container). *size_inout must be set to the actual
// byte count; it becomes the size field of the memory:// URI.
//
// Returning failure leaves the tile blank by design: the only other fill path
// (the offscreen tile render 0x920F75D8) is dead code in this build, its gate
// global 0x9457D05C is never written (re/subsystems/09_item_grid_tiles.md §4).
//
// Icon art comes from closet icons/<guid>.png, the marketplace package's own
// ICON.PNG, imported by `avatarextract --closet-import` / `--closet-icons`.

// Store items the closet does not hold come from the marketplace server, art
// and bytes alike. Each id is asked for once per session.
static std::mutex g_remote_mutex;
static std::unordered_set<std::string> g_remote_missing_icons;
static std::unordered_set<std::string> g_remote_missing_items;

static std::string RemoteItemPath(const avatars::AssetId& id) {
  return fmt::format("/live/{:08x}/avataritems/{}.acp", avatars::TitleIdOf(id), id.to_string());
}

static bool FetchRemoteIcon(const avatars::AssetId& id, std::vector<uint8_t>& png) {
  if (avatars::IsStockPackId(id) || !REXCVAR_GET(avatar_marketplace)) {
    return false;
  }
  const std::string guid = id.to_string();
  {
    std::lock_guard<std::mutex> lock(g_remote_mutex);
    if (g_remote_missing_icons.count(guid)) {
      return false;
    }
  }
  std::string body;
  if (!MarketplaceGet("/icons/" + guid + ".png", &body) || body.empty()) {
    std::lock_guard<std::mutex> lock(g_remote_mutex);
    g_remote_missing_icons.insert(guid);
    return false;
  }
  png.assign(body.begin(), body.end());
  avatars::GetCloset().RegisterCustomIcon(id, png.data(), png.size());
  return true;
}

// The blob comes with the index-row fields, so a purchase writes its row
// without a second round trip.
struct RemoteItem {
  std::vector<uint8_t> blob;
  std::string name;
  uint32_t categories = 0;
  uint8_t bodies = 0;
};

static std::string HeaderValue(const std::string& block, const std::string& name) {
  size_t pos = 0;
  while (pos < block.size()) {
    size_t end = block.find("\r\n", pos);
    if (end == std::string::npos) end = block.size();
    const std::string line = block.substr(pos, end - pos);
    const size_t colon = line.find(':');
    if (colon == name.size() && _strnicmp(line.c_str(), name.c_str(), colon) == 0) {
      size_t v = colon + 1;
      while (v < line.size() && line[v] == ' ') ++v;
      return line.substr(v);
    }
    pos = end + 2;
  }
  return std::string();
}

static std::string PercentDecode(const std::string& text) {
  std::string out;
  for (size_t i = 0; i < text.size(); ++i) {
    if (text[i] == '%' && i + 2 < text.size() && std::isxdigit(uint8_t(text[i + 1])) &&
        std::isxdigit(uint8_t(text[i + 2]))) {
      out.push_back(char(std::strtoul(text.substr(i + 1, 2).c_str(), nullptr, 16)));
      i += 2;
    } else {
      out.push_back(text[i] == '+' ? ' ' : text[i]);
    }
  }
  return out;
}

static bool FetchRemoteItem(const avatars::AssetId& id, RemoteItem& item) {
  if (avatars::IsStockPackId(id) || !REXCVAR_GET(avatar_marketplace)) {
    return false;
  }
  const std::string guid = id.to_string();
  {
    std::lock_guard<std::mutex> lock(g_remote_mutex);
    if (g_remote_missing_items.count(guid)) {
      return false;
    }
  }
  std::string body, headers;
  if (!MarketplaceGet(RemoteItemPath(id), &body, &headers) || body.empty()) {
    std::lock_guard<std::mutex> lock(g_remote_mutex);
    g_remote_missing_items.insert(guid);
    return false;
  }
  item.blob.assign(body.begin(), body.end());
  item.name = PercentDecode(HeaderValue(headers, "X-Avatar-Item-Name"));
  item.categories = uint32_t(
      std::strtoul(HeaderValue(headers, "X-Avatar-Item-Categories").c_str(), nullptr, 16));
  item.bodies =
      uint8_t(std::strtoul(HeaderValue(headers, "X-Avatar-Item-Bodies").c_str(), nullptr, 10));
  // The id carries both masks itself when the server did not say.
  if (!item.categories) item.categories = id.a.get();
  if (!item.bodies) item.bodies = uint8_t(id.c.get() & 0xF);
  if (item.name.empty()) item.name = guid;
  avatars::GetCloset().RegisterCustomItem(id, item.blob.data(), item.blob.size());
  return true;
}

// Ids from a catalog page that the closet cannot serve, parsed once for both
// prefetches.
static std::vector<avatars::AssetId> PrefetchCandidates(const std::vector<std::string>& guids,
                                                        bool icons) {
  std::vector<avatars::AssetId> out;
  for (const auto& guid : guids) {
    avatars::AssetId id{};
    if (!avatars::ParseAssetId(guid, &id) || avatars::IsStockPackId(id)) {
      continue;
    }
    std::vector<uint8_t> have;
    const bool held = icons ? avatars::GetCloset().ReadItemIcon(id, have)
                            : avatars::GetCloset().HasItemBytes(id);
    if (!held) {
      out.push_back(id);
    }
  }
  return out;
}

// A few workers share the list; each takes the next id until it runs dry.
template <typename Fetch>
static void RunPrefetch(std::vector<avatars::AssetId> ids, size_t max_workers, bool wait,
                        Fetch fetch) {
  auto pending = std::make_shared<std::vector<avatars::AssetId>>(std::move(ids));
  auto next = std::make_shared<std::atomic<size_t>>(0);
  const size_t workers = std::min<size_t>(max_workers, pending->size());
  std::vector<std::thread> threads;
  for (size_t w = 0; w < workers; ++w) {
    threads.emplace_back([pending, next, fetch]() {
      for (size_t i = next->fetch_add(1); i < pending->size(); i = next->fetch_add(1)) {
        fetch((*pending)[i]);
      }
    });
  }
  for (auto& t : threads) {
    if (wait) {
      t.join();
    } else {
      t.detach();
    }
  }
}

void MarketplacePrefetchIcons(const std::vector<std::string>& guids, bool wait) {
  if (!REXCVAR_GET(avatar_marketplace)) {
    return;
  }
  auto ids = PrefetchCandidates(guids, true);
  if (ids.empty()) {
    return;
  }
  REXKRNL_INFO("[avatar] prefetching {} tile icons{}", ids.size(), wait ? " (page waits)" : "");
  RunPrefetch(std::move(ids), 6, wait, [](const avatars::AssetId& id) {
    std::vector<uint8_t> png;
    FetchRemoteIcon(id, png);
  });
}

void MarketplacePrefetchItems(const std::vector<std::string>& guids) {
  if (!REXCVAR_GET(avatar_marketplace)) {
    return;
  }
  auto ids = PrefetchCandidates(guids, false);
  if (ids.empty()) {
    return;
  }
  REXKRNL_INFO("[avatar] prefetching {} item packages in the background", ids.size());
  RunPrefetch(std::move(ids), 4, false, [](const avatars::AssetId& id) {
    RemoteItem item;
    FetchRemoteItem(id, item);
  });
}

bool MarketplaceInstallItem(const avatars::AssetId& id) {
  RemoteItem item;
  if (!FetchRemoteItem(id, item)) {
    return false;
  }
  std::vector<uint8_t> png;
  if (!avatars::GetCloset().ReadItemIcon(id, png)) {
    FetchRemoteIcon(id, png);
  }
  const bool ok = avatars::GetCloset().InstallItem(id, item.blob, png, item.categories,
                                                  item.bodies, item.name);
  if (ok) {
    REXKRNL_INFO("[avatar] installed {} '{}' ({:#x} bytes, icon {})", id.to_string(), item.name,
                 item.blob.size(), png.empty() ? "none" : "yes");
  }
  return ok;
}

u32 XamAvatarGetAssetIcon_entry(mapped_void asset_id_ptr, u32 flags, mapped_u32 size_inout,
                                mapped_void buffer, mapped_void overlapped_ptr) {
  auto run = [&]() -> X_RESULT {
    if (!asset_id_ptr || !size_inout || !buffer) {
      return X_ERROR_INVALID_PARAMETER;
    }
    // The closet loads as a side effect of the asset pack load; make sure it
    // has happened (icons are normally requested after enumeration, but do
    // not depend on that ordering).
    if (!g_avatar_asset_pack.is_loaded() && !g_legacy_avatar_asset_pack.is_loaded()) {
      LoadAvatarAssetPack();
    }
    auto* mem = REX_KERNEL_MEMORY();
    const auto* id =
        mem->TranslateVirtual<const avatars::AssetId*>(asset_id_ptr.guest_address());
    std::vector<uint8_t> png;
    const bool closet_icon = avatars::GetCloset().ReadItemIcon(*id, png);
    const bool have_icon = closet_icon || FetchRemoteIcon(*id, png);
    // Receipt: which tiles ask for art and where it came from.
    static std::atomic<uint32_t> icon_calls{0};
    const uint32_t icon_n = icon_calls.fetch_add(1, std::memory_order_relaxed);
    if (icon_n < 48 || (icon_n & 0x3F) == 0) {
      REXKRNL_INFO("[avatar] GetAssetIcon #{} {} -> {}", icon_n, id->to_string(),
                   closet_icon ? "closet" : have_icon ? "server" : "none");
    }
    if (!have_icon) {
      // Only closet-imported marketplace items carry icon art; no icon store
      // has been located in the stock pack. Fail so the editor falls through
      // to its own live mannequin preview render.
      return X_ERROR_FUNCTION_FAILED;
    }
    const uint32_t capacity = *size_inout;
    if (png.size() > capacity) {
      REXKRNL_WARN("[avatar] GetAssetIcon: {} icon {:#x} bytes > capacity {:#x}",
                   id->to_string(), png.size(), capacity);
      return X_ERROR_INSUFFICIENT_BUFFER;
    }
    std::memcpy(mem->TranslateVirtual<uint8_t*>(buffer.guest_address()), png.data(),
                png.size());
    *size_inout = static_cast<uint32_t>(png.size());
    return X_ERROR_SUCCESS;
  };
  if (overlapped_ptr) {
    const X_RESULT result = run();
    REX_KERNEL_STATE()->CompleteOverlappedImmediate(overlapped_ptr.guest_address(), result);
    // Not X_ERROR_IO_PENDING (997) here: the editor's icon state machine
    // (sub_920F6E50) accepts only 0 or E_PENDING (0x8000000A) from this call
    // and drops to its failed state on anything else. The overlapped is
    // completed synchronously above, so reporting success is accurate and the
    // machine then polls an already-complete overlapped. The sibling
    // GetAssetBinary keeps 997 because its consumer tests differently.
    // Only a genuine success may report 0: X_ERROR_* are positive Win32 codes,
    // so a ">= 0" test would report success on failure.
    return result == X_ERROR_SUCCESS ? X_ERROR_SUCCESS : X_ERROR_IO_PENDING;
  }
  return run();
}

// XamAvatarGetAssetBinary(id, flags, size_inout, buffer, overlapped): hand the
// title the asset's raw STRB bytes from the pack. The Avatar Editor fetches
// every selection-grid item through this (2MB buffer, in/out byte count, see
// sub_920B5990) and parses the STRB itself to render the option preview.
u32 XamAvatarGetAssetBinary_entry(mapped_void asset_id_ptr, u32 flags, mapped_u32 size_inout,
                                  mapped_void buffer, mapped_void overlapped_ptr) {
  auto run = [&]() -> X_RESULT {
    if (!asset_id_ptr || !size_inout || !buffer) {
      return X_ERROR_INVALID_PARAMETER;
    }
    avatars::AssetPack* pack =
        (g_title_version < 3) ? &g_legacy_avatar_asset_pack : &g_avatar_asset_pack;
    if (!pack->is_loaded()) {
      LoadAvatarAssetPack();
    }
    if (!pack->is_loaded()) {
      pack = pack == &g_avatar_asset_pack ? &g_legacy_avatar_asset_pack : &g_avatar_asset_pack;
    }
    if (!pack->is_loaded()) {
      return X_ERROR_FUNCTION_FAILED;
    }
    auto* mem = REX_KERNEL_MEMORY();
    const auto* id =
        mem->TranslateVirtual<const avatars::AssetId*>(asset_id_ptr.guest_address());
    const uint8_t* data = nullptr;
    size_t size = 0;
    std::vector<uint8_t> closet_bytes;
    RemoteItem remote;
    if (avatars::GetCloset().ReadItemBytes(*id, closet_bytes) && !closet_bytes.empty()) {
      // Imported marketplace item: hand the title the raw YTGR/STRB blob.
      data = closet_bytes.data();
      size = closet_bytes.size();
    } else if (pack->GetAssetData(*id, data, size) && size) {
      // Stock pack asset.
    } else if (FetchRemoteItem(*id, remote)) {
      // A store item the closet does not hold yet, for its try-on.
      data = remote.blob.data();
      size = remote.blob.size();
    } else {
      REXKRNL_WARN("[avatar] GetAssetBinary: no data for {}", id->to_string());
      return X_ERROR_FUNCTION_FAILED;
    }
    const uint32_t capacity = *size_inout;
    if (size > capacity) {
      REXKRNL_WARN("[avatar] GetAssetBinary: {} needs {:#x} > capacity {:#x}", id->to_string(),
                   size, capacity);
      return X_ERROR_INSUFFICIENT_BUFFER;
    }
    RememberServedBinary(*id, data, size);
    // Receipt: which items the editor pulls bytes for, and from where.
    static std::atomic<uint32_t> binary_calls{0};
    const uint32_t binary_n = binary_calls.fetch_add(1, std::memory_order_relaxed);
    if (binary_n < 48 || (binary_n & 0x3F) == 0) {
      REXKRNL_INFO("[avatar] GetAssetBinary #{} {} from={} bytes={:#x}", binary_n,
                   id->to_string(), closet_bytes.empty() ? "pack" : "closet", size);
    }
    std::memcpy(mem->TranslateVirtual<uint8_t*>(buffer.guest_address()), data, size);
    *size_inout = static_cast<uint32_t>(size);
    return X_ERROR_SUCCESS;
  };
  if (overlapped_ptr) {
    const X_RESULT result = run();
    REX_KERNEL_STATE()->CompleteOverlappedImmediate(overlapped_ptr.guest_address(), result);
    return X_ERROR_IO_PENDING;
  }
  return run();
}

void XamAvatarGetInstalledAssetPackageDescription_entry(ppc_ptr_t<X_ASSET_ID> asset_id_ptr,
                                                        mapped_void content_data_ptr) {
  // rexglue ships no installed avatar-asset packages; report "none" by leaving
  // the descriptor as-is. (xenia fills an XCONTENT_AGGREGATE_DATA from the id.)
  static std::atomic<uint32_t> calls{0};
  const uint32_t n = calls.fetch_add(1, std::memory_order_relaxed);
  if (n < 24 || (n & 0x3F) == 0) {
    const auto* id = asset_id_ptr
                         ? REX_KERNEL_MEMORY()->TranslateVirtual<const avatars::AssetId*>(
                               asset_id_ptr.guest_address())
                         : nullptr;
    REXKRNL_INFO("[avatar] GetInstalledAssetPackageDescription #{} {} (no packages)", n,
                 id ? id->to_string() : "?");
  }
}

void XamAvatarSetMocks_entry() {
  // No-op.
}

// Animation ------------------------------------------------------------------
// Known stock animation asset ids, kept for reference when tracing
// XamAvatarLoadAnimation requests.
static const std::map<uint64_t, std::string> XAnimationTypeMap = {
    {0x0040000000030003, "Animation Generic Stand 0"},
    {0x0040000000040003, "Animation Generic Stand 1"},
    {0x0040000000050003, "Animation Generic Stand 2"},
    {0x0040000000270003, "Animation Generic Stand 3"},
    {0x0040000000280003, "Animation Generic Stand 4"},
    {0x0040000000290003, "Animation Generic Stand 5"},
    {0x00400000002A0003, "Animation Generic Stand 6"},
    {0x00400000002B0003, "Animation Generic Stand 7"},
    {0x0040000000130001, "Animation Male Idle Looks Around"},
    {0x0040000000140001, "Animation Male Idle Stretch"},
    {0x0040000000150001, "Animation Male Idle Shifts Weight"},
    {0x0040000000260001, "Animation Male Idle Checks Hand"},
    {0x0040000000090002, "Animation Female Idle Check Nails"},
    {0x00400000000A0002, "Animation Female Idle Looks Around"},
    {0x00400000000B0002, "Animation Female Idle Shifts Weight"},
    {0x00400000000C0002, "Animation Female Idle Fixes Shoe"},
};

u32 XamAvatarLoadAnimation_entry(mapped_u64 asset_id_ptr, u32 flags, mapped_void output,
                                 mapped_void overlapped_ptr) {
  const uint32_t asset_id_addr = asset_id_ptr.guest_address();
  const uint32_t output_addr = output.guest_address();

  auto run = [=]() -> X_RESULT {
    auto* mem = REX_KERNEL_MEMORY();
    if (!asset_id_addr || !output_addr) {
      return X_E_FAIL;
    }
    avatars::AssetPack* pack =
        (g_title_version >= 3) ? &g_avatar_asset_pack : &g_legacy_avatar_asset_pack;
    if (!pack->is_loaded()) {
      return X_E_FAIL;
    }
    const auto* asset_id = mem->TranslateVirtual<const avatars::AssetId*>(asset_id_addr);
    auto* animation = mem->TranslateVirtual<avatars::X_AVATAR_ANIMATION*>(output_addr);
    auto* compressed = mem->TranslateVirtual<uint8_t*>(animation->compressed_data_buffer_ptr);
    if (!avatars::LoadAnimationToGuest(pack, *asset_id, animation, compressed,
                                       g_coordinate_system)) {
      REXKRNL_WARN("[avatar] LoadAnimation failed");
      return X_E_FAIL;
    }
    return X_ERROR_SUCCESS;
  };

  if (overlapped_ptr) {
    X_RESULT result = run();
    REX_KERNEL_STATE()->CompleteOverlappedImmediate(overlapped_ptr.guest_address(), result);
    return X_ERROR_IO_PENDING;
  }
  return run();
}

u32 XamAvatarGenerateMipMaps_entry(mapped_u32 avatar_assets_ptr, u32 flags, u32 buffer_size,
                                   mapped_u32 mip_map_buffer_ptr, mapped_void overlapped_ptr) {
  if (overlapped_ptr) {
    REX_KERNEL_STATE()->CompleteOverlappedImmediate(overlapped_ptr.guest_address(),
                                                    X_ERROR_SUCCESS);
    return X_ERROR_IO_PENDING;
  }
  return X_STATUS_SUCCESS;
}

// Enum -----------------------------------------------------------------------
// Ported from xenia-canary avatars (xam_avatar_enum.cc). The Avatar Editor's
// feature-selection grids ("Change My Features/Style") are populated entirely
// through this enumerator.
struct X_AVATAR_ASSET_INFO {
  avatars::AssetId asset_id;            // 0x000
  rex::be<uint32_t> categories;         // 0x010
  uint8_t body_mask;                    // 0x014
  uint8_t random_body_mask;             // 0x015
  rex::be<uint32_t> flags;              // 0x018
  uint8_t unknown[164];                 // 0x01C
  rex::be<uint32_t> subcategory;        // 0x0C0
  avatars::AssetId associated_ids[2];   // 0x0C4
  rex::be<uint16_t> name[64];           // 0x0E4
  rex::be<uint32_t> data_offset;        // 0x164
  rex::be<uint32_t> data_size;          // 0x168
  rex::be<uint16_t> description[100];   // 0x16C
  rex::be<uint32_t> title_id;           // 0x234
  rex::be<uint16_t> title_name[64];     // 0x238
  rex::be<uint32_t> time_0;             // 0x2B8
  rex::be<uint32_t> time_1;             // 0x2BC
  uint8_t skeleton_version_mask;        // 0x2C0
};
static_assert(sizeof(X_AVATAR_ASSET_INFO) == 0x2C4, "X_AVATAR_ASSET_INFO size");

// UTF-8 -> UTF-16 (BMP) into a fixed, NUL-terminated big-endian field.
// Closet names/titles/descriptions are UTF-8 (the marketplace is
// multilingual), so a byte-wise copy would mangle anything non-ASCII.
static void CopyUtf8ToUtf16(const std::string& s, rex::be<uint16_t>* dst, size_t cap) {
  size_t n = 0;
  for (size_t i = 0; i < s.size() && n + 1 < cap;) {
    const uint8_t c0 = (uint8_t)s[i];
    uint32_t cp = c0;
    size_t len = 1;
    if (c0 >= 0xF0 && i + 3 < s.size()) {
      cp = ((c0 & 0x07u) << 18) | (((uint8_t)s[i + 1] & 0x3Fu) << 12) |
           (((uint8_t)s[i + 2] & 0x3Fu) << 6) | ((uint8_t)s[i + 3] & 0x3Fu);
      len = 4;
    } else if (c0 >= 0xE0 && i + 2 < s.size()) {
      cp = ((c0 & 0x0Fu) << 12) | (((uint8_t)s[i + 1] & 0x3Fu) << 6) | ((uint8_t)s[i + 2] & 0x3Fu);
      len = 3;
    } else if (c0 >= 0xC0 && i + 1 < s.size()) {
      cp = ((c0 & 0x1Fu) << 6) | ((uint8_t)s[i + 1] & 0x3Fu);
      len = 2;
    }
    i += len;
    dst[n++] = (uint16_t)(cp > 0xFFFF ? '?' : cp);
  }
  dst[n] = 0;
}

namespace {
class AssetEnumerator {
 public:
  explicit AssetEnumerator(uint32_t items_per_enumerate = 50)
      : items_per_enumerate_(items_per_enumerate) {}

  size_t item_size() const { return sizeof(X_AVATAR_ASSET_INFO); }
  size_t items_per_enumerate() const { return items_per_enumerate_; }
  size_t item_count() const { return item_count_; }

  X_AVATAR_ASSET_INFO* AppendItem() {
    size_t offset = buffer_.size();
    buffer_.resize(offset + item_size());
    item_count_++;
    return reinterpret_cast<X_AVATAR_ASSET_INFO*>(&buffer_.data()[offset]);
  }

  uint32_t WriteItems(uint8_t* buffer_data, uint32_t* written_count) {
    size_t count = std::min(item_count_ - current_item_, items_per_enumerate());
    if (!count) {
      return X_ERROR_NO_MORE_FILES;
    }
    size_t size = count * item_size();
    size_t offset = current_item_ * item_size();
    std::memcpy(buffer_data, buffer_.data() + offset, size);
    current_item_ += count;
    if (written_count) {
      *written_count = static_cast<uint32_t>(count);
    }
    return X_ERROR_SUCCESS;
  }

 private:
  size_t items_per_enumerate_;
  size_t item_count_ = 0;
  size_t current_item_ = 0;
  std::vector<uint8_t> buffer_;
};

std::mutex g_asset_enum_mutex;
AssetEnumerator g_asset_enumerator(50);

// --- Catalog search (the Ctrl+F item filter; rex/kernel/xam/avatar_search.h)
std::mutex g_catalog_search_mutex;
std::string g_catalog_search;  // lowercase UTF-8; empty = no filter
std::atomic<uint32_t> g_catalog_search_matches{0};
std::atomic<int> g_catalog_search_scope{-1};  // grid channel 0..24; -1 = all
std::atomic<uint32_t> g_catalog_search_body_mask{0};  // last enum's body mask

// Port of the title's category-bits -> grid-channel classifier (sub_9220F7C8
// over the 22-entry mask table at 0x92056D68). An item belongs to `channel`
// when bits contains required and fits inside required|allowed. The guest
// assigns every catalog entry its grid bucket with this same function (boot
// builder sub_92210208), so scoping the name filter by this value matches what
// the browsed grid shows.
struct ChannelMapEntry {
  uint32_t required, allowed, channel;
};
constexpr ChannelMapEntry kChannelMap[] = {
    {0x00000001, 0x00000000, 1},  {0x00000002, 0x00000000, 2},
    {0x00080000, 0x00000000, 4},  {0x00100000, 0x00000000, 3},
    {0x00200000, 0x00000000, 5},  {0x00002000, 0x00000000, 7},
    {0x00004000, 0x00000000, 8},  {0x00008000, 0x00000000, 6},
    {0x00010000, 0x00000000, 9},  {0x00040000, 0x00000000, 10},
    {0x00020000, 0x00000000, 11}, {0x00000004, 0x00000000, 12},
    {0x00000008, 0x00000200, 13}, {0x00000010, 0x00000000, 14},
    {0x00000020, 0x00000000, 15}, {0x00000040, 0x00800504, 16},
    {0x00000080, 0x00000A00, 17}, {0x00000100, 0x00000000, 18},
    {0x00000200, 0x00000000, 19}, {0x00000400, 0x00000000, 20},
    {0x00000800, 0x00000000, 21}, {0x00001000, 0x00000000, 22},
};
int ChannelForCategories(uint32_t bits) {
  for (const auto& e : kChannelMap) {
    if ((bits & ~(e.required | e.allowed)) == 0 &&
        (bits & e.required) == e.required) {
      return int(e.channel);
    }
  }
  if (!bits) return 0;
  return (bits & 0xFF7FF003u) ? 0 : 23;
}

std::string FoldLowerAscii(std::string_view s) {
  std::string out;
  out.reserve(s.size());
  for (char c : s) {
    out.push_back(c >= 'A' && c <= 'Z' ? char(c - 'A' + 'a') : c);
  }
  return out;
}

// Case-insensitive substring match of the active query against a UTF-16
// display name (item names are ASCII in practice; non-ASCII code units are
// folded to bytes and simply won't match an ASCII query).
bool CatalogSearchMatchesName(const std::u16string& name16,
                              const std::string& needle_lower) {
  std::string hay;
  hay.reserve(name16.size());
  for (char16_t c : name16) {
    char b = c < 128 ? char(c) : char(0x1A);
    hay.push_back(b >= 'A' && b <= 'Z' ? char(b - 'A' + 'a') : b);
  }
  return hay.find(needle_lower) != std::string::npos;
}
bool CatalogSearchMatchesName(const std::string& name8,
                              const std::string& needle_lower) {
  return FoldLowerAscii(name8).find(needle_lower) != std::string::npos;
}
}  // namespace

void SetAvatarCatalogSearch(const std::string& query_utf8) {
  std::lock_guard<std::mutex> lock(g_catalog_search_mutex);
  g_catalog_search = FoldLowerAscii(query_utf8);
  g_catalog_search_matches.store(0, std::memory_order_relaxed);
}

std::string GetAvatarCatalogSearch() {
  std::lock_guard<std::mutex> lock(g_catalog_search_mutex);
  return g_catalog_search;
}

uint32_t GetAvatarCatalogSearchMatches() {
  return g_catalog_search_matches.load(std::memory_order_relaxed);
}

void SetAvatarCatalogSearchScope(int channel) {
  g_catalog_search_scope.store(channel, std::memory_order_relaxed);
}

int GetAvatarCatalogSearchScope() {
  return g_catalog_search_scope.load(std::memory_order_relaxed);
}

uint32_t QueryAvatarCatalog(const std::string& query_utf8, size_t max_results,
                            std::vector<AvatarCatalogMatch>& out) {
  out.clear();
  const std::string needle = FoldLowerAscii(query_utf8);
  if (needle.empty()) {
    return 0;
  }
  // Mirror the grid's population exactly: only items in the scoped channel
  // (the catalog being browsed) and matching the last enumeration's body
  // mask count, so the live "N matches" line equals what the grid shows.
  const int scope = g_catalog_search_scope.load(std::memory_order_relaxed);
  const uint32_t body_mask =
      g_catalog_search_body_mask.load(std::memory_order_relaxed);
  const auto in_scope = [&](uint32_t categories, uint32_t bodies) {
    if (body_mask && !(bodies & body_mask)) return false;
    return scope < 0 || ChannelForCategories(categories) == scope;
  };
  uint32_t total = 0;
  // Stock pack (companion halves excluded, same as the enumeration).
  avatars::AssetPack* pack =
      (g_title_version < 3) ? &g_legacy_avatar_asset_pack : &g_avatar_asset_pack;
  if (!pack->is_loaded()) {
    pack = pack == &g_avatar_asset_pack ? &g_legacy_avatar_asset_pack
                                        : &g_avatar_asset_pack;
  }
  if (pack && pack->is_loaded()) {
    const auto& infos = pack->asset_infos();
    std::vector<bool> is_companion(infos.size(), false);
    for (size_t index = 0; index < infos.size(); ++index) {
      const auto& id0 = infos[index].asset_ids[0];
      const size_t partner = id0.b;
      if (!id0.is_zero() && partner != index && partner < infos.size()) {
        is_companion[partner] = true;
      }
    }
    for (size_t index = 0; index < infos.size(); ++index) {
      if (is_companion[index] || !(infos[index].flags & 1)) {
        continue;
      }
      if (!in_scope(infos[index].categories, infos[index].bodies)) {
        continue;
      }
      auto name16 = pack->GetAssetNameByIndex(index);
      if (name16.empty() || !CatalogSearchMatchesName(name16, needle)) {
        continue;
      }
      total++;
      if (out.size() < max_results) {
        AvatarCatalogMatch m;
        m.name.reserve(name16.size());
        for (char16_t c : name16) {
          m.name.push_back(c < 128 ? char(c) : '?');
        }
        m.categories = infos[index].categories;
        m.from_closet = false;
        out.push_back(std::move(m));
      }
    }
  }
  for (const auto& item : avatars::GetCloset().items()) {
    if (!in_scope(item.categories, item.bodies)) {
      continue;
    }
    if (!CatalogSearchMatchesName(item.name, needle)) {
      continue;
    }
    total++;
    if (out.size() < max_results) {
      out.push_back(AvatarCatalogMatch{item.name, item.categories, true});
    }
  }
  return total;
}

u32 XamAvatarBeginEnumAssets_entry(u32 user_index, u32 items_per_enumerate, u32 category_mask,
                                   u32 body_mask, u32 flags, mapped_void overlapped_ptr) {
  std::lock_guard<std::mutex> lock(g_asset_enum_mutex);
  g_asset_enumerator = AssetEnumerator(items_per_enumerate ? items_per_enumerate : 50);
  REXKRNL_INFO("[avatar] enum assets cats={:#x} bodies={:#x} closet={}", category_mask, body_mask,
               avatars::GetCloset().items().size());

  // Catalog search (Ctrl+F): while a query is armed, enumerate only items
  // whose display name matches, the grid then shows exactly the matches.
  std::string search_lower;
  {
    std::lock_guard<std::mutex> slock(g_catalog_search_mutex);
    search_lower = g_catalog_search;
  }
  const bool search_active = !search_lower.empty();
  uint32_t search_matches = 0;
  // Scope (avatar_search.h): filter only items belonging to the browsed
  // grid's channel, every other category enumerates unfiltered, so the
  // search never empties catalogs the user is not looking at.
  const int search_scope =
      g_catalog_search_scope.load(std::memory_order_relaxed);
  const auto search_applies = [&](uint32_t categories) {
    return search_active && (search_scope < 0 ||
                             ChannelForCategories(categories) == search_scope);
  };
  g_catalog_search_body_mask.store(body_mask, std::memory_order_relaxed);

  avatars::AssetPack* pack =
      (g_title_version < 3) ? &g_legacy_avatar_asset_pack : &g_avatar_asset_pack;
  if (!pack->is_loaded() && !LoadAvatarAssetPack()) {
    pack = nullptr;
  } else if (!pack->is_loaded()) {
    pack = pack == &g_avatar_asset_pack ? &g_legacy_avatar_asset_pack : &g_avatar_asset_pack;
  }
  if (pack && pack->is_loaded()) {
    const auto& infos = pack->asset_infos();
    // Paired wearables ship as catalog + companion entries: the catalog
    // entry's asset_ids[0].b points at its partner (the flattened
    // "worn-under-a-hat" variant, whose name ends in "(Hat)"). Enumerate only
    // the catalog halves: an entry pointed at by another entry's id is a
    // companion and would otherwise duplicate every style in the grid.
    std::vector<bool> is_companion(infos.size(), false);
    for (size_t index = 0; index < infos.size(); ++index) {
      const auto& id0 = infos[index].asset_ids[0];
      const size_t partner = id0.b;
      if (!id0.is_zero() && partner != index && partner < infos.size()) {
        is_companion[partner] = true;
      }
    }
    for (size_t index = 0; index < infos.size(); ++index) {
      const auto& asset_info = infos[index];
      if (!(asset_info.categories & category_mask) || !(asset_info.bodies & body_mask)) {
        continue;
      }
      if (is_companion[index]) {
        continue;
      }
      auto name = pack->GetAssetNameByIndex(index);
      if (search_applies(asset_info.categories)) {
        if (!CatalogSearchMatchesName(name, search_lower)) {
          continue;
        }
        search_matches++;
      }
      // The loader resolves every component id purely through AssetId.b
      // (= pack index, see FindAsset / GetAssetData), so the id handed out must
      // resolve to this entry. Most pack entries carry a zero asset_ids[0], and
      // catalog halves of paired wearables carry an id whose .b points at their
      // companion. Construct the id the way the title's shipped preset
      // manifests encode theirs: {a = category bits, b = pack index,
      // c = bodies mask, d = stock tail}. The .c field is the asset's gender
      // mask (1 = male, 2 = female, 3 = unisex), not a constant: the editor
      // derives .c = bodies for the ids it writes into saved-outfit manifests,
      // and the outfit-display gate (sub_920B9898) requires an exact id match
      // against these enumerated entries.
      avatars::AssetId id = asset_info.asset_ids[0];
      if (id.is_zero() || id.b.get() != static_cast<uint16_t>(index)) {
        id.a = asset_info.categories;
        id.b = static_cast<uint16_t>(index);
        id.c = asset_info.bodies ? asset_info.bodies : 3;
        std::memcpy(id.d, kPackAssetGuidTail, sizeof(id.d));
      }
      auto* guest_info = g_asset_enumerator.AppendItem();
      *guest_info = {};
      guest_info->asset_id = id;
      guest_info->categories = asset_info.categories;
      guest_info->body_mask = asset_info.bodies;
      guest_info->random_body_mask = asset_info.random_bodies;
      guest_info->flags = asset_info.flags;
      guest_info->subcategory = asset_info.subcategory;
      guest_info->data_offset = static_cast<uint32_t>(asset_info.data_offset);
      guest_info->data_size = static_cast<uint32_t>(asset_info.data_size);
      const size_t n = std::min(name.size(), rex::countof(guest_info->name) - 1);
      for (size_t k = 0; k < n; ++k) {
        guest_info->name[k] = static_cast<uint16_t>(name[k]);
      }
    }
    // Imported marketplace items (the closet) appear in the same grids.
    // Their ids are full product GUIDs, resolved by GetAssetBinary /
    // LoadAsset through the closet rather than by pack index.
    for (const auto& item : avatars::GetCloset().items()) {
      if (!(item.categories & category_mask) || !(item.bodies & body_mask)) {
        continue;
      }
      if (search_applies(item.categories)) {
        if (!CatalogSearchMatchesName(item.name, search_lower)) {
          continue;
        }
        search_matches++;
      }
      auto* guest_info = g_asset_enumerator.AppendItem();
      *guest_info = {};
      guest_info->asset_id = item.id;
      guest_info->categories = item.categories;
      guest_info->body_mask = item.bodies;
      guest_info->random_body_mask = item.bodies;
      // Stock pack entries carry flags=0x01 and the editor enumerates with
      // flags=0x1; items without it are filtered out of the grids. 0x400 marks
      // downloaded content: the re-enumeration after a purchase (sub_92210748)
      // only adds items that carry it.
      guest_info->flags = 1 | 0x400;
      // That same pass locks an item (entry+51, the exclamation badge) unless
      // this mask carries the bit for the running skeleton version,
      // 1 << (version >= 3). Closet items load on both, so advertise both.
      guest_info->skeleton_version_mask = 0x3;
      guest_info->subcategory = 0;
      CopyUtf8ToUtf16(item.name, guest_info->name, rex::countof(guest_info->name));
      if (item.is_award) {
        // The editor's database builder (AvatarEditor.xex sub_92210208)
        // routes an entry to the per-game Awards page when flags bit 0x100
        // is set: it files the record under a title table keyed by title_id
        // (title_name = the game's label), shows description as the
        // "reason" and the FILETIME pair as the unlock date. 0x200 = earned
        // (the item is also offered in the regular grids; clear = locked,
        // Awards page only), 0x800 = secret (generic text instead of
        // name/description).
        // 0x400 = installed: the editor's second pass (sub_92210748) only
        // marks an award wearable (entry+50) when this bit is set; otherwise
        // the Awards page legend reads "Download".
        guest_info->flags = 1u | 0x100u | 0x200u | 0x400u;
        guest_info->title_id = item.title_id;
        CopyUtf8ToUtf16(item.title_name, guest_info->title_name,
                        rex::countof(guest_info->title_name));
        CopyUtf8ToUtf16(item.description, guest_info->description,
                        rex::countof(guest_info->description));
        guest_info->time_0 = static_cast<uint32_t>(item.unlock_time & 0xFFFFFFFFu);  // low
        guest_info->time_1 = static_cast<uint32_t>(item.unlock_time >> 32);           // high
      }
    }
  }
  if (search_active) {
    g_catalog_search_matches.store(search_matches, std::memory_order_relaxed);
  }
  if (overlapped_ptr) {
    REX_KERNEL_STATE()->CompleteOverlappedImmediate(overlapped_ptr.guest_address(),
                                                    X_ERROR_SUCCESS);
    return X_ERROR_IO_PENDING;
  }
  return X_STATUS_SUCCESS;
}

u32 XamAvatarEnumAssets_entry(mapped_void buffer, mapped_u32 written_items, u32 unk3) {
  std::lock_guard<std::mutex> lock(g_asset_enum_mutex);
  if (!buffer) {
    return X_ERROR_INVALID_PARAMETER;
  }
  auto* out = REX_KERNEL_MEMORY()->TranslateVirtual<uint8_t*>(buffer.guest_address());
  uint32_t written = 0;
  const uint32_t result = g_asset_enumerator.WriteItems(out, &written);
  if (written_items) {
    *written_items = written;
  }
  return result;
}

u32 XamAvatarEndEnumAssets_entry(mapped_void overlapped_ptr) {
  {
    std::lock_guard<std::mutex> lock(g_asset_enum_mutex);
    g_asset_enumerator = AssetEnumerator(50);
  }
  if (overlapped_ptr) {
    REX_KERNEL_STATE()->CompleteOverlappedImmediate(overlapped_ptr.guest_address(),
                                                    X_ERROR_SUCCESS);
    return X_ERROR_IO_PENDING;
  }
  return X_STATUS_SUCCESS;
}

// Other ----------------------------------------------------------------------
u32 XamAvatarWearNow_entry(u64 unk1, mapped_u32 unk2, mapped_void overlapped_ptr) {
  X_RESULT result = X_ERROR_SUCCESS;
  if (REX_KERNEL_STATE()->title_id() != kAvatarEditorID) {
    if (overlapped_ptr) {
      REX_KERNEL_STATE()->CompleteOverlappedImmediate(overlapped_ptr.guest_address(), result);
      return X_ERROR_IO_PENDING;
    }
  }
  return result;
}

u32 XamAvatarReinstallAwardedAsset_entry(mapped_string string_out_ptr, u32 string_size,
                                         mapped_u32 unk_ptr) {
  return X_ERROR_SUCCESS;
}

}  // namespace xam
}  // namespace kernel
}  // namespace rex

REX_EXPORT(__imp__XamAvatarInitialize, rex::kernel::xam::XamAvatarInitialize_entry)
REX_EXPORT(__imp__XamAvatarShutdown, rex::kernel::xam::XamAvatarShutdown_entry)
REX_EXPORT(__imp__XamAvatarGetManifestLocalUser,
           rex::kernel::xam::XamAvatarGetManifestLocalUser_entry)
REX_EXPORT(__imp__XamAvatarGetManifestsByXuid, rex::kernel::xam::XamAvatarGetManifestsByXuid_entry)
REX_EXPORT(__imp__XamAvatarGetAssetsResultSize,
           rex::kernel::xam::XamAvatarGetAssetsResultSize_entry)
REX_EXPORT(__imp__XamAvatarGetAssets, rex::kernel::xam::XamAvatarGetAssets_entry)
REX_EXPORT(__imp__XamAvatarSetCustomAsset, rex::kernel::xam::XamAvatarSetCustomAsset_entry)
REX_EXPORT(__imp__XamAvatarSetManifest, rex::kernel::xam::XamAvatarSetManifest_entry)
REX_EXPORT(__imp__XamAvatarGetMetadataRandom, rex::kernel::xam::XamAvatarGetMetadataRandom_entry)
REX_EXPORT(__imp__XamAvatarGetMetadataSignedOutProfileCount,
           rex::kernel::xam::XamAvatarGetMetadataSignedOutProfileCount_entry)
REX_EXPORT(__imp__XamAvatarGetMetadataSignedOutProfile,
           rex::kernel::xam::XamAvatarGetMetadataSignedOutProfile_entry)
REX_EXPORT(__imp__XamAvatarManifestGetBodyType,
           rex::kernel::xam::XamAvatarManifestGetBodyType_entry)
REX_EXPORT(__imp__XamAvatarGetInstrumentation,
           rex::kernel::xam::XamAvatarGetInstrumentation_entry)
REX_EXPORT(__imp__XamAvatarGetAssetIcon, rex::kernel::xam::XamAvatarGetAssetIcon_entry)
REX_EXPORT(__imp__XamAvatarGetAssetBinary, rex::kernel::xam::XamAvatarGetAssetBinary_entry)
REX_EXPORT(__imp__XamAvatarGetInstalledAssetPackageDescription,
           rex::kernel::xam::XamAvatarGetInstalledAssetPackageDescription_entry)
REX_EXPORT(__imp__XamAvatarSetMocks, rex::kernel::xam::XamAvatarSetMocks_entry)
REX_EXPORT(__imp__XamAvatarLoadAnimation, rex::kernel::xam::XamAvatarLoadAnimation_entry)
REX_EXPORT(__imp__XamAvatarGenerateMipMaps, rex::kernel::xam::XamAvatarGenerateMipMaps_entry)
REX_EXPORT(__imp__XamAvatarBeginEnumAssets, rex::kernel::xam::XamAvatarBeginEnumAssets_entry)
REX_EXPORT(__imp__XamAvatarEnumAssets, rex::kernel::xam::XamAvatarEnumAssets_entry)
REX_EXPORT(__imp__XamAvatarEndEnumAssets, rex::kernel::xam::XamAvatarEndEnumAssets_entry)
REX_EXPORT(__imp__XamAvatarWearNow, rex::kernel::xam::XamAvatarWearNow_entry)
REX_EXPORT(__imp__XamAvatarReinstallAwardedAsset,
           rex::kernel::xam::XamAvatarReinstallAwardedAsset_entry)
