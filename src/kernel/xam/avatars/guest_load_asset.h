/**
 ******************************************************************************
 * Xenia : Xbox 360 Emulator Research Project                                 *
 ******************************************************************************
 * Copyright 2026 Ben Vanik. All rights reserved.                             *
 * Released under the BSD license - see LICENSE in the root for more details. *
 ******************************************************************************
 */

#ifndef XENIA_AVATARS_GUEST_LOAD_ASSET_H_
#define XENIA_AVATARS_GUEST_LOAD_ASSET_H_

#include <vector>

#include "asset_pack.h"
#include "guest_asset.h"
#include "memory_block.h"
#include <rex/types.h>

#include "xe_compat.h"

namespace rex {
namespace avatars {

bool LoadAssetsToGuest(const X_AVATAR_METADATA& metadata,
                       uint32_t category_mask, uint32_t flags,
                       AssetPack* asset_pack, MemoryBlock* cpu_memory,
                       MemoryBlock* gpu_memory, uint32_t skeleton_version,
                       uint32_t coordinate_system);

// Registers the legacy (pre-Fall-2010) asset pack so the old-body item rescue
// can build an exact per-vertex old->new body correspondence; the legacy body
// shares topology with the current one. Called by xam_avatar.cpp after
// LoadAvatarAssetPack; pass nullptr to clear.
void SetLegacyAssetPack(AssetPack* pack);

}  // namespace avatars
}  // namespace rex

#endif  // XENIA_AVATARS_GUEST_LOAD_ASSET_H_
