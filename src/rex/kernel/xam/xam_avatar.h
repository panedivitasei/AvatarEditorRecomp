/**
 ******************************************************************************
 * Xenia : Xbox 360 Emulator Research Project                                 *
 ******************************************************************************
 * Copyright 2025 Ben Vanik. All rights reserved.                             *
 * Released under the BSD license - see LICENSE in the root for more details. *
 ******************************************************************************
 *
 * @modified    Tom Clay, 2026 - Ported to ReXGlue runtime (xe:: -> rex::)
 */

#ifndef REX_KERNEL_XAM_XAM_AVATAR_H_
#define REX_KERNEL_XAM_XAM_AVATAR_H_

#include <rex/system/xtypes.h>  // static_assert_size, X_* types

namespace rex {
namespace kernel {
namespace xam {

struct X_ASSET_ID {   // X_GUID
  uint64_t data;      // 0x0 sz:0x8
  uint32_t data2;     // 0x8 sz:0x4
  uint32_t title_id;  // 0xC sz:0x4
};
static_assert_size(X_ASSET_ID, 0x10);

// More Research Needed (mostly opaque guest buffer; named fields per xenia).
struct X_AVATAR_METADATA {
  // body type exists between 0x120 and 0x130
  uint8_t data1[0x4];
  uint32_t weight;           // 0x4 sz:0x4
  uint32_t height;           // 0x8 sz:0x4
  uint8_t data3[0xF0];       // 0xC sz:0xF0
  uint32_t skin_color;       // 0xFC sz:0x4
  uint8_t data4[0x4];        // 0x100 sz:0x4
  uint32_t lipstick_color;   // 0x104 sz:0x4
  uint8_t data5[0x8];        // 0x108 sz:0x4
  uint32_t eyeshadow_color;  // 0x110 sz:0x4
  uint8_t data6[0x2D4];
};
static_assert_size(X_AVATAR_METADATA, 0x3E8);

// https://github.com/hetelek/Velocity AvatarAssetDefinitions.h
enum X_AVATAR_BODY_TYPE : uint8_t { Unknown, Male, Female, All };

enum X_BINARY_ASSET_TYPE : uint32_t {
  Component = 1,
  Texture = 2,
  ShapeOverride = 3,
  Animation = 4,
  ShapeOverridePost = 5,
};

}  // namespace xam
}  // namespace kernel

namespace system {
namespace xam {
class UserProfile;
}  // namespace xam
}  // namespace system

namespace kernel {
namespace xam {

// Backs XPROFILE_GAMERCARD_AVATAR_INFO_1 (0x63E80044) with the persisted avatar
// manifest, so titles that read the player's avatar from the profile see the
// saved avatar. Reads the shared manifest file and adds or refreshes the binary
// setting on the profile; no-op when no manifest has been saved yet.
void EnsureAvatarProfileSetting(system::xam::UserProfile* profile);

// Directory holding the persisted avatar manifest (avatar_data_dir cvar or
// <user_data_root>\avatars), created on demand. The editor's gamer pictures are
// saved beside the manifest (xam_user.cpp).
std::string AvatarDataDir();

}  // namespace xam
}  // namespace kernel
}  // namespace rex

#endif  // REX_KERNEL_XAM_XAM_AVATAR_H_
