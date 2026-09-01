/**
 ******************************************************************************
 * Xenia : Xbox 360 Emulator Research Project                                 *
 ******************************************************************************
 * Copyright 2022 Ben Vanik. All rights reserved.                             *
 * Released under the BSD license - see LICENSE in the root for more details. *
 ******************************************************************************
 *
 * @modified    Tom Clay, 2026 - Adapted for ReXGlue runtime
 */

// Disable warnings about unused parameters for kernel functions
#pragma GCC diagnostic ignored "-Wunused-parameter"

#include <cstdio>
#include <cstring>

#include <rex/cvar.h>
#include <rex/kernel/xam/private.h>
#include <rex/kernel/xam/tile_icon.h>
#include <rex/kernel/xam/xam_avatar.h>

#include "avatars/closet.h"
#include <rex/kernel/xnet.h>
#include <rex/filesystem.h>
#include <rex/logging.h>
#include <rex/math.h>
#include <rex/hook.h>
#include <rex/types.h>
#include <rex/string.h>
#include <rex/system/kernel_state.h>
#include <rex/system/xam/user_profile.h>
#include <rex/system/xenumerator.h>
#include <rex/system/xio.h>
#include <rex/system/xthread.h>
#include <rex/system/xtypes.h>

// Decodes the user-configured gamer picture (user_gamerpic cvar).
#define STB_IMAGE_IMPLEMENTATION
#define STBI_NO_STDIO
#define STBI_ONLY_PNG
#define STBI_ONLY_JPEG
#define STBI_ONLY_BMP
#define STBI_ONLY_TGA
#include "stb_image.h"

REXCVAR_DEFINE_UINT32(user_language, 1, "Kernel", "User's language ID");
REXCVAR_DEFINE_STRING(user_gamerpic, "", "Kernel",
                      "Path to an image (PNG/JPG/BMP/TGA) served as the profile's gamer picture "
                      "(64x64 recommended). Relative paths resolve against the exe folder. "
                      "Empty keeps the title's default tile.");

namespace rex {
namespace kernel {
namespace xam {
using namespace rex::system;
using namespace rex::system::xam;

// The runtime's UserProfile reports local sign-in only (signin_state() == 1,
// no online XUID). The editor's Gamer Pic camera gate and other live-gated
// flows need a LIVE sign-in with an online XUID, so report those here.
static constexpr uint64_t kOnlineXuid = 0x0009BABEBABEBABEULL;
static constexpr uint32_t kSigninStateLive = 2;  // 2 = signed in to LIVE

i32 XamUserGetXUID_entry(u32 user_index, u32 type_mask, mapped_u64 xuid_ptr) {
  assert_true(type_mask == 1 || type_mask == 2 || type_mask == 3 || type_mask == 4 ||
              type_mask == 7);
  if (!xuid_ptr) {
    return X_E_INVALIDARG;
  }
  uint32_t result = X_E_NO_SUCH_USER;
  uint64_t xuid = 0;
  if (user_index < 4) {
    if (user_index == 0) {
      const auto& user_profile = REX_KERNEL_STATE()->user_profile();
      auto type = user_profile->type() & type_mask;
      if (type & (2 | 4)) {
        // Online profile: return the online-format XUID (0x0009...).
        xuid = kOnlineXuid;
        result = X_E_SUCCESS;
      } else if (type & 1) {
        // maybe offline profile?
        xuid = user_profile->xuid();
        result = X_E_SUCCESS;
      }
    }
  } else {
    result = X_E_INVALIDARG;
  }
  *xuid_ptr = xuid;
  return result;
}

// The configured sign-in state is reported to every title. The Avatar Editor's
// Gamer Pic camera gate (sub_920DFDF0) needs a live-enabled profile,
// SignedInToLive and the live connection notification, otherwise it shows
// "only available with an Xbox Live profile"; the gamer-pic ops themselves are
// served offline by XamPngEncodeEx / XamWriteGamerTileEx below.
u32 XamUserGetSigninState_entry(u32 user_index) {
  uint32_t signin_state = 0;
  if (user_index < 4) {
    if (user_index == 0) {
      signin_state = kSigninStateLive;
    }
  }
  return signin_state;
}

typedef struct {
  rex::be<uint64_t> xuid;
  rex::be<uint32_t> info_flags;  // XUSER_INFO_FLAGS; bit 0 = XUSER_INFO_FLAG_LIVE_ENABLED
  rex::be<uint32_t> signin_state;
  rex::be<uint32_t> unk10;  // ?
  rex::be<uint32_t> unk14;  // ?
  char name[16];
} X_USER_SIGNIN_INFO;
static_assert_size(X_USER_SIGNIN_INFO, 40);

i32 XamUserGetSigninInfo_entry(u32 user_index, u32 flags, ppc_ptr_t<X_USER_SIGNIN_INFO> info) {
  if (!info) {
    return X_E_INVALIDARG;
  }

  std::memset(info, 0, sizeof(X_USER_SIGNIN_INFO));
  if (user_index) {
    return X_E_NO_SUCH_USER;
  }

  const auto& user_profile = REX_KERNEL_STATE()->user_profile();
  const uint32_t signin_state = kSigninStateLive;
  // Report the online XUID when signed in to live: live-gated titles check that
  // the XUID is online-format (0x0009...) and reject the offline/dummy XUID.
  info->xuid = (signin_state == 2) ? kOnlineXuid : user_profile->xuid();
  info->signin_state = signin_state;
  // unk08 = dwInfoFlags. Bit 0 = XUSER_INFO_FLAG_LIVE_ENABLED. Online-gated
  // titles read this bit via XamUserGetSigninInfo to decide whether the
  // profile is "Xbox live enabled"; with it clear they reject the profile.
  // Set it whenever the user is SignedInToLive (state 2).
  info->info_flags = (signin_state == 2) ? 0x00000001u : 0u;
  rex::string::copy_truncating(info->name, user_profile->name(), rex::countof(info->name));
  return X_E_SUCCESS;
}

u32 XamUserGetName_entry(u32 user_index, mapped_string buffer, u32 buffer_len) {
  if (user_index >= 4) {
    return X_E_INVALIDARG;
  }

  if (user_index) {
    return X_E_NO_SUCH_USER;
  }

  const auto& user_profile = REX_KERNEL_STATE()->user_profile();
  const auto& user_name = user_profile->name();
  rex::string::copy_truncating(buffer, user_name, std::min(buffer_len, uint32_t(16)));
  return X_E_SUCCESS;
}

u32 XamUserGetGamerTag_entry(u32 user_index, mapped_wstring buffer, u32 buffer_len) {
  if (user_index >= 4) {
    return X_E_INVALIDARG;
  }

  if (user_index) {
    return X_E_NO_SUCH_USER;
  }

  if (!buffer || buffer_len < 16) {
    return X_E_INVALIDARG;
  }

  const auto& user_profile = REX_KERNEL_STATE()->user_profile();
  auto user_name = rex::string::to_utf16(user_profile->name());
  rex::string::copy_and_swap_truncating(buffer, user_name, std::min(buffer_len, uint32_t(16)));
  return X_E_SUCCESS;
}

typedef struct {
  rex::be<uint32_t> setting_count;
  rex::be<uint32_t> settings_ptr;
} X_USER_READ_PROFILE_SETTINGS;
static_assert_size(X_USER_READ_PROFILE_SETTINGS, 8);

// https://github.com/oukiar/freestyledash/blob/master/Freestyle/Tools/Generic/xboxtools.cpp
uint32_t XamUserReadProfileSettingsEx(uint32_t title_id, uint32_t user_index, uint32_t xuid_count,
                                      be<uint64_t>* xuids, uint32_t setting_count,
                                      be<uint32_t>* setting_ids, uint32_t unk,
                                      be<uint32_t>* buffer_size_ptr, uint8_t* buffer,
                                      XAM_OVERLAPPED* overlapped) {
  if (!xuid_count) {
    assert_null(xuids);
  } else {
    assert_true(xuid_count == 1);
    assert_not_null(xuids);
    // TODO(gibbed): allow proper lookup of arbitrary XUIDs
    const auto& user_profile = REX_KERNEL_STATE()->user_profile();
    assert_true(static_cast<uint64_t>(xuids[0]) == user_profile->xuid());
    // TODO(gibbed): we assert here, but in case a title passes xuid_count > 1
    // until it's implemented for release builds...
    xuid_count = 1;
  }
  assert_zero(unk);  // probably flags

  // must have at least 1 to 32 settings
  if (setting_count < 1 || setting_count > 32) {
    return X_ERROR_INVALID_PARAMETER;
  }

  // buffer size pointer must be valid
  if (!buffer_size_ptr) {
    return X_ERROR_INVALID_PARAMETER;
  }

  // if buffer size is non-zero, buffer pointer must be valid
  auto buffer_size = static_cast<uint32_t>(*buffer_size_ptr);
  if (buffer_size && !buffer) {
    return X_ERROR_INVALID_PARAMETER;
  }

  uint32_t needed_header_size = 0;
  uint32_t needed_data_size = 0;
  for (uint32_t i = 0; i < setting_count; ++i) {
    needed_header_size += sizeof(X_USER_PROFILE_SETTING);
    UserProfile::Setting::Key setting_key;
    setting_key.value = static_cast<uint32_t>(setting_ids[i]);
    switch (static_cast<UserProfile::Setting::Type>(setting_key.type)) {
      case UserProfile::Setting::Type::WSTRING:
      case UserProfile::Setting::Type::BINARY:
        needed_data_size += setting_key.size;
        break;
      default:
        break;
    }
  }
  if (xuids) {
    needed_header_size *= xuid_count;
    needed_data_size *= xuid_count;
  }
  needed_header_size += sizeof(X_USER_READ_PROFILE_SETTINGS);

  uint32_t needed_size = needed_header_size + needed_data_size;
  if (!buffer || buffer_size < needed_size) {
    if (!buffer_size) {
      *buffer_size_ptr = needed_size;
    }
    return X_ERROR_INSUFFICIENT_BUFFER;
  }

  // Title ID = 0 means us.
  // 0xfffe07d1 = profile?

  if (!xuids && user_index) {
    // Only support user 0.
    if (overlapped) {
      REX_KERNEL_STATE()->CompleteOverlappedImmediate(
          REX_KERNEL_MEMORY()->HostToGuestVirtual(overlapped), X_ERROR_NO_SUCH_USER);
      return X_ERROR_IO_PENDING;
    }
    return X_ERROR_NO_SUCH_USER;
  }

  const auto& user_profile = REX_KERNEL_STATE()->user_profile();

  // Back XPROFILE_GAMERCARD_AVATAR_INFO_1 with the persisted manifest before
  // validity checking; titles read the player's avatar from this profile
  // setting at boot.
  EnsureAvatarProfileSetting(user_profile);

  // First call asks for size (fill buffer_size_ptr).
  // Second call asks for buffer contents with that size.

  // TODO(gibbed): setting validity checking without needing a user profile
  // object.
  bool any_missing = false;
  for (uint32_t i = 0; i < setting_count; ++i) {
    auto setting_id = static_cast<uint32_t>(setting_ids[i]);
    auto setting = user_profile->GetSetting(setting_id);
    if (!setting) {
      any_missing = true;
      REXKRNL_ERROR(
          "xeXamUserReadProfileSettingsEx requested unimplemented setting "
          "{:08X}",
          setting_id);
    }
  }
  if (any_missing) {
    // TODO(benvanik): don't fail? most games don't even check!
    if (overlapped) {
      REX_KERNEL_STATE()->CompleteOverlappedImmediate(
          REX_KERNEL_MEMORY()->HostToGuestVirtual(overlapped), X_ERROR_INVALID_PARAMETER);
      return X_ERROR_IO_PENDING;
    }
    return X_ERROR_INVALID_PARAMETER;
  }

  auto out_header = reinterpret_cast<X_USER_READ_PROFILE_SETTINGS*>(buffer);
  auto out_setting = reinterpret_cast<X_USER_PROFILE_SETTING*>(&out_header[1]);
  out_header->setting_count = static_cast<uint32_t>(setting_count);
  out_header->settings_ptr = REX_KERNEL_MEMORY()->HostToGuestVirtual(out_setting);

  UserProfile::SettingByteStream out_stream(REX_KERNEL_MEMORY()->HostToGuestVirtual(buffer), buffer,
                                            buffer_size, needed_header_size);
  for (uint32_t n = 0; n < setting_count; ++n) {
    uint32_t setting_id = setting_ids[n];
    auto setting = user_profile->GetSetting(setting_id);

    std::memset(out_setting, 0, sizeof(X_USER_PROFILE_SETTING));
    out_setting->from = !setting || !setting->is_set ? 0 : setting->is_title_specific() ? 2 : 1;
    if (xuids) {
      out_setting->xuid = user_profile->xuid();
    } else {
      out_setting->user_index = static_cast<uint32_t>(user_index);
    }
    out_setting->setting_id = setting_id;

    if (setting && setting->is_set) {
      setting->Append(&out_setting->data, &out_stream);
    }
    ++out_setting;
  }

  if (overlapped) {
    REX_KERNEL_STATE()->CompleteOverlappedImmediate(
        REX_KERNEL_MEMORY()->HostToGuestVirtual(overlapped), X_ERROR_SUCCESS);
    return X_ERROR_IO_PENDING;
  }
  return X_ERROR_SUCCESS;
}

u32 XamUserReadProfileSettings_entry(u32 title_id, u32 user_index, u32 xuid_count, mapped_u64 xuids,
                                     u32 setting_count, mapped_u32 setting_ids,
                                     mapped_u32 buffer_size_ptr, mapped_void buffer_ptr,
                                     ppc_ptr_t<XAM_OVERLAPPED> overlapped) {
  return XamUserReadProfileSettingsEx(title_id, user_index, xuid_count, xuids, setting_count,
                                      setting_ids, 0, buffer_size_ptr, buffer_ptr, overlapped);
}

u32 XamUserReadProfileSettingsEx_entry(u32 title_id, u32 user_index, u32 xuid_count,
                                       mapped_u64 xuids, u32 setting_count, mapped_u32 setting_ids,
                                       mapped_u32 buffer_size_ptr, u32 unk_2,
                                       mapped_void buffer_ptr,
                                       ppc_ptr_t<XAM_OVERLAPPED> overlapped) {
  return XamUserReadProfileSettingsEx(title_id, user_index, xuid_count, xuids, setting_count,
                                      setting_ids, unk_2, buffer_size_ptr, buffer_ptr, overlapped);
}

u32 XamUserWriteProfileSettings_entry(u32 title_id, u32 user_index, u32 setting_count,
                                      ppc_ptr_t<X_USER_PROFILE_SETTING> settings,
                                      ppc_ptr_t<XAM_OVERLAPPED> overlapped) {
  if (!setting_count || !settings) {
    return X_ERROR_INVALID_PARAMETER;
  }

  if (user_index) {
    // Only support user 0.
    if (overlapped) {
      REX_KERNEL_STATE()->CompleteOverlappedImmediate(overlapped.guest_address(),
                                                      X_ERROR_NO_SUCH_USER);
      return X_ERROR_IO_PENDING;
    }
    return X_ERROR_NO_SUCH_USER;
  }

  // Update and save settings.
  const auto& user_profile = REX_KERNEL_STATE()->user_profile();

  for (uint32_t n = 0; n < setting_count; ++n) {
    const X_USER_PROFILE_SETTING& setting = settings[n];

    auto setting_type = static_cast<UserProfile::Setting::Type>(setting.data.type);
    if (setting_type == UserProfile::Setting::Type::UNSET) {
      continue;
    }

    REXKRNL_DEBUG(
        "XamUserWriteProfileSettings: setting index [{}]:"
        " from={} setting_id={:08X} data.type={}",
        n, (uint32_t)setting.from, (uint32_t)setting.setting_id, setting.data.type);

    switch (setting_type) {
      case UserProfile::Setting::Type::CONTENT:
      case UserProfile::Setting::Type::BINARY: {
        uint8_t* binary_ptr = REX_KERNEL_MEMORY()->TranslateVirtual(setting.data.binary.ptr);
        size_t binary_size = setting.data.binary.size;
        std::vector<uint8_t> bytes;
        if (setting.data.binary.ptr) {
          // Copy provided data
          bytes.resize(binary_size);
          std::memcpy(bytes.data(), binary_ptr, binary_size);
        } else {
          // Data pointer was NULL, so just fill with zeroes
          bytes.resize(binary_size, 0);
        }
        user_profile->AddSetting(
            std::make_unique<xam::UserProfile::BinarySetting>(setting.setting_id, bytes));
      } break;
      case UserProfile::Setting::Type::WSTRING:
      case UserProfile::Setting::Type::DOUBLE:
      case UserProfile::Setting::Type::FLOAT:
      case UserProfile::Setting::Type::INT32:
      case UserProfile::Setting::Type::INT64:
      case UserProfile::Setting::Type::DATETIME:
      default: {
        REXKRNL_ERROR("XamUserWriteProfileSettings: Unimplemented data type {}", setting_type);
      } break;
    };
  }

  if (overlapped) {
    REX_KERNEL_STATE()->CompleteOverlappedImmediate(overlapped.guest_address(), X_ERROR_SUCCESS);
    return X_ERROR_IO_PENDING;
  }
  return X_ERROR_SUCCESS;
}

u32 XamUserCheckPrivilege_entry(u32 user_index, u32 mask, mapped_u32 out_value) {
  // checking all users?
  if (user_index != 0xFF) {
    if (user_index >= 4) {
      return X_ERROR_INVALID_PARAMETER;
    }

    if (user_index) {
      return X_ERROR_NO_SUCH_USER;
    }
  }

  // Grant all privileges for the signed-in live user. Live-required titles
  // re-prompt sign-in forever if online privileges are denied.
  *out_value = 1;
  return X_ERROR_SUCCESS;
}

u32 XamUserContentRestrictionGetFlags_entry(u32 user_index, mapped_u32 out_flags) {
  if (user_index) {
    return X_ERROR_NO_SUCH_USER;
  }

  // No restrictions?
  *out_flags = 0;
  return X_ERROR_SUCCESS;
}

u32 XamUserContentRestrictionGetRating_entry(u32 user_index, u32 unk1, mapped_u32 out_unk2,
                                             mapped_u32 out_unk3) {
  if (user_index) {
    return X_ERROR_NO_SUCH_USER;
  }

  // Some games have special case paths for 3F that differ from the failure
  // path, so my guess is that's 'don't care'.
  *out_unk2 = 0x3F;
  *out_unk3 = 0;
  return X_ERROR_SUCCESS;
}

u32 XamUserContentRestrictionCheckAccess_entry(u32 user_index, u32 unk1, u32 unk2, u32 unk3,
                                               u32 unk4, mapped_u32 out_unk5, u32 overlapped_ptr) {
  *out_unk5 = 1;

  if (overlapped_ptr) {
    // TODO(benvanik): does this need the access arg on it?
    REX_KERNEL_STATE()->CompleteOverlappedImmediate(overlapped_ptr, X_ERROR_SUCCESS);
  }

  return X_ERROR_SUCCESS;
}

u32 XamUserIsOnlineEnabled_entry(u32 user_index) {
  return 1;
}

u32 XamUserGetMembershipTier_entry(u32 user_index) {
  if (user_index >= 4) {
    return X_ERROR_INVALID_PARAMETER;
  }
  if (user_index) {
    return X_ERROR_NO_SUCH_USER;
  }
  return 6 /* 6 appears to be Gold */;
}

u32 XamUserAreUsersFriends_entry(u32 user_index, u32 unk1, u32 unk2, mapped_u32 out_value,
                                 u32 overlapped_ptr) {
  uint32_t are_friends = 0;
  X_RESULT result;

  if (user_index >= 4) {
    result = X_ERROR_INVALID_PARAMETER;
  } else {
    if (user_index == 0) {
      // Always signed in under the title's live shim (kSigninStateLive).
      // No friends!
      are_friends = 0;
      result = X_ERROR_SUCCESS;
    } else {
      // Only support user 0.
      result = X_ERROR_NO_SUCH_USER;  // if user is local -> X_ERROR_NOT_LOGGED_ON
    }
  }

  if (out_value) {
    assert_true(!overlapped_ptr);
    *out_value = result == X_ERROR_SUCCESS ? are_friends : 0;
    return result;
  } else if (overlapped_ptr) {
    assert_true(!out_value);
    REX_KERNEL_STATE()->CompleteOverlappedImmediateEx(
        overlapped_ptr, result == X_ERROR_SUCCESS ? X_ERROR_SUCCESS : X_ERROR_FUNCTION_FAILED,
        X_HRESULT_FROM_WIN32(result), result == X_ERROR_SUCCESS ? are_friends : 0);
    return X_ERROR_IO_PENDING;
  } else {
    assert_always();
    return X_ERROR_INVALID_PARAMETER;
  }
}

u32 XamShowSigninUI_entry(u32 unk, u32 unk_mask) {
  // Mask values vary. Probably matching user types? Local/remote?

  // To fix game modes that display a 4 profile signin UI (even if playing
  // alone):
  // XN_SYS_SIGNINCHANGED
  REX_KERNEL_STATE()->BroadcastNotification(0x0000000A, 1);
  // Games sit and loop until this notification fires:
  // XN_SYS_UI (off)
  REX_KERNEL_STATE()->BroadcastNotification(0x00000009, 0);
  // XN_LIVE_CONNECTIONCHANGED = connected. Live-required titles sit in an
  // online-init loop until they are told the live link is up.
  REX_KERNEL_STATE()->BroadcastNotification(0x00000014, 1);
  return X_ERROR_SUCCESS;
}

// TODO(gibbed): probably a FILETIME/LARGE_INTEGER, unknown currently
struct X_ACHIEVEMENT_UNLOCK_TIME {
  rex::be<uint32_t> unk_0;
  rex::be<uint32_t> unk_4;
};

struct X_ACHIEVEMENT_DETAILS {
  rex::be<uint32_t> id;
  rex::be<uint32_t> label_ptr;
  rex::be<uint32_t> description_ptr;
  rex::be<uint32_t> unachieved_ptr;
  rex::be<uint32_t> image_id;
  rex::be<uint32_t> gamerscore;
  X_ACHIEVEMENT_UNLOCK_TIME unlock_time;
  rex::be<uint32_t> flags;

  static const size_t kStringBufferSize = 464;
};
static_assert_size(X_ACHIEVEMENT_DETAILS, 36);

class XStaticAchievementEnumerator : public XEnumerator {
 public:
  struct AchievementDetails {
    uint32_t id;
    std::u16string label;
    std::u16string description;
    std::u16string unachieved;
    uint32_t image_id;
    uint32_t gamerscore;
    struct {
      uint32_t unk_0;
      uint32_t unk_4;
    } unlock_time;
    uint32_t flags;
  };

  XStaticAchievementEnumerator(KernelState* kernel_state, size_t items_per_enumerate,
                               uint32_t flags)
      : XEnumerator(kernel_state, items_per_enumerate,
                    sizeof(X_ACHIEVEMENT_DETAILS) +
                        (!!(flags & 7) ? X_ACHIEVEMENT_DETAILS::kStringBufferSize : 0)),
        flags_(flags) {}

  void AppendItem(AchievementDetails item) { items_.push_back(std::move(item)); }

  uint32_t WriteItems(uint32_t buffer_ptr, uint8_t* buffer_data, uint32_t* written_count) override {
    size_t count = std::min(items_.size() - current_item_, items_per_enumerate());
    if (!count) {
      return X_ERROR_NO_MORE_FILES;
    }

    size_t size = count * item_size();

    auto details = reinterpret_cast<X_ACHIEVEMENT_DETAILS*>(buffer_data);
    size_t string_offset = items_per_enumerate() * sizeof(X_ACHIEVEMENT_DETAILS);
    auto string_buffer =
        StringBuffer{buffer_ptr + static_cast<uint32_t>(string_offset), &buffer_data[string_offset],
                     count * X_ACHIEVEMENT_DETAILS::kStringBufferSize};
    for (size_t i = 0, o = current_item_; i < count; ++i, ++current_item_) {
      const auto& item = items_[current_item_];
      details[i].id = item.id;
      details[i].label_ptr = !!(flags_ & 1) ? AppendString(string_buffer, item.label) : 0;
      details[i].description_ptr =
          !!(flags_ & 2) ? AppendString(string_buffer, item.description) : 0;
      details[i].unachieved_ptr = !!(flags_ & 4) ? AppendString(string_buffer, item.unachieved) : 0;
      details[i].image_id = item.image_id;
      details[i].gamerscore = item.gamerscore;
      details[i].unlock_time.unk_0 = item.unlock_time.unk_0;
      details[i].unlock_time.unk_4 = item.unlock_time.unk_4;
      details[i].flags = item.flags;
    }

    if (written_count) {
      *written_count = static_cast<uint32_t>(count);
    }

    return X_ERROR_SUCCESS;
  }

 private:
  struct StringBuffer {
    uint32_t ptr;
    uint8_t* data;
    size_t remaining_bytes;
  };

  uint32_t AppendString(StringBuffer& sb, const std::u16string_view string) {
    size_t count = string.length() + 1;
    size_t size = count * sizeof(char16_t);
    if (size > sb.remaining_bytes) {
      assert_always();
      return 0;
    }
    auto ptr = sb.ptr;
    rex::string::copy_and_swap_truncating(reinterpret_cast<char16_t*>(sb.data), string, count);
    sb.ptr += static_cast<uint32_t>(size);
    sb.data += size;
    sb.remaining_bytes -= size;
    return ptr;
  }

 private:
  uint32_t flags_;
  std::vector<AchievementDetails> items_;
  size_t current_item_ = 0;
};

u32 XamUserCreateAchievementEnumerator_entry(u32 title_id, u32 user_index, u32 xuid, u32 flags,
                                             u32 offset, u32 count, mapped_u32 buffer_size_ptr,
                                             mapped_u32 handle_ptr) {
  if (!count || !buffer_size_ptr || !handle_ptr) {
    return X_ERROR_INVALID_PARAMETER;
  }

  if (user_index >= 4) {
    return X_ERROR_INVALID_PARAMETER;
  }

  size_t entry_size = sizeof(X_ACHIEVEMENT_DETAILS);
  if (flags & 7) {
    entry_size += X_ACHIEVEMENT_DETAILS::kStringBufferSize;
  }

  if (buffer_size_ptr) {
    *buffer_size_ptr = static_cast<uint32_t>(entry_size) * count;
  }

  auto e = object_ref<XStaticAchievementEnumerator>(
      new XStaticAchievementEnumerator(REX_KERNEL_STATE(), count, flags));
  auto result = e->Initialize(user_index, 0xFB, 0xB000A, 0xB000B, 0);
  if (XFAILED(result)) {
    return result;
  }

  const util::XdbfGameData db = REX_KERNEL_STATE()->title_xdbf();

  if (db.is_valid()) {
    const XLanguage language =
        db.GetExistingLanguage(static_cast<XLanguage>(REXCVAR_GET(user_language)));
    const std::vector<util::XdbfAchievementTableEntry> achievement_list = db.GetAchievements();

    for (const util::XdbfAchievementTableEntry& entry : achievement_list) {
      auto item = XStaticAchievementEnumerator::AchievementDetails{
          entry.id,
          rex::string::to_utf16(db.GetStringTableEntry(language, entry.label_id)),
          rex::string::to_utf16(db.GetStringTableEntry(language, entry.description_id)),
          rex::string::to_utf16(db.GetStringTableEntry(language, entry.unachieved_id)),
          entry.image_id,
          entry.gamerscore,
          {0, 0},
          entry.flags};

      e->AppendItem(item);
    }
  }

  *handle_ptr = e->handle();
  return X_ERROR_SUCCESS;
}

// Avatar-award (granted asset) enumerator. The Avatar Editor's wardrobe
// sanitizer (AvatarEditor.xex sub_920BFF80, run on creator/outfit screen
// transitions) validates every worn component whose asset id carries the award
// provenance nibble (((id.c >> 8) & 0xF) == 1, true for every title-granted
// avatar award guid, false for marketplace purchases and the constructed
// stock-pack ids) by enumerating the granting title's awards through this API
// (sub_920BFE40: id.d[4..7] = the granting title id) and requiring a matching
// entry with flags bit 0x20000 ("granted"). Without a match the sanitizer
// replaces the worn award item with a stock default on save, so serve the
// closet's award items as granted.
//
// Entry layout (52 bytes, per the check's disassembly): the worn id is matched
// as two 64-bit compares against entry bytes +4..+19, i.e. the entry carries
// the verbatim 16-byte asset id at +4; flags live at +44 (bit 0x20000 =
// granted).
struct X_AVATAR_AWARD_ENTRY {
  rex::be<uint32_t> title_id;    // +0
  avatars::AssetId id;           // +4..+19, verbatim (matched as 2 qwords)
  uint8_t reserved[24];          // +20
  rex::be<uint32_t> flags;       // +44
  rex::be<uint32_t> reserved2;   // +48
};
static_assert(sizeof(X_AVATAR_AWARD_ENTRY) == 52, "award entry is 52 bytes");

u32 XamUserCreateAvatarAssetEnumerator_entry(u32 user_index, u32 title_id, u32 unk3, u32 unk4,
                                             u32 unk5, u32 item_count, mapped_u32 buffer_size_ptr,
                                             mapped_u32 handle_ptr) {
  if (!handle_ptr || !item_count || user_index >= 4) {
    if (buffer_size_ptr) {
      *buffer_size_ptr = 0;
    }
    return X_ERROR_INVALID_PARAMETER;
  }
  if (buffer_size_ptr) {
    *buffer_size_ptr = static_cast<uint32_t>(sizeof(X_AVATAR_AWARD_ENTRY)) * item_count;
  }
  auto e = make_object<XStaticEnumerator<X_AVATAR_AWARD_ENTRY>>(REX_KERNEL_STATE(), item_count);
  auto result = e->Initialize(user_index, 0xFB, 0x20005, 0x20007, 0);
  if (XFAILED(result)) {
    return result;
  }
  for (const auto& item : avatars::GetCloset().items()) {
    const uint32_t c = item.id.c.get();
    if (((c >> 8) & 0xFu) != 1u) {
      continue;  // marketplace purchase, not an award
    }
    const uint32_t item_title = (uint32_t(item.id.d[4]) << 24) | (uint32_t(item.id.d[5]) << 16) |
                                (uint32_t(item.id.d[6]) << 8) | uint32_t(item.id.d[7]);
    if (title_id && item_title != title_id) {
      continue;
    }
    auto* entry = e->AppendItem();
    *entry = {};
    entry->title_id = item_title;
    entry->id = item.id;
    entry->flags = 0x20000u;  // granted
  }
  *handle_ptr = e->handle();
  return X_ERROR_SUCCESS;
}

u32 XamUserCreateStatsEnumerator_entry(u32 title_id, u32 enumerator_type, u64 pivot_user,
                                       u32 num_rows, u32 num_stats_specs,
                                       ppc_ptr_t<X_USER_STATS_SPEC> specs,
                                       mapped_u32 buffer_size_ptr, mapped_u32 handle_ptr) {
  // Must hand back a valid (empty) leaderboard: titles enumerate the handle and
  // parse the X_USER_STATS_READ_RESULTS without validating it, sorting
  // view->rows with whatever num_rows/rows_ptr they find in the buffer.
  if (!handle_ptr) {
    return X_ERROR_INVALID_PARAMETER;
  }
  *handle_ptr = 0;

  if (!buffer_size_ptr) {
    return X_ERROR_INVALID_PARAMETER;
  }
  *buffer_size_ptr = 0;

  if (!pivot_user || !specs) {
    return X_ERROR_INVALID_PARAMETER;
  }

  if (!num_rows || num_rows > X_STATS_MAX_ROW_COUNT) {
    return X_ERROR_INVALID_PARAMETER;
  }

  if (!num_stats_specs || enumerator_type > X_STATS_ENUMERATOR_TYPE::BY_RATING) {
    return X_ERROR_INVALID_PARAMETER;
  }

  auto* memory = REX_KERNEL_MEMORY();

  const uint32_t views_size =
      num_stats_specs * static_cast<uint32_t>(sizeof(X_USER_STATS_VIEW));
  const uint32_t views_address = memory->SystemHeapAlloc(views_size);
  auto* views = memory->TranslateVirtual<X_USER_STATS_VIEW*>(views_address);
  std::memset(views, 0, views_size);

  const X_USER_STATS_SPEC* spec_array = specs;
  for (uint32_t i = 0; i < num_stats_specs; ++i) {
    X_USER_STATS_VIEW& view = views[i];
    view.view_id = spec_array[i].view_id;
    view.total_view_rows = 0;
    view.num_rows = 0;
    // Some titles dereference rows_ptr even with zero rows, so keep it valid.
    const uint32_t rows_address =
        memory->SystemHeapAlloc(static_cast<uint32_t>(sizeof(X_USER_STATS_ROW)));
    auto* row = memory->TranslateVirtual<X_USER_STATS_ROW*>(rows_address);
    std::memset(row, 0, sizeof(X_USER_STATS_ROW));
    view.rows_ptr = rows_address;
  }

  auto e = make_object<XStaticEnumerator<X_USER_STATS_READ_RESULTS>>(REX_KERNEL_STATE(), 1);
  auto result = e->Initialize(0xFF, 0xFB, 0xB0023, 0xB0024, 0);
  if (XFAILED(result)) {
    return result;
  }

  auto* results = e->AppendItem();
  results->num_views = num_stats_specs;
  results->views_ptr = views_address;

  *buffer_size_ptr = static_cast<uint32_t>(sizeof(X_USER_STATS_READ_RESULTS)) + views_size;
  *handle_ptr = e->handle();

  REXKRNL_DEBUG("XamUserCreateStatsEnumerator({:08X}, type={}, pivot={:016X}, rows={}, specs={}) "
                "-> empty leaderboard",
                title_id, enumerator_type, pivot_user, num_rows, num_stats_specs);
  return X_ERROR_SUCCESS;
}

u32 XamParseGamerTileKey_entry(mapped_u32 key_ptr, mapped_u32 out1_ptr, mapped_u32 out2_ptr,
                               mapped_u32 out3_ptr) {
  REXKRNL_DEBUG("XamParseGamerTileKey()");
  *out1_ptr = 0xC0DE0001;
  *out2_ptr = 0xC0DE0002;
  *out3_ptr = 0xC0DE0003;
  return X_ERROR_SUCCESS;
}

namespace {

// The user_gamerpic image, decoded once on first use into the layout profile
// tiles use: linear 32-bit big-endian A8R8G8B8, i.e. A,R,G,B byte order
// (matches xenia-canary's XamReadTileToTextureEx).
struct CustomGamerPic {
  std::vector<uint8_t> argb;
  uint32_t width = 0;
  uint32_t height = 0;
  bool valid() const { return !argb.empty(); }
};

const CustomGamerPic& GetCustomGamerPic() {
  static const CustomGamerPic pic = [] {
    CustomGamerPic result;
    const std::string configured = REXCVAR_GET(user_gamerpic);
    if (configured.empty()) {
      return result;
    }
    std::filesystem::path path = configured;
    if (path.is_relative()) {
      path = rex::filesystem::GetExecutableFolder() / path;
    }
    std::vector<uint8_t> file_data;
    if (FILE* file = rex::filesystem::OpenFile(path, "rb")) {
      fseek(file, 0, SEEK_END);
      file_data.resize(static_cast<size_t>(ftell(file)));
      fseek(file, 0, SEEK_SET);
      fread(file_data.data(), 1, file_data.size(), file);
      fclose(file);
    }
    if (file_data.empty()) {
      REXKRNL_WARN("[gamerpic] user_gamerpic not readable: {}", path.string());
      return result;
    }
    int width = 0, height = 0, channels = 0;
    stbi_uc* rgba = stbi_load_from_memory(file_data.data(), static_cast<int>(file_data.size()),
                                          &width, &height, &channels, STBI_rgb_alpha);
    if (!rgba) {
      REXKRNL_WARN("[gamerpic] user_gamerpic decode failed ({}): {}", stbi_failure_reason(),
                   path.string());
      return result;
    }
    result.width = static_cast<uint32_t>(width);
    result.height = static_cast<uint32_t>(height);
    const size_t pixel_count = size_t(width) * size_t(height);
    result.argb.resize(pixel_count * 4);
    for (size_t i = 0; i < pixel_count; ++i) {
      const stbi_uc* src = &rgba[i * 4];
      uint8_t* dst = &result.argb[i * 4];
      dst[0] = src[3];  // A
      dst[1] = src[0];  // R
      dst[2] = src[1];  // G
      dst[3] = src[2];  // B
    }
    stbi_image_free(rgba);
    return result;
  }();
  return pic;
}

}  // namespace

// XamReadTile / XamReadTileEx: image file bytes for a tile. The Avatar
// Editor's Awards page builds one tile per game that granted an award and asks
// here for the game's icon (AvatarEditor.xex sub_921FF0A0: a 0x4000-byte
// buffer, 0x10000 for tile type 15, then the image goes through the title's
// PNG/JPEG loader). Served from the closet's titles/<TITLEID>.png (any size:
// resampled and re-encoded when it would not fit the caller's buffer). No art
// gives X_ERROR_FILE_NOT_FOUND, which the editor answers with its default tile.
// Xbox tile types (xenia XTileType): 0 achievement, 1 game icon, 2/3 gamer
// tile, 4/5 local gamer tile, 6 background, 7/8 awarded gamer tile,
// 9 gamer tile by image id, 10/11 personal, 12 by key, 13/14 avatar gamer
// tile, 15 avatar full body.
// Per-thread preferred serve size (tile_icon.h): the title's hook at its XUI
// tile task sets it right before the guest calls XamReadTileEx.
static thread_local uint32_t t_tileSizeHint = 0;

void SetTileSizeHint(uint32_t px) { t_tileSizeHint = px; }

static X_RESULT ReadTileImage(uint32_t tile_type, uint32_t game_id, uint64_t item_id,
                              mapped_void buffer_ptr, mapped_u32 size_inout) {
  const uint32_t size_hint = t_tileSizeHint;
  t_tileSizeHint = 0;
  if (!buffer_ptr || !size_inout) {
    return X_ERROR_INVALID_PARAMETER;
  }
  std::vector<uint8_t> file;
  if (!game_id || !avatars::GetCloset().ReadTitleIcon(game_id, file)) {
    // No art anywhere (not even titles/_default.*): serve a fully transparent
    // 64x64 PNG instead of failing. The catalog popup's logo element only
    // repaints when an image loads, so a failure leaves whatever logo was
    // there before on screen; transparent means "no logo", never wrong art.
    static const std::vector<uint8_t> kBlank = [] {
      std::vector<uint8_t> rgba(64 * 64 * 4, 0);
      std::vector<uint8_t> png;
      avatars::EncodePng(rgba.data(), 64, 64, png);
      return png;
    }();
    if (kBlank.empty() || uint32_t(*size_inout) < kBlank.size()) {
      return X_ERROR_FILE_NOT_FOUND;
    }
    std::memcpy(buffer_ptr.as<uint8_t*>(), kBlank.data(), kBlank.size());
    *size_inout = uint32_t(kBlank.size());
    return X_ERROR_SUCCESS;
  }
  const uint32_t capacity = *size_inout;
  // Decode and fit: the editor draws the tile larger than 64 px, so serve the
  // largest square the caller's buffer can hold (128 -> 96 -> 64 px; the
  // original title icons were 64x64 and XUI upscales them), shrinking with an
  // area average so downscaled logos stay smooth. The stored file may be
  // png/jpg/bmp/gif, so always re-encode to PNG, the only format the editor's
  // tile loader is known to take; an image smaller than 64 px is re-encoded at
  // its own size (never upscaled).
  int w = 0, h = 0, ch = 0;
  {
    stbi_uc* rgba =
        stbi_load_from_memory(file.data(), static_cast<int>(file.size()), &w, &h, &ch, STBI_rgb_alpha);
    if (rgba) {
      // 64 px by default (the console title-icon size; the Awards page
      // highlight box draws the texture at native size, so 128 px would crop
      // there). A requester that scales its tile asks for 128 through
      // SetTileSizeHint and gets the largest of 128/96/64 whose PNG fits its
      // buffer.
      int sizes[3];
      int num_sizes = 0;
      if (size_hint >= 128) {
        sizes[num_sizes++] = 128;
        sizes[num_sizes++] = 96;
      }
      sizes[num_sizes++] = 64;
      std::vector<uint8_t> best;
      for (int si = 0; si < num_sizes; ++si) {
        const int target = sizes[si];
        if (w < target && h < target) {
          continue;  // never upscale
        }
        std::vector<uint8_t> small(size_t(target) * target * 4);
        for (int y = 0; y < target; ++y) {
          const int y0 = y * h / target, y1 = std::max(y0 + 1, (y + 1) * h / target);
          for (int x = 0; x < target; ++x) {
            const int x0 = x * w / target, x1 = std::max(x0 + 1, (x + 1) * w / target);
            uint64_t acc[4] = {0, 0, 0, 0};
            uint64_t n = 0;
            for (int sy = y0; sy < y1; ++sy) {
              const uint8_t* row = &rgba[(size_t(sy) * w + x0) * 4];
              for (int sx = x0; sx < x1; ++sx, row += 4) {
                // Premultiply so transparent fringe pixels do not bleed
                // their (often black) colour into the average.
                const uint64_t a = row[3];
                acc[0] += row[0] * a;
                acc[1] += row[1] * a;
                acc[2] += row[2] * a;
                acc[3] += a;
                ++n;
              }
            }
            uint8_t* out = &small[(size_t(y) * target + x) * 4];
            if (acc[3]) {
              out[0] = uint8_t(acc[0] / acc[3]);
              out[1] = uint8_t(acc[1] / acc[3]);
              out[2] = uint8_t(acc[2] / acc[3]);
            } else {
              out[0] = out[1] = out[2] = 0;
            }
            out[3] = uint8_t(acc[3] / (n ? n : 1));
          }
        }
        std::vector<uint8_t> png;
        if (avatars::EncodePng(small.data(), uint32_t(target), uint32_t(target), png) &&
            png.size() <= capacity) {
          best.swap(png);
          break;
        }
      }
      if (best.empty() && w > 0 && h > 0 && w <= 128 && h <= 128) {
        // Smaller than the ladder (or nothing fit): native size, PNG.
        std::vector<uint8_t> png;
        if (avatars::EncodePng(rgba, uint32_t(w), uint32_t(h), png) && png.size() <= capacity) {
          best.swap(png);
        }
      }
      if (!best.empty()) {
        file.swap(best);
      }
      stbi_image_free(rgba);
    }
  }
  if (file.size() > capacity) {
    REXKRNL_WARN("[tile] title {:08X} icon {:#x} bytes > capacity {:#x}", game_id, file.size(),
                 capacity);
    *size_inout = static_cast<uint32_t>(file.size());
    return X_ERROR_INSUFFICIENT_BUFFER;
  }
  std::memcpy(buffer_ptr.as<uint8_t*>(), file.data(), file.size());
  *size_inout = static_cast<uint32_t>(file.size());
  return X_ERROR_SUCCESS;
}

u32 XamReadTile_entry(u32 tile_type, u32 game_id, u64 item_id, u32 offset, mapped_void buffer_ptr,
                      mapped_u32 size_inout, mapped_void overlapped_ptr) {
  const X_RESULT result = ReadTileImage(tile_type, game_id, item_id, buffer_ptr, size_inout);
  if (overlapped_ptr) {
    REX_KERNEL_STATE()->CompleteOverlappedImmediate(overlapped_ptr.guest_address(), result);
    return X_ERROR_IO_PENDING;
  }
  return result;
}

u32 XamReadTileEx_entry(u32 tile_type, u32 game_id, u64 item_id, u32 offset, u32 unk1, u32 unk2,
                        mapped_void buffer_ptr, mapped_u32 size_inout) {
  return ReadTileImage(tile_type, game_id, item_id, buffer_ptr, size_inout);
}

u32 XamReadTileToTexture_entry(u32 unknown, u32 title_id, u64 tile_id, u32 user_index,
                               mapped_void buffer_ptr, u32 stride, u32 height, u32 overlapped_ptr) {
  REXKRNL_DEBUG("XamReadTileToTexture(type={}, title={:08X}, tile_id={:016X}, user={}, stride={}, "
                "height={}, overlapped={:08X})",
                unknown, title_id, tile_id, user_index, stride, height, overlapped_ptr);
  // TODO(gibbed): unknown=0,2,3,9
  const size_t size = size_t(stride) * size_t(height);

  const auto& pic = GetCustomGamerPic();
  if (pic.valid() && buffer_ptr && stride >= 4 && height) {
    // Serve the configured picture as the tile, including for tile_id == 0,
    // which titles pass for the local user when offline (a title falls back to
    // its own built-in image when this call fails).
    // Games size the buffer as stride x height with width == stride / 4
    // (64x64 big tile, 32x32 small); resample nearest-neighbor to fit.
    auto* dst = buffer_ptr.as<uint8_t*>();
    std::memset(dst, 0, size);
    const uint32_t dst_w = std::min<uint32_t>(stride / 4, 256);
    const uint32_t dst_h = std::min<uint32_t>(height, 256);
    for (uint32_t y = 0; y < dst_h; ++y) {
      uint8_t* dst_row = dst + size_t(y) * stride;
      const size_t src_y = size_t(y) * pic.height / dst_h;
      for (uint32_t x = 0; x < dst_w; ++x) {
        const size_t src_x = size_t(x) * pic.width / dst_w;
        std::memcpy(&dst_row[size_t(x) * 4], &pic.argb[(src_y * pic.width + src_x) * 4], 4);
      }
    }
  } else {
    if (!tile_id) {
      return X_ERROR_INVALID_PARAMETER;
    }
    std::memset(buffer_ptr, 0xFF, size);
  }

  if (overlapped_ptr) {
    REX_KERNEL_STATE()->CompleteOverlappedImmediate(overlapped_ptr, X_ERROR_SUCCESS);
    return X_ERROR_IO_PENDING;
  }
  return X_ERROR_SUCCESS;
}

// ---------------------------------------------------------------------------
// XamGetLiveHiveValueA(name, buffer, size, flags, overlapped): live service
// configuration ("hive") values. The Avatar Editor reads two feature flags
// through 52-byte async readers (sub_920D1CB8: value into a 16-byte buffer,
// ready when the overlapped is no longer pending with extended error 0):
// "AvatarPhotoBoothEnabled" gates the Gamer Pic camera (sub_920FDCE0 compares
// the value with "1"; anything else = "feature isn't currently available")
// and "AvatarMarketplaceEnabled" gates the live marketplace. It also asks for
// "AvatarAssetUriRoot" (download root) and "AvatarAssetRefreshFrequency".
// Offline: the photo booth is on (its ops are served by XamPngEncodeEx /
// XamWriteGamerTileEx below); every other key reports X_ERROR_NOT_FOUND, a
// positive error callers read as "no value, skip", so no marketplace and no
// asset downloads.
// ---------------------------------------------------------------------------
u32 XamGetLiveHiveValueA_entry(mapped_string name, mapped_string buffer, u32 buffer_size, u32 flags,
                               mapped_void overlapped_ptr) {
  const std::string key = name ? std::string(name.value()) : std::string();
  const char* value = nullptr;
  if (key == "AvatarPhotoBoothEnabled") {
    value = "1";
  }
  u32 result = X_ERROR_NOT_FOUND;
  if (value && buffer && buffer_size) {
    const size_t len = std::strlen(value);
    if (len + 1 <= buffer_size) {
      std::memcpy(buffer.as<char*>(), value, len + 1);
      result = X_ERROR_SUCCESS;
    } else {
      result = X_ERROR_INSUFFICIENT_BUFFER;
    }
  }
  if (overlapped_ptr) {
    REX_KERNEL_STATE()->CompleteOverlappedImmediate(overlapped_ptr.guest_address(), result);
    return X_ERROR_IO_PENDING;
  }
  return result;
}

// ---------------------------------------------------------------------------
// XamPngDecode(png, png_size, pixels, pixels_size, 0, 0, 0): the Avatar
// Editor's PNG-to-pixels path for the Gamer Pic backdrops. sub_920B3AE0 reads
// gamerpic_background_%d.png / _s_%d.png from the media package into a 48 KB
// stack buffer, decodes into the caller's w*h*4 buffer, then rotates every
// big-endian dword left by 8: the decoder's layout is ARGB per BE dword =
// memory bytes A,R,G,B, which the rotate turns into R,G,B,A for the
// compositor (sub_920B4488, alpha last).
// ---------------------------------------------------------------------------
u32 XamPngDecode_entry(mapped_void png_ptr, u32 png_size, mapped_void pixels_ptr, u32 pixels_size,
                       u32 unk5, u32 unk6, u32 unk7) {
  if (!png_ptr || !pixels_ptr || png_size < 8) {
    return 0x80070057u;  // E_INVALIDARG
  }
  int w = 0, h = 0, ch = 0;
  stbi_uc* rgba = stbi_load_from_memory(png_ptr.as<const uint8_t*>(), static_cast<int>(png_size),
                                        &w, &h, &ch, STBI_rgb_alpha);
  if (!rgba) {
    REXKRNL_WARN("[pngdecode] {} bytes: not a decodable image ({})", png_size,
                 stbi_failure_reason() ? stbi_failure_reason() : "?");
    return 0x80004005u;  // E_FAIL
  }
  const uint64_t need = uint64_t(w) * uint64_t(h) * 4u;
  u32 result = 0;
  if (need > pixels_size) {
    REXKRNL_WARN("[pngdecode] {}x{} needs {:#x} bytes, caller gave {:#x}", w, h, need,
                 pixels_size);
    result = 0x8007007Au;  // HRESULT_FROM_WIN32(ERROR_INSUFFICIENT_BUFFER)
  } else {
    uint8_t* out = pixels_ptr.as<uint8_t*>();
    const uint8_t* src = rgba;
    for (uint64_t i = 0; i < uint64_t(w) * h; ++i, src += 4, out += 4) {
      out[0] = src[3];  // A
      out[1] = src[0];  // R
      out[2] = src[1];  // G
      out[3] = src[2];  // B
    }
  }
  stbi_image_free(rgba);
  return result;
}

// ---------------------------------------------------------------------------
// Avatar Editor gamer picture. The editor's "Gamer Pic" camera
// composites the avatar render over the chosen background into two RGBA
// buffers (64x64 = 0x4000 and 32x32 = 0x1000, the console's large/small
// gamer picture; alpha forced opaque, sub_920B4488), PNG-encodes each with
// XamPngEncodeEx (XUI task sub_921FF600: buf, w, h, stride 4w, out, &size,
// 1, ovl; HRESULT, pending = 0x800703E5) into its own 0x4000/0x1000 output
// buffer, then writes both through XamWriteGamerTileEx (sub_921FF2E0:
// user, 18|50, 0xFFFE0854, 0, 0, 0, png64, size64, png32, size32, ovl), which
// is a live upload on the console. Offline the pictures are kept beside the
// persisted avatar manifest as gamerpic.png / gamerpic_small.png.
// ---------------------------------------------------------------------------
static constexpr u32 kHresultIoPending = 0x800703E5u;  // HRESULT_FROM_WIN32(ERROR_IO_PENDING)

static bool SaveAvatarSideFile(const char* name, const uint8_t* data, size_t size) {
  if (!data || !size) return false;
  const std::string path = AvatarDataDir() + "\\" + name;
  FILE* f = std::fopen(path.c_str(), "wb");
  if (!f) {
    REXKRNL_WARN("[gamerpic] cannot write {}", path);
    return false;
  }
  std::fwrite(data, 1, size, f);
  std::fclose(f);
  return true;
}

static bool LooksLikePng(const uint8_t* p, u32 size) {
  return p && size > 8 && p[0] == 0x89 && p[1] == 'P' && p[2] == 'N' && p[3] == 'G';
}

u32 XamPngEncodeEx_entry(mapped_void pixels_ptr, u32 width, u32 height, u32 stride,
                         mapped_void out_ptr, mapped_u32 size_inout, u32 flags,
                         mapped_void overlapped_ptr) {
  const u32 capacity = size_inout ? u32(*size_inout) : 0u;
  u32 result = X_ERROR_INVALID_PARAMETER;
  std::vector<uint8_t> png;
  if (pixels_ptr && out_ptr && size_inout && width && height && width <= 4096 && height <= 4096 &&
      stride >= width * 4) {
    const uint8_t* src = pixels_ptr.as<const uint8_t*>();
    if (avatars::EncodePngRgb(src, width, height, stride, png)) {
      if (png.size() <= capacity) {
        std::memcpy(out_ptr.as<uint8_t*>(), png.data(), png.size());
        *size_inout = static_cast<u32>(png.size());
        result = X_ERROR_SUCCESS;
      } else {
        REXKRNL_WARN("[gamerpic] {}x{} PNG {:#x} bytes > capacity {:#x}", width, height,
                     png.size(), capacity);
        *size_inout = static_cast<u32>(png.size());
        result = X_ERROR_INSUFFICIENT_BUFFER;
      }
    } else {
      result = X_ERROR_FUNCTION_FAILED;
    }
  }
  if (overlapped_ptr) {
    REX_KERNEL_STATE()->CompleteOverlappedImmediate(overlapped_ptr.guest_address(), result);
    return kHresultIoPending;
  }
  return result == X_ERROR_SUCCESS ? 0u : (0x80070000u | (result & 0xFFFFu));
}

u32 XamWriteGamerTileEx_entry(u32 user_index, u32 kind, u32 title_id, u32 unk4, u32 unk5,
                              u32 unk6, mapped_void large_ptr, u32 large_size,
                              mapped_void small_ptr, u32 small_size, mapped_void overlapped_ptr) {
  u32 result = X_ERROR_INVALID_PARAMETER;
  // Two PNGs (in whichever order the title packed them): the file name follows
  // the IHDR width, 64 px and up = gamerpic.png, smaller = gamerpic_small.png.
  const uint8_t* bufs[2] = {large_ptr ? large_ptr.as<const uint8_t*>() : nullptr,
                            small_ptr ? small_ptr.as<const uint8_t*>() : nullptr};
  const u32 sizes[2] = {large_size, small_size};
  int saved = 0;
  for (int i = 0; i < 2; i++) {
    if (!LooksLikePng(bufs[i], sizes[i]) || sizes[i] < 24) continue;
    const u32 w = (u32(bufs[i][16]) << 24) | (u32(bufs[i][17]) << 16) |
                  (u32(bufs[i][18]) << 8) | u32(bufs[i][19]);
    const char* name = w >= 64 ? "gamerpic.png" : "gamerpic_small.png";
    if (SaveAvatarSideFile(name, bufs[i], sizes[i])) saved++;
  }
  if (saved) {
    result = X_ERROR_SUCCESS;
  } else {
    REXKRNL_WARN("[gamerpic] XamWriteGamerTileEx: no PNG payload, nothing saved");
  }
  if (overlapped_ptr) {
    REX_KERNEL_STATE()->CompleteOverlappedImmediate(overlapped_ptr.guest_address(), result);
    return X_ERROR_IO_PENDING;
  }
  return result;
}

// XamWriteTile(15 = avatar full-body tile, title 0xFFFE0851, 0, user, data,
// size, ovl): sub_921FF2E0's type-15 branch over sub_921FF830's single payload
// (op+100/+104). The write is acknowledged but nothing is persisted; a
// non-PNG payload is logged.
u32 XamWriteTile_entry(u32 tile_type, u32 title_id, u32 unk3, u32 user_index, mapped_void data_ptr,
                       u32 data_size, mapped_void overlapped_ptr) {
  u32 result = X_ERROR_SUCCESS;
  const uint8_t* data = data_ptr ? data_ptr.as<const uint8_t*>() : nullptr;
  if (tile_type == 15 && LooksLikePng(data, data_size)) {
    // The avatar-body picture is deliberately not written to disk.
  } else if (data && data_size) {
    REXKRNL_WARN("[gamerpic] XamWriteTile type {} payload not a PNG ({:#x} bytes), acknowledged",
                 tile_type, data_size);
  }
  if (overlapped_ptr) {
    REX_KERNEL_STATE()->CompleteOverlappedImmediate(overlapped_ptr.guest_address(), result);
    return X_ERROR_IO_PENDING;
  }
  return result;
}

u32 XamWriteGamerTile_entry(u32 arg1, u32 arg2, u32 arg3, u32 arg4, u32 arg5, u32 overlapped_ptr) {
  REXKRNL_DEBUG("XamWriteGamerTile({:08X}, {:08X}, {:08X}, {:08X}, {:08X}, overlapped={:08X})",
                arg1, arg2, arg3, arg4, arg5, overlapped_ptr);
  if (overlapped_ptr) {
    REX_KERNEL_STATE()->CompleteOverlappedImmediate(overlapped_ptr, X_ERROR_SUCCESS);
    return X_ERROR_IO_PENDING;
  }
  return X_ERROR_SUCCESS;
}

u32 XamSessionCreateHandle_entry(mapped_u32 handle_ptr) {
  *handle_ptr = 0xCAFEDEAD;
  return X_ERROR_SUCCESS;
}

u32 XamSessionRefObjByHandle_entry(u32 handle, mapped_u32 obj_ptr) {
  assert_true(handle == 0xCAFEDEAD);
  // TODO(PermaNull): Implement this properly,
  // For the time being returning 0xDEADF00D will prevent crashing.
  *obj_ptr = 0xDEADF00D;
  return X_ERROR_SUCCESS;
}

// XamUserProfileSync(user_mask): flush/sync profile data. There is no cloud to
// sync with, and local persistence already happened by the time titles call
// this (the Avatar Editor saves the manifest through XamAvatarSetManifest
// first), so report success and let save flows complete.
u32 XamUserProfileSync_entry(u32 user_mask) {
  REXKRNL_DEBUG("XamUserProfileSync(mask={:#x}) -> success (offline no-op)", user_mask);
  return 0;
}

// XamUserValidateAvatarManifest(user, X_AVATAR_METADATA*, overlapped): on the
// console this validates the manifest against live (the
// validateavatarmanifest service). Offline, locally authored manifests are
// valid by definition, so complete with success and let the Avatar Editor's
// save operation advance past its validate stage.
u32 XamUserValidateAvatarManifest_entry(u32 user_index, mapped_void manifest_ptr,
                                        mapped_void overlapped_ptr) {
  REXKRNL_DEBUG("XamUserValidateAvatarManifest(user={}) -> success (offline)", user_index);
  if (overlapped_ptr) {
    REX_KERNEL_STATE()->CompleteOverlappedImmediate(overlapped_ptr.guest_address(),
                                                    X_ERROR_SUCCESS);
    return X_ERROR_IO_PENDING;
  }
  return 0;
}

}  // namespace xam
}  // namespace kernel
}  // namespace rex

REX_EXPORT(__imp__XamUserGetXUID, rex::kernel::xam::XamUserGetXUID_entry)
REX_EXPORT(__imp__XamUserGetSigninState, rex::kernel::xam::XamUserGetSigninState_entry)
REX_EXPORT(__imp__XamUserGetSigninInfo, rex::kernel::xam::XamUserGetSigninInfo_entry)
REX_EXPORT(__imp__XamUserGetName, rex::kernel::xam::XamUserGetName_entry)
REX_EXPORT(__imp__XamUserGetGamerTag, rex::kernel::xam::XamUserGetGamerTag_entry)
REX_EXPORT(__imp__XamUserReadProfileSettings, rex::kernel::xam::XamUserReadProfileSettings_entry)
REX_EXPORT(__imp__XamUserReadProfileSettingsEx,
           rex::kernel::xam::XamUserReadProfileSettingsEx_entry)
REX_EXPORT(__imp__XamUserWriteProfileSettings, rex::kernel::xam::XamUserWriteProfileSettings_entry)
REX_EXPORT(__imp__XamUserCheckPrivilege, rex::kernel::xam::XamUserCheckPrivilege_entry)
REX_EXPORT(__imp__XamUserContentRestrictionGetFlags,
           rex::kernel::xam::XamUserContentRestrictionGetFlags_entry)
REX_EXPORT(__imp__XamUserContentRestrictionGetRating,
           rex::kernel::xam::XamUserContentRestrictionGetRating_entry)
REX_EXPORT(__imp__XamUserContentRestrictionCheckAccess,
           rex::kernel::xam::XamUserContentRestrictionCheckAccess_entry)
REX_EXPORT(__imp__XamUserIsOnlineEnabled, rex::kernel::xam::XamUserIsOnlineEnabled_entry)
REX_EXPORT(__imp__XamUserGetMembershipTier, rex::kernel::xam::XamUserGetMembershipTier_entry)
REX_EXPORT(__imp__XamUserAreUsersFriends, rex::kernel::xam::XamUserAreUsersFriends_entry)
REX_EXPORT(__imp__XamShowSigninUI, rex::kernel::xam::XamShowSigninUI_entry)
REX_EXPORT(__imp__XamUserCreateAchievementEnumerator,
           rex::kernel::xam::XamUserCreateAchievementEnumerator_entry)
REX_EXPORT(__imp__XamUserCreateStatsEnumerator,
           rex::kernel::xam::XamUserCreateStatsEnumerator_entry)
REX_EXPORT(__imp__XamUserCreateAvatarAssetEnumerator,
           rex::kernel::xam::XamUserCreateAvatarAssetEnumerator_entry)
REX_EXPORT(__imp__XamParseGamerTileKey, rex::kernel::xam::XamParseGamerTileKey_entry)
REX_EXPORT(__imp__XamReadTileToTexture, rex::kernel::xam::XamReadTileToTexture_entry)
REX_EXPORT(__imp__XamReadTile, rex::kernel::xam::XamReadTile_entry)
REX_EXPORT(__imp__XamReadTileEx, rex::kernel::xam::XamReadTileEx_entry)
REX_EXPORT(__imp__XamWriteGamerTile, rex::kernel::xam::XamWriteGamerTile_entry)
REX_EXPORT(__imp__XamWriteGamerTileEx, rex::kernel::xam::XamWriteGamerTileEx_entry)
REX_EXPORT(__imp__XamWriteTile, rex::kernel::xam::XamWriteTile_entry)
REX_EXPORT(__imp__XamPngEncodeEx, rex::kernel::xam::XamPngEncodeEx_entry)
REX_EXPORT(__imp__XamGetLiveHiveValueA, rex::kernel::xam::XamGetLiveHiveValueA_entry)
REX_EXPORT(__imp__XamPngDecode, rex::kernel::xam::XamPngDecode_entry)
REX_EXPORT(__imp__XamSessionCreateHandle, rex::kernel::xam::XamSessionCreateHandle_entry)
REX_EXPORT(__imp__XamSessionRefObjByHandle, rex::kernel::xam::XamSessionRefObjByHandle_entry)

REX_EXPORT_STUB(__imp__XamUserAddRecentPlayer);
REX_EXPORT_STUB(__imp__XamUserAllowedToPostToSocialNetwork);
REX_EXPORT_STUB(__imp__XamUserCreatePlayerEnumerator);
REX_EXPORT_STUB(__imp__XamUserCreateTitlesPlayedEnumerator);
REX_EXPORT_STUB(__imp__XamUserFlushLogonQueue);
REX_EXPORT_STUB(__imp__XamUserGetAge);
REX_EXPORT_STUB(__imp__XamUserGetAgeGroup);
REX_EXPORT_STUB(__imp__XamUserGetCachedUserFlags);
REX_EXPORT_STUB(__imp__XamUserGetDeviceId);
REX_EXPORT_STUB(__imp__XamUserGetIndexFromXUID);
REX_EXPORT_STUB(__imp__XamUserGetMembershipTierFromXUID);
REX_EXPORT_STUB(__imp__XamUserGetOnlineCountryFromXUID);
REX_EXPORT_STUB(__imp__XamUserGetOnlineLanguageFromXUID);
REX_EXPORT_STUB(__imp__XamUserGetOnlineXUIDFromOfflineXUID);
REX_EXPORT_STUB(__imp__XamUserGetReportingInfo);
REX_EXPORT_STUB(__imp__XamUserGetRequestedUserIndexMask);
REX_EXPORT_STUB(__imp__XamUserGetSubscriptionType);
REX_EXPORT_STUB(__imp__XamUserGetUserFlags);
REX_EXPORT_STUB(__imp__XamUserGetUserFlagsFromXUID);
REX_EXPORT_STUB(__imp__XamUserGetUserIndexMask);
REX_EXPORT_STUB(__imp__XamUserGetUserTenure);
REX_EXPORT_STUB(__imp__XamUserGetUsersMissingAvatars);
REX_EXPORT_STUB(__imp__XamUserGetXUIDForTFA);
REX_EXPORT_STUB(__imp__XamUserInvalidateProfileSetting);
REX_EXPORT_STUB(__imp__XamUserIsGuest);
REX_EXPORT_STUB(__imp__XamUserIsLogonPreviewModeEnabled);
REX_EXPORT_STUB(__imp__XamUserIsParentalControlled);
REX_EXPORT_STUB(__imp__XamUserIsPartial);
REX_EXPORT_STUB(__imp__XamUserIsPartialProfile);
REX_EXPORT_STUB(__imp__XamUserIsUnsafeProgrammingAllowed);
REX_EXPORT_STUB(__imp__XamUserLockLogonPreviewMode);
REX_EXPORT_STUB(__imp__XamUserLogon);
REX_EXPORT_STUB(__imp__XamUserLogonEx);
REX_EXPORT_STUB(__imp__XamUserLookupDevice);
REX_EXPORT_STUB(__imp__XamUserNuiBind);
REX_EXPORT_STUB(__imp__XamUserNuiEnableBiometric);
REX_EXPORT_STUB(__imp__XamUserNuiGetEnrollmentIndex);
REX_EXPORT_STUB(__imp__XamUserNuiGetUserIndex);
REX_EXPORT_STUB(__imp__XamUserNuiGetUserIndexForBind);
REX_EXPORT_STUB(__imp__XamUserNuiGetUserIndexForSignin);
REX_EXPORT_STUB(__imp__XamUserNuiIsBiometricEnabled);
REX_EXPORT_STUB(__imp__XamUserNuiUnbind);
REX_EXPORT_STUB(__imp__XamUserOverrideBindingCallbacks);
REX_EXPORT_STUB(__imp__XamUserOverrideDeviceBindings);
REX_EXPORT_STUB(__imp__XamUserOverrideGlobalState);
REX_EXPORT_STUB(__imp__XamUserOverrideUserInfo);
REX_EXPORT_STUB(__imp__XamUserPrefetchProfileSettings);
REX_EXPORT(__imp__XamUserProfileSync, rex::kernel::xam::XamUserProfileSync_entry)
REX_EXPORT_STUB(__imp__XamUserReadUserPreference);
REX_EXPORT_STUB(__imp__XamUserResetSubscriptionType);
REX_EXPORT_STUB(__imp__XamUserUnlockLogonPreviewMode);
REX_EXPORT_STUB(__imp__XamUserUpdateRecentPlayer);
REX_EXPORT(__imp__XamUserValidateAvatarManifest,
           rex::kernel::xam::XamUserValidateAvatarManifest_entry)
REX_EXPORT_STUB(__imp__XamUserWriteUserPreference);
REX_EXPORT_STUB(__imp__XamVerifyPasscode);
