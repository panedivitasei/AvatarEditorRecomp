/**
 ******************************************************************************
 * Xenia : Xbox 360 Emulator Research Project                                 *
 ******************************************************************************
 * Copyright 2021 Ben Vanik. All rights reserved.                             *
 * Released under the BSD license - see LICENSE in the root for more details. *
 ******************************************************************************
 *
 * @modified    Tom Clay, 2026 - Adapted for ReXGlue runtime
 */

#include <cstring>
#include <filesystem>
#include <fstream>

#include <rex/filesystem.h>
#include <rex/kernel/xam/apps/xlivebase_app.h>
// SDK apps/modules re-registered by InitializeTitleKernel (below).
#include <rex/kernel/xam/apps/app.h>
#include <rex/kernel/xam/apps/xgi_app.h>
#include <rex/kernel/xam/apps/xmp_app.h>
#include <rex/kernel/xam/module.h>
#include <rex/kernel/xbdm/module.h>
#include <rex/kernel/xboxkrnl/module.h>
#include <rex/kernel/xam/xam_avatar.h>
#include <rex/kernel/xnet.h>
#include <rex/logging.h>
#include <rex/runtime.h>
#include <rex/string.h>
#include <rex/system/thread_state.h>
#include <rex/system/xenumerator.h>
#include <rex/system/xobject.h>
#include <rex/thread.h>

namespace rex {
namespace kernel {
namespace xam {
using namespace rex::system;
using namespace rex::system::xam;
namespace apps {
using namespace rex::system;

XLiveBaseApp::XLiveBaseApp(KernelState* kernel_state) : App(kernel_state, 0xFC) {}

// http://mb.mirage.org/bugzilla/xliveless/main.c

// XOnline async RPC, message 0x00050000 | schema index. The Avatar Editor's
// gamer-pic / save op issues schema 1814 (sub_921FC4D0, index 0x108: the
// "avatar content query", 5 marshalled args: user, 0, 1000, a GUID block,
// xuid) and branches on the single result byte the service writes back
// (CXLiveAsyncTask results_ptr = the op's flag at +64, results_size 1):
// nonzero continues into the gamer-pic encode / avatar-body build, zero or a
// failed overlapped resets the op (E_FAIL). Offline, answer 1.
static X_HRESULT XOnlineAvatarRpc(rex::memory::Memory* memory, uint32_t message,
                                  uint32_t buffer_ptr) {
  if (!buffer_ptr) return X_E_INVALIDARG;
  auto* msg = memory->TranslateVirtual<XLIVEBASE_ASYNC_MESSAGE*>(buffer_ptr);
  if (!msg || !msg->xlive_async_task_ptr) return X_E_INVALIDARG;
  auto* task = memory->TranslateVirtual<XLIVE_ASYNC_TASK*>(
      static_cast<uint32_t>(msg->xlive_async_task_ptr));
  if (!task) return X_E_INVALIDARG;
  const uint32_t ordinal = task->ordinal;
  const uint32_t results_ptr = task->results_ptr;
  const uint32_t results_size = task->results_size;
  if (ordinal == 1814) {
    if (results_ptr && results_size >= 1) {
      auto* out = memory->TranslateVirtual<uint8_t*>(results_ptr);
      std::memset(out, 0, results_size);
      out[0] = 1;  // content query: allowed
    }
    return X_E_SUCCESS;
  }
  REXKRNL_ERROR("[xonline] unimplemented XOnline RPC ordinal {} (msg {:08X})", ordinal, message);
  return X_E_FAIL;
}

X_HRESULT XLiveBaseApp::DispatchMessageSync(uint32_t message, uint32_t buffer_ptr,
                                            uint32_t buffer_length) {
  // NOTE: buffer_length may be zero or valid.
  auto buffer = memory_->TranslateVirtual(buffer_ptr);
  switch (message) {
    case 0x00050009: {
      REXKRNL_DEBUG("XStorageDownloadToMemory({:08X}, {:08X}) guest_lr={:08X}", buffer_ptr,
                    buffer_length,
                    rex::runtime::ThreadState::Get()
                        ? uint32_t(rex::runtime::current_ppc_context()->lr)
                        : 0);
      return XStorageDownloadToMemory(buffer_ptr);
    }
    case 0x0005000B: {
      REXKRNL_DEBUG("XStorageUploadFromMemory({:08X}, {:08X}) guest_lr={:08X}", buffer_ptr,
                    buffer_length,
                    rex::runtime::ThreadState::Get()
                        ? uint32_t(rex::runtime::current_ppc_context()->lr)
                        : 0);
      return XStorageUploadFromMemory(buffer_ptr);
    }
    case 0x00058003: {
      // XLiveBaseLogonGetHR: report the live logon as connected/established so
      // titles stop waiting for the online link to come up.
      REXKRNL_DEBUG("XLiveBaseLogonGetHR({:08X}, {:08X})", buffer_ptr, buffer_length);
      return X_ONLINE_S_LOGON_CONNECTION_ESTABLISHED;
    }
    case 0x00058004: {
      // Called on startup, seems to just return a bool in the buffer.
      assert_true(!buffer_length || buffer_length == 4);
      REXKRNL_DEBUG("XLiveBaseGetLogonId({:08X})", buffer_ptr);
      memory::store_and_swap<uint32_t>(buffer + 0, 1);  // ?
      return X_E_SUCCESS;
    }
    case 0x00058006: {
      assert_true(!buffer_length || buffer_length == 4);
      REXKRNL_DEBUG("XLiveBaseGetNatType({:08X})", buffer_ptr);
      memory::store_and_swap<uint32_t>(buffer + 0, 1);  // XONLINE_NAT_OPEN
      return X_E_SUCCESS;
    }
    case 0x00058007: {
      // Occurs if title calls XOnlineGetServiceInfo, expects dwServiceId
      // and pServiceInfo. pServiceInfo should contain pointer to
      // XONLINE_SERVICE_INFO structure.
      REXKRNL_DEBUG("CXLiveLogon::GetServiceInfo({:08X}, {:08X})", buffer_ptr, buffer_length);
      return 0x80151802;  // ERROR_CONNECTION_INVALID
    }
    case 0x00058020: {
      // CXLiveFriends::Enumerate: marshalled friends-enumerator creation. The
      // title spins on its friends/presence init until it gets a valid
      // enumerator handle it can pump, so build a real (empty) enumerator and
      // write the handle + buffer size back, mirroring xenia-canary's
      // XFriendsCreateEnumerator. The marshalled X_ARGUMENT_LIST is at the
      // second arg (buffer_length), not buffer_ptr.
      REXKRNL_DEBUG("CXLiveFriends::Enumerate({:08X}, {:08X})", buffer_ptr, buffer_length);
      return XFriendsCreateEnumerator(buffer_length);
    }
    case 0x00058023: {
      REXKRNL_DEBUG(
          "CXLiveMessaging::XMessageGameInviteGetAcceptedInfo({:08X}, {:08X}) "
          "unimplemented",
          buffer_ptr, buffer_length);
      return X_E_FAIL;
    }
    case 0x00058035: {
      // Builds the Title Storage server path for a facility (per-title,
      // per-user-title, game clip). Titles poll this every frame during their
      // storage init until it succeeds.
      REXKRNL_DEBUG("XStorageBuildServerPath({:08X}, {:08X}) guest_lr={:08X}", buffer_ptr,
                    buffer_length,
                    rex::runtime::ThreadState::Get()
                        ? uint32_t(rex::runtime::current_ppc_context()->lr)
                        : 0);
      return XStorageBuildServerPath(buffer_ptr);
    }
    case 0x00058046: {
      // Required to be successful for 4D530910 to detect signed-in profile
      // Doesn't seem to set anything in the given buffer, probably only takes
      // input
      REXKRNL_DEBUG("XLiveBaseUnk58046({:08X}, {:08X}) unimplemented", buffer_ptr, buffer_length);
      return X_E_SUCCESS;
    }
  }
  if (message == 0x00050108) {
    return XOnlineAvatarRpc(memory_, message, buffer_ptr);
  }
  REXKRNL_ERROR(
      "Unimplemented XLIVEBASE message app={:08X}, msg={:08X}, arg1={:08X}, "
      "arg2={:08X}",
      app_id(), message, buffer_ptr, buffer_length);
  return X_E_FAIL;
}

X_HRESULT XLiveBaseApp::XFriendsCreateEnumerator(uint32_t buffer_length) {
  if (!buffer_length) {
    return X_E_INVALIDARG;
  }

  auto* fe = memory_->TranslateVirtual<X_CREATE_FRIENDS_ENUMERATOR*>(buffer_length);

  const uint32_t user_index = memory::load_and_swap<uint32_t>(
      memory_->TranslateVirtual(
          static_cast<uint32_t>(fe->user_index.argument_value_ptr)));
  const uint32_t friends_starting_index = memory::load_and_swap<uint32_t>(
      memory_->TranslateVirtual(
          static_cast<uint32_t>(fe->friends_starting_index.argument_value_ptr)));
  const uint32_t friends_amount = memory::load_and_swap<uint32_t>(
      memory_->TranslateVirtual(
          static_cast<uint32_t>(fe->friends_amount.argument_value_ptr)));

  const uint32_t buffer_address =
      static_cast<uint32_t>(fe->buffer_ptr.argument_value_ptr);
  const uint32_t handle_address =
      static_cast<uint32_t>(fe->handle_ptr.argument_value_ptr);

  if (!handle_address) {
    return X_E_INVALIDARG;
  }
  auto* handle_ptr = memory_->TranslateVirtual<uint32_t*>(handle_address);
  *handle_ptr = 0;  // titles expect 0 (not -1) on failure; set as early as possible

  if (!buffer_address) {
    return X_E_INVALIDARG;
  }
  auto* buffer_size_ptr = memory_->TranslateVirtual<uint32_t*>(buffer_address);
  *buffer_size_ptr = 0;

  if (user_index >= 4) {
    return X_E_INVALIDARG;
  }
  if (friends_starting_index >= X_ONLINE_MAX_FRIENDS) {
    return X_E_INVALIDARG;
  }
  if (friends_amount > X_ONLINE_MAX_FRIENDS) {
    return X_E_INVALIDARG;
  }

  // Empty friends list: a valid handle with zero items. The title pumps it via
  // XamEnumerate, gets "no more items", and its presence/friends init settles.
  auto e = object_ref<XStaticUntypedEnumerator>(new XStaticUntypedEnumerator(
      kernel_state_, friends_amount, sizeof(X_ONLINE_FRIEND)));
  auto result = e->Initialize(user_index, app_id(), 0x58021, 0x58022, 0);
  if (XFAILED(result)) {
    return result;
  }

  const uint32_t friends_buffer_size =
      static_cast<uint32_t>(e->items_per_enumerate() * e->item_size());

  *buffer_size_ptr = rex::byte_swap<uint32_t>(friends_buffer_size);
  *handle_ptr = rex::byte_swap<uint32_t>(e->handle());
  return X_E_SUCCESS;
}

X_HRESULT XLiveBaseApp::XStorageBuildServerPath(uint32_t buffer_ptr) {
  // Ported from xenia-canary, minus the backend/VFS mirroring: the title only
  // needs a syntactically valid server path for its storage init to complete.
  // Later XStorage* / XHTTP traffic against the path is served by the
  // in-process interceptors.
  if (!buffer_ptr) {
    return X_E_INVALIDARG;
  }

  auto* args = memory_->TranslateVirtual<X_STORAGE_BUILD_SERVER_PATH*>(buffer_ptr);

  if (!args->file_name_ptr || !args->server_path_length_ptr) {
    return X_E_INVALIDARG;
  }

  const uint32_t user_index = args->user_index;
  if (user_index >= 4 && user_index != 0xFF /* XUserIndexNone */) {
    return X_E_INVALIDARG;
  }

  uint64_t xuid = user_index == 0xFF ? static_cast<uint64_t>(args->xuid) : 0;

  const bool xuid_required =
      args->storage_location == X_STORAGE_FACILITY::FACILITY_PER_USER_TITLE ||
      args->storage_location == X_STORAGE_FACILITY::FACILITY_GAME_CLIP;
  if (!xuid && xuid_required) {
    const auto* profile = kernel_state_->user_profile();
    if (profile) {
      xuid = profile->xuid();
    }
  }

  const std::u16string filename = rex::memory::load_and_swap<std::u16string>(
      memory_->TranslateVirtual(static_cast<uint32_t>(args->file_name_ptr)));
  const std::string filename_str = rex::string::to_utf8(filename);

  // Virtual host; requests never leave the process (path-routed in xhttp).
  const std::string prefix = "http://xstorage.rexglue/xstorage";
  const uint32_t title_id = kernel_state_->title_id();

  std::string server_path;
  std::string storage_type;
  switch (args->storage_location) {
    case X_STORAGE_FACILITY::FACILITY_GAME_CLIP: {
      uint32_t leaderboard_id = 0;
      if (args->storage_location_info_ptr) {
        const auto* clip_info =
            memory_->TranslateVirtual<X_STORAGE_FACILITY_INFO_GAME_CLIP*>(
                static_cast<uint32_t>(args->storage_location_info_ptr));
        leaderboard_id = clip_info->leaderboard_id;
      }
      server_path = fmt::format("{}/clips/title/{:08X}/{:016X}/{:08X}/{}", prefix,
                                title_id, xuid, leaderboard_id, filename_str);
      storage_type = "Game Clip";
    } break;
    case X_STORAGE_FACILITY::FACILITY_PER_TITLE: {
      server_path = fmt::format("{}/title/{:08X}/{}", prefix, title_id, filename_str);
      storage_type = "Per Title";
    } break;
    case X_STORAGE_FACILITY::FACILITY_PER_USER_TITLE: {
      server_path = fmt::format("{}/user/{:016X}/title/{:08X}/{}", prefix, xuid,
                                title_id, filename_str);
      storage_type = "Per User Title";
    } break;
    default:
      return X_ONLINE_E_STORAGE_INVALID_FACILITY;
  }

  const std::u16string server_path16 = rex::string::to_utf16(server_path);

  if (args->server_path_ptr) {
    auto* out = memory_->TranslateVirtual<char16_t*>(
        static_cast<uint32_t>(args->server_path_ptr));
    rex::string::copy_and_swap_truncating(out, server_path16,
                                               X_ONLINE_MAX_PATHNAME_LENGTH);
  }

  auto* length_out = memory_->TranslateVirtual<uint32_t*>(
      static_cast<uint32_t>(args->server_path_length_ptr));
  *length_out =
      rex::byte_swap(static_cast<uint32_t>(server_path16.size() + 1));  // incl. NUL

  REXKRNL_INFO("XStorageBuildServerPath: {} '{}'", storage_type, server_path);
  return X_E_SUCCESS;
}

namespace {

// Both XStorage transfer messages marshal the same request stream:
//   be<u32> user_index, be<u32> path_len, char16[path_len] (BE) server path,
//   be<u32> buffer_size, be<u32> buffer_address
struct StorageTransferArgs {
  uint32_t user_index = 0;
  std::string server_path;  // utf8
  uint32_t buffer_size = 0;
  uint32_t buffer_address = 0;
  XLIVE_ASYNC_TASK* task = nullptr;
};

X_HRESULT UnmarshalStorageTransfer(rex::memory::Memory* memory, uint32_t buffer_ptr,
                                   StorageTransferArgs* out) {
  if (!buffer_ptr) {
    return X_E_INVALIDARG;
  }
  auto* msg = memory->TranslateVirtual<XLIVEBASE_ASYNC_MESSAGE*>(buffer_ptr);
  if (!msg->xlive_async_task_ptr) {
    return X_E_INVALIDARG;
  }
  auto* task = memory->TranslateVirtual<XLIVE_ASYNC_TASK*>(
      static_cast<uint32_t>(msg->xlive_async_task_ptr));
  if (!task->marshalled_request_ptr || task->marshalled_request_size < 16) {
    return X_E_INVALIDARG;
  }
  const uint8_t* p = memory->TranslateVirtual<const uint8_t*>(
      static_cast<uint32_t>(task->marshalled_request_ptr));
  const uint32_t request_size = task->marshalled_request_size;

  out->task = task;
  out->user_index = memory::load_and_swap<uint32_t>(p + 0);
  const uint32_t path_len = memory::load_and_swap<uint32_t>(p + 4);
  if (path_len > X_ONLINE_MAX_PATHNAME_LENGTH + 1 || 16 + path_len * 2 > request_size) {
    return X_E_INVALIDARG;
  }
  std::u16string path16;
  path16.reserve(path_len);
  for (uint32_t i = 0; i < path_len; ++i) {
    const char16_t c = static_cast<char16_t>(memory::load_and_swap<uint16_t>(p + 8 + i * 2));
    if (!c) {
      break;
    }
    path16.push_back(c);
  }
  out->server_path = rex::string::to_utf8(path16);
  out->buffer_size = memory::load_and_swap<uint32_t>(p + 8 + path_len * 2);
  out->buffer_address = memory::load_and_swap<uint32_t>(p + 8 + path_len * 2 + 4);
  if (out->server_path.empty()) {
    return X_ONLINE_E_STORAGE_INVALID_STORAGE_PATH;
  }
  return X_E_SUCCESS;
}

// Maps a server path from XStorageBuildServerPath to its host backing file:
// <user_data_root>/xstorage/<path after "/xstorage/">.
std::filesystem::path LocalXStoragePath(system::KernelState* kernel_state,
                                        const std::string& server_path) {
  // The Avatar Editor's live avatar store: "//avatar/u:%016I64x/<file>"
  // (sub_920C83D0; avatarpic-l.png 64x64 + avatarpic-s.png 32x32 = the gamer
  // picture pair the booth takes, avatar-body.png 150x300 at Save). Keep them
  // beside the persisted avatar manifest under their plain names, for the
  // upload and the download direction (same mapper).
  static constexpr std::string_view kAvatar = "//avatar/u:";
  if (server_path.compare(0, kAvatar.size(), kAvatar) == 0) {
    const size_t slash = server_path.find('/', kAvatar.size());
    if (slash != std::string::npos) {
      std::string name = server_path.substr(slash + 1);
      if (name == "avatarpic-l.png") name = "gamerpic.png";
      else if (name == "avatarpic-s.png") name = "gamerpic_small.png";
      else if (name == "avatar-body.png") return {};  // deliberately not stored
      if (!name.empty() && name.find("..") == std::string::npos &&
          name.find('/') == std::string::npos && name.find('\\') == std::string::npos) {
        return rex::to_path(rex::kernel::xam::AvatarDataDir()) / rex::to_path(name);
      }
    }
    return {};
  }
  static constexpr std::string_view kMarker = "/xstorage/";
  const size_t pos = server_path.find(kMarker);
  if (pos == std::string::npos) {
    return {};
  }
  std::string tail = server_path.substr(pos + kMarker.size());
  if (tail.empty() || tail.find("..") != std::string::npos) {
    return {};
  }
  return kernel_state->emulator()->user_data_root() / "xstorage" / rex::to_path(tail);
}

}  // namespace

X_HRESULT XLiveBaseApp::XStorageDownloadToMemory(uint32_t buffer_ptr) {
  StorageTransferArgs args;
  if (X_HRESULT hr = UnmarshalStorageTransfer(memory_, buffer_ptr, &args)) {
    return hr;
  }
  if (!args.buffer_address) {
    return X_E_INVALIDARG;
  }

  // Zero the results block up front, like xenia's unmarshaller does.
  if (args.task->results_ptr && args.task->results_size) {
    std::memset(memory_->TranslateVirtual(static_cast<uint32_t>(args.task->results_ptr)), 0,
                args.task->results_size);
  }

  const auto local = LocalXStoragePath(kernel_state_, args.server_path);
  if (local.empty()) {
    REXKRNL_WARN("XStorage: no local mapping for '{}' ({} bytes)", args.server_path,
                 args.buffer_size);
    return X_ONLINE_E_STORAGE_INVALID_STORAGE_PATH;
  }

  std::error_code ec;
  if (!std::filesystem::exists(local, ec)) {
    REXKRNL_INFO("XStorageDownloadToMemory: no stored file for '{}'", args.server_path);
    return X_ONLINE_E_STORAGE_FILE_NOT_FOUND;
  }

  std::ifstream file(local, std::ios::binary | std::ios::ate);
  if (!file) {
    return X_ONLINE_E_STORAGE_FILE_NOT_FOUND;
  }
  const auto size = static_cast<uint64_t>(file.tellg());
  if (size > args.buffer_size) {
    REXKRNL_WARN("XStorageDownloadToMemory: stored file {}b exceeds title buffer {}b ('{}')",
                 size, args.buffer_size, args.server_path);
    return X_E_FAIL;
  }
  file.seekg(0);
  file.read(memory_->TranslateVirtual<char*>(args.buffer_address),
            static_cast<std::streamsize>(size));

  if (args.task->results_ptr &&
      args.task->results_size >= sizeof(X_STORAGE_DOWNLOAD_TO_MEMORY_RESULTS)) {
    auto* results = memory_->TranslateVirtual<X_STORAGE_DOWNLOAD_TO_MEMORY_RESULTS*>(
        static_cast<uint32_t>(args.task->results_ptr));
    results->bytes_total = static_cast<uint32_t>(size);
    const auto* profile = kernel_state_->user_profile();
    results->xuid_owner = profile ? profile->xuid() : 0;
  }

  REXKRNL_INFO("XStorageDownloadToMemory: {}b <- '{}'", size, args.server_path);
  return X_E_SUCCESS;
}

X_HRESULT XLiveBaseApp::XStorageUploadFromMemory(uint32_t buffer_ptr) {
  StorageTransferArgs args;
  if (X_HRESULT hr = UnmarshalStorageTransfer(memory_, buffer_ptr, &args)) {
    return hr;
  }
  if (!args.buffer_address || args.buffer_size > X_STORAGE_MAX_MEMORY_BUFFER_SIZE) {
    return X_E_INVALIDARG;
  }

  const auto local = LocalXStoragePath(kernel_state_, args.server_path);
  if (local.empty()) {
    if (args.server_path.find("/avatar-body.png") != std::string::npos) {
      REXKRNL_INFO("XStorage: avatar-body.png upload accepted and discarded ({} bytes)",
                   args.buffer_size);
      return X_E_SUCCESS;
    }
    REXKRNL_WARN("XStorage: no local mapping for '{}' ({} bytes)", args.server_path,
                 args.buffer_size);
    return X_ONLINE_E_STORAGE_INVALID_STORAGE_PATH;
  }

  std::error_code ec;
  std::filesystem::create_directories(local.parent_path(), ec);
  std::ofstream file(local, std::ios::binary | std::ios::trunc);
  if (!file) {
    REXKRNL_WARN("XStorageUploadFromMemory: cannot open '{}' for write", local.string());
    return X_E_FAIL;
  }
  file.write(memory_->TranslateVirtual<const char*>(args.buffer_address), args.buffer_size);
  file.close();

  REXKRNL_INFO("XStorageUploadFromMemory: {}b -> '{}'", args.buffer_size, args.server_path);
  return X_E_SUCCESS;
}

// Title kernel init (see the header). Keep in step with the SDK's
// rex::kernel::InitializeKernel: same app set and module load order, with the
// title's XLiveBaseApp taking app id 0xFC instead of the SDK built-in.
void InitializeTitleKernel(Runtime* runtime, system::KernelState* kernel_state) {
  (void)runtime;
  auto* app_mgr = kernel_state->app_manager();
  app_mgr->RegisterApp(std::make_unique<XmpApp>(kernel_state));
  app_mgr->RegisterApp(std::make_unique<XgiApp>(kernel_state));
  app_mgr->RegisterApp(std::make_unique<XLiveBaseApp>(kernel_state));
  app_mgr->RegisterApp(std::make_unique<XamApp>(kernel_state));

  kernel_state->LoadKernelModule<xboxkrnl::XboxkrnlModule>();
  kernel_state->LoadKernelModule<XamModule>();
  kernel_state->LoadKernelModule<xbdm::XbdmModule>();
}

}  // namespace apps
}  // namespace xam
}  // namespace kernel
}  // namespace rex
