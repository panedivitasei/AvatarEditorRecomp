#pragma once
/**
 ******************************************************************************
 * Xenia : Xbox 360 Emulator Research Project                                 *
 ******************************************************************************
 * Copyright 2015 Ben Vanik. All rights reserved.                             *
 * Released under the BSD license - see LICENSE in the root for more details. *
 ******************************************************************************
 *
 * @modified    Tom Clay, 2026 - Adapted for ReXGlue runtime
 */

#include <rex/system/kernel_state.h>
#include <rex/system/xam/app_manager.h>

namespace rex {
namespace kernel {
namespace xam {
namespace apps {

// Title kernel init, assigned to RuntimeConfig::kernel_init in OnPreSetup.
// Mirrors the SDK's rex::kernel::InitializeKernel but registers the title's
// XLiveBaseApp for app id 0xFC. The SDK default registers its own built-in
// xlivebase app for the same id, and AppManager::RegisterApp asserts on a
// duplicate id (and keeps the first registration), so the SDK default is
// replaced wholesale rather than wrapped.
void InitializeTitleKernel(Runtime* runtime, system::KernelState* kernel_state);

class XLiveBaseApp : public system::xam::App {
 public:
  explicit XLiveBaseApp(system::KernelState* kernel_state);

  X_HRESULT DispatchMessageSync(uint32_t message, uint32_t buffer_ptr,
                                uint32_t buffer_length) override;

 private:
  // CXLiveFriends::Enumerate (msg 0x58020). Reads the marshalled
  // X_CREATE_FRIENDS_ENUMERATOR at buffer_length and returns a valid (empty)
  // enumerator handle so the title's friends/presence init completes.
  X_HRESULT XFriendsCreateEnumerator(uint32_t buffer_length);

  // XStorageBuildServerPath (msg 0x58035). Formats the Title Storage server
  // path for the requested facility and writes it (+ its length) back to the
  // caller. Titles retry-spin on failure (another title polls this every frame), so a
  // valid path + success is required for their storage init to complete.
  X_HRESULT XStorageBuildServerPath(uint32_t buffer_ptr);

  // XStorageDownloadToMemory (msg 0x50009) / XStorageUploadFromMemory
  // (msg 0x5000B). Title Storage transfers, backed by plain host files under
  // <user_data_root>/xstorage/, keyed by the tail of the server path handed
  // out by XStorageBuildServerPath (another title: demoTime.bin/demoCheck.bin ghosts).
  X_HRESULT XStorageDownloadToMemory(uint32_t buffer_ptr);
  X_HRESULT XStorageUploadFromMemory(uint32_t buffer_ptr);
};

}  // namespace apps
}  // namespace xam
}  // namespace kernel
}  // namespace rex
