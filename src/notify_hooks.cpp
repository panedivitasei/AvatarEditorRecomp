// notify_hooks.cpp: Xbox LIVE connection notification.
//
// The runtime's notification startup batch covers sign-in and input device
// events but never reports the LIVE link. The editor's live-gated flows
// (the Gamer Pic camera among them) latch XN_LIVE_CONNECTIONCHANGED through
// NETMANXA and show "only available with an Xbox Live profile" until it
// arrives, so deliver it whenever a listener that watches the live area is
// created. The connection state is idempotent; re-broadcasting to earlier
// listeners is harmless.

#include <cstdint>

#include <windows.h>

#include <rex/hook.h>
#include <rex/logging.h>
#include <rex/ppc/context.h>
#include <rex/system/kernel_state.h>

namespace {

constexpr uint32_t kXnLiveConnectionChanged = 0x02000001;
constexpr uint32_t kLogonConnectionEstablished = 0x001510F0;
constexpr uint32_t kXnSysSigninChanged = 0x0000000A;
constexpr uint64_t kLiveAreaMask = 0x00000002;

PPCFunc* OriginalExport(const char* name) {
  HMODULE runtime = GetModuleHandleW(L"rexruntimerd.dll");
  if (!runtime) runtime = GetModuleHandleW(L"rexruntime.dll");
  if (!runtime) return nullptr;
  return reinterpret_cast<PPCFunc*>(GetProcAddress(runtime, name));
}

void NotifyLiveConnected(uint64_t mask, uint32_t handle) {
  // Mask 0 subscribes to every area.
  if (handle == 0 || (mask != 0 && !(mask & kLiveAreaMask))) {
    return;
  }
  auto* kernel_state = rex::system::kernel_state();
  kernel_state->BroadcastNotification(kXnLiveConnectionChanged,
                                      kLogonConnectionEstablished);
  // The front end re-evaluates live availability on a sign-in change.
  kernel_state->BroadcastNotification(kXnSysSigninChanged, 1);
  REXKRNL_INFO("Live connection reported to listener {:08X} (mask {:#x})", handle, mask);
}

}  // namespace

REX_HOOK_RAW(__imp__XamNotifyCreateListener) {
  const uint64_t mask = ctx.r3.u64;
  static PPCFunc* original = OriginalExport("__imp__XamNotifyCreateListener");
  if (!original) {
    ctx.r3.u64 = 0;
    return;
  }
  original(ctx, base);
  NotifyLiveConnected(mask, ctx.r3.u32);
}

// System executables create their listeners through the internal variant.
REX_HOOK_RAW(__imp__XamNotifyCreateListenerInternal) {
  const uint64_t mask = ctx.r3.u64;
  static PPCFunc* original =
      OriginalExport("__imp__XamNotifyCreateListenerInternal");
  if (!original) {
    ctx.r3.u64 = 0;
    return;
  }
  original(ctx, base);
  NotifyLiveConnected(mask, ctx.r3.u32);
}
