// content_hooks.cpp: the editor's Save Outfit lands as a new OUTFIT* content package, which unlocks Fresh Fit.

#include <cstdint>
#include <cstring>

#include <windows.h>

#include <rex/hook.h>
#include <rex/ppc/context.h>
#include <rex/system/kernel_state.h>
#include <rex/system/xam/content_manager.h>

#include "ae_achievements.h"

namespace {

PPCFunc* OriginalExport(const char* name) {
  HMODULE runtime = GetModuleHandleW(L"rexruntimerd.dll");
  if (!runtime) runtime = GetModuleHandleW(L"rexruntime.dll");
  if (!runtime) return nullptr;
  return reinterpret_cast<PPCFunc*>(GetProcAddress(runtime, name));
}

// CREATE_NEW and CREATE_ALWAYS write a package; the other dispositions only open one.
void NoteCreate(uint32_t content_data, uint32_t flags) {
  const uint32_t disposition = flags & 0xF;
  if (!content_data || (disposition != 1 && disposition != 2)) {
    return;
  }
  const auto* data =
      rex::system::kernel_memory()->TranslateVirtual<const rex::system::xam::XCONTENT_DATA*>(
          content_data);
  if (data && std::strncmp(data->file_name_raw, "OUTFIT", 6) == 0) {
    ae::achievements::Unlock(ae::achievements::kFreshFit);
  }
}

}  // namespace

REX_HOOK_RAW(__imp__XamContentCreateEx) {
  static PPCFunc* original = OriginalExport("__imp__XamContentCreateEx");
  NoteCreate(ctx.r5.u32, ctx.r6.u32);
  if (!original) {
    ctx.r3.u64 = 0x57;  // ERROR_INVALID_PARAMETER
    return;
  }
  original(ctx, base);
}

REX_HOOK_RAW(__imp__XamContentCreateInternal) {
  static PPCFunc* original = OriginalExport("__imp__XamContentCreateInternal");
  NoteCreate(ctx.r4.u32, ctx.r5.u32);
  if (!original) {
    ctx.r3.u64 = 0x57;  // ERROR_INVALID_PARAMETER
    return;
  }
  original(ctx, base);
}
