// System UI exports the title needs answered without a dialog. The runtime's
// XamShowKeyboardUI opens an ImGui prompt the native video path never shows,
// so the request never completes and "Saving" spins forever.
#include <cstring>
#include <functional>

#include <rex/hook.h>
#include <rex/logging.h>
#include <rex/string.h>
#include <rex/system/kernel_state.h>
#include <rex/system/xtypes.h>
#include <rex/types.h>

namespace rex {
namespace kernel {
namespace xam {

namespace {
constexpr uint32_t kNotifySystemUI = 0x9;  // XN_SYS_UI
}

// Accept the default text the title offers ("Outfit 1" for a new outfit) as
// the answer, the way the console's keyboard would if the user pressed Done.
// The XN_SYS_UI pair around the completion mirrors a shown-and-closed dialog.
u32 XamShowKeyboardUI_entry(u32 user_index, u32 flags, mapped_wstring default_text,
                            mapped_wstring title, mapped_wstring description,
                            mapped_wstring buffer, u32 buffer_length, mapped_void overlapped) {
  if (!buffer || !buffer_length) {
    return X_ERROR_INVALID_PARAMETER;
  }
  const std::u16string accepted(default_text ? default_text.value() : std::u16string_view());
  REXKRNL_INFO("XamShowKeyboardUI: accepting the default text ({} chars, {} max)",
               accepted.size(), buffer_length);
  auto run = [buffer, buffer_length, accepted](uint32_t& extended_error,
                                              uint32_t& length) -> X_RESULT {
    // Both sides hold guest byte order, so the copy stays as-is.
    rex::string::copy_truncating(buffer.host_address(), accepted, buffer_length);
    extended_error = X_ERROR_SUCCESS;
    length = 0;
    return X_ERROR_SUCCESS;
  };
  if (!overlapped) {
    uint32_t extended_error = 0, length = 0;
    return run(extended_error, length);
  }
  auto* kernel_state = REX_KERNEL_STATE();
  kernel_state->CompleteOverlappedDeferredEx(
      run, overlapped.guest_address(),
      [kernel_state]() { kernel_state->BroadcastNotification(kNotifySystemUI, 1); },
      [kernel_state]() { kernel_state->BroadcastNotification(kNotifySystemUI, 0); });
  return X_ERROR_IO_PENDING;
}

}  // namespace xam
}  // namespace kernel
}  // namespace rex

REX_EXPORT(__imp__XamShowKeyboardUI, rex::kernel::xam::XamShowKeyboardUI_entry)
