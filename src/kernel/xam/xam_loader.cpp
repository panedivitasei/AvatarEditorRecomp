// Title launch exports the runtime leaves stubbed. The Avatar Editor's Save
// and Exit ends in XamLoaderLaunchTitleEx(NULL), a return to the dashboard.

#include <rex/hook.h>
#include <rex/logging.h>
#include <rex/system/kernel_state.h>
#include <rex/types.h>

namespace rex {
namespace kernel {
namespace xam {

// There is no dashboard to return to currently.
void XamLoaderLaunchTitleEx_entry(mapped_string path, u32 flags, u32 unk3, u32 unk4) {
  if (path && !path.value().empty()) {
    REXKRNL_WARN("XamLoaderLaunchTitleEx({}, {:#x}): launching titles is not supported, exiting",
                 path.value(), flags);
  } else {
    REXKRNL_INFO("XamLoaderLaunchTitleEx(NULL): exit to dashboard, terminating");
  }
  REX_KERNEL_STATE()->TerminateTitle();
}

}  // namespace xam
}  // namespace kernel
}  // namespace rex

REX_EXPORT(__imp__XamLoaderLaunchTitleEx, rex::kernel::xam::XamLoaderLaunchTitleEx_entry)
