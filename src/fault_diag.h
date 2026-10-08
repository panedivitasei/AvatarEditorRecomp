// Names the guest function behind an unhandled host fault, so a crash log says where the title died.
#pragma once

namespace ae::diag {

// Installs after the SDK's own MMIO handler, so it only sees faults nothing else claimed.
void InstallFaultReporter();

}  // namespace ae::diag
