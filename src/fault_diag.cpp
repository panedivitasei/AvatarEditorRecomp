// Guest-function attribution for unhandled host faults, resolved through the generated function table.

#include "fault_diag.h"

#include <algorithm>
#include <cstdint>
#include <string>
#include <utility>
#include <vector>

#include <rex/exception_handler.h>
#include <rex/logging.h>
#include <rex/ppc/func.h>

#if defined(_WIN32)
#include <windows.h>
#endif

namespace ae::diag {

namespace {

// (host entry, guest address), sorted by host entry so a pc maps to the function that starts before it.
std::vector<std::pair<uintptr_t, uint32_t>> g_functions;
uintptr_t g_module_base = 0;
uintptr_t g_module_end = 0;

// The recompiled function a host address falls in, or 0 when it sits outside the generated code.
uint32_t GuestFunction(uintptr_t pc, uintptr_t* offset) {
  auto it = std::upper_bound(g_functions.begin(), g_functions.end(), std::make_pair(pc, uint32_t(0xFFFFFFFF)));
  if (it == g_functions.begin()) return 0;
  // Host code linked after a generated function would otherwise be charged to it, so the next entry bounds it.
  const uintptr_t next = it == g_functions.end() ? it[-1].first + 0x10000 : it->first;
  --it;
  *offset = pc - it->first;
  return pc < next ? it->second : 0;
}

std::string Describe(uintptr_t pc) {
  uintptr_t offset = 0;
  if (const uint32_t guest = GuestFunction(pc, &offset)) {
    return fmt::format("guest sub_{:08X}+{:#x}", guest, offset);
  }
  if (pc >= g_module_base && pc < g_module_end) {
    return fmt::format("exe+{:#x}", pc - g_module_base);
  }
  return fmt::format("host {:#x}", pc);
}

bool OnFault(rex::arch::Exception* ex, void*) {
  const uintptr_t pc = ex->pc();
  REXLOG_ERROR("[fault] pc {:#x} = {}, fault address {:#x}", pc, Describe(pc), ex->fault_address());
#if defined(_WIN32) && defined(_M_X64)
  // Return addresses near the top of the stack name the guest path that reached the faulting host code.
  const auto* ctx = ex->thread_context();
  const auto* sp = reinterpret_cast<const uintptr_t*>(ctx->rsp);
  std::string chain;
  int hits = 0;
  for (int i = 0; i < 512 && hits < 16; ++i) {
    MEMORY_BASIC_INFORMATION info{};
    if (!VirtualQuery(sp + i, &info, sizeof(info)) || !(info.State & MEM_COMMIT)) break;
    const uintptr_t value = sp[i];
    if (value < g_module_base || value >= g_module_end) continue;
    chain += fmt::format(" [{}]{}", i, Describe(value));
    ++hits;
  }
  REXLOG_ERROR("[fault] stack:{}", chain);
#endif
  return false;
}

}  // namespace

void InstallFaultReporter() {
  for (const PPCFuncMapping* m = PPCFuncMappings; m->host; ++m) {
    g_functions.emplace_back(reinterpret_cast<uintptr_t>(m->host), uint32_t(m->guest));
  }
  std::sort(g_functions.begin(), g_functions.end());
#if defined(_WIN32)
  HMODULE module = GetModuleHandleW(nullptr);
  g_module_base = reinterpret_cast<uintptr_t>(module);
  const auto* dos = reinterpret_cast<const IMAGE_DOS_HEADER*>(module);
  const auto* nt = reinterpret_cast<const IMAGE_NT_HEADERS*>(g_module_base + dos->e_lfanew);
  g_module_end = g_module_base + nt->OptionalHeader.SizeOfImage;
#endif
  rex::arch::ExceptionHandler::Install(OnFault, nullptr);
  REXLOG_INFO("[fault] reporter installed over {} guest functions, exe at {:#x}", g_functions.size(), g_module_base);
}

}  // namespace ae::diag
