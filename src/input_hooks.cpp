// input_hooks.cpp: keyboard keystroke synthesis for XUI navigation.
//
// The runtime's mouse/keyboard driver maps keybinds to gamepad state but
// never feeds the keystroke queue, and the editor navigates through
// XamInputGetKeystrokeEx. Capture the window's key events, map them through
// the same keybind cvars the driver uses, and serve VK_PAD keystrokes with
// key repeat whenever the runtime has none of its own. Real controllers keep
// their native keystroke path through the delegated call.

#include <atomic>
#include <chrono>
#include <cstdint>
#include <deque>
#include <mutex>
#include <string>
#include <string_view>
#include <vector>

#include <windows.h>

#include <rex/cvar.h>
#include <rex/hook.h>
#include <rex/logging.h>
#include <rex/ppc/context.h>
#include <rex/system/kernel_state.h>
#include <rex/ui/keybinds.h>
#include <rex/ui/virtual_key.h>

#include "input_hooks.h"

namespace {

using Clock = std::chrono::steady_clock;
using rex::ui::VirtualKey;

constexpr uint16_t kKeystrokeKeyDown = 0x0001;
constexpr uint16_t kKeystrokeKeyUp = 0x0002;
constexpr uint16_t kKeystrokeRepeat = 0x0004;

struct PadBind {
  uint16_t pad_vk;         // kXInputPad* value served to the guest
  const char* cvar_name;   // keybind cvar holding the bind spec
};

const PadBind kPadBinds[] = {
    {uint16_t(VirtualKey::kXInputPadA), "keybind_a"},
    {uint16_t(VirtualKey::kXInputPadB), "keybind_b"},
    {uint16_t(VirtualKey::kXInputPadX), "keybind_x"},
    {uint16_t(VirtualKey::kXInputPadY), "keybind_y"},
    {uint16_t(VirtualKey::kXInputPadDpadUp), "keybind_dpad_up"},
    {uint16_t(VirtualKey::kXInputPadDpadDown), "keybind_dpad_down"},
    {uint16_t(VirtualKey::kXInputPadDpadLeft), "keybind_dpad_left"},
    {uint16_t(VirtualKey::kXInputPadDpadRight), "keybind_dpad_right"},
    {uint16_t(VirtualKey::kXInputPadStart), "keybind_start"},
    {uint16_t(VirtualKey::kXInputPadBack), "keybind_back"},
    {uint16_t(VirtualKey::kXInputPadLShoulder), "keybind_left_shoulder"},
    {uint16_t(VirtualKey::kXInputPadRShoulder), "keybind_right_shoulder"},
    {uint16_t(VirtualKey::kXInputPadLThumbUp), "keybind_lstick_up"},
    {uint16_t(VirtualKey::kXInputPadLThumbDown), "keybind_lstick_down"},
    {uint16_t(VirtualKey::kXInputPadLThumbLeft), "keybind_lstick_left"},
    {uint16_t(VirtualKey::kXInputPadLThumbRight), "keybind_lstick_right"},
};

struct QueuedStroke {
  uint16_t pad_vk;
  uint16_t flags;
};

struct HeldPad {
  uint16_t pad_vk;
  VirtualKey source_key;
  Clock::time_point down_time;
  Clock::time_point last_repeat;
};

std::mutex g_mutex;
std::deque<QueuedStroke> g_queue;
std::vector<HeldPad> g_held;

std::string_view TrimSpaces(std::string_view s) {
  while (!s.empty() && s.front() == ' ') s.remove_prefix(1);
  while (!s.empty() && s.back() == ' ') s.remove_suffix(1);
  return s;
}

bool EqualsNoCase(std::string_view a, std::string_view b) {
  if (a.size() != b.size()) return false;
  for (size_t i = 0; i < a.size(); ++i) {
    if (::tolower(uint8_t(a[i])) != ::tolower(uint8_t(b[i]))) return false;
  }
  return true;
}

// One bind token: [Shift+|Ctrl+|Alt+]* KeyName. A token with modifiers
// requires them held; a bare token requires none, so a Shift chord does not
// also fire the unmodified bind.
bool TokenMatches(std::string_view token, VirtualKey key, bool shift, bool ctrl, bool alt) {
  bool want_shift = false, want_ctrl = false, want_alt = false;
  for (;;) {
    size_t plus = token.find('+');
    if (plus == std::string_view::npos) break;
    std::string_view mod = TrimSpaces(token.substr(0, plus));
    if (EqualsNoCase(mod, "Shift")) want_shift = true;
    else if (EqualsNoCase(mod, "Ctrl")) want_ctrl = true;
    else if (EqualsNoCase(mod, "Alt")) want_alt = true;
    else return false;
    token.remove_prefix(plus + 1);
  }
  token = TrimSpaces(token);
  if (token.empty()) return false;
  if (rex::ui::ParseVirtualKey(token) != key) return false;
  return shift == want_shift && ctrl == want_ctrl && alt == want_alt;
}

bool BindMatches(const std::string& spec, VirtualKey key, bool shift, bool ctrl, bool alt) {
  std::string_view rest(spec);
  while (!rest.empty()) {
    size_t comma = rest.find(',');
    std::string_view token = rest.substr(0, comma);
    rest = (comma == std::string_view::npos) ? std::string_view()
                                             : rest.substr(comma + 1);
    if (TokenMatches(TrimSpaces(token), key, shift, ctrl, alt)) return true;
  }
  return false;
}

std::string QueryBind(const char* name) {
  if (!rex::cvar::GetFlagInfo(name)) return {};
  return rex::cvar::Query<std::string>(name);
}

}  // namespace

namespace ae_input {

void OnHostKey(VirtualKey key, bool down, bool shift, bool ctrl, bool alt) {
  std::lock_guard<std::mutex> lock(g_mutex);
  if (down) {
    for (const PadBind& bind : kPadBinds) {
      if (!BindMatches(QueryBind(bind.cvar_name), key, shift, ctrl, alt)) continue;
      bool already = false;
      for (const HeldPad& h : g_held) {
        if (h.pad_vk == bind.pad_vk) { already = true; break; }
      }
      if (already) continue;
      const auto now = Clock::now();
      g_held.push_back({bind.pad_vk, key, now, now});
      g_queue.push_back({bind.pad_vk, kKeystrokeKeyDown});
    }
  } else {
    for (size_t i = 0; i < g_held.size();) {
      if (g_held[i].source_key == key) {
        g_queue.push_back({g_held[i].pad_vk, kKeystrokeKeyUp});
        g_held.erase(g_held.begin() + i);
      } else {
        ++i;
      }
    }
  }
}

}  // namespace ae_input

namespace {

// Drain one synthesized keystroke, generating repeats for held keys.
bool NextStroke(QueuedStroke* out) {
  std::lock_guard<std::mutex> lock(g_mutex);
  if (!g_queue.empty()) {
    *out = g_queue.front();
    g_queue.pop_front();
    return true;
  }
  const auto now = Clock::now();
  for (HeldPad& h : g_held) {
    using std::chrono::milliseconds;
    if (now - h.down_time < milliseconds(400)) continue;
    if (now - h.last_repeat < milliseconds(100)) continue;
    h.last_repeat = now;
    *out = {h.pad_vk, uint16_t(kKeystrokeKeyDown | kKeystrokeRepeat)};
    return true;
  }
  return false;
}

// The runtime's own export, reachable by name through the DLL's export table
// even though this exe defines the same symbol for generated code.
PPCFunc* OriginalGetKeystrokeEx() {
  static PPCFunc* original = []() -> PPCFunc* {
    HMODULE runtime = GetModuleHandleW(L"rexruntimerd.dll");
    if (!runtime) runtime = GetModuleHandleW(L"rexruntime.dll");
    if (!runtime) return nullptr;
    return reinterpret_cast<PPCFunc*>(
        GetProcAddress(runtime, "__imp__XamInputGetKeystrokeEx"));
  }();
  return original;
}

}  // namespace

REX_HOOK_RAW(__imp__XamInputGetKeystrokeEx) {
  const uint32_t user_index_ptr = ctx.r3.u32;
  const uint32_t flags = ctx.r4.u32;
  const uint32_t keystroke_ptr = ctx.r5.u32;

  // Real devices first: controllers deliver their own keystrokes.
  if (PPCFunc* original = OriginalGetKeystrokeEx()) {
    original(ctx, base);
    if (ctx.r3.u32 == 0) {
      return;
    }
  }

  if (!keystroke_ptr) return;
  if ((flags & 0xFF) && (flags & 0x01) == 0) return;  // gamepad queries only

  QueuedStroke stroke;
  if (!NextStroke(&stroke)) return;

  auto* memory = rex::system::kernel_state()->memory();
  uint8_t* ks = memory->TranslateVirtual<uint8_t*>(keystroke_ptr);
  if (!ks) return;
  ks[0] = uint8_t(stroke.pad_vk >> 8);
  ks[1] = uint8_t(stroke.pad_vk);
  ks[2] = 0;  // unicode
  ks[3] = 0;
  ks[4] = uint8_t(stroke.flags >> 8);
  ks[5] = uint8_t(stroke.flags);
  ks[6] = 0;  // user index
  ks[7] = 0;  // hid code
  if (user_index_ptr) {
    if (uint8_t* user = memory->TranslateVirtual<uint8_t*>(user_index_ptr)) {
      user[0] = user[1] = user[2] = user[3] = 0;
    }
  }
  ctx.r3.u64 = 0;  // X_ERROR_SUCCESS
}
