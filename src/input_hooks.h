// Keyboard keystroke synthesis (input_hooks.cpp): the app's window listener
// feeds host key events here; the XamInputGetKeystrokeEx hook serves them to
// the guest as VK_PAD keystrokes.

#pragma once

#include <rex/ui/virtual_key.h>

namespace ae_input {
void OnHostKey(rex::ui::VirtualKey key, bool down, bool shift, bool ctrl, bool alt);
}
