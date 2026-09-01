/**
 ******************************************************************************
 * Xenia : Xbox 360 Emulator Research Project                                 *
 ******************************************************************************
 * Copyright 2025 Xenia Canary. All rights reserved.                          *
 * Released under the BSD license - see LICENSE in the root for more details. *
 ******************************************************************************
 *
 * @modified    Ported to ReXGlue runtime, 2026
 */

#ifndef REX_KERNEL_UTIL_XFILETIME_H_
#define REX_KERNEL_UTIL_XFILETIME_H_

#include <cstdint>

#include <rex/types.h>

namespace rex {
namespace kernel {

struct X_FILETIME {
  static constexpr uint64_t minimal_valid_time = 125911584000000000ull;
  static constexpr uint64_t maximal_valid_time = 157469184000000000ull;

  rex::be<uint32_t> high_part;
  rex::be<uint32_t> low_part;

  X_FILETIME() {
    high_part = 0;
    low_part = 0;
  }

  X_FILETIME(uint64_t filetime) {
    high_part = static_cast<uint32_t>(filetime >> 32);
    low_part = static_cast<uint32_t>(filetime);
  }

  uint64_t to_uint64() const {
    return (static_cast<uint64_t>(high_part) << 32) | low_part;
  }

  bool is_valid() const {
    const uint64_t filetime = to_uint64();
    return filetime >= minimal_valid_time && filetime <= maximal_valid_time;
  }
};
static_assert_size(X_FILETIME, 0x8);

}  // namespace kernel
}  // namespace rex

#endif
