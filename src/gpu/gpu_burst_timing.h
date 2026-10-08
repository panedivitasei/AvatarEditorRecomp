#pragma once

#include <chrono>
#include "gpu/gpu.h"

namespace ae::gpu {

bool BurstTimingEnabled();

// Diagnostic scopes are inclusive; nested texture costs must not be added to binding time.
class BurstTiming {
 public:
  explicit BurstTiming(int kind) : kind_(kind), enabled_(BurstTimingEnabled()) {
    if (enabled_) start_ = Clock::now();
  }
  ~BurstTiming() {
    if (enabled_) AddGuestTiming(kind_, uint64_t(std::chrono::duration_cast<std::chrono::nanoseconds>(
        Clock::now() - start_).count()));
  }
 private:
  using Clock = std::chrono::steady_clock;
  int kind_;
  bool enabled_;
  Clock::time_point start_;
};

}  // namespace ae::gpu
