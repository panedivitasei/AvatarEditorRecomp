// gamma_ramp.h: the display gamma ramp the XDK uploads, 768 host-endian words as R[256] G[256] B[256].
#pragma once

#include <array>
#include <atomic>
#include <cstdint>

inline std::array<uint16_t, 768> ae_gpu_gamma_ramp{};
// Bumped after each ramp write so a renderer can spot a fresh table.
inline std::atomic<uint32_t> ae_gpu_gamma_ramp_gen{0};
