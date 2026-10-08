// Avatar marketplace key for online mode.

#include "marketplace_key.h"

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <mutex>
#include <string>

#include <rex/cvar.h>

#if defined(_WIN32)
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#endif

// Defined at global scope in xam_user.cpp.
REXCVAR_DECLARE(std::string, avatar_marketplace_key);

namespace ae::marketplace {
namespace {

constexpr char kMagic[] = "JMKEY1";
constexpr size_t kMagicLen = sizeof(kMagic) - 1;
// Key pad.
constexpr unsigned char kPad[] = {0x4a, 0x4d, 0x73, 0x74, 0x75, 0x64, 0x69, 0x6f, 0x73, 0x2e, 0x6d,
                                  0x61, 0x72, 0x6b, 0x65, 0x74, 0x70, 0x6c, 0x61, 0x63, 0x65, 0x2e,
                                  0x76, 0x31, 0x20, 0x4a, 0x4d, 0x73, 0x74, 0x75, 0x64, 0x69};

std::string FromTrailer() {
#if defined(_WIN32)
  wchar_t path[MAX_PATH];
  const DWORD n = GetModuleFileNameW(nullptr, path, MAX_PATH);
  if (!n) return {};
  FILE* f = _wfopen(path, L"rb");
  if (!f) return {};
  std::string key;
  unsigned char tail[4 + kMagicLen];
  if (fseek(f, -long(sizeof tail), SEEK_END) == 0 && fread(tail, 1, sizeof tail, f) == sizeof tail &&
      memcmp(tail + 4, kMagic, kMagicLen) == 0) {
    const uint32_t len = uint32_t(tail[0]) | (uint32_t(tail[1]) << 8) | (uint32_t(tail[2]) << 16) |
                         (uint32_t(tail[3]) << 24);
    if (len && len <= 256 && fseek(f, -long(sizeof tail + len), SEEK_END) == 0) {
      key.resize(len);
      if (fread(key.data(), 1, len, f) != len) key.clear();
      for (size_t i = 0; i < key.size(); ++i) key[i] = char(key[i] ^ kPad[i % sizeof kPad]);
    }
  }
  fclose(f);
  return key;
#else
  return {};
#endif
}

}  // namespace

const std::string& Key() {
  static std::string key;
  static std::once_flag once;
  std::call_once(once, [] {
    key = REXCVAR_GET(avatar_marketplace_key);
    if (key.empty()) key = FromTrailer();
  });
  return key;
}

}  // namespace ae::marketplace
