// Compatibility shims so the ported xenia avatar code builds against ReXGlue.
// Injected at the top of every avatars/*.{h,cpp} file.
#ifndef REX_AVATARS_XE_COMPAT_H_
#define REX_AVATARS_XE_COMPAT_H_

#include <cstdint>
#include <string>

#include <rex/assert.h>
#include <rex/math.h>
#include <rex/string.h>
#include <rex/types.h>

// --- Xenia assert_* macros -> ReXGlue ---------------------------------------
#ifndef assert_true
#define assert_true(...) rex_assert((__VA_ARGS__))
#endif
#ifndef assert_false
#define assert_false(...) rex_assert(!(__VA_ARGS__))
#endif
#ifndef assert_always
#define assert_always(...) rex_assert(false)
#endif
#ifndef assert_unhandled_case
#define assert_unhandled_case(...) rex_assert(false)
#endif
#ifndef assert_zero
#define assert_zero(x) rex_assert((x) == 0)
#endif
#ifndef assert_not_null
#define assert_not_null(x) rex_assert((x) != nullptr)
#endif
#ifndef assert_size
#define assert_size(type, size) static_assert_size(type, size)
#endif

namespace rex {

// xe::fourcc_t -> a 4-char-code stored big-endian.
using fourcc_t = uint32_t;

// xe::make_fourcc("ABCD") -> packed big-endian 4-char-code.
constexpr fourcc_t make_fourcc(const char (&s)[5]) {
  return (static_cast<fourcc_t>(static_cast<uint8_t>(s[0])) << 24) |
         (static_cast<fourcc_t>(static_cast<uint8_t>(s[1])) << 16) |
         (static_cast<fourcc_t>(static_cast<uint8_t>(s[2])) << 8) |
         static_cast<fourcc_t>(static_cast<uint8_t>(s[3]));
}

// xe::load_and_swap<T>(ptr): read a big-endian value at ptr, byte-swapped to
// host. The avatar code only uses the std::u16string specialization (asset
// names: NUL-terminated UTF-16BE).
template <typename T>
inline T load_and_swap(const void* mem) {
  return byte_swap(*reinterpret_cast<const T*>(mem));
}
template <>
inline std::u16string load_and_swap<std::u16string>(const void* mem) {
  auto p = reinterpret_cast<const char16_t*>(mem);
  std::u16string value;
  for (; *p; ++p) {
    value.push_back(static_cast<char16_t>(byte_swap(static_cast<uint16_t>(*p))));
  }
  return value;
}

}  // namespace rex

#endif  // REX_AVATARS_XE_COMPAT_H_
