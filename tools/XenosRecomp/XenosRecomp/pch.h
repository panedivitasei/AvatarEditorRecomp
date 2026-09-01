#pragma once

#ifdef _WIN32
#include <Windows.h>
#endif

#include <dxcapi.h>

#include <bit>
#include <cassert>
#include <cstdint>
#include <stdexcept>

#ifdef XENOS_COVERAGE
// Coverage builds turn asserts into exceptions so a per-shader harness can
// catch and catalog unsupported constructs instead of aborting (or silently
// continuing in release).
struct CoverageError : std::runtime_error
{
    using std::runtime_error::runtime_error;
};
#undef assert
#define assert(expr) ((expr) ? (void)0 : throw CoverageError("assert: " #expr))
#endif
#include <execution>
#include <filesystem>
#include <map>
#include <set>
#include <smolv.h>
#include <fmt/core.h>
#include <string>
#include <tuple>
#include <unordered_map>
#include <vector>
#include <xxhash.h>
#include <zstd.h>

template<typename T>
static T byteSwap(T value)
{
    if constexpr (sizeof(T) == 1)
        return value;
    else if constexpr (sizeof(T) == 2)
        return static_cast<T>(__builtin_bswap16(static_cast<uint16_t>(value)));
    else if constexpr (sizeof(T) == 4)
        return static_cast<T>(__builtin_bswap32(static_cast<uint32_t>(value)));
    else if constexpr (sizeof(T) == 8) 
        return static_cast<T>(__builtin_bswap64(static_cast<uint64_t>(value)));

    assert(false && "Unexpected byte size.");
    return value;
}

template<typename T>
struct be
{
    T value;

    T get() const
    {
        if constexpr (std::is_enum_v<T>)
            return T(byteSwap(std::underlying_type_t<T>(value)));
        else
            return byteSwap(value);
    }

    operator T() const
    {
        return get();
    }
};  
