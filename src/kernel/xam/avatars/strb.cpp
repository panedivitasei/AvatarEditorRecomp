/**
 ******************************************************************************
 * Xenia : Xbox 360 Emulator Research Project                                 *
 ******************************************************************************
 * Copyright 2026 Ben Vanik. All rights reserved.                             *
 * Released under the BSD license - see LICENSE in the root for more details. *
 ******************************************************************************
 */

#include "strb.h"
#include <rex/math.h>

#include "xe_compat.h"

namespace rex {
namespace avatars {
namespace strb {
using namespace rex::avatars;

template <std::endian E>
size_t GetSTRBValue(const uint8_t* buffer, size_t size) {
  switch (size) {
    case 1:
      return buffer[0];
    case 2:
      return *reinterpret_cast<const endian_store<uint16_t, E>*>(buffer);
    case 4:
      return *reinterpret_cast<const endian_store<uint32_t, E>*>(buffer);
    case 8:
      return *reinterpret_cast<const endian_store<uint64_t, E>*>(buffer);
    default:
      assert_unhandled_case(size);
  }
  return 8;
}

size_t ReadSTRBValue(const uint8_t* buffer, size_t size,
                     bool is_little_endian) {
  auto value = is_little_endian
                   ? GetSTRBValue<std::endian::little>(buffer, size)
                   : GetSTRBValue<std::endian::big>(buffer, size);
  return value;
}

// Walk every block; the visitor returns true to stop the walk.
template <typename Visitor>
static bool WalkSTRBBlocks(const uint8_t* strb_buffer, size_t strb_size,
                           Visitor&& visit) {
  auto magic = *reinterpret_cast<const be<fourcc_t>*>(&strb_buffer[0]);
  if (magic == make_fourcc("YTGR")) {
    // skip verification signature
    // TODO(gibbed): check header?
    if (strb_size < 0x140) {
      return false;
    }
    strb_buffer += 0x140;
    strb_size -= 0x140;
    magic = *reinterpret_cast<const be<fourcc_t>*>(&strb_buffer[0]);
  }

  if (strb_size < 24) {
    return false;
  }

  if (magic != make_fourcc("STRB")) {
    return false;
  }

  auto has_block_alignment = strb_buffer[4] != 0;
  auto is_little_endian = strb_buffer[5] != 0;
  uint8_t guid[16];
  std::memcpy(guid, &strb_buffer[6], sizeof(guid));
  auto block_id_size = size_t(strb_buffer[22]);
  auto block_size_size = size_t(strb_buffer[23]);
  auto block_alignment = size_t(has_block_alignment ? strb_buffer[26] : 1);

  size_t block_header_size =
      align(block_id_size + block_size_size + block_size_size, block_alignment);

  size_t block_start_offset =
      align(size_t(has_block_alignment ? 30 : 26), block_alignment);

  for (size_t block_offset = block_start_offset;
       block_offset + block_header_size <= strb_size;) {
    auto block_id = STRBBlockId(ReadSTRBValue(&strb_buffer[block_offset],
                                              block_id_size, is_little_endian));
    size_t block_data_size =
        ReadSTRBValue(&strb_buffer[block_offset + block_id_size],
                      block_size_size, is_little_endian);
    size_t block_entry_size = ReadSTRBValue(
        &strb_buffer[block_offset + block_id_size + block_size_size],
        block_size_size, is_little_endian);
    block_offset += block_header_size;

    if (visit(block_id, &strb_buffer[block_offset],
              block_data_size * block_entry_size)) {
      return true;
    }

    block_offset += align(block_data_size, block_alignment);
  }

  return false;
}

bool GetSTRBBlock(const uint8_t* strb_buffer, size_t strb_size,
                  STRBBlockId target_id, const uint8_t*& target_buffer,
                  size_t& target_size) {
  return GetSTRBBlockN(strb_buffer, strb_size, target_id, 0, target_buffer,
                       target_size);
}

bool GetSTRBBlockN(const uint8_t* strb_buffer, size_t strb_size,
                   STRBBlockId target_id, size_t occurrence,
                   const uint8_t*& target_buffer, size_t& target_size) {
  size_t seen = 0;
  return WalkSTRBBlocks(strb_buffer, strb_size,
                        [&](STRBBlockId id, const uint8_t* data, size_t size) {
                          if (id != target_id || seen++ != occurrence) {
                            return false;
                          }
                          target_buffer = data;
                          target_size = size;
                          return true;
                        });
}

size_t CountSTRBBlocks(const uint8_t* strb_buffer, size_t strb_size,
                       STRBBlockId target_id) {
  size_t count = 0;
  WalkSTRBBlocks(strb_buffer, strb_size,
                 [&](STRBBlockId id, const uint8_t*, size_t) {
                   if (id == target_id) {
                     ++count;
                   }
                   return false;
                 });
  return count;
}

}  // namespace strb
}  // namespace avatars
}  // namespace rex
