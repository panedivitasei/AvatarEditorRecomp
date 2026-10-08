// Guest thread: turns a texture fetch constant into a host texture description and decodes its guest memory
// (untile, endian swap, format conversion) into upload bytes. Uses the SDK texture layout helpers.
#pragma once

#include <array>
#include <cstdint>
#include <memory>

#include <rex/graphics/xenos.h>

#include "gpu/gpu_records.h"

namespace ae::gpu {

struct GuestTextureDesc {
  rex::graphics::xenos::xe_gpu_texture_fetch_t fetch{};
  uint32_t width = 0;
  uint32_t height = 0;
  uint32_t depth = 1;         // 3D depth; 1 otherwise
  uint32_t layers = 1;        // 6 for cube maps, the stack depth for stacked 2D
  uint32_t mip_levels = 1;    // host levels, starting at the base
  uint32_t mip_min = 0;       // fetch-constant LOD range after the SDK's clean-up
  uint32_t mip_max = 0;
  uint32_t base_address = 0;  // physical
  uint32_t base_size = 0;
  uint32_t mip_address = 0;
  uint32_t mip_size = 0;
  HostFormat host_format = HostFormat::kUnsupported;
  TextureShape shape = TextureShape::k2D;
  bool decompress = false;    // BC data expanded to RGBA8 (sizes not a multiple of 4)
  bool expand = false;        // per-component conversion to float (signs, gamma, packed or depth formats)
};

// False for dimensions or formats the decoder does not handle; the reason is logged once per format.
bool DescribeTexture(const rex::graphics::xenos::xe_gpu_texture_fetch_t& fetch, GuestTextureDesc& out);

// Physical base and mip ranges a texture fetch constant covers, without format support checks.
bool TextureMemory(const rex::graphics::xenos::xe_gpu_texture_fetch_t& fetch, uint32_t& base, uint32_t& base_size,
                   uint32_t& mip, uint32_t& mip_size);

// Reads guest memory; levels are ordered layer-major (layer 0 all mips, then layer 1), 3D levels hold every slice.
std::shared_ptr<TextureUpload> DecodeTexture(const GuestTextureDesc& desc);

// Copies the texture's guest bytes now and decodes them on a worker thread; the upload's Wait() blocks until the
// levels are filled. Small textures decode inline.
std::shared_ptr<TextureUpload> DecodeTextureAsync(const GuestTextureDesc& desc);

// Decodes one BC1-BC5 block into a 4x4 RGBA8 tile, row-major.
void DecodeBlockRGBA(HostFormat format, const uint8_t* block, uint8_t out[16][4]);

// Bytes per block row of one host level, and the block height in texels (4 for BC).
uint32_t HostBlockBytes(HostFormat format);
uint32_t HostBlockSize(HostFormat format);

// Composes a fetch-constant swizzle with the host format's storage: a one-component format replicates X into every
// component and a two-component one replicates Y into Z and W, as Xenos does before the swizzle.
std::array<uint8_t, 4> ComposeSwizzle(HostFormat format, const std::array<uint8_t, 4>& fetch_swizzle);

}  // namespace ae::gpu
