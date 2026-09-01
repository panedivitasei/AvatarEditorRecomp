# plume (vendored)

- Upstream: https://github.com/renderbag/plume.git
- Commit: 4f556be ("Metal: Track resource residency at the device level (#101)")
- License: MIT (see LICENSE)
- Vendored: 2026-07-13 for the native video layer port
  (UnleashedRecomp-style guest D3D implementation).

Trims relative to upstream:
- examples/, contrib/metal-cpp, plume_metal.cpp/.mm dropped (no Apple build).
- contrib submodules flattened at their pinned commits:
  - D3D12MemoryAllocator 9ef66bc (include/ + src/D3D12MemAlloc.* only)
  - Vulkan-Headers 2fa2034 (include/ only, C headers only — the C++
    vulkan.hpp family is unused by plume and was 16MB)
  - VulkanMemoryAllocator 29b35ea (include/ only)
  - volk be3dbd4 (volk.c/.h + CMakeLists + LICENSE)

Note: the SDK's thirdparty/vulkan-headers, volk, vulkan-memory-allocator
(gated behind REXGLUE_USE_VULKAN) are intentionally NOT reused here — plume
expects its own pinned contrib paths and must build regardless of that
option. Keep this copy self-contained.

Local patch: CMakeLists.txt links d3d12+dxgi on WIN32 (upstream expects
the consuming app to do it).
