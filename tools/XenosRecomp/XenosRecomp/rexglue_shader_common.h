#ifndef REXGLUE_SHADER_COMMON_H_INCLUDED
#define REXGLUE_SHADER_COMMON_H_INCLUDED

// Runtime contract for the REXGLUE codegen mode: recompiled shaders bind
// exactly like rexglue's DxbcShaderTranslator output against the BINDLESS
// graphics root signature (see rexglue-sdk/docs/native_shaders.md).
//
// TWO binding models from one source (the UnleashedRecomp pattern): DXC
// defines __spirv__ under -spirv, so the DXIL half binds the b0-b4 root
// CBVs + shared-memory root SRV/UAV, while the SPIR-V half reads the same
// data through vk::RawBufferLoad off a push-constant block of buffer
// device addresses, plume's Vulkan backend has no root-descriptor
// realization at all ("D3D12 only"), push constants + BDA is its native
// model. Shader BODIES are identical: every constant is reached through a
// name that is a cbuffer member on DXIL and a macro/function on SPIR-V.

#ifdef __spirv__

// Slot layout must MATCH the runtime's SetRootBuffer slots
// (renderer.cpp kPushSlot*): one uint64 device address each.
struct XePushConstants
{
    uint64_t System;     // slot 0: b0 system constants
    uint64_t FloatsVs;   // slot 1: b1 float constants, vertex stage
    uint64_t FloatsPs;   // slot 2: b1 float constants, pixel stage
    uint64_t BoolLoop;   // slot 3: b2 bool/loop constants
    uint64_t Fetch;      // slot 4: b3 fetch constants
    uint64_t IdxVs;      // slot 5: b4 descriptor indices, vertex stage
    uint64_t IdxPs;      // slot 6: b4 descriptor indices, pixel stage
    uint64_t SharedMem;  // slot 7: guest shared memory (SRV and UAV alias)
};

[[vk::push_constant]] ConstantBuffer<XePushConstants> xe_push;

// The recompiler emits XE_PIXEL_SHADER before this header for pixel
// shaders; geometry shaders ride the vertex-stage members.
#ifdef XE_PIXEL_SHADER
#define XE_PUSH_FLOATS (xe_push.FloatsPs)
#define XE_PUSH_IDX    (xe_push.IdxPs)
#else
#define XE_PUSH_FLOATS (xe_push.FloatsVs)
#define XE_PUSH_IDX    (xe_push.IdxVs)
#endif

// b0 members by byte offset (c-register * 16 + component * 4).
#define xe_flags                   vk::RawBufferLoad<uint>(xe_push.System + 0)
#define xe_line_loop_closing_index vk::RawBufferLoad<uint>(xe_push.System + 12)
#define xe_vertex_index_endian     vk::RawBufferLoad<uint>(xe_push.System + 16)
#define xe_vertex_index_offset     vk::RawBufferLoad<uint>(xe_push.System + 20)
#define xe_vertex_index_min        vk::RawBufferLoad<uint>(xe_push.System + 24)
#define xe_vertex_index_max        vk::RawBufferLoad<uint>(xe_push.System + 28)
#define xe_ndc_scale               vk::RawBufferLoad<float3>(xe_push.System + 128, 4)
#define xe_ndc_offset              vk::RawBufferLoad<float3>(xe_push.System + 144, 4)
#define xe_alpha_test_reference    vk::RawBufferLoad<float>(xe_push.System + 220)
#define xe_color_exp_bias          vk::RawBufferLoad<float4>(xe_push.System + 240, 4)

#else  // ---- DXIL: root-CBV binding model ----

// ---- System constants (subset; offsets match DxbcShaderTranslator::SystemConstants) ----

cbuffer xe_system_cbuffer : register(b0, space0)
{
    uint   xe_flags                   : packoffset(c0.x);
    uint   xe_line_loop_closing_index : packoffset(c0.w);
    uint   xe_vertex_index_endian     : packoffset(c1.x);
    uint   xe_vertex_index_offset     : packoffset(c1.y);
    uint   xe_vertex_index_min        : packoffset(c1.z);
    uint   xe_vertex_index_max        : packoffset(c1.w);
    // Enabled user clip planes, compacted, PRE-TRANSFORMED by the runtime
    // into post-ndc_scale/offset position space (consumed by the
    // clip_planes GS, not by VS/PS).
    float4 xe_user_clip_planes[6]     : packoffset(c2);
    float3 xe_ndc_scale               : packoffset(c8.x);
    float3 xe_ndc_offset              : packoffset(c9.x);
    float  xe_alpha_test_reference    : packoffset(c13.w);
    float4 xe_color_exp_bias          : packoffset(c15);
};

#endif  // __spirv__

// kSysFlag_* bit values (dxbc_translator.h)
#define XE_FLAG_SHARED_MEMORY_IS_UAV   (1u << 0)
#define XE_FLAG_XY_DIVIDED_BY_W        (1u << 1)
#define XE_FLAG_Z_DIVIDED_BY_W         (1u << 2)
#define XE_FLAG_W_NOT_RECIPROCAL       (1u << 3)
#define XE_FLAG_ALPHA_PASS_IF_LESS     (1u << 7)
#define XE_FLAG_ALPHA_PASS_IF_EQUAL    (1u << 8)
#define XE_FLAG_ALPHA_PASS_IF_GREATER  (1u << 9)

// ---- Bool/loop constants ----

#ifndef __spirv__
cbuffer xe_bool_loop_cbuffer : register(b2, space0)
{
    uint4 xe_bool_constants[2]; // 256 bools
    uint4 xe_loop_constants[8]; // 32 loop constants
};
#endif

// The codegen defines g_Booleans per stage over xe_bool_word:
//   VS: guest bools b0-b31   -> word 0
//   PS: guest bools b128-159 -> word 1 << 16 (bool defines use 1 << (reg + 16))
uint xe_bool_word(uint i)
{
#ifdef __spirv__
    return vk::RawBufferLoad<uint>(xe_push.BoolLoop + i * 16);
#else
    return xe_bool_constants[i].x;
#endif
}

// ---- Fetch constants (32 x 6 dwords) ----

#ifndef __spirv__
cbuffer xe_fetch_cbuffer : register(b3, space0)
{
    uint4 xe_fetch_constants[48];
};
#endif

uint xe_fetch_dword(uint dwordIndex)
{
#ifdef __spirv__
    return vk::RawBufferLoad<uint>(xe_push.Fetch + dwordIndex * 4);
#else
    return xe_fetch_constants[dwordIndex >> 2][dwordIndex & 3];
#endif
}

// ---- Bindless descriptor indices (per-draw, per-binding order) ----

#ifndef __spirv__
cbuffer xe_descriptor_indices_cbuffer : register(b4, space0)
{
    uint4 xe_descriptor_indices[8]; // up to 32 indices; runtime uploads the used count
};
#endif

uint xe_raw_descriptor_index(uint bindingIndex)
{
#ifdef __spirv__
    return vk::RawBufferLoad<uint>(XE_PUSH_IDX + bindingIndex * 4);
#else
    return xe_descriptor_indices[bindingIndex >> 2][bindingIndex & 3];
#endif
}

uint xe_sampler_index(uint bindingIndex)
{
    // Sampler heap bound (must MATCH kSamplerDescriptorCount 16; all 16
    // slots prefilled). The 2047 texture clamp is 128x past this heap, so
    // sampler indices need their own bound.
    return min(xe_raw_descriptor_index(bindingIndex), 15u);
}

uint xe_descriptor_index(uint bindingIndex)
{
    // Clamp to the descriptor heap bounds (must MATCH renderer.cpp
    // kTextureDescriptorCount 2048; every slot is prefilled with a valid
    // blank texture). A stale/garbage per-draw index past the heap is
    // UNDEFINED descriptor access: AMD reads null-ish, Intel fabricates a
    // resource VA from whatever bytes follow the heap -> GPU page fault ->
    // device removal.
    // Must MATCH renderer.cpp kTextureDescriptorCount; packs are keyed to
    // this constant, so regenerate every pack built from this header.
    return min(xe_raw_descriptor_index(bindingIndex), 16383u);
}

// ---- Resources ----

#ifndef __spirv__
ByteAddressBuffer xe_shared_memory_srv : register(t0, space0);
RWByteAddressBuffer xe_shared_memory_uav : register(u0, space0);
#endif

// Texture2DArray: the runtime creates TEXTURE2DARRAY descriptor views to
// match (plume REXGLUE fix), object type and view dimension must agree or
// sampling is UB (zero on AMD).
Texture2DArray<float4> xe_textures_2d[]  : register(t0, space1);
Texture3D<float4>      xe_textures_3d[]  : register(t0, space2);
TextureCube<float4>    xe_textures_cube[] : register(t0, space3);
SamplerState           xe_samplers[]     : register(s0, space0);

// ---- Vertex index (mirrors StartVertexShader_LoadVertexIndex) ----

// xenos::Endian: 0 = none, 1 = 8-in-16, 2 = 8-in-32 (both stages), 3 = 16-in-32.
uint xe_endian_swap_32(uint v, uint endian)
{
    // 8-in-16, or one half of 8-in-32.
    if (endian == 1u || endian == 2u)
        v = ((v & 0x00FF00FFu) << 8) | ((v >> 8) & 0x00FF00FFu);
    // 16-in-32, or the other half of 8-in-32.
    if (endian == 2u || endian == 3u)
        v = (v << 16) | (v >> 16);
    return v;
}

uint4 xe_endian_swap_32(uint4 v, uint endian)
{
    if (endian == 1u || endian == 2u)
        v = ((v & 0x00FF00FFu) << 8) | ((v >> 8) & 0x00FF00FFu);
    if (endian == 2u || endian == 3u)
        v = (v << 16) | (v >> 16);
    return v;
}

float xe_vertex_index(uint vertexId)
{
    // Zero the closing vertex of a non-indexed line loop.
    uint index = (vertexId != xe_line_loop_closing_index) ? vertexId : 0u;
    index = xe_endian_swap_32(index, xe_vertex_index_endian);
    // Base vertex, 24-bit wrap, then clamp.
    index = (index + xe_vertex_index_offset) & 0xFFFFFFu;
    index = clamp(index, xe_vertex_index_min, xe_vertex_index_max);
    return float(index);
}

// ---- Vertex fetch (address + raw load; format decode is emitted inline) ----

// Shared memory is bound as a ROOT descriptor (t0 space0), root SRVs are
// Not bounds-checked by D3D12, so an out-of-range address (one frame of a
// stale/garbage fetch constant during streaming) is UNDEFINED behavior:
// AMD reads garbage harmlessly, Intel iGPUs wedge into a TDR. Clamp every load to
// the ring bounds, a no-op for valid data, defined zeros for garbage.
// Must MATCH renderer.cpp kUploadBufferSize (128 MB).
#define XE_SHARED_MEMORY_CLAMP(addr) min((addr), 0x8000000u - 16u)

// fetchDwordIndex = constIndex * 6 + constIndexSelect * 2
// Returns the base byte address of the element.
uint xe_vfetch_address(uint fetchDwordIndex, float indexFloat, uint strideDwords, bool indexRounded)
{
    uint base = xe_fetch_dword(fetchDwordIndex) & ~3u;
    int index = int(indexRounded ? floor(indexFloat + 0.5) : indexFloat);
    return uint(int(base) + index * int(strideDwords * 4u));
}

// Raw shared-memory loads: SRV/UAV root views on DXIL; on SPIR-V both views
// alias the same buffer, so XE_FLAG_SHARED_MEMORY_IS_UAV collapses to one
// device-address load (alignment 4, guest addresses are dword-aligned).
uint4 xe_shared_load4(uint byteAddress)
{
#ifdef __spirv__
    return vk::RawBufferLoad<uint4>(xe_push.SharedMem + byteAddress, 4);
#else
    if (xe_flags & XE_FLAG_SHARED_MEMORY_IS_UAV)
        return xe_shared_memory_uav.Load4(byteAddress);
    return xe_shared_memory_srv.Load4(byteAddress);
#endif
}

uint3 xe_shared_load3(uint byteAddress)
{
#ifdef __spirv__
    return vk::RawBufferLoad<uint3>(xe_push.SharedMem + byteAddress, 4);
#else
    if (xe_flags & XE_FLAG_SHARED_MEMORY_IS_UAV)
        return xe_shared_memory_uav.Load3(byteAddress);
    return xe_shared_memory_srv.Load3(byteAddress);
#endif
}

uint2 xe_shared_load2(uint byteAddress)
{
#ifdef __spirv__
    return vk::RawBufferLoad<uint2>(xe_push.SharedMem + byteAddress, 4);
#else
    if (xe_flags & XE_FLAG_SHARED_MEMORY_IS_UAV)
        return xe_shared_memory_uav.Load2(byteAddress);
    return xe_shared_memory_srv.Load2(byteAddress);
#endif
}

uint xe_shared_load1(uint byteAddress)
{
#ifdef __spirv__
    return vk::RawBufferLoad<uint>(xe_push.SharedMem + byteAddress, 4);
#else
    if (xe_flags & XE_FLAG_SHARED_MEMORY_IS_UAV)
        return xe_shared_memory_uav.Load(byteAddress);
    return xe_shared_memory_srv.Load(byteAddress);
#endif
}

// Xenos vfetch BOUNDS SEMANTICS: the
// hardware clamps every vertex fetch against the fetch constant's own SIZE
// field, out-of-bounds fetches return zero. Without this, a fetch past
// the pair's declared size read the next ring allocation's bytes: garbage
// that differed per pass (the exploding skinned pieces were sane in one
// pass and NaN shards in another, different ring neighbors). A garbage
// bone INDEX likewise sails past the 6912-byte palette without it.
// dword1 bits 2..25 = size in bytes (dwords<<2), dword0 & ~3 = base.
bool xe_vfetch_in_bounds(uint byteAddress, uint fetchDwordIndex, uint bytes)
{
    uint base = xe_fetch_dword(fetchDwordIndex) & ~3u;
    uint size = xe_fetch_dword(fetchDwordIndex + 1) & 0x3FFFFFCu;
    return byteAddress >= base && (byteAddress + bytes) <= (base + size);
}

uint4 xe_vfetch_load4(uint byteAddress, uint fetchDwordIndex)
{
    if (!xe_vfetch_in_bounds(byteAddress, fetchDwordIndex, 16u))
        return uint4(0u, 0u, 0u, 0u);
    uint4 words = xe_shared_load4(XE_SHARED_MEMORY_CLAMP(byteAddress));
    return xe_endian_swap_32(words, xe_fetch_dword(fetchDwordIndex + 1) & 0b11u);
}

uint2 xe_vfetch_load2(uint byteAddress, uint fetchDwordIndex)
{
    if (!xe_vfetch_in_bounds(byteAddress, fetchDwordIndex, 8u))
        return uint2(0u, 0u);
    uint2 words = xe_shared_load2(XE_SHARED_MEMORY_CLAMP(byteAddress));
    return xe_endian_swap_32(uint4(words, 0, 0), xe_fetch_dword(fetchDwordIndex + 1) & 0b11u).xy;
}

uint3 xe_vfetch_load3(uint byteAddress, uint fetchDwordIndex)
{
    if (!xe_vfetch_in_bounds(byteAddress, fetchDwordIndex, 12u))
        return uint3(0u, 0u, 0u);
    uint3 words = xe_shared_load3(XE_SHARED_MEMORY_CLAMP(byteAddress));
    return xe_endian_swap_32(uint4(words, 0), xe_fetch_dword(fetchDwordIndex + 1) & 0b11u).xyz;
}

uint xe_vfetch_load1(uint byteAddress, uint fetchDwordIndex)
{
    if (!xe_vfetch_in_bounds(byteAddress, fetchDwordIndex, 4u))
        return 0u;
    uint word = xe_shared_load1(XE_SHARED_MEMORY_CLAMP(byteAddress));
    return xe_endian_swap_32(word, xe_fetch_dword(fetchDwordIndex + 1) & 0b11u);
}

// ---- Texture fetch (bindless; binding SLOT indices into b4 are assigned by
// codegen in first-encounter order and resolved through xe_descriptor_index;
// helper names/signatures match the stock emitter) ----

#define FLT_MIN asfloat(0xff7fffff)
#define FLT_MAX asfloat(0x7f7fffff)

struct CubeMapData
{
    float3 cubeMapDirections[2];
    uint cubeMapIndex;
};

float4 cube(float4 value, inout CubeMapData cubeMapData)
{
    uint index = cubeMapData.cubeMapIndex;
    cubeMapData.cubeMapDirections[index] = value.xyz;
    ++cubeMapData.cubeMapIndex;

    return float4(0.0, 0.0, 0.0, index);
}

// Rounding epsilon the translator adds to texel coordinates (resolves
// point-sampling ambiguity between texels; dxbc_translator_fetch.cpp:713).
#define XE_TEXEL_ROUNDING_OFFSET (1.5 / 1024.0)

// 2D texture size from the fetch constant (fields are size-1, 13 bits each).
float2 xe_tfetch_size_2d(uint fetchDword)
{
    uint sizeDword = xe_fetch_dword(fetchDword + 2u);
    return float2(float((sizeDword & 0x1FFFu) + 1u), float(((sizeDword >> 13) & 0x1FFFu) + 1u));
}

// Fetch-constant RESULT EXPONENT BIAS (dword 3 bits 13-18, signed): the
// sampled color is scaled by 2^bias. The ring translator applies this after
// every texture fetch (dxbc_translator_fetch: IBFE 6@13 of word 3 + exponent
// add), games use it to read scaled fixed-point buffers (e.g. one title's
// deferred light accumulation is written /8 and fetched with bias +3).
float4 xe_tfetch_exp_adjust(float4 color, uint fetchDword)
{
    int expAdjust = (int(xe_fetch_dword(fetchDword + 3u)) << 13) >> 26;
    return color * asfloat(uint(0x3F800000) + uint(expAdjust << 23));
}

// 1D textures are bound as 2D arrays, like the runtime's texture cache.
float4 tfetch1D(uint textureSlot, uint samplerSlot, uint fetchDword, bool denorm, float x)
{
    uint sizeDword = xe_fetch_dword(fetchDword + 2u);
    float width = float((sizeDword & 0xFFFFFFu) + 1u);
    float texel = (denorm ? x : x * width) + XE_TEXEL_ROUNDING_OFFSET;
    return xe_tfetch_exp_adjust(xe_textures_2d[xe_descriptor_index(textureSlot)].Sample(
        xe_samplers[xe_sampler_index(samplerSlot)], float3(texel / width, 0.0, 0.0)), fetchDword);
}

float4 tfetch2D(uint textureSlot, uint samplerSlot, uint fetchDword, bool denorm, float2 uv, float2 offsetTexels)
{
    float2 size = xe_tfetch_size_2d(fetchDword);
    float2 texel = (denorm ? uv : uv * size) + offsetTexels + XE_TEXEL_ROUNDING_OFFSET;
    return xe_tfetch_exp_adjust(xe_textures_2d[xe_descriptor_index(textureSlot)].Sample(
        xe_samplers[xe_sampler_index(samplerSlot)], float3(texel / size, 0.0)), fetchDword);
}

float4 tfetch3D(uint textureSlot, uint samplerSlot, uint fetchDword, bool denorm, float3 uvw)
{
    // NOTE: stacked-2D textures (3D fetches of 2D arrays) are not handled;
    // this samples the 3D view. Size fields: 11/11/10 bits, size-1.
    uint sizeDword = xe_fetch_dword(fetchDword + 2u);
    float3 size = float3(float((sizeDword & 0x7FFu) + 1u), float(((sizeDword >> 11) & 0x7FFu) + 1u),
                         float(((sizeDword >> 22) & 0x3FFu) + 1u));
    float3 texel = (denorm ? uvw : uvw * size) + XE_TEXEL_ROUNDING_OFFSET;
    return xe_tfetch_exp_adjust(xe_textures_3d[xe_descriptor_index(textureSlot)].Sample(
        xe_samplers[xe_sampler_index(samplerSlot)], texel / size), fetchDword);
}

float4 tfetchCube(uint textureSlot, uint samplerSlot, uint fetchDword, bool denorm, float3 texCoord,
                  inout CubeMapData cubeMapData)
{
    return xe_tfetch_exp_adjust(xe_textures_cube[xe_descriptor_index(textureSlot)].Sample(
        xe_samplers[xe_sampler_index(samplerSlot)], cubeMapData.cubeMapDirections[texCoord.z]), fetchDword);
}

// ---- Explicit-LOD (_lod0) tfetch variants ----
// Emitted for fetches inside divergent (per-pixel p0) control flow and for
// vertex shaders: Sample()'s implicit derivatives require the 2x2 quad in
// lockstep; under divergence Intel's helper-lane logic deadlocks the EU,
// while AMD reconverges.
// Signatures match the Sample variants exactly; the recompiler appends
// "_lod0" to the call when the fetch's cf index sits in a divergent span.

float4 tfetch1D_lod0(uint textureSlot, uint samplerSlot, uint fetchDword, bool denorm, float x)
{
    uint sizeDword = xe_fetch_dword(fetchDword + 2u);
    float width = float((sizeDword & 0xFFFFFFu) + 1u);
    float texel = (denorm ? x : x * width) + XE_TEXEL_ROUNDING_OFFSET;
    return xe_tfetch_exp_adjust(xe_textures_2d[xe_descriptor_index(textureSlot)].SampleLevel(
        xe_samplers[xe_sampler_index(samplerSlot)], float3(texel / width, 0.0, 0.0), 0.0), fetchDword);
}

float4 tfetch2D_lod0(uint textureSlot, uint samplerSlot, uint fetchDword, bool denorm, float2 uv, float2 offsetTexels)
{
    float2 size = xe_tfetch_size_2d(fetchDword);
    float2 texel = (denorm ? uv : uv * size) + offsetTexels + XE_TEXEL_ROUNDING_OFFSET;
    return xe_tfetch_exp_adjust(xe_textures_2d[xe_descriptor_index(textureSlot)].SampleLevel(
        xe_samplers[xe_sampler_index(samplerSlot)], float3(texel / size, 0.0), 0.0), fetchDword);
}

float4 tfetch3D_lod0(uint textureSlot, uint samplerSlot, uint fetchDword, bool denorm, float3 uvw)
{
    uint sizeDword = xe_fetch_dword(fetchDword + 2u);
    float3 size = float3(float((sizeDword & 0x7FFu) + 1u), float(((sizeDword >> 11) & 0x7FFu) + 1u),
                         float(((sizeDword >> 22) & 0x3FFu) + 1u));
    float3 texel = (denorm ? uvw : uvw * size) + XE_TEXEL_ROUNDING_OFFSET;
    return xe_tfetch_exp_adjust(xe_textures_3d[xe_descriptor_index(textureSlot)].SampleLevel(
        xe_samplers[xe_sampler_index(samplerSlot)], texel / size, 0.0), fetchDword);
}

float4 tfetchCube_lod0(uint textureSlot, uint samplerSlot, uint fetchDword, bool denorm, float3 texCoord,
                       inout CubeMapData cubeMapData)
{
    return xe_tfetch_exp_adjust(xe_textures_cube[xe_descriptor_index(textureSlot)].SampleLevel(
        xe_samplers[xe_sampler_index(samplerSlot)], cubeMapData.cubeMapDirections[texCoord.z], 0.0), fetchDword);
}

// getWeights reads the texture size from the fetch constant; no texture
// binding is involved, matching the translator.
float2 getWeights2D(uint fetchDword, bool denorm, float2 uv, float2 offsetTexels)
{
    float2 size = xe_tfetch_size_2d(fetchDword);
    float2 texel = (denorm ? uv : uv * size) + offsetTexels + XE_TEXEL_ROUNDING_OFFSET;
    return select(isnan(uv), 0.0, frac(texel - 0.5));
}

// ---- VS position epilogue (mirrors CompleteVertexOrDomainShader steps 1-6;
// user clip planes and vertex kill are not emitted by the REXGLUE mode) ----

float4 xe_apply_position(float4 pos)
{
    if (!(xe_flags & XE_FLAG_W_NOT_RECIPROCAL))
        pos.w = 1.0 / pos.w;
    if (xe_flags & XE_FLAG_XY_DIVIDED_BY_W)
        pos.xy *= pos.w;
    if (xe_flags & XE_FLAG_Z_DIVIDED_BY_W)
        pos.z *= pos.w;
    pos.xyz = pos.xyz * xe_ndc_scale + xe_ndc_offset * pos.w;
    return pos;
}

// ---- PS epilogue helpers ----

// Alpha test (RTV path): discard when the comparison against
// xe_alpha_test_reference fails. Pass mask bits: less/equal/greater.
void xe_alpha_test(float alpha)
{
    uint passFlags = xe_flags & (XE_FLAG_ALPHA_PASS_IF_LESS | XE_FLAG_ALPHA_PASS_IF_EQUAL |
                                 XE_FLAG_ALPHA_PASS_IF_GREATER);
    // "Always pass" = all three bits set; skip the test entirely then.
    if (passFlags != (XE_FLAG_ALPHA_PASS_IF_LESS | XE_FLAG_ALPHA_PASS_IF_EQUAL |
                      XE_FLAG_ALPHA_PASS_IF_GREATER))
    {
        bool pass = false;
        if (passFlags & XE_FLAG_ALPHA_PASS_IF_LESS)
            pass = pass || (alpha < xe_alpha_test_reference);
        if (passFlags & XE_FLAG_ALPHA_PASS_IF_EQUAL)
            pass = pass || (alpha == xe_alpha_test_reference);
        if (passFlags & XE_FLAG_ALPHA_PASS_IF_GREATER)
            pass = pass || (alpha > xe_alpha_test_reference);
        if (!pass)
            discard;
    }
}

// ---- Misc ALU helpers (names match the stock emitter) ----

float4 dst(float4 src0, float4 src1)
{
    float4 dest;
    dest.x = 1.0;
    dest.y = src0.y * src1.y;
    dest.z = src0.z;
    dest.w = src1.w;
    return dest;
}

float4 max4(float4 src0)
{
    return max(max(src0.x, src0.y), max(src0.z, src0.w));
}

#endif
