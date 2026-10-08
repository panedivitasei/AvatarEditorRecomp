#ifndef REXGLUE_SHADER_COMMON_H_INCLUDED
#define REXGLUE_SHADER_COMMON_H_INCLUDED

// Runtime contract for the REXGLUE codegen mode: recompiled shaders bind like rexglue's
// DxbcShaderTranslator output against the bindless graphics root signature (rexglue-sdk/docs/native_shaders.md).
//
// Two binding models from one source: the DXIL half binds the b0-b4 root CBVs and the shared-memory
// root SRV/UAV, the SPIR-V half (__spirv__) reads the same data with vk::RawBufferLoad off push-constant
// device addresses because plume's Vulkan backend has no root descriptors; the bodies are identical.

#ifdef __spirv__

// One uint64 device address per slot, in the order src/gpu declares its push slots (pack_contract.md 1.2).
// The pack header records the slot count, so a layout change refuses stale packs.
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
    uint64_t VfetchVs;   // slot 8: b5 vfetch layout table, vertex stage
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
#define xe_pixel_position_scale    vk::RawBufferLoad<float2>(xe_push.System + 4, 4)
#define xe_line_loop_closing_index vk::RawBufferLoad<uint>(xe_push.System + 12)
#define xe_vertex_index_endian     vk::RawBufferLoad<uint>(xe_push.System + 16)
#define xe_vertex_index_offset     vk::RawBufferLoad<uint>(xe_push.System + 20)
#define xe_vertex_index_min        vk::RawBufferLoad<uint>(xe_push.System + 24)
#define xe_vertex_index_max        vk::RawBufferLoad<uint>(xe_push.System + 28)
#define xe_ndc_scale               vk::RawBufferLoad<float3>(xe_push.System + 128, 4)
#define xe_ndc_offset              vk::RawBufferLoad<float3>(xe_push.System + 144, 4)
#define xe_alpha_test_reference    vk::RawBufferLoad<float>(xe_push.System + 220)
#define xe_color_exp_bias          vk::RawBufferLoad<float4>(xe_push.System + 240, 4)
#define xe_cc2_feature_mask        vk::RawBufferLoad<uint>(xe_push.System + 256)
#define xe_shadow_texel            vk::RawBufferLoad<float2>(xe_push.System + 264, 4)

#else  // ---- DXIL: root-CBV binding model ----

// ---- System constants (subset; offsets match DxbcShaderTranslator::SystemConstants) ----

cbuffer xe_system_cbuffer : register(b0, space0)
{
    uint   xe_flags                   : packoffset(c0.x);
    float2 xe_pixel_position_scale    : packoffset(c0.y);
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
    // CC2 region from c16 on, clear of the SDK-mirrored offsets; b0 is 272 bytes.
    uint   xe_cc2_feature_mask        : packoffset(c16.x);
    float2 xe_shadow_texel            : packoffset(c16.z);
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
    // The pack ABI reserves 1024 exact sampler states independently of the texture heap.
    return min(xe_raw_descriptor_index(bindingIndex), 1023u);
}

uint xe_descriptor_index(uint bindingIndex)
{
    // A garbage index past the 16384-entry texture heap page-faults Intel GPUs into device removal.
    // The pack header records the heap size, so changing this clamp refuses packs built with the old one.
    return min(xe_raw_descriptor_index(bindingIndex), 16383u);
}

// ---- Resources ----

#ifndef __spirv__
ByteAddressBuffer xe_shared_memory_srv : register(t0, space0);
// No UAV view of shared memory: the runtime never sets XE_FLAG_SHARED_MEMORY_IS_UAV and no title shader exports
// to memory, and a UAV referenced by every draw makes Gen9-class GPUs serialise the draws.
#endif

// Texture2DArray: the runtime creates TEXTURE2DARRAY descriptor views to
// match (plume REXGLUE fix), object type and view dimension must agree or
// sampling is UB (zero on AMD).
Texture2DArray<float4> xe_textures_2d[]  : register(t0, space1);
Texture3D<float4>      xe_textures_3d[]  : register(t0, space2);
TextureCube<float4>    xe_textures_cube[] : register(t0, space3);
SamplerState           xe_samplers[]     : register(s0, space0);

// ---- Vertex index (mirrors StartVertexShader_LoadVertexIndex) ----

// xenos::Endian: 0 = none, 1 = 8-in-16, 2 = 8-in-32 (both swaps), 3 = 16-in-32.
// Built from select() so the swap stays branch-free; HLSL 2021 turns ?: and || into control flow.
uint xe_endian_swap_32(uint v, uint endian)
{
    v = select(or(endian == 1u, endian == 2u), ((v & 0x00FF00FFu) << 8) | ((v >> 8) & 0x00FF00FFu), v);
    return select(or(endian == 2u, endian == 3u), (v << 16) | (v >> 16), v);
}

uint4 xe_endian_swap_32(uint4 v, uint endian)
{
    v = select(or(endian == 1u, endian == 2u), ((v & 0x00FF00FFu) << 8) | ((v >> 8) & 0x00FF00FFu), v);
    return select(or(endian == 2u, endian == 3u), (v << 16) | (v >> 16), v);
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

// ---- Vertex fetch (address, raw load, and the X3 layout-record decode) ----

// D3D12 never bounds-checks the t0 root descriptor, and a stale fetch constant wedges Intel iGPUs into a TDR.
// Every load is clamped to the 128 MB buffer, whose size the pack header records.
#define XE_SHARED_MEMORY_CLAMP(addr) min((addr), 0x8000000u - 16u)

// Raw shared-memory load: SRV/UAV root views on DXIL; on SPIR-V both views alias one buffer,
// so XE_FLAG_SHARED_MEMORY_IS_UAV collapses to one device-address load (alignment 4).
uint4 xe_shared_load4(uint byteAddress)
{
#ifdef __spirv__
    return vk::RawBufferLoad<uint4>(xe_push.SharedMem + byteAddress, 4);
#else
    return xe_shared_memory_srv.Load4(byteAddress);
#endif
}

// The two dwords of a vertex fetch constant at fetchDwordIndex = constIndex * 6 + constIndexSelect * 2.
// The index is even, so the pair is one half of a uint4 and a dynamic index costs no local array.
uint2 xe_vfetch_constant(uint fetchDwordIndex)
{
#ifdef __spirv__
    return vk::RawBufferLoad<uint2>(xe_push.Fetch + fetchDwordIndex * 4, 4);
#else
    uint4 v = xe_fetch_constants[fetchDwordIndex >> 2];
    return select((fetchDwordIndex & 2u) != 0u, v.zw, v.xy);
#endif
}

// Per-draw vfetch layout table: one 16-byte record per vfetch ordinal (pack_contract.md 2.4).
//   x: bits 0-7 fetch dword index, 8-13 format, 14 signed, 15 integer, 16 rf no-zero, 31 valid
//   y: stride in dwords, z: signed offset in dwords, w: destination swizzle (3 bits per component)
#ifndef __spirv__
cbuffer xe_vfetch_cbuffer : register(b5, space0)
{
    uint4 xe_vfetch_records[32];
};
#endif

#ifndef XE_VFETCH_STATIC
uint4 xe_vfetch_record(uint ordinal)
{
#ifdef __spirv__
    return vk::RawBufferLoad<uint4>(xe_push.VfetchVs + ordinal * 16, 4);
#else
    return xe_vfetch_records[ordinal];
#endif
}
#else
// The shader defines it after this header from its baked record table.
uint4 xe_vfetch_record(uint ordinal);
#endif

#define XE_VFETCH_SIGNED      1u
#define XE_VFETCH_INTEGER     2u
#define XE_VFETCH_RF_NO_ZERO  4u

#define XE_VFETCH_KIND_INT    0u
#define XE_VFETCH_KIND_HALF   1u
#define XE_VFETCH_KIND_FLOAT  2u

// Rows indexed by xenos::VertexFormat; each component is offset | width << 8 | word << 16, width 0 = absent.
// x bits 24-26 carry the word count (0 = a format fetched as zero, like the SDK) and bits 28-29 the kind.
static const uint4 xe_vfetch_layouts[64] =
{
    uint4(0x00000000u, 0x00000u, 0x00000u, 0x00000u),
    uint4(0x00000000u, 0x00000u, 0x00000u, 0x00000u),
    uint4(0x00000000u, 0x00000u, 0x00000u, 0x00000u),
    uint4(0x00000000u, 0x00000u, 0x00000u, 0x00000u),
    uint4(0x00000000u, 0x00000u, 0x00000u, 0x00000u),
    uint4(0x00000000u, 0x00000u, 0x00000u, 0x00000u),
    uint4(0x01000800u, 0x00808u, 0x00810u, 0x00818u), // 6 k_8_8_8_8
    uint4(0x01000A00u, 0x00A0Au, 0x00A14u, 0x0021Eu), // 7 k_2_10_10_10
    uint4(0x00000000u, 0x00000u, 0x00000u, 0x00000u),
    uint4(0x00000000u, 0x00000u, 0x00000u, 0x00000u),
    uint4(0x00000000u, 0x00000u, 0x00000u, 0x00000u),
    uint4(0x00000000u, 0x00000u, 0x00000u, 0x00000u),
    uint4(0x00000000u, 0x00000u, 0x00000u, 0x00000u),
    uint4(0x00000000u, 0x00000u, 0x00000u, 0x00000u),
    uint4(0x00000000u, 0x00000u, 0x00000u, 0x00000u),
    uint4(0x00000000u, 0x00000u, 0x00000u, 0x00000u),
    uint4(0x01000B00u, 0x00B0Bu, 0x00A16u, 0x00000u), // 16 k_10_11_11
    uint4(0x01000A00u, 0x00B0Au, 0x00B15u, 0x00000u), // 17 k_11_11_10
    uint4(0x00000000u, 0x00000u, 0x00000u, 0x00000u),
    uint4(0x00000000u, 0x00000u, 0x00000u, 0x00000u),
    uint4(0x00000000u, 0x00000u, 0x00000u, 0x00000u),
    uint4(0x00000000u, 0x00000u, 0x00000u, 0x00000u),
    uint4(0x00000000u, 0x00000u, 0x00000u, 0x00000u),
    uint4(0x00000000u, 0x00000u, 0x00000u, 0x00000u),
    uint4(0x00000000u, 0x00000u, 0x00000u, 0x00000u),
    uint4(0x01001000u, 0x01010u, 0x00000u, 0x00000u), // 25 k_16_16
    uint4(0x02001000u, 0x01010u, 0x11000u, 0x11010u), // 26 k_16_16_16_16
    uint4(0x00000000u, 0x00000u, 0x00000u, 0x00000u),
    uint4(0x00000000u, 0x00000u, 0x00000u, 0x00000u),
    uint4(0x00000000u, 0x00000u, 0x00000u, 0x00000u),
    uint4(0x00000000u, 0x00000u, 0x00000u, 0x00000u),
    uint4(0x11001000u, 0x01010u, 0x00000u, 0x00000u), // 31 k_16_16_FLOAT
    uint4(0x12001000u, 0x01010u, 0x11000u, 0x11010u), // 32 k_16_16_16_16_FLOAT
    uint4(0x01002000u, 0x00000u, 0x00000u, 0x00000u), // 33 k_32
    uint4(0x02002000u, 0x12000u, 0x00000u, 0x00000u), // 34 k_32_32
    uint4(0x04002000u, 0x12000u, 0x22000u, 0x32000u), // 35 k_32_32_32_32
    uint4(0x21002000u, 0x00000u, 0x00000u, 0x00000u), // 36 k_32_FLOAT
    uint4(0x22002000u, 0x12000u, 0x00000u, 0x00000u), // 37 k_32_32_FLOAT
    uint4(0x24002000u, 0x12000u, 0x22000u, 0x32000u), // 38 k_32_32_32_32_FLOAT
    uint4(0x00000000u, 0x00000u, 0x00000u, 0x00000u),
    uint4(0x00000000u, 0x00000u, 0x00000u, 0x00000u),
    uint4(0x00000000u, 0x00000u, 0x00000u, 0x00000u),
    uint4(0x00000000u, 0x00000u, 0x00000u, 0x00000u),
    uint4(0x00000000u, 0x00000u, 0x00000u, 0x00000u),
    uint4(0x00000000u, 0x00000u, 0x00000u, 0x00000u),
    uint4(0x00000000u, 0x00000u, 0x00000u, 0x00000u),
    uint4(0x00000000u, 0x00000u, 0x00000u, 0x00000u),
    uint4(0x00000000u, 0x00000u, 0x00000u, 0x00000u),
    uint4(0x00000000u, 0x00000u, 0x00000u, 0x00000u),
    uint4(0x00000000u, 0x00000u, 0x00000u, 0x00000u),
    uint4(0x00000000u, 0x00000u, 0x00000u, 0x00000u),
    uint4(0x00000000u, 0x00000u, 0x00000u, 0x00000u),
    uint4(0x00000000u, 0x00000u, 0x00000u, 0x00000u),
    uint4(0x00000000u, 0x00000u, 0x00000u, 0x00000u),
    uint4(0x00000000u, 0x00000u, 0x00000u, 0x00000u),
    uint4(0x00000000u, 0x00000u, 0x00000u, 0x00000u),
    uint4(0x00000000u, 0x00000u, 0x00000u, 0x00000u),
    uint4(0x23002000u, 0x12000u, 0x22000u, 0x00000u), // 57 k_32_32_32_FLOAT
    uint4(0x00000000u, 0x00000u, 0x00000u, 0x00000u),
    uint4(0x00000000u, 0x00000u, 0x00000u, 0x00000u),
    uint4(0x00000000u, 0x00000u, 0x00000u, 0x00000u),
    uint4(0x00000000u, 0x00000u, 0x00000u, 0x00000u),
    uint4(0x00000000u, 0x00000u, 0x00000u, 0x00000u),
    uint4(0x00000000u, 0x00000u, 0x00000u, 0x00000u)
};

// Normalization factors as exact float bits for the field widths the formats use: 1/(2^w-1) unsigned,
// 1/(2^(w-1)-1) signed, 2/(2^w-1) kNoZero scale with the unsigned factor as bias (dxbc_translator_fetch.cpp).
float4 xe_vfetch_scale_by_width(uint4 width, uint4 w2, uint4 w8, uint4 w10, uint4 w11, uint4 w16, uint4 w32)
{
    return asfloat(select(width == 2u, w2, select(width == 8u, w8, select(width == 10u, w10,
                   select(width == 11u, w11, select(width == 16u, w16, w32))))));
}

// All four components at once, so each inlined fetch costs a few vector operations per step.
// For 32-bit fields the -1 clamp never triggers, since -2^31 scales to exactly -1 as the SDK notes.
float4 xe_vfetch_decode(uint4 words, uint4 layout, uint flags)
{
    uint kind = (layout.x >> 28) & 3u;
    uint4 desc = layout & 0x3FFFFu;
    uint4 offset = desc & 0xFFu;
    uint4 width = min((desc >> 8) & 0xFFu, 32u);
    uint4 wordSel = desc >> 16;
    uint4 word = select(wordSel == 0u, words.xxxx, select(wordSel == 1u, words.yyyy,
                 select(wordSel == 2u, words.zzzz, words.wwww)));

    bool4 isSigned = ((flags & XE_VFETCH_SIGNED) != 0u).xxxx;
    float4 s = float4(asint(word << (32u - offset - width)) >> (32u - width));
    float4 u = float4((word >> offset) & (0xFFFFFFFFu >> (32u - width)));
    float4 v = select(isSigned, s, u);

    float4 unormScale = xe_vfetch_scale_by_width(width, 0x3EAAAAABu, 0x3B808081u, 0x3A802008u, 0x3A001002u,
                                                 0x37800080u, 0x2F800000u);
    float4 snormScale = xe_vfetch_scale_by_width(width, 0x3F800000u, 0x3C010204u, 0x3B004020u, 0x3A802008u,
                                                 0x38000100u, 0x30000000u);
    float4 rfScale = xe_vfetch_scale_by_width(width, 0x3F2AAAABu, 0x3C008081u, 0x3B002008u, 0x3A801002u,
                                              0x38000080u, 0x30000000u);
    float4 signedNorm = select(((flags & XE_VFETCH_RF_NO_ZERO) != 0u).xxxx, v * rfScale + unormScale,
                               max(v * snormScale, -1.0));
    float4 normalized = select(isSigned, signedNorm, v * unormScale);
    float4 integerOrNorm = select(((flags & XE_VFETCH_INTEGER) != 0u).xxxx, v, normalized);

    float4 decoded = select((kind == XE_VFETCH_KIND_HALF).xxxx, f16tof32(word >> offset),
                     select((kind == XE_VFETCH_KIND_FLOAT).xxxx, asfloat(word), integerOrNorm));
    return select(width != 0u, decoded, 0.0);
}

// Destination swizzle per component: 0-3 = xyzw, 4 = 0, 5 = 1, 7 = keep; the invalid 6 writes 0 as the SDK does.
float4 xe_vfetch_merge(float4 v, float4 old, uint swizzle)
{
    uint4 s = (swizzle.xxxx >> uint4(0u, 3u, 6u, 9u)) & 7u;
    float4 picked = select(s == 0u, v.xxxx, select(s == 1u, v.yyyy, select(s == 2u, v.zzzz, v.wwww)));
    return select(s <= 3u, picked, select(s == 5u, 1.0, select(s == 7u, old, 0.0)));
}

// Vfetch ordinal i through record i; the index operand, rounding, exp adjust and predication stay in the
// instruction. A record without the valid bit leaves the register untouched.
float4 xe_vfetch_exp(uint ordinal, float indexFloat, bool indexRounded, float expScale, float4 old)
{
    uint4 record = xe_vfetch_record(ordinal);
    uint4 layout = xe_vfetch_layouts[(record.x >> 8) & 0x3Fu];
    uint2 fetchConstant = xe_vfetch_constant(record.x & 0xFEu);

    // Element base, floored (or rounded) index times the stride, then the instruction offset.
    uint base = fetchConstant.x & ~3u;
    int index = int(floor(select(indexRounded, indexFloat + 0.5, indexFloat)));
    uint address = uint(int(base) + index * int(record.y * 4u) + int(record.z) * 4);

    // The hardware returns zero past the fetch constant's size (dword1 bits 2-25); without the check a fetch
    // read the next ring allocation. A format this decoder lacks has no words and decodes to zero.
    uint bytes = ((layout.x >> 24) & 7u) * 4u;
    uint size = fetchConstant.y & 0x3FFFFFCu;
    bool inBounds = and(address >= base, address + bytes <= base + size);
    uint4 words = xe_endian_swap_32(xe_shared_load4(XE_SHARED_MEMORY_CLAMP(address)), fetchConstant.y & 3u);
    words = select(inBounds, words, 0u);

    float4 v = xe_vfetch_decode(words, layout, (record.x >> 14) & 7u) * expScale;
    return select((record.x & 0x80000000u) != 0u, xe_vfetch_merge(v, old, record.w), old);
}

float4 xe_vfetch(uint ordinal, float indexFloat, bool indexRounded, float4 old)
{
    return xe_vfetch_exp(ordinal, indexFloat, indexRounded, 1.0, old);
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

// Fetch-constant LOD bias (dword 4 bits 12-21, signed, 1/32 steps). The ring
// translator adds it to every computed or explicit LOD (dxbc_translator_fetch.cpp:1411).
float xe_tfetch_lod_bias(uint fetchDword)
{
    return float(int(xe_fetch_dword(fetchDword + 4u) << 10) >> 22) * (1.0 / 32.0);
}

// 1D textures are bound as 2D arrays, like the runtime's texture cache.
float3 xe_tfetch_coord_1d(uint fetchDword, bool denorm, float x)
{
    uint sizeDword = xe_fetch_dword(fetchDword + 2u);
    float width = float((sizeDword & 0xFFFFFFu) + 1u);
    float texel = (denorm ? x : x * width) + XE_TEXEL_ROUNDING_OFFSET;
    return float3(texel / width, 0.0, 0.0);
}

float3 xe_tfetch_coord_2d(uint fetchDword, bool denorm, float2 uv, float2 offsetTexels)
{
    float2 size = xe_tfetch_size_2d(fetchDword);
    float2 texel = (denorm ? uv : uv * size) + offsetTexels + XE_TEXEL_ROUNDING_OFFSET;
    return float3(texel / size, 0.0);
}

float3 xe_tfetch_size_3d(uint fetchDword)
{
    uint sizeDword = xe_fetch_dword(fetchDword + 2u);
    return float3(float((sizeDword & 0x7FFu) + 1u), float(((sizeDword >> 11) & 0x7FFu) + 1u),
                  float(((sizeDword >> 22) & 0x3FFu) + 1u));
}

// Stacked-2D textures (3D fetches of 2D arrays) are not handled; these sample the 3D view.
float3 xe_tfetch_coord_3d(uint fetchDword, bool denorm, float3 uvw)
{
    float3 size = xe_tfetch_size_3d(fetchDword);
    return ((denorm ? uvw : uvw * size) + XE_TEXEL_ROUNDING_OFFSET) / size;
}

// Computed-LOD fetches (implicit derivatives). The fetch-constant LOD bias rides on SampleBias.
float4 tfetch1D(uint textureSlot, uint samplerSlot, uint fetchDword, bool denorm, float x)
{
    return xe_tfetch_exp_adjust(xe_textures_2d[xe_descriptor_index(textureSlot)].SampleBias(
        xe_samplers[xe_sampler_index(samplerSlot)], xe_tfetch_coord_1d(fetchDword, denorm, x),
        xe_tfetch_lod_bias(fetchDword)), fetchDword);
}

float4 tfetch2D(uint textureSlot, uint samplerSlot, uint fetchDword, bool denorm, float2 uv, float2 offsetTexels)
{
    return xe_tfetch_exp_adjust(xe_textures_2d[xe_descriptor_index(textureSlot)].SampleBias(
        xe_samplers[xe_sampler_index(samplerSlot)], xe_tfetch_coord_2d(fetchDword, denorm, uv, offsetTexels),
        xe_tfetch_lod_bias(fetchDword)), fetchDword);
}

float4 tfetch3D(uint textureSlot, uint samplerSlot, uint fetchDword, bool denorm, float3 uvw)
{
    return xe_tfetch_exp_adjust(xe_textures_3d[xe_descriptor_index(textureSlot)].SampleBias(
        xe_samplers[xe_sampler_index(samplerSlot)], xe_tfetch_coord_3d(fetchDword, denorm, uvw),
        xe_tfetch_lod_bias(fetchDword)), fetchDword);
}

float4 tfetchCube(uint textureSlot, uint samplerSlot, uint fetchDword, bool denorm, float3 texCoord,
                  inout CubeMapData cubeMapData)
{
    return xe_tfetch_exp_adjust(xe_textures_cube[xe_descriptor_index(textureSlot)].SampleBias(
        xe_samplers[xe_sampler_index(samplerSlot)], cubeMapData.cubeMapDirections[texCoord.z],
        xe_tfetch_lod_bias(fetchDword)), fetchDword);
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

// ---- LOD-controlled tfetch variants ----
// _bias adds the instruction bias and register LOD (setTexLOD) to a computed LOD, _level is an explicit
// LOD (UseComputedLOD=false or any VS fetch), _grad uses register gradients scaled by 2^bias.

float4 tfetch1D_bias(uint textureSlot, uint samplerSlot, uint fetchDword, bool denorm, float x, float bias)
{
    return xe_tfetch_exp_adjust(xe_textures_2d[xe_descriptor_index(textureSlot)].SampleBias(
        xe_samplers[xe_sampler_index(samplerSlot)], xe_tfetch_coord_1d(fetchDword, denorm, x),
        xe_tfetch_lod_bias(fetchDword) + bias), fetchDword);
}

float4 tfetch2D_bias(uint textureSlot, uint samplerSlot, uint fetchDword, bool denorm, float2 uv, float2 offsetTexels,
                     float bias)
{
    return xe_tfetch_exp_adjust(xe_textures_2d[xe_descriptor_index(textureSlot)].SampleBias(
        xe_samplers[xe_sampler_index(samplerSlot)], xe_tfetch_coord_2d(fetchDword, denorm, uv, offsetTexels),
        xe_tfetch_lod_bias(fetchDword) + bias), fetchDword);
}

float4 tfetch3D_bias(uint textureSlot, uint samplerSlot, uint fetchDword, bool denorm, float3 uvw, float bias)
{
    return xe_tfetch_exp_adjust(xe_textures_3d[xe_descriptor_index(textureSlot)].SampleBias(
        xe_samplers[xe_sampler_index(samplerSlot)], xe_tfetch_coord_3d(fetchDword, denorm, uvw),
        xe_tfetch_lod_bias(fetchDword) + bias), fetchDword);
}

float4 tfetchCube_bias(uint textureSlot, uint samplerSlot, uint fetchDword, bool denorm, float3 texCoord,
                       inout CubeMapData cubeMapData, float bias)
{
    return xe_tfetch_exp_adjust(xe_textures_cube[xe_descriptor_index(textureSlot)].SampleBias(
        xe_samplers[xe_sampler_index(samplerSlot)], cubeMapData.cubeMapDirections[texCoord.z],
        xe_tfetch_lod_bias(fetchDword) + bias), fetchDword);
}

float4 tfetch1D_level(uint textureSlot, uint samplerSlot, uint fetchDword, bool denorm, float x, float lod)
{
    return xe_tfetch_exp_adjust(xe_textures_2d[xe_descriptor_index(textureSlot)].SampleLevel(
        xe_samplers[xe_sampler_index(samplerSlot)], xe_tfetch_coord_1d(fetchDword, denorm, x),
        xe_tfetch_lod_bias(fetchDword) + lod), fetchDword);
}

float4 tfetch2D_level(uint textureSlot, uint samplerSlot, uint fetchDword, bool denorm, float2 uv, float2 offsetTexels,
                      float lod)
{
    return xe_tfetch_exp_adjust(xe_textures_2d[xe_descriptor_index(textureSlot)].SampleLevel(
        xe_samplers[xe_sampler_index(samplerSlot)], xe_tfetch_coord_2d(fetchDword, denorm, uv, offsetTexels),
        xe_tfetch_lod_bias(fetchDword) + lod), fetchDword);
}

float4 tfetch3D_level(uint textureSlot, uint samplerSlot, uint fetchDword, bool denorm, float3 uvw, float lod)
{
    return xe_tfetch_exp_adjust(xe_textures_3d[xe_descriptor_index(textureSlot)].SampleLevel(
        xe_samplers[xe_sampler_index(samplerSlot)], xe_tfetch_coord_3d(fetchDword, denorm, uvw),
        xe_tfetch_lod_bias(fetchDword) + lod), fetchDword);
}

float4 tfetchCube_level(uint textureSlot, uint samplerSlot, uint fetchDword, bool denorm, float3 texCoord,
                        inout CubeMapData cubeMapData, float lod)
{
    return xe_tfetch_exp_adjust(xe_textures_cube[xe_descriptor_index(textureSlot)].SampleLevel(
        xe_samplers[xe_sampler_index(samplerSlot)], cubeMapData.cubeMapDirections[texCoord.z],
        xe_tfetch_lod_bias(fetchDword) + lod), fetchDword);
}

float4 tfetch1D_grad(uint textureSlot, uint samplerSlot, uint fetchDword, bool denorm, float x, float bias,
                     float3 gradH, float3 gradV)
{
    float scale = exp2(xe_tfetch_lod_bias(fetchDword) + bias);
    float width = float((xe_fetch_dword(fetchDword + 2u) & 0xFFFFFFu) + 1u);
    float norm = denorm ? 1.0 / width : 1.0;
    return xe_tfetch_exp_adjust(xe_textures_2d[xe_descriptor_index(textureSlot)].SampleGrad(
        xe_samplers[xe_sampler_index(samplerSlot)], xe_tfetch_coord_1d(fetchDword, denorm, x),
        float2(gradH.x * scale * norm, 0.0), float2(gradV.x * scale * norm, 0.0)), fetchDword);
}

float4 tfetch2D_grad(uint textureSlot, uint samplerSlot, uint fetchDword, bool denorm, float2 uv, float2 offsetTexels,
                     float bias, float3 gradH, float3 gradV)
{
    float scale = exp2(xe_tfetch_lod_bias(fetchDword) + bias);
    float2 norm = denorm ? 1.0 / xe_tfetch_size_2d(fetchDword) : 1.0;
    return xe_tfetch_exp_adjust(xe_textures_2d[xe_descriptor_index(textureSlot)].SampleGrad(
        xe_samplers[xe_sampler_index(samplerSlot)], xe_tfetch_coord_2d(fetchDword, denorm, uv, offsetTexels),
        gradH.xy * scale * norm, gradV.xy * scale * norm), fetchDword);
}

float4 tfetch3D_grad(uint textureSlot, uint samplerSlot, uint fetchDword, bool denorm, float3 uvw, float bias,
                     float3 gradH, float3 gradV)
{
    float scale = exp2(xe_tfetch_lod_bias(fetchDword) + bias);
    float3 norm = denorm ? 1.0 / xe_tfetch_size_3d(fetchDword) : 1.0;
    return xe_tfetch_exp_adjust(xe_textures_3d[xe_descriptor_index(textureSlot)].SampleGrad(
        xe_samplers[xe_sampler_index(samplerSlot)], xe_tfetch_coord_3d(fetchDword, denorm, uvw),
        gradH * scale * norm, gradV * scale * norm), fetchDword);
}

// Cube register gradients are already in cube space and never get denormalized (translator behavior).
float4 tfetchCube_grad(uint textureSlot, uint samplerSlot, uint fetchDword, bool denorm, float3 texCoord,
                       inout CubeMapData cubeMapData, float bias, float3 gradH, float3 gradV)
{
    float scale = exp2(xe_tfetch_lod_bias(fetchDword) + bias);
    return xe_tfetch_exp_adjust(xe_textures_cube[xe_descriptor_index(textureSlot)].SampleGrad(
        xe_samplers[xe_sampler_index(samplerSlot)], cubeMapData.cubeMapDirections[texCoord.z],
        gradH * scale, gradV * scale), fetchDword);
}

// getCompTexLOD: unclamped LOD in X, sampled with linear mip filtering (the binding forces it).
float4 getCompTexLOD1D(uint textureSlot, uint samplerSlot, uint fetchDword, bool denorm, float x)
{
    return float4(xe_textures_2d[xe_descriptor_index(textureSlot)].CalculateLevelOfDetailUnclamped(
        xe_samplers[xe_sampler_index(samplerSlot)], xe_tfetch_coord_1d(fetchDword, denorm, x).xy), 0.0, 0.0, 0.0);
}

float4 getCompTexLOD2D(uint textureSlot, uint samplerSlot, uint fetchDword, bool denorm, float2 uv)
{
    return float4(xe_textures_2d[xe_descriptor_index(textureSlot)].CalculateLevelOfDetailUnclamped(
        xe_samplers[xe_sampler_index(samplerSlot)], xe_tfetch_coord_2d(fetchDword, denorm, uv, 0.0).xy), 0.0, 0.0, 0.0);
}

float4 getCompTexLOD3D(uint textureSlot, uint samplerSlot, uint fetchDword, bool denorm, float3 uvw)
{
    return float4(xe_textures_3d[xe_descriptor_index(textureSlot)].CalculateLevelOfDetailUnclamped(
        xe_samplers[xe_sampler_index(samplerSlot)], xe_tfetch_coord_3d(fetchDword, denorm, uvw)), 0.0, 0.0, 0.0);
}

float4 getCompTexLODCube(uint textureSlot, uint samplerSlot, uint fetchDword, bool denorm, float3 texCoord,
                         inout CubeMapData cubeMapData)
{
    return float4(xe_textures_cube[xe_descriptor_index(textureSlot)].CalculateLevelOfDetailUnclamped(
        xe_samplers[xe_sampler_index(samplerSlot)], cubeMapData.cubeMapDirections[texCoord.z]), 0.0, 0.0, 0.0);
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

// Xenos multiply: a +-0 or denormal factor gives +0 whatever the other one is, inf and NaN
// included (dxbc_translator_alu.cpp:85). The recompiler skips it when both operands are identical.
#define XE_FLT_MIN_NORMAL 1.17549435e-38

template<typename T> T xe_mul(T a, T b)
{
    return select(or(abs(a) < XE_FLT_MIN_NORMAL, abs(b) < XE_FLT_MIN_NORMAL), (T)0.0, a * b);
}

// Dot products apply the rule per product and add in order, as dp4/dp3/dp2add do on Xenos.
float xe_dot(float2 a, float2 b)
{
    float2 p = xe_mul(a, b);
    return p.x + p.y;
}

float xe_dot(float3 a, float3 b)
{
    float3 p = xe_mul(a, b);
    return (p.x + p.y) + p.z;
}

float xe_dot(float4 a, float4 b)
{
    float4 p = xe_mul(a, b);
    return ((p.x + p.y) + p.z) + p.w;
}

// Shader Model 3 max/min: a >= b ? a : b, so a NaN in a picks b (dxbc_translator_alu.cpp:123).
template<typename T> T xe_max(T a, T b)
{
    return select(a >= b, a, b);
}

template<typename T> T xe_min(T a, T b)
{
    return select(a < b, a, b);
}

float4 dst(float4 src0, float4 src1)
{
    float4 dest;
    dest.x = 1.0;
    dest.y = xe_mul(src0.y, src1.y);
    dest.z = src0.z;
    dest.w = src1.w;
    return dest;
}

// First component that is >= every later one, the interpreter's max4 order (interpreter.cpp:462).
float4 max4(float4 src0)
{
    if (src0.x >= src0.y && src0.x >= src0.z && src0.x >= src0.w)
        return src0.x;
    if (src0.y >= src0.z && src0.y >= src0.w)
        return src0.y;
    return src0.z >= src0.w ? src0.z : src0.w;
}

#endif
