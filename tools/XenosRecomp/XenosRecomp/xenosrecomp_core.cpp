#include "xenosrecomp.h"

#include "shader.h"
#include "shader_recompiler.h"
#include "dxc_compiler.h"
#include "ucode_fingerprint.h"
#include "xenosrecomp_abi_gen.h"

#include <cstring>
#include <fstream>
#include <mutex>
#include <numeric>
#include <thread>

namespace xenosrecomp {

namespace {

template <typename T>
void appendPod(std::vector<uint8_t>& out, const T& value)
{
    const auto* p = reinterpret_cast<const uint8_t*>(&value);
    out.insert(out.end(), p, p + sizeof(T));
}

void alignTo(std::vector<uint8_t>& out, size_t alignment)
{
    out.resize((out.size() + alignment - 1) / alignment * alignment, 0);
}

std::vector<uint8_t> blobBytes(IDxcBlob* blob)
{
    const auto* p = reinterpret_cast<const uint8_t*>(blob->GetBufferPointer());
    std::vector<uint8_t> bytes(p, p + blob->GetBufferSize());
    blob->Release();
    return bytes;
}

// Metadata-only container for shaders with none: an empty constant table gives xe_fc{reg} names, PS inputs
// are identity-wired and the PS output mask comes from the ucode. There is no definition table.
std::vector<uint8_t> buildSyntheticContainer(bool isPixelShader, uint32_t psExportMask)
{
    constexpr uint32_t containerSize = sizeof(ShaderContainer);
    constexpr uint32_t constantTableOffset = containerSize;
    constexpr uint32_t constantTableSize = sizeof(ConstantTableContainer);
    constexpr uint32_t shaderOffset = constantTableOffset + constantTableSize;
    constexpr uint32_t psInterpolatorCount = 16;

    const uint32_t shaderSize = isPixelShader
        ? sizeof(PixelShader) + psInterpolatorCount * sizeof(uint32_t)
        : sizeof(VertexShader);

    std::vector<uint8_t> data(shaderOffset + shaderSize);
    auto put32 = [&](uint32_t offset, uint32_t value)
        {
            value = byteSwap(value);
            memcpy(data.data() + offset, &value, sizeof(value));
        };

    put32(offsetof(ShaderContainer, flags), 0x102A1100 | (isPixelShader ? 0 : 1));
    put32(offsetof(ShaderContainer, virtualSize), uint32_t(data.size()));
    put32(offsetof(ShaderContainer, constantTableOffset), constantTableOffset);
    put32(offsetof(ShaderContainer, shaderOffset), shaderOffset);

    put32(constantTableOffset + offsetof(ConstantTableContainer, size), constantTableSize);
    put32(constantTableOffset + offsetof(ConstantTableContainer, constantTable) +
        offsetof(ConstantTable, size), sizeof(ConstantTable));

    if (isPixelShader)
    {
        // No pixel-position register without a container, and 16 identity interpolators (reg in bits 8-11).
        put32(shaderOffset + offsetof(Shader, fieldC), 0xFF << 8);
        put32(shaderOffset + offsetof(Shader, interpolatorInfo), psInterpolatorCount << 5);
        put32(shaderOffset + offsetof(PixelShader, outputs), psExportMask);
        for (uint32_t i = 0; i < psInterpolatorCount; i++)
            put32(shaderOffset + sizeof(PixelShader) + i * sizeof(uint32_t), i << 8);
    }

    return data;
}

// Checks the parts of an embedded container the translator reads as metadata.
void validateContainer(std::span<const uint8_t> bytes, bool isPixelShader)
{
    if (bytes.size() < sizeof(ShaderContainer))
        throw std::runtime_error("container smaller than its header");
    auto container = reinterpret_cast<const ShaderContainer*>(bytes.data());
    if ((container->flags & 0xFFFFFF00) != 0x102A1100)
        throw std::runtime_error("container signature mismatch");
    if (((container->flags & 0x1) == 0) != isPixelShader)
        throw std::runtime_error("container stage differs from the requested stage");
    const uint32_t virtualSize = container->virtualSize;
    if (virtualSize > bytes.size())
        throw std::runtime_error("container virtual size past the buffer");
    if (container->constantTableOffset == 0 ||
        container->constantTableOffset + sizeof(ConstantTableContainer) > virtualSize)
        throw std::runtime_error("constant table offset out of bounds");
    const size_t shaderSize = isPixelShader ? sizeof(PixelShader) : sizeof(VertexShader);
    if (container->shaderOffset == 0 || container->shaderOffset + shaderSize > virtualSize)
        throw std::runtime_error("shader offset out of bounds");
    if (container->definitionTableOffset != 0 && container->definitionTableOffset + sizeof(DefinitionTable) > virtualSize)
        throw std::runtime_error("definition table offset out of bounds");
}

// Drops the unwritten oVar/oPts outputs and their zero-inits; each parameter carries its own leading comma.
std::string trimVsOutputs(const std::string& hlsl, uint32_t writtenMask, bool wroteOPts)
{
    std::string s = hlsl;
    auto removeAll = [&s](const std::string& what)
        {
            size_t pos = 0;
            while ((pos = s.find(what)) != std::string::npos)
                s.erase(pos, what.size());
        };
    for (uint32_t i = 0; i < 16; i++)
    {
        if (writtenMask & (1u << i))
            continue;
        removeAll(fmt::format(",\n\t[[vk::location({0})]] out float4 oVar{0} : TEXCOORD{0}", i));
        removeAll(fmt::format("\toVar{} = 0.0;\n", i));
    }
    if (!wroteOPts)
    {
        removeAll(",\n\t[[vk::location(16)]] out float4 oPts : TEXCOORD16");
        removeAll("\toPts = 0.0;\n");
    }
    return s;
}

uint8_t filterKey(uint32_t value, uint32_t fromFetchConstant)
{
    return value == fromFetchConstant ? 0xFF : uint8_t(value);
}

std::vector<uint8_t> serializeReflection(const ShaderRecompiler& rc, const std::set<uint32_t>& usedFloats,
    bool floatsDynamic, bool containerless, bool hasTrim)
{
    XeReflHeader h{};
    h.stage = rc.isPixelShader ? 1 : 0;
    h.float_mode = floatsDynamic ? 1 : 0;

    uint16_t flags = 0;
    if (hasTrim)
        flags |= kReflHasTrim;
    if (rc.rexWroteOPts)
        flags |= kReflWritesPointSize;
    if (rc.isPixelShader && (rc.psOutputsMask & PIXEL_SHADER_OUTPUT_DEPTH))
        flags |= kReflWritesDepth;
    if (rc.rexUsesKill)
        flags |= kReflUsesKill;
    const bool pixelPos = rc.isPixelShader && rc.rexPixelPosReg < 32;
    if (pixelPos)
        flags |= kReflUsesPixelPos | kReflUsesFace;
    if (containerless)
        flags |= kReflContainerless;
    if (rc.rexPcMachine)
        flags |= kReflPcMachineFlow;
    h.flags = flags;

    for (uint32_t reg : usedFloats)
    {
        if (reg < 256)
            h.float_bitmap[reg >> 5] |= 1u << (reg & 31);
    }
    if (floatsDynamic)
        std::fill(std::begin(h.float_bitmap), std::end(h.float_bitmap), ~0u);

    h.bool_mask = rc.rexBoolMask;
    h.vs_written_ovar = rc.isPixelShader ? 0 : uint16_t(rc.rexWrittenOVarMask);
    h.ps_read_ivar = rc.isPixelShader ? uint16_t(rc.rexPsReadMask) : 0;
    h.ps_outputs = rc.isPixelShader ? uint8_t(rc.psOutputsMask & 0x1F) : 0;
    h.pixel_pos_reg = pixelPos ? uint8_t(rc.rexPixelPosReg) : 0xFF;

    if (rc.rexVfetchRefl.size() > kMaxVfetch)
        throw std::runtime_error("too many vfetches for the layout table");
    if (rc.rexBindings.size() > 32)
        throw std::runtime_error(fmt::format("{} b4 slots exceed the 32-entry descriptor index table", rc.rexBindings.size()));
    if (rc.rexLiterals.size() > 255 || rc.rexInterps.size() > 255)
        throw std::runtime_error("reflection array count overflow");
    h.vfetch_count = uint8_t(rc.rexVfetchRefl.size());
    h.binding_count = uint8_t(rc.rexBindings.size());
    h.literal_count = uint8_t(rc.rexLiterals.size());
    h.interp_count = uint8_t(rc.rexInterps.size());

    std::vector<uint8_t> out;
    appendPod(out, h);
    for (auto& v : rc.rexVfetchRefl)
        appendPod(out, XeReflVfetch{ uint16_t(v.slot), uint8_t(v.usage), uint8_t(v.usageIndex) });
    for (auto& b : rc.rexBindings)
    {
        XeReflBinding rb{};
        rb.kind = b.isSampler ? 1 : 0;
        rb.fetch_constant = uint8_t(b.fetchConstant);
        rb.dimension = b.isSampler ? 0 : uint8_t(b.dimension);
        rb.is_signed = b.isSigned ? 1 : 0;
        if (b.isSampler)
        {
            // TextureFilter 3 and AnisoFilter 7 defer to the fetch constant.
            rb.mag = filterKey(b.magFilter, 3);
            rb.min = filterKey(b.minFilter, 3);
            rb.mip = filterKey(b.mipFilter, 3);
            rb.aniso = filterKey(b.aniso, 7);
        }
        appendPod(out, rb);
    }
    for (auto& i : rc.rexInterps)
        appendPod(out, XeReflInterp{ uint8_t(i.reg), uint8_t(i.usage), uint8_t(i.usageIndex), 0 });
    for (auto& l : rc.rexLiterals)
    {
        XeReflLiteral rl{};
        rl.reg = uint16_t(l.reg);
        rl.flags = (floatsDynamic || usedFloats.count(l.reg)) ? kLiteralRead : 0;
        if (!l.hasValue)
            rl.flags |= kLiteralNoValue;
        memcpy(rl.value, l.value, sizeof(rl.value));
        appendPod(out, rl);
    }
    return out;
}

// The rect GS completes a Xenos RECTLIST parallelogram, clip_planes turns the pre-transformed b0 planes into
// SV_ClipDistance, and point_expand sizes sprites from oPts.x. All share the one VS output signature.
const char kRectGs[] = R"(
struct XeRectVertex
{
    float4 pos : SV_Position;
    float4 var[16] : TEXCOORD0;
};

[maxvertexcount(4)]
void main(triangle XeRectVertex v[3], inout TriangleStream<XeRectVertex> stream)
{
    // The corner off the longest edge is the right angle; the 4th vertex mirrors it across that edge's midpoint.
    float2 p0 = v[0].pos.xy / v[0].pos.w;
    float2 p1 = v[1].pos.xy / v[1].pos.w;
    float2 p2 = v[2].pos.xy / v[2].pos.w;
    float2 e01 = p1 - p0;
    float2 e02 = p2 - p0;
    float2 e12 = p2 - p1;
    float d01 = dot(e01, e01);
    float d02 = dot(e02, e02);
    float d12 = dot(e12, e12);
    uint corner, hypA, hypB;
    if (d01 >= d02 && d01 >= d12) { corner = 2; hypA = 0; hypB = 1; }
    else if (d02 >= d12)          { corner = 1; hypA = 0; hypB = 2; }
    else                          { corner = 0; hypA = 1; hypB = 2; }

    XeRectVertex v3;
    v3.pos = v[hypA].pos + v[hypB].pos - v[corner].pos;
    [unroll]
    for (uint i = 0; i < 16; i++)
        v3.var[i] = v[hypA].var[i] + v[hypB].var[i] - v[corner].var[i];

    // Xenos never culls rectangles and the renderer disables culling on rect pipelines, so winding is free.
    stream.Append(v[corner]);
    stream.Append(v[hypA]);
    stream.Append(v[hypB]);
    stream.Append(v3);
}
)";

const char kClipGs[] = R"(
#ifdef __spirv__
struct XePushConstants { uint64_t System; uint64_t FloatsVs; uint64_t FloatsPs; uint64_t BoolLoop; uint64_t Fetch; uint64_t IdxVs; uint64_t IdxPs; uint64_t SharedMem; uint64_t VfetchVs; };
[[vk::push_constant]] ConstantBuffer<XePushConstants> xe_push;
#define XE_UCP(i) vk::RawBufferLoad<float4>(xe_push.System + 32 + (i) * 16, 4)
#else
cbuffer xe_system_cbuffer : register(b0, space0)
{
    float4 xe_user_clip_planes[6] : packoffset(c2);
};
#define XE_UCP(i) xe_user_clip_planes[i]
#endif

struct XeClipVertexIn
{
    float4 pos : SV_Position;
    float4 var[16] : TEXCOORD0;
};

struct XeClipVertexOut
{
    float4 pos : SV_Position;
    float4 var[16] : TEXCOORD0;
    float4 clip03 : SV_ClipDistance0;
    float2 clip45 : SV_ClipDistance1;
};

[maxvertexcount(3)]
void main(triangle XeClipVertexIn v[3], inout TriangleStream<XeClipVertexOut> stream)
{
    [unroll]
    for (uint i = 0; i < 3; i++)
    {
        XeClipVertexOut o;
        o.pos = v[i].pos;
        [unroll]
        for (uint j = 0; j < 16; j++)
            o.var[j] = v[i].var[j];
        o.clip03 = float4(dot(v[i].pos, XE_UCP(0)),
                          dot(v[i].pos, XE_UCP(1)),
                          dot(v[i].pos, XE_UCP(2)),
                          dot(v[i].pos, XE_UCP(3)));
        o.clip45 = float2(dot(v[i].pos, XE_UCP(4)),
                          dot(v[i].pos, XE_UCP(5)));
        stream.Append(o);
    }
}
)";

const char kPointGs[] = R"(
#ifdef __spirv__
struct XePushConstants { uint64_t System; uint64_t FloatsVs; uint64_t FloatsPs; uint64_t BoolLoop; uint64_t Fetch; uint64_t IdxVs; uint64_t IdxPs; uint64_t SharedMem; uint64_t VfetchVs; };
[[vk::push_constant]] ConstantBuffer<XePushConstants> xe_push;
#define XE_SYS(i) vk::RawBufferLoad<float4>(xe_push.System + (i) * 16, 4)
#else
cbuffer XeSystemCb : register(b0, space0)
{
    float4 xe_sys[16];
};
#define XE_SYS(i) xe_sys[i]
#endif

struct XePointVertex
{
    float4 pos : SV_Position;
    float4 var[16] : TEXCOORD0;
    float4 pts : TEXCOORD16;
};

[maxvertexcount(4)]
void main(point XePointVertex v[1], inout TriangleStream<XePointVertex> stream)
{
    // oPts.x is the vertex diameter in pixels clamped to c10.zw, and 0 selects the constant size in c10.xy.
    // The NDC half extent is size * |c9.xy|, times w because the offset is applied before the divide.
    float2 size_px;
    if (v[0].pts.x > 0.0)
        size_px = clamp(v[0].pts.x, XE_SYS(10).z, XE_SYS(10).w).xx;
    else
        size_px = XE_SYS(10).xy;
    float2 half_ndc = size_px * abs(XE_SYS(9).xy);
    float2 corners[4] = { float2(-1.0, -1.0), float2(1.0, -1.0),
                          float2(-1.0, 1.0),  float2(1.0, 1.0) };
    [unroll]
    for (uint i = 0; i < 4; i++)
    {
        XePointVertex o = v[0];
        o.pos.xy += corners[i] * half_ndc * v[0].pos.w;
        stream.Append(o);
    }
}
)";

const char kBlitVs[] = R"(
void main(in uint vid : SV_VertexID, out float4 pos : SV_Position, out float2 uv : TEXCOORD0)
{
    uv = float2((vid << 1) & 2, vid & 2);
    pos = float4(uv * float2(2.0, -2.0) + float2(-1.0, 1.0), 0.0, 1.0);
}
)";

const char kBlitPs[] = R"(
#ifdef __spirv__
struct XePushConstants { uint64_t System; uint64_t FloatsVs; uint64_t FloatsPs; uint64_t BoolLoop; uint64_t Fetch; uint64_t IdxVs; uint64_t IdxPs; uint64_t SharedMem; uint64_t VfetchVs; };
[[vk::push_constant]] ConstantBuffer<XePushConstants> xe_push;
#define XE_IDX(i) vk::RawBufferLoad<uint>(xe_push.IdxPs + (i) * 4)
#else
cbuffer XeDescriptorIndices : register(b4, space0)
{
    uint4 xe_descriptor_indices[8];
};
#define XE_IDX(i) (xe_descriptor_indices[(i) >> 2][(i) & 3])
#endif
Texture2DArray<float4> xe_textures2d[] : register(t0, space1);
SamplerState xe_samplers[] : register(s0, space0);

void main(in float4 pos : SV_Position, in float2 uv : TEXCOORD0, out float4 color : SV_Target0)
{
    color = xe_textures2d[XE_IDX(0)].SampleLevel(xe_samplers[0], float3(uv, 0.0), 0.0);
    // b4 slot 1 is an optional 256x3 R16_UNORM gamma LUT (rows R, G, B), 0 = identity.
    uint lut = XE_IDX(1);
    if (lut != 0)
    {
        uint3 idx = uint3(saturate(color.rgb) * 255.0 + 0.5);
        color.r = xe_textures2d[lut].Load(int4(idx.r, 0, 0, 0)).x;
        color.g = xe_textures2d[lut].Load(int4(idx.g, 1, 0, 0)).x;
        color.b = xe_textures2d[lut].Load(int4(idx.b, 2, 0, 0)).x;
    }
    // b4 slot 2 is an optional warmth grade strength as float bits, 0 = off; it trades blue for red and green.
    uint warmthBits = XE_IDX(2);
    if (warmthBits != 0)
    {
        float w = asfloat(warmthBits);
        color.rgb *= lerp(float3(1.0, 1.0, 1.0), float3(1.12, 1.06, 0.82), saturate(w));
        if (w > 1.0)
            color.rgb *= lerp(float3(1.0, 1.0, 1.0), float3(1.12, 1.06, 0.82), saturate(w - 1.0));
    }
    color.a = 1.0;
}
)";

const BuiltinShader kBuiltins[] = {
    { "rect_expand", kPackGS, kRectGs },
    { "point_expand", kPackGS, kPointGs },
    { "clip_planes", kPackGS, kClipGs },
    { "blit", kPackBlitVS, kBlitVs },
    { "blit", kPackBlitPS, kBlitPs },
};

DxcCompiler& threadCompiler()
{
    thread_local DxcCompiler compiler;
    return compiler;
}

bool dxilSigned(const std::vector<uint8_t>& dxil)
{
    // The container digest (bytes 4-19) stays zero unless dxil.dll validated and signed the blob.
    if (dxil.size() < 32)
        return false;
    for (size_t i = 4; i < 20; i++)
    {
        if (dxil[i] != 0)
            return true;
    }
    return false;
}

std::vector<uint8_t> compileOrThrow(const std::string& hlsl, bool pixel, bool spirv, const char* what,
    const wchar_t* target = nullptr)
{
    DxcCompiler& dxc = threadCompiler();
    IDxcBlob* blob = dxc.compile(hlsl, pixel, false, spirv, target);
    if (blob == nullptr)
    {
        std::string first = dxc.lastError.substr(0, dxc.lastError.find('\n'));
        throw std::runtime_error(fmt::format("{} compile failed: {}", what, first));
    }
    std::vector<uint8_t> bytes = blobBytes(blob);
    if (!spirv && !dxilSigned(bytes))
        throw std::runtime_error(fmt::format("{} is not signed (dxil.dll missing?)", what));
    return bytes;
}

std::string hex16(uint64_t v)
{
    return fmt::format("{:016X}", v);
}

}  // namespace

std::vector<XeReflLiteral> ReadLiterals(Stage stage, std::span<const uint8_t> container, std::span<const uint8_t> physical)
{
    std::vector<XeReflLiteral> literals;
    if (container.size() < sizeof(ShaderContainer))
        return literals;
    auto c = reinterpret_cast<const ShaderContainer*>(container.data());
    const uint32_t tableOffset = c->definitionTableOffset;
    if (tableOffset == 0 || tableOffset + sizeof(DefinitionTable) > container.size())
        return literals;

    // Float4 definitions are {register u16, count u16, physical offset u32} pairs up to a zero dword.
    const uint8_t* end = container.data() + container.size();
    auto definitions = reinterpret_cast<const DefinitionTable*>(container.data() + tableOffset)->definitions;
    while (reinterpret_cast<const uint8_t*>(definitions + 2) <= end && *definitions != 0)
    {
        auto definition = reinterpret_cast<const Float4Definition*>(definitions);
        const uint32_t registers = (uint32_t(definition->count) + 3) / 4;
        const uint8_t* base = nullptr;
        size_t available = 0;
        if (!physical.empty())
        {
            base = physical.data();
            available = physical.size();
        }
        else if (c->virtualSize <= container.size())
        {
            base = container.data() + c->virtualSize;
            available = container.size() - c->virtualSize;
        }
        const bool inRange = base != nullptr && size_t(definition->physicalOffset) + registers * 16 <= available;
        for (uint32_t i = 0; i < registers; i++)
        {
            XeReflLiteral literal{};
            literal.reg = uint16_t(definition->registerIndex + i - (stage == Stage::kPixel ? 256 : 0));
            literal.flags = inRange ? 0 : kLiteralNoValue;
            if (inRange)
            {
                auto words = reinterpret_cast<const be<uint32_t>*>(base + definition->physicalOffset) + i * 4;
                for (uint32_t k = 0; k < 4; k++)
                    literal.value[k] = words[k].get();
            }
            literals.push_back(literal);
        }
        definitions += 2;
    }
    return literals;
}

std::string_view CommonHeader()
{
    return std::string_view(reinterpret_cast<const char*>(kXrCommonHeader), sizeof(kXrCommonHeader));
}

std::span<const BuiltinShader> Builtins()
{
    return kBuiltins;
}

uint64_t BuiltinKey(std::string_view name)
{
    return XXH3_64bits(name.data(), name.size());
}

uint64_t AbiHash()
{
    static const uint64_t hash = []
        {
            XXH3_state_t* state = XXH3_createState();
            XXH3_64bits_reset(state);
            auto add = [&](const void* data, size_t size) { XXH3_64bits_update(state, data, size); };
            auto addString = [&](std::string_view s)
                {
                    uint64_t size = s.size();
                    add(&size, sizeof(size));
                    add(s.data(), s.size());
                };

            addString(CommonHeader());
            addString(std::string_view(reinterpret_cast<const char*>(kXrBaseFile), sizeof(kXrBaseFile)));
            const uint64_t sourceHash = kXrTranslatorSourceHash;
            add(&sourceHash, sizeof(sourceHash));
            addString(threadCompiler().version());

            // Every argument list the pack compiles with: VS/PS for both targets and the GS profile.
            const struct { bool pixel, spirv; const wchar_t* target; } lists[] = {
                { false, false, nullptr }, { true, false, nullptr }, { false, true, nullptr }, { true, true, nullptr },
                { false, false, L"-T gs_6_0" }, { false, true, L"-T gs_6_0" },
            };
            for (auto& l : lists)
            {
                std::string joined;
                for (const wchar_t* arg : DxcCompiler::arguments(l.pixel, false, l.spirv, l.target))
                {
                    for (const wchar_t* c = arg; *c != 0; c++)
                        joined += char(*c);
                    joined += '\n';
                }
                addString(joined);
            }

            const uint32_t constants[] = { kB0Size, kPushSlots, kTextureHeap, kSamplerHeap, kSharedMemSize, kMaxVfetch };
            add(constants, sizeof(constants));

            uint64_t value = XXH3_64bits_digest(state);
            XXH3_freeState(state);
            return value;
        }();
    return hash;
}

uint64_t Fingerprint(std::span<const uint32_t> ucode, bool big_endian)
{
    return ucodeFingerprint2(ucode.data(), ucode.size(), big_endian);
}

uint32_t ReadVfetchRecords(std::span<const uint32_t> patched_ucode, bool big_endian, std::span<XeVfetchRecord> out)
{
    struct Walk
    {
        std::span<XeVfetchRecord> out;
        uint32_t count = 0;
        bool haveFull = false;
        uint32_t fetchDwordIndex = 0;
        uint32_t stride = 0;
    } walk{ out };

    ucodeVisitVfetchSlots(patched_ucode.data(), patched_ucode.size(), big_endian,
        [](uint32_t, const uint32_t* d, void* context)
        {
            auto& w = *static_cast<Walk*>(context);
            union
            {
                VertexFetchInstruction vf;
                struct { uint32_t d0, d1, d2; };
            };
            d0 = d[0];
            d1 = d[1];
            d2 = d[2];

            // A mini fetch takes its fetch constant and stride from the last full fetch, as the hardware does.
            if (!vf.isMiniFetch)
            {
                w.haveFull = true;
                w.fetchDwordIndex = uint32_t(vf.constIndex) * 6 + uint32_t(vf.constIndexSelect) * 2;
                w.stride = vf.stride;
            }

            if (w.count < w.out.size())
            {
                XeVfetchRecord& r = w.out[w.count];
                r.word0 = (w.fetchDwordIndex & 0xFF) | ((uint32_t(vf.format) & 0x3F) << 8) |
                    (vf.formatCompAll ? kVfetchSigned : 0) | (vf.numFormatAll ? kVfetchInteger : 0) |
                    (vf.signedRfModeAll ? kVfetchRfNoZero : 0) | (w.haveFull ? kVfetchValid : 0);
                r.stride_dwords = w.stride;
                r.offset_dwords = int32_t(vf.offset);
                r.dst_swizzle = vf.dstSwizzle;
            }
            w.count++;
        }, &walk);

    return walk.count;
}

TranslateResult Translate(const TranslateInput& in)
{
    TranslateResult result;
    try
    {
        if (in.ucode.empty())
            throw std::runtime_error("empty ucode");

        const bool isPixelShader = in.stage == Stage::kPixel;
        const auto* codeBytes = reinterpret_cast<const uint8_t*>(in.ucode.data());
        const uint32_t codeSize = uint32_t(in.ucode.size() * sizeof(uint32_t));
        result.runtime_ucode_hash = XXH3_64bits(codeBytes, codeSize);
        result.fp2 = ucodeFingerprint2(in.ucode.data(), in.ucode.size(), true);

        std::vector<uint8_t> synthetic;
        const uint8_t* container = nullptr;
        size_t containerSize = 0;
        const bool containerless = in.container.empty();
        if (containerless)
        {
            const uint32_t exportMask = isPixelShader ? ucodePsExportMask(in.ucode.data(), in.ucode.size(), true) : 0;
            synthetic = buildSyntheticContainer(isPixelShader, exportMask);
            container = synthetic.data();
            containerSize = synthetic.size();
        }
        else
        {
            validateContainer(in.container, isPixelShader);
            container = in.container.data();
            containerSize = in.container.size();
        }

        const std::string_view include = CommonHeader();
        thread_local ShaderRecompiler recompiler;

        auto setup = [&]
            {
                recompiler = {};
                recompiler.rexglueMode = true;
                recompiler.rexAbsoluteFloatFile = containerless;
                recompiler.rexCodeOverride = codeBytes;
                recompiler.rexCodeOverrideSize = codeSize;
                recompiler.rexContainerSize = containerSize;
                recompiler.rexPhysicalData = in.physical.empty() ? nullptr : in.physical.data();
                recompiler.rexPhysicalSize = in.physical.size();
            };

        // Pass 1 discovers the float registers read, pass 2 emits the absolute layout b1 is uploaded in.
        setup();
        recompiler.recompile(container, include);
        const std::set<uint32_t> usedFloats = std::move(recompiler.rexUsedFloatConstants);
        const bool floatsDynamic = recompiler.rexFloatsDynamic;
        // The named layout of a dynamic shader has no member for unnamed registers, so they get xe_fc{N} members.
        std::set<uint32_t> extraFloats;
        if (floatsDynamic && !containerless)
        {
            for (uint32_t reg : usedFloats)
            {
                if (recompiler.float4Constants.find(reg) == recompiler.float4Constants.end())
                    extraFloats.insert(reg);
            }
        }

        setup();
        recompiler.rexExtraFloatRegs = extraFloats;
        if (!floatsDynamic && !containerless)
        {
            for (uint32_t reg : usedFloats)
                recompiler.rexFloatRank.emplace(reg, reg);
        }
        recompiler.recompile(container, include);

        if (recompiler.isPixelShader != isPixelShader)
            throw std::runtime_error("container stage differs from the requested stage");

        const std::string& hlsl = recompiler.out;
        if (in.targets & kDxil)
            result.dxil = compileOrThrow(hlsl, isPixelShader, false, "DXIL");
        if (in.targets & kSpirv)
            result.spirv = compileOrThrow(hlsl, isPixelShader, true, "SPIR-V");

        // Container-less shaders have no interpolator table to prove a trim safe.
        std::string hlslTrim;
        const bool wantTrim = (in.targets & kTrim) && !isPixelShader && !containerless &&
            (recompiler.rexWrittenOVarMask != 0xFFFFu || !recompiler.rexWroteOPts);
        if (wantTrim)
        {
            hlslTrim = trimVsOutputs(hlsl, recompiler.rexWrittenOVarMask, recompiler.rexWroteOPts);
            if (in.targets & kDxil)
                result.dxil_trim = compileOrThrow(hlslTrim, false, false, "trimmed DXIL");
            if (in.targets & kSpirv)
                result.spirv_trim = compileOrThrow(hlslTrim, false, true, "trimmed SPIR-V");
        }

        result.reflection = serializeReflection(recompiler, usedFloats, floatsDynamic, containerless, wantTrim);
        if (in.keep_hlsl)
        {
            result.hlsl = hlsl;
            result.hlsl_trim = std::move(hlslTrim);
        }
        result.ok = true;
    }
    catch (const std::exception& e)
    {
        result.ok = false;
        result.error = e.what();
        result.dxil.clear();
        result.spirv.clear();
        result.dxil_trim.clear();
        result.spirv_trim.clear();
        result.reflection.clear();
    }
    return result;
}

// ---- Reflection ----

size_t ReflectionSize(const XeReflHeader& h)
{
    return sizeof(XeReflHeader) + h.vfetch_count * sizeof(XeReflVfetch) + h.binding_count * sizeof(XeReflBinding) +
        h.interp_count * sizeof(XeReflInterp) + h.literal_count * sizeof(XeReflLiteral);
}

bool ParseReflection(std::span<const uint8_t> bytes, ReflectionView& out)
{
    if (bytes.size() < sizeof(XeReflHeader))
        return false;
    auto h = reinterpret_cast<const XeReflHeader*>(bytes.data());
    if (bytes.size() < ReflectionSize(*h))
        return false;
    const uint8_t* p = bytes.data() + sizeof(XeReflHeader);
    out.header = h;
    out.vfetches = { reinterpret_cast<const XeReflVfetch*>(p), h->vfetch_count };
    p += h->vfetch_count * sizeof(XeReflVfetch);
    out.bindings = { reinterpret_cast<const XeReflBinding*>(p), h->binding_count };
    p += h->binding_count * sizeof(XeReflBinding);
    out.interps = { reinterpret_cast<const XeReflInterp*>(p), h->interp_count };
    p += h->interp_count * sizeof(XeReflInterp);
    out.literals = { reinterpret_cast<const XeReflLiteral*>(p), h->literal_count };
    return true;
}

// ---- Pack writer ----

std::vector<uint8_t> BuildPack(std::vector<PackInput> inputs, int zstd_level, std::string* error)
{
    auto fail = [&](std::string message)
        {
            if (error != nullptr)
                *error = std::move(message);
            return std::vector<uint8_t>{};
        };

    std::sort(inputs.begin(), inputs.end(), [](const PackInput& a, const PackInput& b)
        {
            return a.stage != b.stage ? a.stage < b.stage : a.fp2 < b.fp2;
        });
    for (size_t i = 1; i < inputs.size(); i++)
    {
        if (inputs[i].stage == inputs[i - 1].stage && inputs[i].fp2 == inputs[i - 1].fp2)
            return fail(fmt::format("duplicate pack key stage {} fp2 {:016X}", inputs[i].stage, inputs[i].fp2));
    }

    // One independent zstd frame per blob, compressed in parallel.
    struct Frame
    {
        const std::vector<uint8_t>* source;
        std::vector<uint8_t> compressed;
    };
    std::vector<Frame> frames;
    frames.reserve(inputs.size() * 4);
    for (auto& input : inputs)
    {
        for (auto* blob : { &input.dxil, &input.spirv, &input.dxil_trim, &input.spirv_trim })
            frames.push_back({ blob, {} });
    }
    std::atomic<bool> compressFailed = false;
    std::for_each(std::execution::par, frames.begin(), frames.end(), [&](Frame& f)
        {
            if (f.source->empty())
                return;
            f.compressed.resize(ZSTD_compressBound(f.source->size()));
            size_t size = ZSTD_compress(f.compressed.data(), f.compressed.size(), f.source->data(), f.source->size(),
                zstd_level);
            if (ZSTD_isError(size))
                compressFailed = true;
            else
                f.compressed.resize(size);
        });
    if (compressFailed)
        return fail("zstd compression failed");

    struct AliasRow
    {
        uint64_t hash;
        uint32_t entry;
        uint8_t stage;
    };
    std::vector<AliasRow> aliases;
    for (uint32_t i = 0; i < inputs.size(); i++)
    {
        for (uint64_t hash : inputs[i].aliases)
            aliases.push_back({ hash, i, inputs[i].stage });
    }
    std::sort(aliases.begin(), aliases.end(), [](const AliasRow& a, const AliasRow& b) { return a.hash < b.hash; });
    aliases.erase(std::unique(aliases.begin(), aliases.end(), [](const AliasRow& a, const AliasRow& b)
        {
            return a.hash == b.hash && a.entry == b.entry;
        }), aliases.end());
    for (size_t i = 1; i < aliases.size(); i++)
    {
        if (aliases[i].hash == aliases[i - 1].hash)
            return fail(fmt::format("runtime hash {:016X} resolves to two entries", aliases[i].hash));
    }

    std::vector<uint8_t> refl;
    std::vector<uint32_t> reflOffsets;
    for (auto& input : inputs)
    {
        alignTo(refl, 16);
        reflOffsets.push_back(uint32_t(refl.size()));
        refl.insert(refl.end(), input.reflection.begin(), input.reflection.end());
    }

    std::vector<uint8_t> blobs;
    std::vector<PackEntry> entries(inputs.size());
    uint32_t packFlags = 0;
    uint32_t builtinCount = 0;
    for (size_t i = 0; i < inputs.size(); i++)
    {
        PackEntry& e = entries[i];
        e = {};
        e.fp2 = inputs[i].fp2;
        e.stage = inputs[i].stage;
        e.flags = inputs[i].flags;
        e.refl_offset = reflOffsets[i];
        PackBlob* targets[] = { &e.dxil, &e.spirv, &e.dxil_trim, &e.spirv_trim };
        for (size_t k = 0; k < 4; k++)
        {
            const Frame& f = frames[i * 4 + k];
            if (f.compressed.empty())
                continue;
            if (blobs.size() + f.compressed.size() > UINT32_MAX)
                return fail("blob region exceeds 4 GB");
            targets[k]->offset = uint32_t(blobs.size());
            targets[k]->csize = uint32_t(f.compressed.size());
            blobs.insert(blobs.end(), f.compressed.begin(), f.compressed.end());
        }
        if (e.dxil.csize != 0)
            packFlags |= kPackHasDxil;
        if (e.spirv.csize != 0)
            packFlags |= kPackHasSpirv;
        if (e.dxil_trim.csize != 0 || e.spirv_trim.csize != 0)
            packFlags |= kPackHasTrim;
        if (e.stage >= kPackGS)
            builtinCount++;
    }

    std::vector<uint8_t> file(sizeof(PackHeader), 0);
    PackHeader h{};
    memcpy(h.magic, "CC2XPACK", 8);
    h.version = kPackVersion;
    h.fingerprint_ver = kFingerprintVersion;
    h.flags = packFlags;
    h.abi_hash = AbiHash();
    h.b0_size = kB0Size;
    h.push_slots = kPushSlots;
    h.texture_heap = kTextureHeap;
    h.sampler_heap = kSamplerHeap;
    h.shared_mem_size = kSharedMemSize;
    h.max_vfetch = kMaxVfetch;
    h.entry_count = uint32_t(entries.size());
    h.alias_count = uint32_t(aliases.size());
    h.builtin_count = builtinCount;

    alignTo(file, 16);
    h.index_offset = file.size();
    for (auto& e : entries)
        appendPod(file, e);
    alignTo(file, 16);
    h.alias_offset = file.size();
    for (auto& a : aliases)
    {
        PackAlias pa{};
        pa.runtime_ucode_hash = a.hash;
        pa.entry_index = a.entry;
        pa.stage = a.stage;
        appendPod(file, pa);
    }
    alignTo(file, 16);
    h.refl_offset = file.size();
    h.refl_size = refl.size();
    file.insert(file.end(), refl.begin(), refl.end());
    alignTo(file, 16);
    h.blob_offset = file.size();
    h.blob_size = blobs.size();
    file.insert(file.end(), blobs.begin(), blobs.end());
    alignTo(file, 16);

    memcpy(file.data(), &h, sizeof(h));
    h.file_xxh3 = XXH3_64bits(file.data() + sizeof(PackHeader), file.size() - sizeof(PackHeader));
    memcpy(file.data(), &h, sizeof(h));
    return file;
}

// ---- Pack reader ----

bool Pack::Open(std::span<const uint8_t> bytes, std::string* error, bool check_abi)
{
    auto fail = [&](std::string message)
        {
            if (error != nullptr)
                *error = std::move(message);
            header_ = nullptr;
            return false;
        };

    if (bytes.size() < sizeof(PackHeader))
        return fail("file smaller than the pack header");
    auto h = reinterpret_cast<const PackHeader*>(bytes.data());
    if (memcmp(h->magic, "CC2XPACK", 8) != 0)
        return fail("magic");
    if (h->version != kPackVersion)
        return fail(fmt::format("version {} (expected {})", h->version, kPackVersion));
    if (h->fingerprint_ver != kFingerprintVersion)
        return fail(fmt::format("fingerprint_ver {} (expected {})", h->fingerprint_ver, kFingerprintVersion));
    const struct { const char* name; uint32_t have, want; } constants[] = {
        { "b0_size", h->b0_size, kB0Size },
        { "push_slots", h->push_slots, kPushSlots },
        { "texture_heap", h->texture_heap, kTextureHeap },
        { "sampler_heap", h->sampler_heap, kSamplerHeap },
        { "shared_mem_size", h->shared_mem_size, kSharedMemSize },
        { "max_vfetch", h->max_vfetch, kMaxVfetch },
    };
    for (auto& c : constants)
    {
        if (c.have != c.want)
            return fail(fmt::format("{} {} (expected {})", c.name, c.have, c.want));
    }
    if (check_abi && h->abi_hash != AbiHash())
        return fail(fmt::format("abi_hash {:016X} (expected {:016X})", h->abi_hash, AbiHash()));
    if (XXH3_64bits(bytes.data() + sizeof(PackHeader), bytes.size() - sizeof(PackHeader)) != h->file_xxh3)
        return fail("file_xxh3");

    auto inRange = [&](uint64_t offset, uint64_t size) { return offset <= bytes.size() && size <= bytes.size() - offset; };
    if (!inRange(h->index_offset, uint64_t(h->entry_count) * sizeof(PackEntry)))
        return fail("index_offset");
    if (!inRange(h->alias_offset, uint64_t(h->alias_count) * sizeof(PackAlias)))
        return fail("alias_offset");
    if (!inRange(h->refl_offset, h->refl_size))
        return fail("refl_offset");
    if (!inRange(h->blob_offset, h->blob_size))
        return fail("blob_offset");

    bytes_ = bytes;
    header_ = h;
    entries_ = { reinterpret_cast<const PackEntry*>(bytes.data() + h->index_offset), h->entry_count };
    aliases_ = { reinterpret_cast<const PackAlias*>(bytes.data() + h->alias_offset), h->alias_count };
    for (auto& e : entries_)
    {
        for (const PackBlob* b : { &e.dxil, &e.spirv, &e.dxil_trim, &e.spirv_trim })
        {
            if (uint64_t(b->offset) + b->csize > h->blob_size)
                return fail(fmt::format("blob bounds of entry {:016X}", e.fp2));
        }
        if (e.refl_offset + sizeof(XeReflHeader) > h->refl_size)
            return fail(fmt::format("reflection bounds of entry {:016X}", e.fp2));
    }
    for (auto& a : aliases_)
    {
        if (a.entry_index >= h->entry_count)
            return fail("alias entry_index");
    }
    return true;
}

bool Pack::OpenFile(const std::filesystem::path& path, std::string* error, bool check_abi)
{
    std::ifstream file(path, std::ios::binary);
    if (!file)
    {
        if (error != nullptr)
            *error = "cannot open " + path.string();
        return false;
    }
    owned_.assign(std::istreambuf_iterator<char>(file), std::istreambuf_iterator<char>());
    return Open(owned_, error, check_abi);
}

const PackEntry* Pack::Find(uint8_t stage, uint64_t fp2) const
{
    auto it = std::lower_bound(entries_.begin(), entries_.end(), std::make_pair(stage, fp2),
        [](const PackEntry& e, const std::pair<uint8_t, uint64_t>& key)
        {
            return e.stage != key.first ? e.stage < key.first : e.fp2 < key.second;
        });
    if (it == entries_.end() || it->stage != stage || it->fp2 != fp2)
        return nullptr;
    return &*it;
}

const PackEntry* Pack::FindAlias(uint64_t runtime_ucode_hash) const
{
    auto it = std::lower_bound(aliases_.begin(), aliases_.end(), runtime_ucode_hash,
        [](const PackAlias& a, uint64_t key) { return a.runtime_ucode_hash < key; });
    if (it == aliases_.end() || it->runtime_ucode_hash != runtime_ucode_hash)
        return nullptr;
    return &entries_[it->entry_index];
}

const PackEntry* Pack::FindBuiltin(uint8_t stage, std::string_view name) const
{
    return Find(stage, BuiltinKey(name));
}

std::span<const uint8_t> Pack::Reflection(const PackEntry& entry) const
{
    const uint8_t* base = bytes_.data() + header_->refl_offset + entry.refl_offset;
    const size_t available = header_->refl_size - entry.refl_offset;
    auto h = reinterpret_cast<const XeReflHeader*>(base);
    return { base, (std::min)(ReflectionSize(*h), available) };
}

bool Pack::Decompress(const PackEntry& entry, BlobKind kind, std::vector<uint8_t>& out, std::string* error) const
{
    const PackBlob& blob = kind == BlobKind::kDxil ? entry.dxil
        : kind == BlobKind::kSpirv ? entry.spirv
        : kind == BlobKind::kDxilTrim ? entry.dxil_trim
        : entry.spirv_trim;
    out.clear();
    if (blob.csize == 0)
        return true;
    const uint8_t* src = bytes_.data() + header_->blob_offset + blob.offset;
    unsigned long long size = ZSTD_getFrameContentSize(src, blob.csize);
    if (size == ZSTD_CONTENTSIZE_ERROR || size == ZSTD_CONTENTSIZE_UNKNOWN || size > (1ull << 30))
    {
        if (error != nullptr)
            *error = "zstd frame header";
        return false;
    }
    out.resize(size_t(size));
    size_t got = ZSTD_decompress(out.data(), out.size(), src, blob.csize);
    if (ZSTD_isError(got) || got != size)
    {
        if (error != nullptr)
            *error = ZSTD_isError(got) ? ZSTD_getErrorName(got) : "zstd size mismatch";
        out.clear();
        return false;
    }
    return true;
}

// ---- Disk cache ----

std::filesystem::path CacheDirectory(const std::filesystem::path& root)
{
    return root / hex16(AbiHash());
}

std::filesystem::path CacheFile(const std::filesystem::path& dir, Stage stage, uint64_t fp2)
{
    return dir / fmt::format("{}_{:016X}.xsc", stage == Stage::kPixel ? "ps" : "vs", fp2);
}

namespace {

bool writeAtomically(const std::filesystem::path& path, std::span<const uint8_t> bytes, std::string* error)
{
    std::error_code ec;
    std::filesystem::create_directories(path.parent_path(), ec);
    const std::filesystem::path temp = path.string() +
        fmt::format(".{:x}.tmp", std::hash<std::thread::id>{}(std::this_thread::get_id()));
    {
        std::ofstream file(temp, std::ios::binary | std::ios::trunc);
        if (!file.write(reinterpret_cast<const char*>(bytes.data()), std::streamsize(bytes.size())))
        {
            if (error != nullptr)
                *error = "cannot write " + temp.string();
            return false;
        }
    }
    std::filesystem::rename(temp, path, ec);
    if (ec)
    {
        std::filesystem::remove(temp, ec);
        if (error != nullptr)
            *error = "cannot rename into " + path.string();
        return false;
    }
    return true;
}

}  // namespace

bool CacheStore(const std::filesystem::path& dir, Stage stage, const TranslateResult& result, std::string* error)
{
    if (!result.ok)
    {
        if (error != nullptr)
            *error = "refusing to cache a failed translation";
        return false;
    }
    PackInput input;
    input.stage = uint8_t(stage);
    input.fp2 = result.fp2;
    ReflectionView view;
    if (ParseReflection(result.reflection, view))
    {
        if (view.header->flags & kReflHasTrim)
            input.flags |= kEntryHasTrim;
        if (view.header->flags & kReflContainerless)
            input.flags |= kEntryContainerless;
    }
    input.reflection = result.reflection;
    input.dxil = result.dxil;
    input.spirv = result.spirv;
    input.dxil_trim = result.dxil_trim;
    input.spirv_trim = result.spirv_trim;
    input.aliases.push_back(result.runtime_ucode_hash);

    std::vector<PackInput> inputs;
    inputs.push_back(std::move(input));
    std::vector<uint8_t> bytes = BuildPack(std::move(inputs), 3, error);
    if (bytes.empty())
        return false;
    return writeAtomically(CacheFile(dir, stage, result.fp2), bytes, error);
}

bool CacheLoad(const std::filesystem::path& dir, Stage stage, uint64_t fp2, Pack& out, std::string* error)
{
    const std::filesystem::path path = CacheFile(dir, stage, fp2);
    std::error_code ec;
    if (!std::filesystem::exists(path, ec))
    {
        if (error != nullptr)
            *error = "miss";
        return false;
    }
    if (!out.OpenFile(path, error))
        return false;
    if (out.Find(uint8_t(stage), fp2) == nullptr)
    {
        if (error != nullptr)
            *error = "cache file holds a different key";
        return false;
    }
    return true;
}

bool CacheWriteMiss(const std::filesystem::path& dir, Stage stage, uint64_t runtime_ucode_hash,
                    std::span<const uint32_t> ucode, std::span<const uint8_t> container, std::string* error)
{
    const std::filesystem::path base = dir / "misses" / fmt::format("{:016X}.{}", runtime_ucode_hash,
        stage == Stage::kPixel ? "ps" : "vs");
    const auto* code = reinterpret_cast<const uint8_t*>(ucode.data());
    if (!writeAtomically(base.string() + ".ucode", { code, ucode.size() * sizeof(uint32_t) }, error))
        return false;
    if (!container.empty() && !writeAtomically(base.string() + ".container", container, error))
        return false;
    return true;
}

}  // namespace xenosrecomp
