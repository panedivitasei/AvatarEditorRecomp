#pragma once

#include "shader.h"
#include "shader_code.h"

struct StringBuffer
{
    std::string out;

    template<class... Args>
    void print(fmt::format_string<Args...> fmt, Args&&... args)
    {
        fmt::vformat_to(std::back_inserter(out), fmt.get(), fmt::make_format_args(args...));
    }

    template<class... Args>
    void println(fmt::format_string<Args...> fmt, Args&&... args)
    {
        fmt::vformat_to(std::back_inserter(out), fmt.get(), fmt::make_format_args(args...));
        out += '\n';
    }
};

struct VertexElementInfo
{
    DeclUsage usage;
    uint32_t usageIndex; // original index (drives the g_SwappedTexcoords bit)
    std::string name;    // unique HLSL parameter name (duplicate semantics get a suffix)
};

// REXGLUE mode: b4 descriptor-index slots replicate the DxbcShaderTranslator allocation
// (FindOrAddTextureBinding / FindOrAddSamplerBinding order and keys), so the runtime's per-draw upload
// from the translated shader's binding lists feeds the same slots.
// Per tfetch: sampler first, then unsigned+signed texture pairs (3D adds a stacked-2D pair); the
// native shader reads the unsigned slot only (signed-texture selection is a known gap).
struct RexglueBinding
{
    uint32_t slot;          // index into the b4 descriptor-index array
    bool isSampler;         // false = texture view
    uint32_t fetchConstant; // tfetch constIndex (guest texture fetch constant)
    uint32_t dimension;     // FetchOpDimension of the binding
    bool isSigned;
    // Sampler key after the translator's normalization (raw TextureFilter / AnisoFilter values), 0 for textures.
    uint32_t magFilter = 0, minFilter = 0, mipFilter = 0, aniso = 0;
};

// Reflection facts the pack and the fallback translator serialize (pack_contract.md 2.3).
struct RexVfetchRefl
{
    uint32_t slot;       // clause instruction slot, the container VertexElement address
    uint32_t usage;      // DeclUsage, 0xFF when the container names no element there
    uint32_t usageIndex;
};

struct RexInterpRefl
{
    uint32_t reg, usage, usageIndex;
};

struct RexLiteralRefl
{
    uint32_t reg;        // stage-local register (PS registers 0-255)
    uint32_t value[4];   // raw bits, host order
    bool hasValue;       // false when the literal bits were not in the provided data
};

struct ShaderRecompiler : StringBuffer
{
    uint32_t indentation = 0;
    bool isPixelShader = false;
    const uint8_t* constantTableData = nullptr;
    std::unordered_map<uint32_t, VertexElementInfo> vertexElements;
    std::unordered_map<uint32_t, std::string> interpolators;
    std::unordered_map<uint32_t, const ConstantInfo*> float4Constants;
    std::unordered_map<uint32_t, const char*> boolConstants;
    std::unordered_map<uint32_t, const char*> samplers;
    std::unordered_map<uint32_t, uint32_t> ifEndLabels;
    // Structured if/else emission: instruction indices where a "} else {" is
    // emitted (then-branch close is part of the event, not ifEndLabels).
    std::unordered_set<uint32_t> elseLabels;
    uint32_t specConstantsMask = 0;

    // Divergent-flow tfetch tracking: Sample()'s implicit derivatives
    // inside per-pixel (p0) control
    // flow deadlock Intel's helper-lane logic when a quad diverges, fetches
    // reachable under divergence must use the SampleLevel(0) helper variants.
    // divergentPcs = cf indices inside a [predicated-jump+1, target) span or
    // predicated execs; instruction-level predication is handled at the
    // fetch site.
    std::set<uint32_t> divergentPcs;
    bool allPcsDivergent = false;
    bool inDivergentFlow = false;

    // REXGLUE mode (see rexglue-sdk/docs/native_shaders.md): emit against
    // rexglue's bindless root signature with in-shader vertex fetch.
    bool rexglueMode = false;
    // Bind-time-patched runtime ucode (byte-swapped to BE) used as the code
    // source instead of the container's; the container provides metadata only.
    const uint8_t* rexCodeOverride = nullptr;
    uint32_t rexCodeOverrideSize = 0;
    std::vector<RexglueBinding> rexBindings;
    // rexglue VS mode: which oVar exports the ucode writes (the
    // interpolant trim, the pack emits a second, trimmed-signature dxil for
    // non-GS pipelines; the full signature stays for GS linkage).
    uint32_t rexWrittenOVarMask = 0;
    bool rexWroteOPts = false;
    // (fetchConstant, dimension, isSigned) -> b4 slot
    std::map<std::tuple<uint32_t, uint32_t, uint32_t>, uint32_t> rexTextureSlots;
    // (fetchConstant, mag, min, mip, aniso) after translator normalization -> b4 slot
    std::map<std::tuple<uint32_t, uint32_t, uint32_t, uint32_t, uint32_t>, uint32_t> rexSamplerSlots;
    uint32_t rexFetchCounter = 0;
    uint32_t psOutputsMask = 0;
    // setTexLOD / setGradientH/V or a tfetch reading their state was seen.
    bool usesRegisterLod = false;
    bool usesRegisterGradients = false;
    // Float-constant usage discovery (pass 1) and compacted layout (pass 2).
    // The runtime uploads only the registers the shader reads, packed in
    // ascending order, unless any read is register-relative, in which case
    // the full 256-register file is uploaded at absolute offsets.
    std::set<uint32_t> rexUsedFloatConstants;
    bool rexFloatsDynamic = false;
    // Guest register -> compacted cbuffer slot; empty on the discovery pass
    // or when the layout is absolute (dynamic).
    std::map<uint32_t, uint32_t> rexFloatRank;
    // Container-less generation: no constant table, so declare b1 as the
    // whole 256-register file and read it by absolute index (identical to
    // the layout the runtime uploads when native shaders are enabled).
    bool rexAbsoluteFloatFile = false;
    // Container size in bytes when known, so out-of-range metadata reads fail instead of reading past it.
    size_t rexContainerSize = 0;
    // Physical data of a runtime shader object (literal bits at their physicalOffset); null = after the container.
    const uint8_t* rexPhysicalData = nullptr;
    size_t rexPhysicalSize = 0;
    // Float registers the named layout reads without a constant table entry (the def literals), declared as xe_fc{N}.
    std::set<uint32_t> rexExtraFloatRegs;
    // X3 vfetch ordinals: slot of every vfetch in control-flow walk order, and the next ordinal to emit.
    std::vector<uint32_t> rexVfetchSlots;
    std::vector<RexVfetchRefl> rexVfetchRefl;
    // Source fields of the last full vfetch; a mini fetch must repeat them (pack_contract.md 2.4).
    bool rexHaveFullVfetch = false;
    uint32_t rexFullVfetchSrc = 0;
    // Reflection state gathered while emitting.
    uint32_t rexBoolMask = 0;
    bool rexUsesKill = false;
    bool rexPcMachine = false;
    uint32_t rexPsReadMask = 0;
    uint32_t rexPixelPosReg = 0xFF;
    std::vector<RexInterpRefl> rexInterps;
    std::vector<RexLiteralRefl> rexLiterals;

#ifdef UNLEASHED_RECOMP
    bool hasMtxProjection = false;
    bool hasMtxPrevInvViewProjection = false;
#endif

#ifdef REBLUE_RECOMP
    bool hasShadowTexture = false;
    uint32_t shadowTapUVCount = 0;
    std::unordered_map<uint32_t, uint32_t> shadowTapSlots;
#endif

    void indent()
    {
        for (uint32_t i = 0; i < indentation; i++)
            out += '\t';
    }

    void printDstSwizzle(uint32_t dstSwizzle, bool operand);
    void printDstSwizzle01(uint32_t dstRegister, uint32_t dstSwizzle);

    void emitRexglueDeclarations(const uint8_t* shaderData);
    void recompileRexglueVfetch(const VertexFetchInstruction& instr, uint32_t address);
    uint32_t rexAddSamplerBinding(const TextureFetchInstruction& instr);
    // Condition text for a cexec/cjmp/ccall on a bool constant, true when the bit equals whenSet.
    std::string boolCondition(uint32_t boolAddress, bool whenSet);

    void recordRexFloatConstant(uint32_t reg, bool relative)
    {
        rexUsedFloatConstants.insert(reg);
        if (relative)
            rexFloatsDynamic = true;
    }

    void recompile(const VertexFetchInstruction& instr, uint32_t address);
    void recompile(const TextureFetchInstruction& instr, bool bicubic);
    void recompile(const AluInstruction& instr);

    void recompile(const uint8_t* shaderData, const std::string_view& include);
};
