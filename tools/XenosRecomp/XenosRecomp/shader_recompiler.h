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

// REXGLUE mode: b4 descriptor-index slots replicate rexglue's
// DxbcShaderTranslator allocation exactly (FindOrAddTextureBinding /
// FindOrAddSamplerBinding order and keys), so the native DXIL is a drop-in
// bytecode replacement, the runtime's per-draw descriptor-index upload,
// driven by the translated shader's binding lists, feeds the same slots.
// Per tfetch: sampler first, then unsigned+signed texture pairs (3D fetches
// allocate a 3D pair then a stacked-2D pair). The native shader reads the
// UNSIGNED texture slot (signed-texture selection is a known v1 gap).
struct RexglueBinding
{
    uint32_t slot;          // index into the b4 descriptor-index array
    bool isSampler;         // false = texture view
    uint32_t fetchConstant; // tfetch constIndex (guest texture fetch constant)
    uint32_t dimension;     // FetchOpDimension of the binding
    bool isSigned;
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
    // rexglue VS mode: which oVar exports the ucode actually writes (the
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
    // Last full vfetch state, inherited by mini fetches.
    uint32_t rexLastVfetchDwordIndex = 0;
    uint32_t rexLastVfetchStride = 0;
    std::string rexLastVfetchIndex;

#ifdef UNLEASHED_RECOMP
    bool hasMtxProjection = false;
    bool hasMtxPrevInvViewProjection = false;
#endif

    void indent()
    {
        for (uint32_t i = 0; i < indentation; i++)
            out += '\t';
    }

    void printDstSwizzle(uint32_t dstSwizzle, bool operand);
    void printDstSwizzle01(uint32_t dstRegister, uint32_t dstSwizzle);

    void emitRexglueDeclarations(const uint8_t* shaderData);
    void recompileRexglueVfetch(const VertexFetchInstruction& instr);
    uint32_t rexAddSamplerBinding(const TextureFetchInstruction& instr);

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
