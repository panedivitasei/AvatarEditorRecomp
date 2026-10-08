#include "shader_recompiler.h"
#include "shader_common.h"
#include "ucode_fingerprint.h"
#include "xenosrecomp.h"

#include <cmath>
#include <cstring>

static constexpr char SWIZZLES[] = 
{ 
    'x',
    'y', 
    'z', 
    'w', 
    '0', 
    '1',
    '_',
    '_'
};

static constexpr const char* USAGE_TYPES[] =
{
    "float4", // POSITION
    "float4", // BLENDWEIGHT
    "uint4", // BLENDINDICES
#ifdef REBLUE_RECOMP
    // BD IA-decodes SNORM normals to float; swapFloats() undoes the engine int16-pair swap, DEC3N goes through tfetchR11G11B10().
    "float4", // NORMAL
#else
    "uint4", // NORMAL
#endif
    "float4", // PSIZE
    "float4", // TEXCOORD
#ifdef REBLUE_RECOMP
    "float4", // TANGENT
    "float4", // BINORMAL
#else
    "uint4", // TANGENT
    "uint4", // BINORMAL
#endif
    "float4", // TESSFACTOR
    "float4", // POSITIONT
    "float4", // COLOR
    "float4", // FOG
    "float4", // DEPTH
    "float4", // SAMPLE
};

static constexpr const char* USAGE_VARIABLES[] =
{
    "Position",
    "BlendWeight",
    "BlendIndices",
    "Normal",
    "PointSize",
    "TexCoord",
    "Tangent",
    "Binormal",
    "TessFactor",
    "PositionT",
    "Color",
    "Fog",
    "Depth",
    "Sample"
};

static constexpr const char* USAGE_SEMANTICS[] =
{
    "POSITION",
    "BLENDWEIGHT",
    "BLENDINDICES",
    "NORMAL",
    "PSIZE",
    "TEXCOORD",
    "TANGENT",
    "BINORMAL",
    "TESSFACTOR",
    "POSITIONT",
    "COLOR",
    "FOG",
    "DEPTH",
    "SAMPLE"
};

struct DeclUsageLocation
{
    DeclUsage usage;
    uint32_t usageIndex;
    uint32_t location;
};

static constexpr DeclUsageLocation USAGE_LOCATIONS[] =
{
#ifdef REBLUE_RECOMP
    #define REBLUE_VERTEX_LOCATION_ROW(usage, index, location) { DeclUsage::usage, index, location },
    REBLUE_VERTEX_INPUT_LOCATIONS(REBLUE_VERTEX_LOCATION_ROW)
    #undef REBLUE_VERTEX_LOCATION_ROW
#else
    { DeclUsage::Position, 0, 0 },
    { DeclUsage::Normal, 0, 1 },
    { DeclUsage::Tangent, 0, 2 },
    { DeclUsage::Binormal, 0, 3 },
    { DeclUsage::TexCoord, 0, 4 },
    { DeclUsage::TexCoord, 1, 5 },
    { DeclUsage::TexCoord, 2, 6 },
    { DeclUsage::TexCoord, 3, 7 },
    { DeclUsage::Color, 0, 8 },
    { DeclUsage::BlendIndices, 0, 9 },
    { DeclUsage::BlendWeight, 0, 10 },
    { DeclUsage::Color, 1, 11 },
    { DeclUsage::TexCoord, 4, 12 },
    { DeclUsage::TexCoord, 5, 13 },
    { DeclUsage::TexCoord, 6, 14 },
    { DeclUsage::TexCoord, 7, 15 },
    { DeclUsage::Position, 1, 15 },
#endif
};

static constexpr std::pair<DeclUsage, size_t> INTERPOLATORS[] =
{
    { DeclUsage::TexCoord, 0 },
    { DeclUsage::TexCoord, 1 },
    { DeclUsage::TexCoord, 2 },
    { DeclUsage::TexCoord, 3 },
    { DeclUsage::TexCoord, 4 },
    { DeclUsage::TexCoord, 5 },
    { DeclUsage::TexCoord, 6 },
    { DeclUsage::TexCoord, 7 },
    { DeclUsage::TexCoord, 8 },
    { DeclUsage::TexCoord, 9 },
    { DeclUsage::TexCoord, 10 },
    { DeclUsage::TexCoord, 11 },
    { DeclUsage::TexCoord, 12 },
    { DeclUsage::TexCoord, 13 },
    { DeclUsage::TexCoord, 14 },
    { DeclUsage::TexCoord, 15 },
    { DeclUsage::Color, 0 },
    { DeclUsage::Color, 1 }
};

static constexpr std::string_view TEXTURE_DIMENSIONS[] = 
{
    "2D",
    "3D", 
    "Cube" 
};

static FetchDestinationSwizzle getDestSwizzle(uint32_t dstSwizzle, uint32_t index)
{
    return FetchDestinationSwizzle((dstSwizzle >> (index * 3)) & 0x7);
}

void ShaderRecompiler::printDstSwizzle(uint32_t dstSwizzle, bool operand)
{
    for (size_t i = 0; i < 4; i++)
    {
        const auto swizzle = getDestSwizzle(dstSwizzle, i);
        if (swizzle >= FetchDestinationSwizzle::X && swizzle <= FetchDestinationSwizzle::W)
            out += SWIZZLES[operand ? uint32_t(swizzle) : i];
    }
}

void ShaderRecompiler::printDstSwizzle01(uint32_t dstRegister, uint32_t dstSwizzle)
{
    for (size_t i = 0; i < 4; i++)
    {
        const auto swizzle = getDestSwizzle(dstSwizzle, i);
        if (swizzle == FetchDestinationSwizzle::Zero)
        {
            indent();
            println("r{}.{} = 0.0;", dstRegister, SWIZZLES[i]);
        }
        else if (swizzle == FetchDestinationSwizzle::One)
        {
            indent();
            println("r{}.{} = 1.0;", dstRegister, SWIZZLES[i]);
        }
    }
}

// X3 vfetch: format, stride, offset, fetch constant and destination swizzle come from the per-draw layout
// record, so the declaration-dependent instruction fields are never read (pack_contract.md 2.4).
void ShaderRecompiler::recompileRexglueVfetch(const VertexFetchInstruction& instr, uint32_t address)
{
    if (isPixelShader)
        throw std::runtime_error("vfetch in a pixel shader (the layout table is vertex-only)");
    if (instr.srcRegisterAm || instr.dstRegisterAam)
        throw std::runtime_error("vfetch with an aL-relative register");

    uint32_t n = rexFetchCounter++;
    if (n >= rexVfetchSlots.size() || rexVfetchSlots[n] != address)
        throw std::runtime_error(fmt::format("vfetch at slot {} is not ordinal {} of the control-flow walk", address, n));

    // A mini fetch reuses the full fetch's address, which the translation recomputes from its own source fields.
    uint32_t src = uint32_t(instr.srcRegister) | (uint32_t(instr.srcSwizzle) << 8) | (uint32_t(instr.isIndexRounded) << 10);
    if (instr.isMiniFetch)
    {
        if (!rexHaveFullVfetch)
            throw std::runtime_error(fmt::format("vfetch_mini at slot {} has no preceding vfetch_full", address));
        if (src != rexFullVfetchSrc)
            throw std::runtime_error(fmt::format("vfetch_mini at slot {} src fields differ from the preceding vfetch_full", address));
    }
    else
    {
        rexHaveFullVfetch = true;
        rexFullVfetchSrc = src;
    }

    if (instr.isPredicated)
    {
        indent();
        println("if ({}p0)", instr.predicateCondition ? "" : "!");
        indent();
        out += "{\n";
        ++indentation;
    }

    const uint32_t dst = uint32_t(instr.dstRegister);
    const std::string index = fmt::format("r{}.{}", uint32_t(instr.srcRegister), SWIZZLES[instr.srcSwizzle & 0x3]);
    const char* rounded = instr.isIndexRounded ? "true" : "false";

    indent();
    if (instr.expAdjust != 0)
    {
        float scale = std::ldexp(1.0f, instr.expAdjust);
        uint32_t scaleBits;
        memcpy(&scaleBits, &scale, sizeof(scaleBits));
        println("r{0} = xe_vfetch_exp({1}u, {2}, {3}, asfloat(0x{4:08X}u), r{0});", dst, n, index, rounded, scaleBits);
    }
    else
    {
        println("r{0} = xe_vfetch({1}u, {2}, {3}, r{0});", dst, n, index, rounded);
    }

    if (instr.isPredicated)
    {
        --indentation;
        indent();
        out += "}\n";
    }
}

void ShaderRecompiler::recompile(const VertexFetchInstruction& instr, uint32_t address)
{
    if (rexglueMode)
    {
        recompileRexglueVfetch(instr, address);
        return;
    }

    if (instr.isPredicated)
    {
        indent();
        println("if ({}p0)", instr.predicateCondition ? "" : "!");

        indent();
        out += "{\n";
        ++indentation;
    }

    indent();
    print("r{}.", instr.dstRegister);
    printDstSwizzle(instr.dstSwizzle, false);

    out += " = ";

    auto findResult = vertexElements.find(address);
    assert(findResult != vertexElements.end());

#ifdef REBLUE_RECOMP
    // Wrap each 16-bit-packed semantic in swapFloats() (per-usage mask); TEXCOORD also runs sintTexcoord() for raw-int bindings.
    switch (findResult->second.usage)
    {
    case DeclUsage::Normal:
        specConstantsMask |= SPEC_CONSTANT_R11G11B10_NORMAL;
        print("tfetchR11G11B10(swapFloats(g_SwappedNormals, ");
        break;
    case DeclUsage::Tangent:
        specConstantsMask |= SPEC_CONSTANT_R11G11B10_NORMAL;
        print("tfetchR11G11B10(swapFloats(g_SwappedTangents, ");
        break;
    case DeclUsage::Binormal:
        specConstantsMask |= SPEC_CONSTANT_R11G11B10_NORMAL;
        print("tfetchR11G11B10(swapFloats(g_SwappedBinormals, ");
        break;
    case DeclUsage::BlendWeight:
        print("swapFloats(g_SwappedBlendWeights, ");
        break;
    case DeclUsage::TexCoord:
        print("sintTexcoord(g_SintTexcoords, swapFloats(g_SwappedTexcoords, ");
        break;
    case DeclUsage::Position:
        print("swapFloats(g_SwappedPositions, ");
        break;
    }
#else
    switch (findResult->second.usage)
    {
    case DeclUsage::Normal:
    case DeclUsage::Tangent:
    case DeclUsage::Binormal:
        specConstantsMask |= SPEC_CONSTANT_R11G11B10_NORMAL;
        print("tfetchR11G11B10(");
        break;

    case DeclUsage::TexCoord:
        print("tfetchTexcoord(g_SwappedTexcoords, ");
        break;
    }
#endif

    print("{}", findResult->second.name);

#ifdef REBLUE_RECOMP
    switch (findResult->second.usage)
    {
    case DeclUsage::Normal:
    case DeclUsage::Tangent:
    case DeclUsage::Binormal:
        print(", {}))", uint32_t(findResult->second.usageIndex));
        break;

    case DeclUsage::TexCoord:
        print(", {}), {})", uint32_t(findResult->second.usageIndex),
              uint32_t(findResult->second.usageIndex));
        break;

    case DeclUsage::BlendWeight:
    case DeclUsage::Position:
        print(", {})", uint32_t(findResult->second.usageIndex));
        break;
    }
#else
    switch (findResult->second.usage)
    {
    case DeclUsage::Normal:
    case DeclUsage::Tangent:
    case DeclUsage::Binormal:
        out += ')';
        break;

    case DeclUsage::TexCoord:
        print(", {})", uint32_t(findResult->second.usageIndex));
        break;
    }
#endif

    out += '.';
    printDstSwizzle(instr.dstSwizzle, true);

    out += ";\n";

    printDstSwizzle01(instr.dstRegister, instr.dstSwizzle);

    if (instr.isPredicated)
    {
        --indentation;
        indent();
        out += "}\n";
    }
}

static const char* fetchOpcodeName(FetchOpcode opcode)
{
    switch (opcode)
    {
    case FetchOpcode::VertexFetch: return "vfetch";
    case FetchOpcode::TextureFetch: return "tfetch";
    case FetchOpcode::GetTextureBorderColorFrac: return "getBCF";
    case FetchOpcode::GetTextureComputedLod: return "getCompTexLOD";
    case FetchOpcode::GetTextureGradients: return "getGradients";
    case FetchOpcode::GetTextureWeights: return "getWeights";
    case FetchOpcode::SetTextureLod: return "setTexLOD";
    case FetchOpcode::SetTextureGradientsHorz: return "setGradientH";
    case FetchOpcode::SetTextureGradientsVert: return "setGradientV";
    default: return "unknown";
    }
}

void ShaderRecompiler::recompile(const TextureFetchInstruction& instr, bool bicubic)
{
    // Every opcode either translates or fails the shader; nothing is dropped
    // (the ring translator's set, dxbc_translator_fetch.cpp:570-668).
    switch (instr.opcode)
    {
    case FetchOpcode::TextureFetch:
    case FetchOpcode::GetTextureWeights:
    case FetchOpcode::GetTextureComputedLod:
    case FetchOpcode::GetTextureGradients:
    case FetchOpcode::SetTextureLod:
    case FetchOpcode::SetTextureGradientsHorz:
    case FetchOpcode::SetTextureGradientsVert:
        break;
    case FetchOpcode::GetTextureBorderColorFrac:
        // The ring translator has no getBCF either (dxbc_translator_fetch.cpp:654).
        throw std::runtime_error("unimplemented fetch opcode 16 (getBCF)");
    default:
        throw std::runtime_error(fmt::format("unknown fetch opcode {}", uint32_t(instr.opcode)));
    }

    if (!isPixelShader && (instr.opcode == FetchOpcode::GetTextureComputedLod ||
                           instr.opcode == FetchOpcode::GetTextureGradients))
        throw std::runtime_error(fmt::format("fetch opcode {} ({}) needs pixel shader derivatives",
            uint32_t(instr.opcode), fetchOpcodeName(instr.opcode)));

    if (instr.opcode == FetchOpcode::GetTextureComputedLod && (!instr.useCompLod || instr.useRegGradients))
        throw std::runtime_error("fetch opcode 17 (getCompTexLOD) with explicit LOD or register gradients");

    if (instr.opcode == FetchOpcode::GetTextureWeights && instr.dimension != TextureDimension::Texture2D)
        throw std::runtime_error(fmt::format("unimplemented fetch opcode 19 (getWeights) for dimension {}",
            uint32_t(instr.dimension)));

    if (instr.isPredicated)
    {
        indent();
        println("if ({}p0)", instr.predCondition ? "" : "!");

        indent();
        out += "{\n";
        ++indentation;
    }

    auto printSrcRegister = [&](size_t componentCount)
        {
            print("r{}.", instr.srcRegister);

            for (size_t i = 0; i < componentCount; i++)
                out += SWIZZLES[((instr.srcSwizzle >> (i * 2))) & 0x3];
        };

    auto closePredicate = [&]()
        {
            if (instr.isPredicated)
            {
                --indentation;
                indent();
                out += "}\n";
            }
        };

    // Register LOD and gradients live in shader-wide state that the tfetch
    // variants read (dxbc_translator_fetch.cpp:570).
    switch (instr.opcode)
    {
    case FetchOpcode::SetTextureLod:
        indent();
        out += "xe_lod = ";
        printSrcRegister(1);
        out += ";\n";
        usesRegisterLod = true;
        closePredicate();
        return;

    case FetchOpcode::SetTextureGradientsHorz:
    case FetchOpcode::SetTextureGradientsVert:
        indent();
        out += instr.opcode == FetchOpcode::SetTextureGradientsHorz ? "xe_grad_h = " : "xe_grad_v = ";
        printSrcRegister(3);
        out += ";\n";
        usesRegisterGradients = true;
        closePredicate();
        return;
    }

    bool hasResult = false;
    for (uint32_t i = 0; i < 4; i++)
    {
        auto swizzle = getDestSwizzle(instr.dstSwizzle, i);
        if (swizzle >= FetchDestinationSwizzle::X && swizzle <= FetchDestinationSwizzle::W)
            hasResult = true;
    }

    if (!hasResult)
    {
        // Constant 0/1 writes only, nothing is fetched.
        printDstSwizzle01(instr.dstRegister, instr.dstSwizzle);
        closePredicate();
        return;
    }

    if (instr.opcode == FetchOpcode::GetTextureGradients)
    {
        // XZ = ddx(src.xy), YW = ddy(src.xy), coarse like the texture unit
        // (dxbc_translator_fetch.cpp:625).
        std::string srcX = fmt::format("r{}.{}", instr.srcRegister, SWIZZLES[instr.srcSwizzle & 0x3]);
        std::string srcY = fmt::format("r{}.{}", instr.srcRegister, SWIZZLES[(instr.srcSwizzle >> 2) & 0x3]);
        indent();
        print("r{}.", instr.dstRegister);
        printDstSwizzle(instr.dstSwizzle, false);
        print(" = float4(ddx_coarse({0}), ddy_coarse({0}), ddx_coarse({1}), ddy_coarse({1})).", srcX, srcY);
        printDstSwizzle(instr.dstSwizzle, true);
        out += ";\n";
        printDstSwizzle01(instr.dstRegister, instr.dstSwizzle);
        closePredicate();
        return;
    }

    std::string constName;
    const char* constNamePtr = nullptr;
#ifdef UNLEASHED_RECOMP
    bool subtractFromOne = false;
#endif

    auto findResult = samplers.find(instr.constIndex);
    if (findResult != samplers.end())
    {
        constNamePtr = findResult->second;

    #ifdef UNLEASHED_RECOMP
        subtractFromOne = hasMtxPrevInvViewProjection && strcmp(constNamePtr, "sampZBuffer") == 0;
    #endif
    }
    else
    {
        constName = fmt::format("s{}", instr.constIndex);
        constNamePtr = constName.c_str();
    }

#ifdef UNLEASHED_RECOMP
    if (instr.constIndex == 0 && instr.dimension == TextureDimension::Texture2D)
    {
        indent();
        print("pixelCoord = getPixelCoord({}_Texture2DDescriptorIndex, ", constNamePtr);
        printSrcRegister(2);
        out += ");\n";
    }
#endif

    // LOD selection for tfetch, as dxbc_translator_fetch.cpp:676 and :1401.
    // BaseMap mip filtering samples level 0; otherwise the LOD (or the bias
    // on a computed LOD) is register LOD + instruction bias, and the helpers
    // add the fetch-constant bias.
    std::string lodVariant;
    std::string lodArguments;
    if (instr.opcode == FetchOpcode::TextureFetch)
    {
        const bool computedLod = instr.useCompLod && (isPixelShader || instr.useRegGradients);

        std::string lodSum;
        if (instr.useRegLod)
        {
            lodSum = "xe_lod";
            usesRegisterLod = true;
        }
        if (instr.lodBias != 0)
        {
            std::string bias = fmt::format("{}", instr.lodBias / 16.0f);
            if (bias.find_first_of(".e") == std::string::npos)
                bias += ".0";
            lodSum = lodSum.empty() ? bias : fmt::format("{} + {}", lodSum, bias);
        }

        if (instr.mipFilter == 2) // TextureFilter::kBaseMap
        {
            lodVariant = "_lod0";
        }
        else if (!computedLod)
        {
            lodVariant = "_level";
            lodArguments = fmt::format(", {}", lodSum.empty() ? "0.0" : lodSum);
        }
        else if (instr.useRegGradients)
        {
            lodVariant = "_grad";
            lodArguments = fmt::format(", {}, xe_grad_h, xe_grad_v", lodSum.empty() ? "0.0" : lodSum);
            usesRegisterGradients = true;
        }
        else if (rexglueMode && (inDivergentFlow || instr.isPredicated))
        {
            // Implicit derivatives under per-pixel flow deadlock Intel EUs, so
            // these keep the mip-0 workaround until derivative hoisting replaces it.
            lodVariant = "_lod0";
        }
        else if (!lodSum.empty())
        {
            lodVariant = "_bias";
            lodArguments = fmt::format(", {}", lodSum);
        }
    }

#ifdef REBLUE_RECOMP
    // Stashed before the fetch statement: the kernel reuses UV registers as
    // fetch destinations (r6.y = tfetch2D(..., r6.yw)).
    const bool shadowTap = hasShadowTexture && !instr.isPredicated &&
        instr.opcode == FetchOpcode::TextureFetch &&
        instr.dimension == TextureDimension::Texture2D &&
        strcmp(constNamePtr, "ShadowTexture") == 0 &&
        instr.offsetX == 0 && instr.offsetY == 0 && shadowTapUVCount < 8;
    if (shadowTap)
    {
        indent();
        print("shadowTapUV[{}] = ", shadowTapUVCount);
        printSrcRegister(2);
        out += ";\n";
    }
#endif

    indent();
    print("r{}.", instr.dstRegister);
    printDstSwizzle(instr.dstSwizzle, false);

    out += " = ";
    switch (instr.opcode)
    {
    case FetchOpcode::TextureFetch:
    {
    #ifdef UNLEASHED_RECOMP
        if (subtractFromOne)
            out += "1.0 - ";
    #endif

        out += "tfetch";
        break;
    }
    case FetchOpcode::GetTextureWeights:
    {
        out += "getWeights";
        break;
    }
    case FetchOpcode::GetTextureComputedLod:
    {
        out += "getCompTexLOD";
        break;
    }
    }

    std::string_view dimension;
    uint32_t componentCount = 0;

    switch (instr.dimension)
    {
    case TextureDimension::Texture1D:
        dimension = "1D";
        componentCount = 1;
        break;
    case TextureDimension::Texture2D:
        dimension = "2D";
        componentCount = 2;
        break;
    case TextureDimension::Texture3D:
        dimension = "3D";
        componentCount = 3;
        break;
    case TextureDimension::TextureCube:
        dimension = "Cube";
        componentCount = 3;
        break;
    }

    out += dimension;
    out += lodVariant;

#ifdef UNLEASHED_RECOMP
    if (bicubic)
        out += "Bicubic";
#endif

    if (rexglueMode)
    {
        const char* denorm = instr.texCoordDenorm ? "true" : "false";
        if (instr.opcode == FetchOpcode::GetTextureWeights)
        {
            // getWeights reads the texture size from the fetch constant; the
            // translator allocates no binding for it.
            print("({}u, {}, ", uint32_t(instr.constIndex) * 6, denorm);
        }
        else
        {
            uint32_t srvDimension = uint32_t(instr.dimension);
            if (srvDimension == 0) // 1D is bound as 2D
                srvDimension = 1;
            auto slotIt = rexTextureSlots.find(std::make_tuple(uint32_t(instr.constIndex), srvDimension, 0u));
            if (slotIt == rexTextureSlots.end())
                throw std::runtime_error(fmt::format(
                    "tfetch binding not pre-allocated: opcode {} constIndex {} dimension {}",
                    uint32_t(instr.opcode), uint32_t(instr.constIndex), srvDimension));
            uint32_t textureSlot = slotIt->second;
            uint32_t samplerSlot = rexAddSamplerBinding(instr);
            print("({}u, {}u, {}u, {}, ", textureSlot, samplerSlot, uint32_t(instr.constIndex) * 6, denorm);
        }
    }
    else
    {
        print("({0}_Texture{1}DescriptorIndex, {0}_SamplerDescriptorIndex, ", constNamePtr, dimension);
    }
    printSrcRegister(componentCount);

    // getCompTexLOD takes no offsets (dxbc_translator_fetch.cpp:694).
    switch (instr.dimension)
    {
    case TextureDimension::Texture2D:
        if (instr.opcode != FetchOpcode::GetTextureComputedLod)
            print(", float2({}, {})", instr.offsetX * 0.5f, instr.offsetY * 0.5f);
        break;
    case TextureDimension::TextureCube:
        out += ", cubeMapData";
        break;
    }

    out += lodArguments;
    out += ").";

    printDstSwizzle(instr.dstSwizzle, true);

    out += ";\n";

    printDstSwizzle01(instr.dstRegister, instr.dstSwizzle);

#ifdef REBLUE_RECOMP
    if (hasShadowTexture)
    {
        for (size_t i = 0; i < 4; i++)
        {
            if (getDestSwizzle(instr.dstSwizzle, i) <= FetchDestinationSwizzle::One)
                shadowTapSlots.erase(uint32_t(instr.dstRegister * 4 + i));
        }
        if (shadowTap)
        {
            for (size_t i = 0; i < 4; i++)
            {
                if (getDestSwizzle(instr.dstSwizzle, i) == FetchDestinationSwizzle::X)
                    shadowTapSlots[uint32_t(instr.dstRegister * 4 + i)] = shadowTapUVCount;
            }
            ++shadowTapUVCount;
        }
    }
#endif

    closePredicate();
}

static const char* aluVectorOpcodeName(AluVectorOpcode opcode)
{
    static constexpr const char* NAMES[] =
    {
        "add", "mul", "max", "min", "seq", "sgt", "sge", "sne", "frc", "trunc", "floor", "mad",
        "cndeq", "cndge", "cndgt", "dp4", "dp3", "dp2add", "cube", "max4", "setp_eq_push",
        "setp_ne_push", "setp_gt_push", "setp_ge_push", "kill_eq", "kill_gt", "kill_ge", "kill_ne",
        "dst", "maxa"
    };
    return uint32_t(opcode) < std::size(NAMES) ? NAMES[uint32_t(opcode)] : "unknown";
}

static const char* aluScalarOpcodeName(AluScalarOpcode opcode)
{
    static constexpr const char* NAMES[] =
    {
        "adds", "adds_prev", "muls", "muls_prev", "muls_prev2", "maxs", "mins", "seqs", "sgts",
        "sges", "snes", "frcs", "truncs", "floors", "exp", "logc", "log", "rcpc", "rcpf", "rcp",
        "rsqc", "rsqf", "rsq", "maxas", "maxasf", "subs", "subs_prev", "setp_eq", "setp_ne",
        "setp_gt", "setp_ge", "setp_inv", "setp_pop", "setp_clr", "setp_rstr", "kills_eq",
        "kills_gt", "kills_ge", "kills_ne", "kills_one", "sqrt", "unknown", "mulsc0", "mulsc1",
        "addsc0", "addsc1", "subsc0", "subsc1", "sin", "cos", "retain_prev"
    };
    return uint32_t(opcode) < std::size(NAMES) ? NAMES[uint32_t(opcode)] : "unknown";
}

void ShaderRecompiler::recompile(const AluInstruction& instr)
{
    // Unknown opcodes fail the shader instead of emitting an empty expression.
    if (uint32_t(instr.vectorOpcode) > uint32_t(AluVectorOpcode::MaxA))
        throw std::runtime_error(fmt::format("unknown ALU vector opcode {}", uint32_t(instr.vectorOpcode)));
    if (uint32_t(instr.scalarOpcode) == 41 || uint32_t(instr.scalarOpcode) > uint32_t(AluScalarOpcode::RetainPrev))
        throw std::runtime_error(fmt::format("unknown ALU scalar opcode {}", uint32_t(instr.scalarOpcode)));

    // Loop-relative destinations (r[aL + n]) are not modeled.
    if (!instr.exportData && (instr.vectorDestRelative || instr.scalarDestRelative))
        throw std::runtime_error(fmt::format("unimplemented aL-relative ALU destination ({} / {})",
            aluVectorOpcodeName(instr.vectorOpcode), aluScalarOpcodeName(instr.scalarOpcode)));

    if (instr.isPredicated)
    {
        indent();
        println("if ({}p0)", instr.predicateCondition ? "" : "!");

        indent();
        out += "{\n";
        ++indentation;
    }

    enum
    {
        VECTOR_0,
        VECTOR_1,
        VECTOR_2,
        SCALAR_0,
        SCALAR_1,
        SCALAR_CONSTANT_0,
        SCALAR_CONSTANT_1
    };

    // maskOverride selects which swizzled source components a vector operand
    // prints, independent of the write mask.
    auto op = [&](size_t operand, uint32_t maskOverride = 0)
        {
            size_t reg = 0;
            size_t swizzle = 0;
            bool select = true;
            bool negate = false;
            bool abs = false;

            // Which relative flag a constant operand uses depends on how many
            // constants precede it (ucode.h src_const_is_addressed).
            bool constRelative = false;
            switch (operand)
            {
            case VECTOR_0:
                constRelative = instr.const0Relative;
                break;
            case VECTOR_1:
                constRelative = instr.src1Select ? instr.const0Relative : instr.const1Relative;
                break;
            case VECTOR_2:
            case SCALAR_0:
            case SCALAR_1:
            case SCALAR_CONSTANT_0:
                constRelative = (instr.src1Select && instr.src2Select) ? instr.const0Relative : instr.const1Relative;
                break;
            }

            switch (operand)
            {
            case SCALAR_CONSTANT_0:
                reg = instr.src3Register;
                swizzle = instr.src3Swizzle;
                select = false;
                negate = instr.src3Negate;
                abs = instr.absConstants;
                break;

            case SCALAR_CONSTANT_1:
                reg = (uint32_t(instr.scalarOpcode) & 1) | (instr.src3Select << 1) | (instr.src3Swizzle & 0x3C);
                swizzle = instr.src3Swizzle;
                select = true;
                negate = instr.src3Negate;
                abs = instr.absConstants;
                break;

            default:
                switch (operand)
                {
                case VECTOR_0:
                    reg = instr.src1Register;
                    swizzle = instr.src1Swizzle;
                    select = instr.src1Select;
                    negate = instr.src1Negate;
                    break;
                case VECTOR_1:
                    reg = instr.src2Register;
                    swizzle = instr.src2Swizzle;
                    select = instr.src2Select;
                    negate = instr.src2Negate;
                    break;
                case VECTOR_2:
                case SCALAR_0:
                case SCALAR_1:
                    reg = instr.src3Register;
                    swizzle = instr.src3Swizzle;
                    select = instr.src3Select;
                    negate = instr.src3Negate;
                    break;
                }

                if (select)
                {
                    abs = (reg & 0x80) != 0;
                    reg &= 0x3F;
                }
                else
                {
                    abs = instr.absConstants;
                }

                break;
            }

            std::string regFormatted;
            const char* relativeSuffix = constRelative ? (instr.constAddressRegisterRelative ? " + a0" : " + aL") : "";

            if (select)
            {
                regFormatted = fmt::format("r{}", reg);
            }
            else if (rexglueMode && (recordRexFloatConstant(reg, instr.const0Relative || instr.const1Relative),
                                     rexAbsoluteFloatFile))
            {
                // Container-less: absolute register file (relative reads
                // mirror the named multi-register indexing below).
                regFormatted = fmt::format("xe_fcfile_read({}{})", reg, relativeSuffix);
            }
            else if (rexglueMode && !rexFloatRank.empty())
            {
                // Compacted static layout (pass 2): per-register variables at
                // the ranks the runtime's packed float-constant upload uses.
                regFormatted = fmt::format("xe_fc{}", reg);
            }
            else
            {
                auto findResult = float4Constants.find(reg);
                if (findResult != float4Constants.end())
                {
                    const char* constantName = reinterpret_cast<const char*>(constantTableData + findResult->second->name);
                    if (findResult->second->registerCount > 1)
                    {
                    #ifdef UNLEASHED_RECOMP
                        if (hasMtxProjection && strcmp(constantName, "g_MtxProjection") == 0)
                        {
                            regFormatted = fmt::format("(iterationIndex == 0 ? mtxProjectionReverseZ[{0}] : mtxProjection[{0}])",
                                reg - findResult->second->registerIndex);
                        }
                        else
                    #endif
                        {
                            regFormatted = fmt::format("{}({}{})", constantName,
                                reg - findResult->second->registerIndex, relativeSuffix);
                        }
                    }
                    else
                    {
                        assert(!constRelative);
                        regFormatted = constantName;
                    }
                }
                else
                {
                    assert(!constRelative);
                    regFormatted = rexglueMode ? fmt::format("xe_fc{}", reg) : fmt::format("c{}", reg);
                }
            }

            std::string result;

            if (negate)
                result += '-';

            if (abs)
                result += "abs(";

            result += regFormatted;
            result += '.';

            switch (operand)
            {
            case VECTOR_0:
            case VECTOR_1:
            case VECTOR_2:
            {
                uint32_t mask;

                if (maskOverride != 0)
                {
                    mask = maskOverride;
                }
                else
                {
                    switch (instr.vectorOpcode)
                    {
                    case AluVectorOpcode::Dp2Add:
                        mask = (operand == VECTOR_2) ? 0b1 : 0b11;
                        break;

                    case AluVectorOpcode::Dp3:
                        mask = 0b111;
                        break;

                    // Kills compare all four components, dst reads a full
                    // float4 (ucode.h GetAluVectorOpNeededSourceComponents).
                    case AluVectorOpcode::Dp4:
                    case AluVectorOpcode::Max4:
                    case AluVectorOpcode::KillEq:
                    case AluVectorOpcode::KillGt:
                    case AluVectorOpcode::KillGe:
                    case AluVectorOpcode::KillNe:
                    case AluVectorOpcode::Dst:
                        mask = 0b1111;
                        break;

                    default:
                        mask = instr.vectorWriteMask != 0 ? instr.vectorWriteMask : 0b1;
                        break;
                    }
                }

                for (size_t i = 0; i < 4; i++)
                {
                    if ((mask >> i) & 0x1)
                        result += SWIZZLES[((swizzle >> (i * 2)) + i) & 0x3];
                }

                break;
            }

            case SCALAR_0:
            case SCALAR_CONSTANT_0:
                result += SWIZZLES[((swizzle >> 6) + 3) & 0x3];
                break;

            case SCALAR_1:
            case SCALAR_CONSTANT_1:
                result += SWIZZLES[swizzle & 0x3];
                break;
            }

            if (abs)
                result += ")";

            return result;
        };

    // Xenos multiply (0 * anything = +0) and SM3 max/min. Identical operands
    // cannot hit the zero case or a lone NaN, so they keep the plain form.
    auto mul = [](const std::string& a, const std::string& b)
        {
            return a == b ? fmt::format("{} * {}", a, b) : fmt::format("xe_mul({}, {})", a, b);
        };
    auto dot = [](const std::string& a, const std::string& b)
        {
            return a == b ? fmt::format("dot({}, {})", a, b) : fmt::format("xe_dot({}, {})", a, b);
        };
    auto maxOp = [](const std::string& a, const std::string& b)
        {
            return a == b ? a : fmt::format("xe_max({}, {})", a, b);
        };
    auto minOp = [](const std::string& a, const std::string& b)
        {
            return a == b ? a : fmt::format("xe_min({}, {})", a, b);
        };

    switch (instr.vectorOpcode)
    {
    case AluVectorOpcode::KillEq:
        indent();
        rexUsesKill = true;
        println("clip(any({} == {}) ? -1 : 1);", op(VECTOR_0), op(VECTOR_1));
        break;

    case AluVectorOpcode::KillGt:
        indent();
        rexUsesKill = true;
        println("clip(any({} > {}) ? -1 : 1);", op(VECTOR_0), op(VECTOR_1));
        break;

    case AluVectorOpcode::KillGe:
        indent();
        rexUsesKill = true;
        println("clip(any({} >= {}) ? -1 : 1);", op(VECTOR_0), op(VECTOR_1));
        break;

    case AluVectorOpcode::KillNe:
        indent();
        rexUsesKill = true;
        println("clip(any({} != {}) ? -1 : 1);", op(VECTOR_0), op(VECTOR_1));
        break;
    }

    bool closeIfBracket = false;

    std::string_view exportRegister;
    if (instr.exportData)
    {
        if (isPixelShader)
        {
            switch (ExportRegister(instr.vectorDest))
            {
            case ExportRegister::PSColor0:
                exportRegister = "oC0";
                break;
            case ExportRegister::PSColor1:
                exportRegister = "oC1";
                break;
            case ExportRegister::PSColor2:
                exportRegister = "oC2";
                break;
            case ExportRegister::PSColor3:
                exportRegister = "oC3";
                break;
            case ExportRegister::PSDepth:
                exportRegister = "oDepth";
                break;
            default:
                throw std::runtime_error(fmt::format("unimplemented pixel shader export register {}{}",
                    uint32_t(instr.vectorDest), instr.vectorDest >= 32 && instr.vectorDest <= 37 ? " (memexport)" : ""));
            }
        }
        else
        {
            switch (ExportRegister(instr.vectorDest))
            {
            case ExportRegister::VSPosition:
                exportRegister = "oPos";

            #ifdef UNLEASHED_RECOMP
                if (hasMtxProjection)
                {
                    indent();
                    out += "if ((g_SpecConstants() & SPEC_CONSTANT_REVERSE_Z) == 0 || iterationIndex == 0)\n";
                    indent();
                    out += "{\n";
                    ++indentation;

                    closeIfBracket = true;
                }
            #endif

                break;

            default:
            {
                if (rexglueMode && instr.vectorDest < 16)
                {
                    // Identity wiring: export o{n} is always oVar{n}.
                    thread_local std::string rexExportName;
                    rexExportName = fmt::format("oVar{}", uint32_t(instr.vectorDest));
                    exportRegister = rexExportName;
                    rexWrittenOVarMask |= 1u << uint32_t(instr.vectorDest);
                    break;
                }
                if (rexglueMode && ExportRegister(instr.vectorDest) ==
                    ExportRegister::VSPointSizeEdgeFlagKillVertex)
                {
                    rexWroteOPts = true;
                    // Register 63: x = point size (pixels), y = edge flag,
                    // z = kill vertex. Routed to the oPts output every rexglue
                    // VS declares; the runtime's point_expand GS reads .x to
                    // size POINTLIST sprites (deferred point-sprite lights).
                    exportRegister = "oPts";
                    break;
                }
                auto findResult = interpolators.find(instr.vectorDest);
#ifdef XENOS_COVERAGE
                if (findResult == interpolators.end())
                    throw CoverageError(fmt::format("assert: unmapped VS export register {}{}", uint32_t(instr.vectorDest),
                        instr.vectorDest >= 32 && instr.vectorDest <= 37 ? " (memexport)" : ""));
#endif
                assert(findResult != interpolators.end());
                exportRegister = findResult->second;
                break;
            }
            }
        }
    }

    uint32_t vectorWriteMask = instr.vectorWriteMask;
    if (instr.exportData)
        vectorWriteMask &= ~instr.scalarWriteMask;

    // Both ops read their sources before either writes (the ring translator
    // stores both results after both ops, dxbc_translator_alu.cpp:954). A vector result
    // headed for a register the scalar op reads is held in a temporary, and
    // maxa's a0 update waits until the scalar op has read its constants.
    const bool scalarActive = instr.scalarOpcode != AluScalarOpcode::RetainPrev;
    int scalarSourceGpr = -1;
    uint32_t scalarSourceComponents = 0;
    if (scalarActive)
    {
        const uint32_t componentA = 1u << (((instr.src3Swizzle >> 6) + 3) & 0x3);
        const uint32_t componentB = 1u << (instr.src3Swizzle & 0x3);
        switch (instr.scalarOpcode)
        {
        case AluScalarOpcode::Mulsc0:
        case AluScalarOpcode::Mulsc1:
        case AluScalarOpcode::Addsc0:
        case AluScalarOpcode::Addsc1:
        case AluScalarOpcode::Subsc0:
        case AluScalarOpcode::Subsc1:
            scalarSourceGpr = int((uint32_t(instr.scalarOpcode) & 1) | (instr.src3Select << 1) | (instr.src3Swizzle & 0x3C));
            scalarSourceComponents = componentB;
            break;
        default:
            if (instr.src3Select)
            {
                scalarSourceGpr = int(instr.src3Register & 0x3F);
                scalarSourceComponents = componentA | componentB;
            }
            break;
        }
    }
    const bool deferVector = vectorWriteMask != 0 && exportRegister.empty() && scalarActive &&
        scalarSourceGpr == int(instr.vectorDest) && (vectorWriteMask & scalarSourceComponents) != 0;
    const bool deferA0 = instr.vectorOpcode == AluVectorOpcode::MaxA;
    const bool scoped = deferVector || deferA0;

    if (scoped)
    {
        indent();
        out += "{\n";
        ++indentation;
    }

    if (deferA0)
    {
        // maxa: a0 = clamp(floor(src0.w + 0.5)) (dxbc_translator_alu.cpp:552).
        indent();
        println("int xe_a0 = (int)clamp(floor({} + 0.5), -256.0, 255.0);", op(VECTOR_0, 0b1000));
    }

    if (instr.vectorOpcode >= AluVectorOpcode::SetpEqPush && instr.vectorOpcode <= AluVectorOpcode::SetpGePush)
    {
        // The predicate comes from the W components (interpreter.cpp:477).
        const char* compare = "==";
        switch (instr.vectorOpcode)
        {
        case AluVectorOpcode::SetpNePush:
            compare = "!=";
            break;
        case AluVectorOpcode::SetpGtPush:
            compare = ">";
            break;
        case AluVectorOpcode::SetpGePush:
            compare = ">=";
            break;
        }
        indent();
        println("p0 = {} == 0.0 && {} {} 0.0;", op(VECTOR_0, 0b1000), op(VECTOR_1, 0b1000), compare);
    }

    std::string vectorWriteTarget;

#ifdef REBLUE_RECOMP
    // A Sgt/Sge whose sources are ShadowTexture taps re-emits per lane as a
    // filtered compare from the stashed tap UV, turning the guest's binary PCF
    // into hardware-equivalent bilinear PCF. Anything else falls through.
    bool shadowCmpRewritten = false;
    if (vectorWriteMask != 0 && !instr.exportData && !deferVector && hasShadowTexture &&
        (instr.vectorOpcode == AluVectorOpcode::Sgt || instr.vectorOpcode == AluVectorOpcode::Sge) &&
        instr.src1Select && !instr.src1Negate && (instr.src1Register & 0x80) == 0 &&
        instr.src2Select && !instr.src2Negate && (instr.src2Register & 0x80) == 0)
    {
        const uint32_t srcA = instr.src1Register & 0x3F;
        const uint32_t srcB = instr.src2Register & 0x3F;
        bool allTaps = instr.vectorDest != srcA && instr.vectorDest != srcB;
        for (size_t i = 0; i < 4 && allTaps; i++)
        {
            if ((vectorWriteMask >> i) & 0x1)
                allTaps = shadowTapSlots.find(srcA * 4 + (((instr.src1Swizzle >> (i * 2)) + i) & 0x3)) != shadowTapSlots.end();
        }
        if (allTaps)
        {
            for (size_t i = 0; i < 4; i++)
            {
                if (((vectorWriteMask >> i) & 0x1) == 0)
                    continue;
                const uint32_t compA = ((instr.src1Swizzle >> (i * 2)) + i) & 0x3;
                const uint32_t compB = ((instr.src2Swizzle >> (i * 2)) + i) & 0x3;
                indent();
                println("r{}.{} = shadowCmp2D(ShadowTexture_Texture2DDescriptorIndex, shadowTapUV[{}], r{}.{});",
                    instr.vectorDest, SWIZZLES[i], shadowTapSlots[srcA * 4 + compA], srcB, SWIZZLES[compB]);
            }
            shadowCmpRewritten = true;
        }
    }

    if (vectorWriteMask != 0 && !shadowCmpRewritten)
#else
    if (vectorWriteMask != 0)
#endif
    {
        if (!exportRegister.empty())
            vectorWriteTarget = fmt::format("{}.", exportRegister);
        else
            vectorWriteTarget = fmt::format("r{}.", instr.vectorDest);

        uint32_t componentCount = 0;
        for (size_t i = 0; i < 4; i++)
        {
            if ((vectorWriteMask >> i) & 0x1)
            {
                vectorWriteTarget += SWIZZLES[i];
                ++componentCount;
            }
        }

        indent();
        if (deferVector)
        {
            if (componentCount == 1)
                out += "float xe_vr = ";
            else
                print("float{} xe_vr = ", componentCount);
        }
        else
        {
            out += vectorWriteTarget;
            out += " = ";
        }

        if (instr.vectorSaturate)
            out += "saturate(";

        switch (instr.vectorOpcode)
        {
        case AluVectorOpcode::Add:
            print("{} + {}", op(VECTOR_0), op(VECTOR_1));
            break;

        case AluVectorOpcode::Mul:
            out += mul(op(VECTOR_0), op(VECTOR_1));
            break;

        case AluVectorOpcode::Max:
        case AluVectorOpcode::MaxA:
            out += maxOp(op(VECTOR_0), op(VECTOR_1));
            break;

        case AluVectorOpcode::Min:
            out += minOp(op(VECTOR_0), op(VECTOR_1));
            break;

        case AluVectorOpcode::Seq:
            print("{} == {}", op(VECTOR_0), op(VECTOR_1));
            break;

        case AluVectorOpcode::Sgt:
            print("{} > {}", op(VECTOR_0), op(VECTOR_1));
            break;

        case AluVectorOpcode::Sge:
            print("{} >= {}", op(VECTOR_0), op(VECTOR_1));
            break;

        case AluVectorOpcode::Sne:
            print("{} != {}", op(VECTOR_0), op(VECTOR_1));
            break;

        case AluVectorOpcode::Frc:
            print("frac({})", op(VECTOR_0));
            break;

        case AluVectorOpcode::Trunc:
            print("trunc({})", op(VECTOR_0));
            break;

        case AluVectorOpcode::Floor:
            print("floor({})", op(VECTOR_0));
            break;

        case AluVectorOpcode::Mad:
            print("{} + {}", mul(op(VECTOR_0), op(VECTOR_1)), op(VECTOR_2));
            break;

        case AluVectorOpcode::CndEq:
            print("select({} == 0.0, {}, {})", op(VECTOR_0), op(VECTOR_1), op(VECTOR_2));
            break;

        case AluVectorOpcode::CndGe:
            print("select({} >= 0.0, {}, {})", op(VECTOR_0), op(VECTOR_1), op(VECTOR_2));
            break;

        case AluVectorOpcode::CndGt:
            print("select({} > 0.0, {}, {})", op(VECTOR_0), op(VECTOR_1), op(VECTOR_2));
            break;

        case AluVectorOpcode::Dp4:
        case AluVectorOpcode::Dp3:
            out += dot(op(VECTOR_0), op(VECTOR_1));
            break;

        case AluVectorOpcode::Dp2Add:
            print("{} + {}", dot(op(VECTOR_0), op(VECTOR_1)), op(VECTOR_2));
            break;

        case AluVectorOpcode::Cube:
        {
            // Xenos cube takes the direction from src1's swizzled lanes (z, w, x);
            // the canonical emit is src.zzxy which makes this src.xyz, but other
            // compilers emit e.g. src.yyzx (Blue Dragon), so honor the swizzle.
            const uint32_t reg = instr.src1Register & 0x3F;
            const bool srcAbs = (instr.src1Register & 0x80) != 0;
            auto lane = [&](uint32_t i) { return SWIZZLES[((instr.src1Swizzle >> (i * 2)) + i) & 0x3]; };
            print("cube({}{}r{}.{}{}{}{}{}, cubeMapData)",
                instr.src1Negate ? "-" : "", srcAbs ? "abs(" : "", reg,
                lane(2), lane(3), lane(0), lane(0), srcAbs ? ")" : "");
            break;
        }

        case AluVectorOpcode::Max4:
            print("max4({})", op(VECTOR_0));
            break;

        case AluVectorOpcode::SetpEqPush:
        case AluVectorOpcode::SetpNePush:
        case AluVectorOpcode::SetpGtPush:
        case AluVectorOpcode::SetpGePush:
        {
            // The result comes from the X components, replicated
            // (interpreter.cpp:477).
            const char* compare = "==";
            switch (instr.vectorOpcode)
            {
            case AluVectorOpcode::SetpNePush:
                compare = "!=";
                break;
            case AluVectorOpcode::SetpGtPush:
                compare = ">";
                break;
            case AluVectorOpcode::SetpGePush:
                compare = ">=";
                break;
            }
            std::string x0 = op(VECTOR_0, 0b0001);
            print("({0} == 0.0 && {1} {2} 0.0) ? 0.0 : {0} + 1.0", x0, op(VECTOR_1, 0b0001), compare);
            break;
        }

        case AluVectorOpcode::KillEq:
            print("any({} == {})", op(VECTOR_0), op(VECTOR_1));
            break;

        case AluVectorOpcode::KillGt:
            print("any({} > {})", op(VECTOR_0), op(VECTOR_1));
            break;

        case AluVectorOpcode::KillGe:
            print("any({} >= {})", op(VECTOR_0), op(VECTOR_1));
            break;

        case AluVectorOpcode::KillNe:
            print("any({} != {})", op(VECTOR_0), op(VECTOR_1));
            break;

        case AluVectorOpcode::Dst:
            print("dst({}, {})", op(VECTOR_0), op(VECTOR_1));
            break;
        }

        if (instr.vectorSaturate)
            out += ')';

        out += ";\n";
    }

#ifdef REBLUE_RECOMP
    if (hasShadowTexture && vectorWriteMask != 0 && exportRegister.empty())
    {
        for (size_t i = 0; i < 4; i++)
        {
            if ((vectorWriteMask >> i) & 0x1)
                shadowTapSlots.erase(uint32_t(instr.vectorDest * 4 + i));
        }
    }
#endif

    if (scalarActive)
    {
        if (instr.scalarOpcode >= AluScalarOpcode::SetpEq && instr.scalarOpcode <= AluScalarOpcode::SetpRstr)
        {
            indent();
            out += "p0 = ";

            switch (instr.scalarOpcode)
            {
            case AluScalarOpcode::SetpEq:
                print("{} == 0.0", op(SCALAR_0));
                break;

            case AluScalarOpcode::SetpNe:
                print("{} != 0.0", op(SCALAR_0));
                break;

            case AluScalarOpcode::SetpGt:
                print("{} > 0.0", op(SCALAR_0));
                break;

            case AluScalarOpcode::SetpGe:
                print("{} >= 0.0", op(SCALAR_0));
                break;

            case AluScalarOpcode::SetpInv:
                print("{} == 1.0", op(SCALAR_0));
                break;

            case AluScalarOpcode::SetpPop:
                print("{} - 1.0 <= 0.0", op(SCALAR_0));
                break;

            case AluScalarOpcode::SetpClr:
                out += "false";
                break;

            case AluScalarOpcode::SetpRstr:
                print("{} == 0.0", op(SCALAR_0));
                break;
            }

            out += ";\n";
        }

        indent();
        out += "ps = ";
        if (instr.scalarSaturate)
            out += "saturate(";

        switch (instr.scalarOpcode)
        {
        case AluScalarOpcode::Adds:
            print("{} + {}", op(SCALAR_0), op(SCALAR_1));
            break;

        case AluScalarOpcode::AddsPrev:
            print("{} + ps", op(SCALAR_0));
            break;

        case AluScalarOpcode::Muls:
            out += mul(op(SCALAR_0), op(SCALAR_1));
            break;

        case AluScalarOpcode::MulsPrev:
            out += mul(op(SCALAR_0), "ps");
            break;

        case AluScalarOpcode::MulsPrev2:
            // -FLT_MAX unless ps and src.b are finite, src.b > 0 and ps is not
            // already -FLT_MAX (dxbc_translator_alu.cpp:638, interpreter.cpp:650).
            print("(ps == -FLT_MAX || !(abs(ps) <= FLT_MAX) || !(abs({0}) <= FLT_MAX) || !({0} > 0.0)) ? -FLT_MAX : {1}",
                op(SCALAR_1), mul(op(SCALAR_0), "ps"));
            break;

        case AluScalarOpcode::Maxs:
        case AluScalarOpcode::MaxAs:
        case AluScalarOpcode::MaxAsf:
            out += maxOp(op(SCALAR_0), op(SCALAR_1));
            break;

        case AluScalarOpcode::Mins:
            out += minOp(op(SCALAR_0), op(SCALAR_1));
            break;

        case AluScalarOpcode::Seqs:
            print("{} == 0.0", op(SCALAR_0));
            break;

        case AluScalarOpcode::Sgts:
            print("{} > 0.0", op(SCALAR_0));
            break;

        case AluScalarOpcode::Sges:
            print("{} >= 0.0", op(SCALAR_0));
            break;

        case AluScalarOpcode::Snes:
            print("{} != 0.0", op(SCALAR_0));
            break;

        case AluScalarOpcode::Frcs:
            print("frac({})", op(SCALAR_0));
            break;

        case AluScalarOpcode::Truncs:
            print("trunc({})", op(SCALAR_0));
            break;

        case AluScalarOpcode::Floors:
            print("floor({})", op(SCALAR_0));
            break;

        case AluScalarOpcode::Exp:
            print("exp2({})", op(SCALAR_0));
            break;

        case AluScalarOpcode::Logc:
        case AluScalarOpcode::Log:
            // Xenos LOG_CLAMP: -inf (log of 0) becomes -FLT_MAX, the bound
            // is the most-negative float, not FLT_MIN (smallest POSITIVE
            // normal): with FLT_MIN every negative log2 result (any input
            // < 1.0) collapsed to ~0.
            print("clamp(log2({}), -FLT_MAX, FLT_MAX)", op(SCALAR_0));
            break;

        case AluScalarOpcode::Rcpc:
        case AluScalarOpcode::Rcpf:
        case AluScalarOpcode::Rcp:
            // Xenos RECIP_CLAMP: ±inf -> ±FLT_MAX, sign-preserving. The old
            // FLT_MIN lower bound zeroed every NEGATIVE reciprocal (e.g. a
            // deferred depth-linearization rcp(-near*far)).
            print("clamp(rcp({}), -FLT_MAX, FLT_MAX)", op(SCALAR_0));
            break;

        case AluScalarOpcode::Rsqc:
        case AluScalarOpcode::Rsqf:
        case AluScalarOpcode::Rsq:
            // Xenos RECIPSQ_CLAMP: same ±FLT_MAX semantics as RECIP_CLAMP.
            print("clamp(rsqrt({}), -FLT_MAX, FLT_MAX)", op(SCALAR_0));
            break;

        case AluScalarOpcode::Subs:
            print("{} - {}", op(SCALAR_0), op(SCALAR_1));
            break;

        case AluScalarOpcode::SubsPrev:
            print("{} - ps", op(SCALAR_0));
            break;

        case AluScalarOpcode::SetpEq:
        case AluScalarOpcode::SetpNe:
        case AluScalarOpcode::SetpGt:
        case AluScalarOpcode::SetpGe:
            out += "p0 ? 0.0 : 1.0";
            break;

        case AluScalarOpcode::SetpInv:
            // p0 = src == 1; ps = 0 when p0 is set, else (src == 0 ? 1 : src)
            // (dxbc_translator_alu.cpp:829).
            print("p0 ? 0.0 : ({0} == 0.0 ? 1.0 : {0})", op(SCALAR_0));
            break;

        case AluScalarOpcode::SetpPop:
            print("p0 ? 0.0 : ({} - 1.0)", op(SCALAR_0));
            break;

        case AluScalarOpcode::SetpClr:
            out += "FLT_MAX";
            break;

        case AluScalarOpcode::SetpRstr:
            print("p0 ? 0.0 : {}", op(SCALAR_0));
            break;

        case AluScalarOpcode::KillsEq:
            print("{} == 0.0", op(SCALAR_0));
            break;

        case AluScalarOpcode::KillsGt:
            print("{} > 0.0", op(SCALAR_0));
            break;

        case AluScalarOpcode::KillsGe:
            print("{} >= 0.0", op(SCALAR_0));
            break;

        case AluScalarOpcode::KillsNe:
            print("{} != 0.0", op(SCALAR_0));
            break;

        case AluScalarOpcode::KillsOne:
            print("{} == 1.0", op(SCALAR_0));
            break;

        case AluScalarOpcode::Sqrt:
            print("sqrt({})", op(SCALAR_0));
            break;

        case AluScalarOpcode::Mulsc0:
        case AluScalarOpcode::Mulsc1:
            out += mul(op(SCALAR_CONSTANT_0), op(SCALAR_CONSTANT_1));
            break;

        case AluScalarOpcode::Addsc0:
        case AluScalarOpcode::Addsc1:
            print("{} + {}", op(SCALAR_CONSTANT_0), op(SCALAR_CONSTANT_1));
            break;

        case AluScalarOpcode::Subsc0:
        case AluScalarOpcode::Subsc1:
            print("{} - {}", op(SCALAR_CONSTANT_0), op(SCALAR_CONSTANT_1));
            break;

        case AluScalarOpcode::Sin:
            print("sin({})", op(SCALAR_0));
            break;

        case AluScalarOpcode::Cos:
            print("cos({})", op(SCALAR_0));
            break;
        }

        if (instr.scalarSaturate)
            out += ')';

        out += ";\n";

        // maxas/maxasf: a0 from src.a, rounded or floored, clamped to
        // [-256, 255] (dxbc_translator_alu.cpp:774).
        switch (instr.scalarOpcode)
        {
        case AluScalarOpcode::MaxAs:
            indent();
            println("a0 = (int)clamp(floor({} + 0.5), -256.0, 255.0);", op(SCALAR_0));
            break;
        case AluScalarOpcode::MaxAsf:
            indent();
            println("a0 = (int)clamp(floor({}), -256.0, 255.0);", op(SCALAR_0));
            break;
        }
    }

    if (deferVector)
    {
        indent();
        println("{} = xe_vr;", vectorWriteTarget);
    }

    if (deferA0)
    {
        indent();
        out += "a0 = xe_a0;\n";
    }

    if (scoped)
    {
        --indentation;
        indent();
        out += "}\n";
    }

    uint32_t scalarWriteMask = instr.scalarWriteMask;
    if (instr.exportData)
        scalarWriteMask &= ~instr.vectorWriteMask;

    if (scalarWriteMask != 0)
    {
        indent();
        if (!exportRegister.empty())
        {
            out += exportRegister;
            out += '.';
        }
        else
        {
            print("r{}.", instr.scalarDest);
        }

        for (size_t i = 0; i < 4; i++)
        {
            if ((scalarWriteMask >> i) & 0x1)
                out += SWIZZLES[i];
        }

        out += " = ps;\n";
    }

#ifdef REBLUE_RECOMP
    if (hasShadowTexture && scalarWriteMask != 0 && exportRegister.empty())
    {
        for (size_t i = 0; i < 4; i++)
        {
            if ((scalarWriteMask >> i) & 0x1)
                shadowTapSlots.erase(uint32_t(instr.scalarDest * 4 + i));
        }
    }
#endif

    if (instr.exportData)
    {
        uint32_t zeroMask = instr.scalarDestRelative ? (0b1111 & ~(instr.vectorWriteMask | instr.scalarWriteMask)) : 0;
        uint32_t oneMask = instr.vectorWriteMask & instr.scalarWriteMask;

        for (size_t i = 0; i < 4; i++)
        {
            uint32_t mask = 1 << i;
            if (zeroMask & mask)
            {
                indent();
                println("{}.{} = 0.0;", exportRegister, SWIZZLES[i]);
            }
            else if (oneMask & mask)
            {
                indent();
                println("{}.{} = 1.0;", exportRegister, SWIZZLES[i]);
            }
        }
    }

    if (instr.scalarOpcode >= AluScalarOpcode::KillsEq && instr.scalarOpcode <= AluScalarOpcode::KillsOne)
    {
        indent();
        rexUsesKill = true;
        out += "clip(ps != 0.0 ? -1 : 1);\n";
    }

    if (closeIfBracket)
    {
        --indentation;
        indent();
        out += "}\n";
    }

    if (instr.isPredicated)
    {
        --indentation;
        indent();
        out += "}\n";
    }
}

void ShaderRecompiler::emitRexglueDeclarations(const uint8_t* shaderData)
{
    const auto shaderContainer = reinterpret_cast<const ShaderContainer*>(shaderData);
    const auto constantTableContainer = reinterpret_cast<const ConstantTableContainer*>(shaderData + shaderContainer->constantTableOffset);

    // Float constants at b1: the runtime uploads only the registers the shader reads, packed in
    // ascending order, or the full 256-register file at absolute offsets when any read is
    // register-relative (the command processor's float_bitmap gather). Pass 2 with a static-usage
    // shader gets per-register variables at their compacted ranks, the discovery pass and dynamic
    // shaders get the named absolute layout.
    if (rexAbsoluteFloatFile)
    {
        // Container-less generation: the whole guest register file by
        // absolute index; out-of-range dynamic reads return 0 like the
        // named multi-register tail clamp.
        out += "#ifndef __spirv__\n";
        println("cbuffer XeFloatConstants : register(b1, space0)");
        out += "{\n";
        out += "\tfloat4 xe_fcfile[256];\n";
        out += "};\n\n";
        println("#define xe_fcfile_read(INDEX) select((INDEX) < 256, xe_fcfile[min(INDEX, 255)], 0.0)");
        out += "#else\n";
        println("#define xe_fcfile_read(INDEX) (((INDEX) < 256) ? vk::RawBufferLoad<float4>(XE_PUSH_FLOATS + uint(min(INDEX, 255)) * 16, 4) : (float4)0.0)");
        out += "#endif\n";
        out += "\n";
    }
    else if (!rexFloatRank.empty())
    {
        out += "#ifndef __spirv__\n";
        println("cbuffer XeFloatConstants : register(b1, space0)");
        out += "{\n";
        for (auto& [reg, rank] : rexFloatRank)
            println("\tfloat4 xe_fc{} : packoffset(c{});", reg, rank);
        out += "};\n";
        out += "#else\n";
        // Same names off the push-constant float buffer (compacted rank
        // offsets, the runtime uploads the compacted layout).
        for (auto& [reg, rank] : rexFloatRank)
            println("#define xe_fc{} vk::RawBufferLoad<float4>(XE_PUSH_FLOATS + {} * 16, 4)", reg, rank);
        out += "#endif\n\n";

        // Samplers still come from the constant table.
        for (uint32_t i = 0; i < constantTableContainer->constantTable.constants; i++)
        {
            const auto constantInfo = reinterpret_cast<const ConstantInfo*>(
                constantTableData + constantTableContainer->constantTable.constantInfo + i * sizeof(ConstantInfo));
            if (constantInfo->registerSet == RegisterSet::Sampler)
            {
                const char* constantName = reinterpret_cast<const char*>(constantTableData + constantInfo->name);
                samplers.emplace(constantInfo->registerIndex, constantName);
            }
        }
    }
    else
    {
    out += "#ifndef __spirv__\n";
    println("cbuffer XeFloatConstants : register(b1, space0)");
    out += "{\n";

    // Emit float constants SORTED BY REGISTER: declaration order carries no
    // meaning with packoffset, but DXC's SPIR-V layout checker misreports
    // out-of-order members as "packoffset caused overlap" and fails the
    // compile (the container's table is name-ordered, e.g. g_cameraPos c13
    // before g_matWorld c0).
    std::vector<const ConstantInfo*> floatInfos;
    for (uint32_t i = 0; i < constantTableContainer->constantTable.constants; i++)
    {
        const auto constantInfo = reinterpret_cast<const ConstantInfo*>(
            constantTableData + constantTableContainer->constantTable.constantInfo + i * sizeof(ConstantInfo));

        if (constantInfo->registerSet == RegisterSet::Float4)
        {
            floatInfos.push_back(constantInfo);
        }
        else if (constantInfo->registerSet == RegisterSet::Sampler)
        {
            const char* constantName = reinterpret_cast<const char*>(constantTableData + constantInfo->name);
            samplers.emplace(constantInfo->registerIndex, constantName);
        }
    }
    std::stable_sort(floatInfos.begin(), floatInfos.end(),
        [](const ConstantInfo* a, const ConstantInfo* b) { return a->registerIndex.get() < b->registerIndex.get(); });
    // Registers the table does not name (the def literals) get their own members, merged in register order.
    auto extra = rexExtraFloatRegs.begin();
    auto printExtraBelow = [&](uint32_t limit)
        {
            for (; extra != rexExtraFloatRegs.end() && *extra < limit; ++extra)
                println("\tfloat4 xe_fc{0} : packoffset(c{0});", *extra);
        };
    for (const ConstantInfo* constantInfo : floatInfos)
    {
        const char* constantName = reinterpret_cast<const char*>(constantTableData + constantInfo->name);

        printExtraBelow(constantInfo->registerIndex.get());
        print("\tfloat4 {}", constantName);
        if (constantInfo->registerCount > 1)
            print("[{}]", constantInfo->registerCount.get());
        println(" : packoffset(c{});", constantInfo->registerIndex.get());

        for (uint16_t j = 0; j < constantInfo->registerCount; j++)
            float4Constants.emplace(constantInfo->registerIndex + j, constantInfo);
    }
    printExtraBelow(UINT32_MAX);

    out += "};\n\n";

    // Dynamic-indexing tail clamp for multi-register constants (guest float
    // file is 256 registers for both stages).
    for (uint32_t i = 0; i < constantTableContainer->constantTable.constants; i++)
    {
        const auto constantInfo = reinterpret_cast<const ConstantInfo*>(
            constantTableData + constantTableContainer->constantTable.constantInfo + i * sizeof(ConstantInfo));

        if (constantInfo->registerSet == RegisterSet::Float4 && constantInfo->registerCount > 1)
        {
            const char* constantName = reinterpret_cast<const char*>(constantTableData + constantInfo->name);
            uint32_t tailCount = 256 - constantInfo->registerIndex;
            println("#define {0}(INDEX) select((INDEX) < {1}, {0}[min(INDEX, {2})], 0.0)", constantName, tailCount, tailCount - 1);
        }
    }

    // SPIR-V half: identical names as RawBufferLoad macros off the push-
    // constant float buffer (the runtime uploads the ABSOLUTE register
    // layout for table shaders, offsets are registerIndex * 16).
    out += "#else\n";
    for (const ConstantInfo* constantInfo : floatInfos)
    {
        const char* constantName = reinterpret_cast<const char*>(constantTableData + constantInfo->name);
        if (constantInfo->registerCount > 1)
        {
            uint32_t tailCount = 256 - constantInfo->registerIndex;
            println("#define {0}(INDEX) (((INDEX) < {1}) ? vk::RawBufferLoad<float4>(XE_PUSH_FLOATS + ({2} + uint(min(INDEX, {3}))) * 16, 4) : (float4)0.0)",
                constantName, tailCount, constantInfo->registerIndex.get(), tailCount - 1);
        }
        else
        {
            println("#define {} vk::RawBufferLoad<float4>(XE_PUSH_FLOATS + {} * 16, 4)",
                constantName, constantInfo->registerIndex.get());
        }
    }
    for (uint32_t reg : rexExtraFloatRegs)
        println("#define xe_fc{0} vk::RawBufferLoad<float4>(XE_PUSH_FLOATS + {0} * 16, 4)", reg);
    out += "#endif\n";
    }

    // Guest booleans out of the b2 bool/loop word pair: VS uses guest
    // b0-b15 (word 0), PS uses guest b128-143 (word 1) shifted to bits
    // 16-31 to match the (1 << (reg + 16)) bool defines.
    println("#define g_Booleans {}", isPixelShader ? "((xe_bool_word(1)) << 16)" : "(xe_bool_word(0))");
    out += "\n";

    // Pre-scan the code and replicate the translator's binding allocation
    // (FindOrAddSamplerBinding / FindOrAddTextureBinding order and keys) so
    // the b4 descriptor-index slots the native shader reads are the ones the
    // runtime uploads for the translated shader's binding lists.
    const auto shader = reinterpret_cast<const Shader*>(shaderData + shaderContainer->shaderOffset);
    const uint32_t* codeWords;
    size_t codeDwords;
    if (rexCodeOverride != nullptr)
    {
        codeWords = reinterpret_cast<const uint32_t*>(rexCodeOverride);
        codeDwords = rexCodeOverrideSize / sizeof(uint32_t);
    }
    else
    {
        codeWords = reinterpret_cast<const uint32_t*>(shaderData + shaderContainer->virtualSize + shader->physicalOffset);
        codeDwords = shader->size / sizeof(uint32_t);
    }

    ucodeVisitFetchSlots(codeWords, codeDwords, true,
        [](const uint32_t* instructionDwords, void* context)
        {
            auto& self = *static_cast<ShaderRecompiler*>(context);

            union
            {
                TextureFetchInstruction tfetch;
                struct { uint32_t d0, d1, d2; };
            };
            d0 = instructionDwords[0];
            d1 = instructionDwords[1];
            d2 = instructionDwords[2];

            // Only these two opcodes allocate bindings in the translator
            // (ProcessTextureFetchInstruction); GetTextureWeights reads sizes
            // from the fetch constant and allocates nothing.
            if (tfetch.opcode != FetchOpcode::TextureFetch &&
                tfetch.opcode != FetchOpcode::GetTextureComputedLod)
                return;

            uint32_t constIndex = tfetch.constIndex;

            // Sampler first, matching the translator's call order.
            self.rexAddSamplerBinding(tfetch);

            // Textures: unsigned then signed; 3D fetches allocate the 3D pair
            // then the stacked-2D pair.
            uint32_t dimension = uint32_t(tfetch.dimension);
            uint32_t passCount = (dimension == 2) ? 2u : 1u; // 3DOrStacked
            for (uint32_t pass = 0; pass < passCount; pass++)
            {
                uint32_t srvDimension = pass ? 1u /* k2D */ : dimension;
                if (srvDimension == 0) // 1D bound as 2D
                    srvDimension = 1;
                for (uint32_t isSigned = 0; isSigned < 2; isSigned++)
                {
                    auto key = std::make_tuple(constIndex, srvDimension, isSigned);
                    if (self.rexTextureSlots.try_emplace(key, uint32_t(self.rexBindings.size())).second)
                        self.rexBindings.push_back({ uint32_t(self.rexBindings.size()), false, constIndex,
                                                     srvDimension, isSigned != 0 });
                }
            }
        }, this);

    // Vfetch ordinals in control-flow walk order, which is the record order of the per-draw layout table.
    ucodeVisitVfetchSlots(codeWords, codeDwords, true,
        [](uint32_t slot, const uint32_t*, void* context)
        {
            static_cast<ShaderRecompiler*>(context)->rexVfetchSlots.push_back(slot);
        }, this);

    if (rexVfetchSlots.size() > 32)
        throw std::runtime_error(fmt::format("{} vfetches exceed the 32-record layout table", rexVfetchSlots.size()));

    if (!isPixelShader)
    {
        const std::string marker = "//@XE_VFETCH_STATIC_RECORDS@\n";
        const size_t at = out.find(marker);
        if (at != std::string::npos)
        {
            xenosrecomp::XeVfetchRecord records[64] = {};
            const uint32_t count = xenosrecomp::ReadVfetchRecords(std::span<const uint32_t>(codeWords, codeDwords), true, records);
            if (count != rexVfetchSlots.size())
                throw std::runtime_error(fmt::format("{} layout records for {} vfetch ordinals", count, rexVfetchSlots.size()));
            std::string text = "static const uint4 xe_vfetch_static_records[32] =\n{\n";
            for (uint32_t i = 0; i < 32; i++)
            {
                const xenosrecomp::XeVfetchRecord r = i < count ? records[i] : xenosrecomp::XeVfetchRecord{};
                text += fmt::format("    uint4(0x{:08X}u, 0x{:08X}u, 0x{:08X}u, 0x{:08X}u),\n", r.word0, r.stride_dwords,
                                    uint32_t(r.offset_dwords), r.dst_swizzle);
            }
            text += "};\n\nuint4 xe_vfetch_record(uint ordinal)\n{\n    return xe_vfetch_static_records[ordinal];\n}\n";
            out.replace(at, marker.size(), text);
        }
    }

    std::unordered_map<uint32_t, std::pair<uint32_t, uint32_t>> elementUsage;
    if (!isPixelShader)
    {
        auto vertexShader = reinterpret_cast<const VertexShader*>(shader);
        for (uint32_t i = 0; i < vertexShader->vertexElementCount; i++)
        {
            union
            {
                VertexElement vertexElement;
                uint32_t value;
            };
            value = vertexShader->vertexElementsAndInterpolators[vertexShader->field18 + i];
            elementUsage.try_emplace(uint32_t(vertexElement.address),
                uint32_t(vertexElement.usage), uint32_t(vertexElement.usageIndex));
        }
    }
    for (uint32_t slot : rexVfetchSlots)
    {
        auto it = elementUsage.find(slot);
        if (it != elementUsage.end())
            rexVfetchRefl.push_back({ slot, it->second.first, it->second.second });
        else
            rexVfetchRefl.push_back({ slot, 0xFF, 0 });
    }
}

// Mirrors FindOrAddSamplerBinding's key normalization and slot allocation.
std::string ShaderRecompiler::boolCondition(uint32_t boolAddress, bool whenSet)
{
#ifdef REBLUE_RECOMP
    if (!rexglueMode)
    {
        auto findResult = boolConstants.find(boolAddress);
        if (findResult != boolConstants.end())
            return fmt::format("{}{}", whenSet ? "" : "!", findResult->second);
        return fmt::format("{}BOOL_BIT({})", whenSet ? "" : "!", boolAddress);
    }
#endif
    // PS bools sit at hardware slots 128-255 while the constant table indexes them from 0.
    // Without a table entry the g_Booleans bit is used directly, PS registers from bit 16.
    uint32_t localAddress = boolAddress;
    if (isPixelShader && localAddress >= 128)
        localAddress -= 128;

    const char* compare = whenSet ? "!" : "=";
    const uint32_t maskBit = localAddress + (isPixelShader ? 16 : 0);
    if (rexglueMode && maskBit < 32)
        rexBoolMask |= 1u << maskBit;
    auto findResult = boolConstants.find(localAddress);
    if (findResult != boolConstants.end())
        return fmt::format("(g_Booleans & {}) {}= 0", findResult->second, compare);
    if (rexglueMode)
        return fmt::format("(g_Booleans & (1 << {})) {}= 0", localAddress + (isPixelShader ? 16 : 0), compare);
    return fmt::format("b{} {}= 0", boolAddress, compare);
}

uint32_t ShaderRecompiler::rexAddSamplerBinding(const TextureFetchInstruction& instr)
{
    uint32_t magFilter = instr.magFilter, minFilter = instr.minFilter, mipFilter = instr.mipFilter;
    uint32_t aniso;
    if (instr.opcode == FetchOpcode::GetTextureComputedLod)
    {
        // getCompTexLOD forces linear mip filtering.
        mipFilter = 1;
        aniso = instr.anisoFilter;
    }
    else
    {
        // Anisotropic filtering only applies to computed LODs, which vertex
        // shaders get only with register gradients (dxbc_translator_fetch.cpp:676).
        bool computedLod = instr.useCompLod && (isPixelShader || instr.useRegGradients);
        aniso = computedLod ? uint32_t(instr.anisoFilter) : 0u;
    }
    // Direct3D 12 can't mix anisotropic and point filtering (0=disabled,
    // 7=use fetch constant, 5=max 16:1).
    if (aniso != 0 && aniso != 7)
    {
        magFilter = minFilter = mipFilter = 1;
        if (aniso > 5)
            aniso = 5;
    }

    auto key = std::make_tuple(uint32_t(instr.constIndex), magFilter, minFilter, mipFilter, aniso);
    auto result = rexSamplerSlots.try_emplace(key, uint32_t(rexBindings.size()));
    if (result.second)
        rexBindings.push_back({ uint32_t(rexBindings.size()), true, uint32_t(instr.constIndex), 0, false,
                                magFilter, minFilter, mipFilter, aniso });
    return result.first->second;
}

void ShaderRecompiler::recompile(const uint8_t* shaderData, const std::string_view& include)
{
    const auto shaderContainer = reinterpret_cast<const ShaderContainer*>(shaderData);

    assert((shaderContainer->flags & 0xFFFFFF00) == 0x102A1100);
    assert(shaderContainer->constantTableOffset != NULL);

    isPixelShader = (shaderContainer->flags & 0x1) == 0;

    // The rexglue common header selects the per-stage push-constant members
    // (XE_PUSH_FLOATS / XE_PUSH_IDX) from this define under __spirv__.
    if (rexglueMode && isPixelShader)
        out += "#define XE_PIXEL_SHADER\n";

    // Vertex fetch layout records are fixed per patched shader (the pack keys on the patched code and the runtime
    // fills b5 from the same walk), so the vertex shader gets them as literals: every layout and fetch-constant
    // lookup then folds to a static index instead of a per-vertex dynamic constant-buffer read.
    if (rexglueMode && !isPixelShader)
        out += "#define XE_VFETCH_STATIC 1\n";

    out += include;
    out += '\n';
    if (rexglueMode && !isPixelShader)
        out += "//@XE_VFETCH_STATIC_RECORDS@\n";

    const auto constantTableContainer = reinterpret_cast<const ConstantTableContainer*>(shaderData + shaderContainer->constantTableOffset);
    constantTableData = reinterpret_cast<const uint8_t*>(&constantTableContainer->constantTable);

    if (rexglueMode)
    {
        emitRexglueDeclarations(shaderData);
    }
    else
    {
    // NOTE: this brace scopes the UNLEASHED_RECOMP locals declared below out
    // of the element-emission code, the fork does not build UNLEASHED_RECOMP.
    out += "#ifdef __spirv__\n\n";

#ifdef UNLEASHED_RECOMP
    bool isMetaInstancer = false;
    bool hasIndexCount = false;
#endif

    for (uint32_t i = 0; i < constantTableContainer->constantTable.constants; i++)
    {
        const auto constantInfo = reinterpret_cast<const ConstantInfo*>(
            constantTableData + constantTableContainer->constantTable.constantInfo + i * sizeof(ConstantInfo));

        const char* constantName = reinterpret_cast<const char*>(constantTableData + constantInfo->name);

#ifdef REBLUE_RECOMP
        if (isPixelShader && constantInfo->registerSet == RegisterSet::Sampler &&
            strcmp(constantName, "ShadowTexture") == 0)
        {
            hasShadowTexture = true;
        }
#endif

    #ifdef UNLEASHED_RECOMP
        if (!isPixelShader)
        {
            if (strcmp(constantName, "g_MtxProjection") == 0)
                hasMtxProjection = true;
            else if (strcmp(constantName, "g_InstanceTypes") == 0)
                isMetaInstancer = true;
            else if (strcmp(constantName, "g_IndexCount") == 0)
                hasIndexCount = true;
        }
        else
        {
            if (strcmp(constantName, "g_MtxPrevInvViewProjection") == 0)
                hasMtxPrevInvViewProjection = true;
        }
    #endif

        switch (constantInfo->registerSet)
        {
        case RegisterSet::Float4:
        {
            const char* shaderName = isPixelShader ? "Pixel" : "Vertex";

            if (constantInfo->registerCount > 1)
            {
                uint32_t tailCount = (isPixelShader ? 224 : 256) - constantInfo->registerIndex;

                println("#define {}(INDEX) select((INDEX) < {}, vk::RawBufferLoad<float4>(g_PushConstants.{}ShaderConstants + ({} + min(INDEX, {})) * 16, 0x10), 0.0)",
                    constantName, tailCount, shaderName, constantInfo->registerIndex.get(), tailCount - 1);
            }
            else
            {
                println("#define {} vk::RawBufferLoad<float4>(g_PushConstants.{}ShaderConstants + {}, 0x10)",
                    constantName, shaderName, constantInfo->registerIndex * 16);
            }
            
#ifdef REBLUE_RECOMP
            // BD aliases a singleton constant over an array slot; the wider registrant wins so body and cbuffer agree.
            for (uint16_t j = 0; j < constantInfo->registerCount; j++)
            {
                uint32_t reg = constantInfo->registerIndex + j;
                auto it = float4Constants.find(reg);
                if (it == float4Constants.end() || it->second->registerCount < constantInfo->registerCount)
                    float4Constants[reg] = constantInfo;
            }
#else
            for (uint16_t j = 0; j < constantInfo->registerCount; j++)
                float4Constants.emplace(constantInfo->registerIndex + j, constantInfo);
#endif

            break;
        }

        case RegisterSet::Sampler:
        {
            for (size_t j = 0; j < std::size(TEXTURE_DIMENSIONS); j++)
            {
                println("#define {}_Texture{}DescriptorIndex vk::RawBufferLoad<uint>(g_PushConstants.SharedConstants + {})",
                    constantName, TEXTURE_DIMENSIONS[j], j * 64 + constantInfo->registerIndex * 4);
            }

            println("#define {}_SamplerDescriptorIndex vk::RawBufferLoad<uint>(g_PushConstants.SharedConstants + {})",
                constantName, std::size(TEXTURE_DIMENSIONS) * 64 + constantInfo->registerIndex * 4);

            samplers.emplace(constantInfo->registerIndex, constantName);
            break;
        }

        }
    }

#ifdef REBLUE_RECOMP
    // BD bodies reference unnamed sampler slots; emit fallback descriptor-index defines for any slot 0..15 the table didn't name.
    for (uint32_t r = 0; r < 16; r++)
    {
        if (samplers.find(r) != samplers.end())
            continue;
        for (size_t j = 0; j < std::size(TEXTURE_DIMENSIONS); j++)
        {
            println("#define s{}_Texture{}DescriptorIndex vk::RawBufferLoad<uint>(g_PushConstants.SharedConstants + {})",
                r, TEXTURE_DIMENSIONS[j], j * 64 + r * 4);
        }
        println("#define s{}_SamplerDescriptorIndex vk::RawBufferLoad<uint>(g_PushConstants.SharedConstants + {})",
            r, std::size(TEXTURE_DIMENSIONS) * 64 + r * 4);
    }
#endif

    out += "\n#else\n\n";

    println("cbuffer {}ShaderConstants : register(b{}, space4)", isPixelShader ? "Pixel" : "Vertex", isPixelShader ? 1 : 0);
    out += "{\n";

    for (uint32_t i = 0; i < constantTableContainer->constantTable.constants; i++)
    {
        const auto constantInfo = reinterpret_cast<const ConstantInfo*>(
            constantTableData + constantTableContainer->constantTable.constantInfo + i * sizeof(ConstantInfo));

        if (constantInfo->registerSet == RegisterSet::Float4)
        {
#ifdef REBLUE_RECOMP
            // Only the alias winner gets a packoffset slot; a loser would overlap it in the cbuffer and fail DXC.
            auto winner = float4Constants.find(constantInfo->registerIndex);
            if (winner == float4Constants.end() || winner->second != constantInfo)
                continue;
#endif

            const char* constantName = reinterpret_cast<const char*>(constantTableData + constantInfo->name);

            print("\tfloat4 {}", constantName);

            if (constantInfo->registerCount > 1)
                print("[{}]", constantInfo->registerCount.get());

            println(" : packoffset(c{});", constantInfo->registerIndex.get());

            if (constantInfo->registerCount > 1)
            {
                uint32_t tailCount = (isPixelShader ? 224 : 256) - constantInfo->registerIndex;
                println("#define {0}(INDEX) select((INDEX) < {1}, {0}[min(INDEX, {2})], 0.0)", constantName, tailCount, tailCount - 1);
            }
        }
    }

    out += "};\n\n";

    out += "cbuffer SharedConstants : register(b2, space4)\n";
    out += "{\n";

    for (uint32_t i = 0; i < constantTableContainer->constantTable.constants; i++)
    {
        const auto constantInfo = reinterpret_cast<const ConstantInfo*>(
            constantTableData + constantTableContainer->constantTable.constantInfo + i * sizeof(ConstantInfo));

        if (constantInfo->registerSet == RegisterSet::Sampler)
        {
            const char* constantName = reinterpret_cast<const char*>(constantTableData + constantInfo->name);

            for (size_t j = 0; j < std::size(TEXTURE_DIMENSIONS); j++)
            {
                println("\tuint {}_Texture{}DescriptorIndex : packoffset(c{}.{});",
                    constantName, TEXTURE_DIMENSIONS[j], j * 4 + constantInfo->registerIndex / 4, SWIZZLES[constantInfo->registerIndex % 4]);
            }

            println("\tuint {}_SamplerDescriptorIndex : packoffset(c{}.{});",
                constantName, 4 * std::size(TEXTURE_DIMENSIONS) + constantInfo->registerIndex / 4, SWIZZLES[constantInfo->registerIndex % 4]);
        }
    }

#ifdef REBLUE_RECOMP
    // Mirror the SPIR-V fallback: packoffset slots for any unnamed sampler index 0..15.
    for (uint32_t r = 0; r < 16; r++)
    {
        if (samplers.find(r) != samplers.end())
            continue;
        for (size_t j = 0; j < std::size(TEXTURE_DIMENSIONS); j++)
        {
            println("\tuint s{}_Texture{}DescriptorIndex : packoffset(c{}.{});",
                r, TEXTURE_DIMENSIONS[j], j * 4 + r / 4, SWIZZLES[r % 4]);
        }
        println("\tuint s{}_SamplerDescriptorIndex : packoffset(c{}.{});",
            r, 4 * std::size(TEXTURE_DIMENSIONS) + r / 4, SWIZZLES[r % 4]);
    }
#endif

    out += "\tDEFINE_SHARED_CONSTANTS();\n";
    out += "};\n\n";

    out += "#endif\n";
    }

    for (uint32_t i = 0; i < constantTableContainer->constantTable.constants; i++)
    {
        const auto constantInfo = reinterpret_cast<const ConstantInfo*>(
            constantTableData + constantTableContainer->constantTable.constantInfo + i * sizeof(ConstantInfo));

        if (constantInfo->registerSet == RegisterSet::Bool)
        {
            const char* constantName = reinterpret_cast<const char*>(constantTableData + constantInfo->name);
#ifdef REBLUE_RECOMP
            if (!rexglueMode)
            {
                // Key named bools by the unified VS(0..127)/PS(128..255) bit so CF tests (also unified) resolve them.
                const uint32_t unifiedBit = uint32_t(constantInfo->registerIndex) + (isPixelShader ? 128u : 0u);
                println("\t#define {} BOOL_BIT({})", constantName, unifiedBit);
                boolConstants.emplace(unifiedBit, constantName);
                continue;
            }
#endif
            println("\t#define {} (1 << {})", constantName, constantInfo->registerIndex + (isPixelShader ? 16 : 0));
            boolConstants.emplace(constantInfo->registerIndex, constantName);
        }
    }

    out += '\n';

    const auto shader = reinterpret_cast<const Shader*>(shaderData + shaderContainer->shaderOffset);

    out += "#ifndef __spirv__\n";

    if (isPixelShader)
        out += "[shader(\"pixel\")]\n";
    else
        out += "[shader(\"vertex\")]\n";

    out += "#endif\n";

    out += "void main(\n";

    if (isPixelShader)
    {
        out += "\tin float4 iPos : SV_Position,\n";

        if (rexglueMode)
        {
            // Xenos wires VS exports to PS input registers by index (o{i} -> interpolator entry i),
            // whatever D3DX semantic names the paired containers carry.
            //
            // Only the registers the interpolator table maps are declared: D3D12 links PS inputs by
            // semantic against the VS's full oVar0-15 signature, so a sparse subset is legal and unread
            // inputs stop costing interpolation.
            {
                auto psShader = reinterpret_cast<const PixelShader*>(shader);
                uint32_t readMask = 0;
                uint32_t readCount = (shader->interpolatorInfo >> 5) & 0x1F;
                for (uint32_t i = 0; i < readCount; i++)
                {
                    union
                    {
                        Interpolator interpolator;
                        uint32_t value;
                    };
                    value = psShader->interpolators[i];
                    if (uint32_t(interpolator.reg) < 16)
                        readMask |= 1u << uint32_t(interpolator.reg);
                }
                rexPsReadMask = readMask;
                // Explicit locations keep a sparse input list linked to the right VS outputs on Vulkan.
                for (uint32_t i = 0; i < 16; i++)
                {
                    if (readMask & (1u << i))
                        println("\t[[vk::location({0})]] in float4 iVar{0} : TEXCOORD{0},", i);
                }
            }
        }
        else
        for (auto& [usage, usageIndex] : INTERPOLATORS)
        {
#ifdef REBLUE_RECOMP
            const char* interpolation = (usage == DeclUsage::Color) ? "centroid " : "";
#else
            const char* interpolation = "";
#endif
            println("\tin {3}float4 i{0}{1} : {2}{1},", USAGE_VARIABLES[uint32_t(usage)], usageIndex, USAGE_SEMANTICS[uint32_t(usage)], interpolation);
        }

        out += "#ifdef __spirv__\n";
        out += "\tin bool iFace : SV_IsFrontFace\n";
        out += "#else\n";
        out += "\tin uint iFace : SV_IsFrontFace\n";
        out += "#endif\n";

        auto pixelShader = reinterpret_cast<const PixelShader*>(shader);
        psOutputsMask = pixelShader->outputs;
        if (pixelShader->outputs & PIXEL_SHADER_OUTPUT_COLOR0)
            out += ",\n\tout float4 oC0 : SV_Target0";
        if (pixelShader->outputs & PIXEL_SHADER_OUTPUT_COLOR1)
            out += ",\n\tout float4 oC1 : SV_Target1";
        if (pixelShader->outputs & PIXEL_SHADER_OUTPUT_COLOR2)
            out += ",\n\tout float4 oC2 : SV_Target2";
        if (pixelShader->outputs & PIXEL_SHADER_OUTPUT_COLOR3)
            out += ",\n\tout float4 oC3 : SV_Target3";
        if (pixelShader->outputs & PIXEL_SHADER_OUTPUT_DEPTH)
            out += ",\n\tout float oDepth : SV_Depth";
    }
    else
    {
        auto vertexShader = reinterpret_cast<const VertexShader*>(shader);

        if (rexglueMode)
        {
            // No input assembler: vertex data is fetched in-shader from the
            // shared-memory buffer (guest data is big-endian). Only the
            // vertex id comes in.
            out += "\tin uint iVertexId : SV_VertexID,\n";
        }
        else
        {
        // Duplicate (usage, usageIndex) declarations are legal on 360: the same
        // semantic fetched from several vertex streams (blend-shape streams
        // declare TexCoord6-8 several times at different fetch addresses).
        // Uniquify the HLSL parameter name/semantic per duplicate; the fetch
        // site resolves by address through vertexElements.
        std::unordered_map<uint32_t, uint32_t> usageCounts;
        uint32_t extraLocation = 16;
        uint32_t extraSemantic = 90;

        for (uint32_t i = 0; i < vertexShader->vertexElementCount; i++)
        {
            union
            {
                VertexElement vertexElement;
                uint32_t value;
            };

            value = vertexShader->vertexElementsAndInterpolators[vertexShader->field18 + i];

            const char* usageType = USAGE_TYPES[uint32_t(vertexElement.usage)];

        #ifdef UNLEASHED_RECOMP
            if ((vertexElement.usage == DeclUsage::TexCoord && vertexElement.usageIndex == 2 && isMetaInstancer) ||
                (vertexElement.usage == DeclUsage::Position && vertexElement.usageIndex == 1))
            {
                usageType = "uint4";
            }
        #endif

            uint32_t duplicate = usageCounts[(uint32_t(vertexElement.usage) << 8) | vertexElement.usageIndex]++;

            std::string name = fmt::format("i{}{}", USAGE_VARIABLES[uint32_t(vertexElement.usage)],
                uint32_t(vertexElement.usageIndex));
            std::string semantic = fmt::format("{}{}", USAGE_SEMANTICS[uint32_t(vertexElement.usage)],
                uint32_t(vertexElement.usageIndex));

            if (duplicate != 0)
            {
                name += fmt::format("_{}", duplicate);
                semantic = fmt::format("TEXCOORD{}", extraSemantic++);
            }

            out += '\t';

            bool foundLocation = false;
            if (duplicate == 0)
            {
                for (auto& usageLocation : USAGE_LOCATIONS)
                {
                    if (usageLocation.usage == vertexElement.usage && usageLocation.usageIndex == vertexElement.usageIndex)
                    {
                        print("[[vk::location({})]] ", usageLocation.location);
                        foundLocation = true;
                        break;
                    }
                }
            }
            if (!foundLocation)
                print("[[vk::location({})]] ", extraLocation++);

            println("in {0} {1} : {2},", usageType, name, semantic);

            vertexElements.emplace(uint32_t(vertexElement.address),
                VertexElementInfo{ vertexElement.usage, uint32_t(vertexElement.usageIndex), std::move(name) });
        }
        }

    #ifdef UNLEASHED_RECOMP
        if (hasIndexCount)
        {
            out += "\tin uint iVertexId : SV_VertexID,\n";
            out += "\tin uint iInstanceId : SV_InstanceID,\n";
        }
    #endif

        out += "\tout float4 oPos : SV_Position";

        if (rexglueMode)
        {
            // Index-wired varyings (see the pixel shader input note).
            for (uint32_t i = 0; i < 16; i++)
                print(",\n\t[[vk::location({0})]] out float4 oVar{0} : TEXCOORD{0}", i);
            // Point size (export register 63, x component), declared on
            // Every rexglue VS so the whole pack shares one output signature
            // and the runtime's point_expand GS links against any of them.
            out += ",\n\t[[vk::location(16)]] out float4 oPts : TEXCOORD16";
        }
        else
        for (auto& [usage, usageIndex] : INTERPOLATORS)
            print(",\n\tout float4 o{0}{1} : {2}{1}", USAGE_VARIABLES[uint32_t(usage)], usageIndex, USAGE_SEMANTICS[uint32_t(usage)]);
    }

    out += ")\n";
    out += "{\n";

#ifdef UNLEASHED_RECOMP
    if (hasMtxProjection)
    {
        specConstantsMask |= SPEC_CONSTANT_REVERSE_Z;

        out += "\toPos = 0.0;\n";

        out += "\tfloat4x4 mtxProjection = float4x4(g_MtxProjection(0), g_MtxProjection(1), g_MtxProjection(2), g_MtxProjection(3));\n";
        out += "\tfloat4x4 mtxProjectionReverseZ = mul(mtxProjection, float4x4(1, 0, 0, 0, 0, 1, 0, 0, 0, 0, -1, 0, 0, 0, 1, 1));\n";

        out += "\t[unroll] for (int iterationIndex = 0; iterationIndex < 2; iterationIndex++)\n";
        out += "\t{\n";
    }
#endif

    if (shaderContainer->definitionTableOffset != NULL)
    {
        auto definitionTable = reinterpret_cast<const DefinitionTable*>(shaderData + shaderContainer->definitionTableOffset);
        auto definitions = definitionTable->definitions;
        while (*definitions != 0)
        {
            auto definition = reinterpret_cast<const Float4Definition*>(definitions);
            const size_t literalBytes = size_t((definition->count + 3) / 4) * 16;
            // Literal bits live in the physical data: after the virtual part of a full container, or in the
            // separate physical span a runtime shader object provides.
            const be<uint32_t>* value = nullptr;
            if (rexPhysicalData != nullptr)
            {
                if (size_t(definition->physicalOffset) + literalBytes <= rexPhysicalSize)
                    value = reinterpret_cast<const be<uint32_t>*>(rexPhysicalData + definition->physicalOffset);
            }
            else if (rexContainerSize == 0 ||
                size_t(shaderContainer->virtualSize) + definition->physicalOffset + literalBytes <= rexContainerSize)
            {
                value = reinterpret_cast<const be<uint32_t>*>(shaderData + shaderContainer->virtualSize + definition->physicalOffset);
            }

            // Rexglue mode reads every literal register from b1 and only reports the literals, since their values
            // belong to the shader object rather than the microcode (pack_contract.md O-1).
            if (rexglueMode)
            {
                for (uint16_t i = 0; i < (definition->count + 3) / 4; i++)
                {
                    RexLiteralRefl literal{ uint32_t(definition->registerIndex + i - (isPixelShader ? 256 : 0)), {}, value != nullptr };
                    if (value != nullptr)
                    {
                        for (uint32_t k = 0; k < 4; k++)
                            literal.value[k] = value[i * 4 + k].get();
                    }
                    rexLiterals.push_back(literal);
                }
                definitions += 2;
                continue;
            }
            if (value == nullptr)
                throw std::runtime_error("definition table literals lie outside the container");

            for (uint16_t i = 0; i < (definition->count + 3) / 4; i++)
            {
#ifdef REBLUE_RECOMP
                // BD bakes its sun-shadow PCF tap offsets as literals in 1/1024 UV units around 0.5 and 1.0 anchors, and
                // those within 3/1024 of an anchor are rescaled by g_ShadowPcfScale (c20.z), which is identity at 1.0.
                auto shadowKernelComponent = [&](uint32_t u) -> std::string
                {
                    if (hasShadowTexture)
                    {
                        if (u == 0x3A800000)
                            return fmt::format("(asfloat(0x{:X}u) * g_ShadowPcfScale)", u);
                        float f;
                        memcpy(&f, &u, sizeof(f));
                        for (float anchor : { 0.5f, 1.0f })
                        {
                            float d = f > anchor ? f - anchor : anchor - f;
                            if (f != anchor && d <= 3.0f / 1024.0f)
                                return fmt::format("({} + (asfloat(0x{:X}u) - {}) * g_ShadowPcfScale)", anchor, u, anchor);
                        }
                    }
                    return fmt::format("asfloat(0x{:X}u)", u);
                };
                println("\tfloat4 c{} = float4({}, {}, {}, {});",
                    definition->registerIndex + i - (isPixelShader ? 256 : 0),
                    shadowKernelComponent(value[0].get()), shadowKernelComponent(value[1].get()),
                    shadowKernelComponent(value[2].get()), shadowKernelComponent(value[3].get()));
#else
                println("\tfloat4 c{} = asfloat(uint4(0x{:X}, 0x{:X}, 0x{:X}, 0x{:X}));",
                    definition->registerIndex + i - (isPixelShader ? 256 : 0), value[0].get(), value[1].get(), value[2].get(), value[3].get());
#endif

                value += 4;
            }
            definitions += 2;
        }
        ++definitions;
        while (*definitions != 0)
        {
            auto definition = reinterpret_cast<const Int4Definition*>(definitions);
            for (uint16_t i = 0; i < definition->count; i++)
            {
                union
                {
                    uint32_t value;
                    struct
                    {
                        int8_t x;
                        int8_t y;
                        int8_t z;
                        int8_t w;
                    };
                };

                value = definition->values[i].get();

                println("\tint4 i{} = int4({}, {}, {}, {});",
                    (definition->registerIndex - 8992) / 4 + i, x, y, z, w);
            }
            definitions += 2;
            definitions += definition->count;
        }

        out += "\n";
    }

    bool printedRegisters[32]{};

    uint32_t interpolatorCount = (shader->interpolatorInfo >> 5) & 0x1F;

    for (uint32_t i = 0; i < interpolatorCount; i++)
    {
        union
        {
            Interpolator interpolator;
            uint32_t value;
        };
    
        if (isPixelShader)
        {
            value = reinterpret_cast<const PixelShader*>(shader)->interpolators[i];
            rexInterps.push_back({ uint32_t(interpolator.reg), uint32_t(interpolator.usage), uint32_t(interpolator.usageIndex) });
            if (rexglueMode)
            {
                // Identity wiring: PS register r{n} receives VS export o{n}.
                // Container entries are in semantic-table order, not register
                // order, use the entry's register on both sides.
                println("\tfloat4 r{0} = iVar{0};", uint32_t(interpolator.reg));
            }
            else
                println("\tfloat4 r{} = i{}{};", uint32_t(interpolator.reg), USAGE_VARIABLES[uint32_t(interpolator.usage)], uint32_t(interpolator.usageIndex));
            printedRegisters[interpolator.reg] = true;
        }
        else
        {
            auto vertexShader = reinterpret_cast<const VertexShader*>(shader);
            value = vertexShader->vertexElementsAndInterpolators[vertexShader->field18 + vertexShader->vertexElementCount + i];
            rexInterps.push_back({ uint32_t(interpolator.reg), uint32_t(interpolator.usage), uint32_t(interpolator.usageIndex) });
            if (rexglueMode)
            {
                // Not used in rexglue mode (the export emitter goes straight
                // to oVar{vectorDest}), kept for map completeness.
                interpolators.emplace(i, fmt::format("oVar{}", i));
                // Trim safety: paired PS input tables link against the VS's
                // declared interpolators, and pairs routinely list registers
                // the ucode never writes; trimming those away turns a
                // harmless read into a PSO linkage failure. Keep every
                // table-declared register (and index, in case .reg is not
                // the register on the VS side) in the trimmed signature.
                rexWrittenOVarMask |= 1u << i;
                if (uint32_t(interpolator.reg) < 16)
                    rexWrittenOVarMask |= 1u << uint32_t(interpolator.reg);
            }
            else
                interpolators.emplace(i, fmt::format("o{}{}", USAGE_VARIABLES[uint32_t(interpolator.usage)], uint32_t(interpolator.usageIndex)));
        }
    }

    if (!isPixelShader)
    {
    #if defined(UNLEASHED_RECOMP)
        if (!hasMtxProjection)
            out += "\toPos = 0.0;\n";
    #elif defined(REBLUE_RECOMP)
        // Always define SV_Position so a skipped position-write block doesn't leave it undef.
        out += "\toPos = 0.0;\n";
    #endif

        if (rexglueMode)
        {
            for (uint32_t i = 0; i < 16; i++)
                println("\toVar{} = 0.0;", i);
            out += "\toPts = 0.0;\n";
        }
        else
        for (auto& [usage, usageIndex] : INTERPOLATORS)
            println("\to{}{} = 0.0;", USAGE_VARIABLES[uint32_t(usage)], usageIndex);

        out += "\n";
    }

    for (size_t i = 0; i < 32; i++)
    {
        if (!printedRegisters[i])
        {
            print("\tfloat4 r{} = ", i);
            if (isPixelShader && i == ((shader->fieldC >> 8) & 0xFF))
            {
                rexPixelPosReg = uint32_t(i);
                if (rexglueMode)
                {
                    // Guest pixel position register: xy = pixel coords
                    // (centers at .5), w sign = facing.
                    out += "float4(iPos.xy * xe_pixel_position_scale, 0.0, iFace ? 1.0 : -1.0);\n";
                }
                else
                {
                    out += "float4((iPos.xy - 0.5) * float2(iFace ? 1.0 : -1.0, 1.0), 0.0, 0.0);\n";
                }
            }
            else if (rexglueMode && !isPixelShader && i == 0)
            {
                // Guest vertex index register (see StartVertexShader_LoadVertexIndex).
                out += "float4(xe_vertex_index(iVertexId), 0.0, 0.0, 0.0);\n";
            }
        #ifdef UNLEASHED_RECOMP
            else if (!isPixelShader && hasIndexCount && i == 0)
            {
                out += "float4(iVertexId + g_IndexCount.x * iInstanceId, 0.0, 0.0, 0.0);\n";
            }
        #endif
            else
            {
                out += "0.0;\n";
            }
        }
    }

    out += "\tint a0 = 0;\n";
    out += "\tint aL = 0;\n";
    out += "\tbool p0 = false;\n";
    out += "\tfloat ps = 0.0;\n";
    out += "\tuint returnPc = 0;\n";
    const size_t fetchStatePos = out.size();
    if (isPixelShader)
    {
#ifdef UNLEASHED_RECOMP
        out += "\tfloat2 pixelCoord = 0.0;\n";
#endif
        out += "\tCubeMapData cubeMapData = (CubeMapData)0;\n";
#ifdef REBLUE_RECOMP
        if (hasShadowTexture)
            out += "\tfloat2 shadowTapUV[8] = (float2[8])0;\n";
#endif
    }

    const be<uint32_t>* code = rexCodeOverride != nullptr
        ? reinterpret_cast<const be<uint32_t>*>(rexCodeOverride)
        : reinterpret_cast<const be<uint32_t>*>(shaderData + shaderContainer->virtualSize + shader->physicalOffset);

    union
    {
        ControlFlowInstruction controlFlow[2];
        struct
        {
            uint32_t code0;
            uint32_t code1;
            uint32_t code2;
            uint32_t code3;
        };
    };

    uint32_t instrAddress = 0;
    uint32_t instrSize = rexCodeOverride != nullptr ? rexCodeOverrideSize : shader->size;

    // Subroutine inlining. A call that is unconditional and unpredicated, whose body is a straight run of clauses
    // ending in a return, is spliced in at its call site and the body dropped, so the program becomes call-free
    // and the structured or guard emitters below can take it instead of the pc dispatch loop. Anything else
    // (conditional calls, bodies with transfers, jumps into a body, growth past the cf region) keeps the original
    // stream. XENOSRECOMP_KEEP_CALLS=1 disables the pass for A/B builds.
    std::vector<be<uint32_t>> inlinedCf;
    if (getenv("XENOSRECOMP_KEEP_CALLS") == nullptr)
    {
        struct CfWord
        {
            uint32_t lo = 0;
            uint32_t hi = 0;
            ControlFlowInstruction instr{};
        };
        std::vector<CfWord> cfs;
        uint32_t cfSize = instrSize;
        uint32_t at = 0;
        auto scan = code;
        while (at < cfSize)
        {
            const uint32_t w0 = scan[0], w1 = scan[1], w2 = scan[2];
            const uint32_t lo[2] = { w0, (w1 >> 16) | (w2 << 16) };
            const uint32_t hi[2] = { w1 & 0xFFFF, w2 >> 16 };
            for (int k = 0; k < 2; k++)
            {
                CfWord c;
                c.lo = lo[k];
                c.hi = hi[k];
                const uint32_t words[2] = { c.lo, c.hi };
                memcpy(&c.instr, words, sizeof(words));
                uint32_t clauseAddress = 0;
                switch (c.instr.opcode)
                {
                case ControlFlowOpcode::Exec:
                case ControlFlowOpcode::ExecEnd:
                    clauseAddress = c.instr.exec.address;
                    break;
                case ControlFlowOpcode::CondExec:
                case ControlFlowOpcode::CondExecEnd:
                case ControlFlowOpcode::CondExecPredClean:
                case ControlFlowOpcode::CondExecPredCleanEnd:
                    clauseAddress = c.instr.condExec.address;
                    break;
                case ControlFlowOpcode::CondExecPred:
                case ControlFlowOpcode::CondExecPredEnd:
                    clauseAddress = c.instr.condExecPred.address;
                    break;
                default:
                    break;
                }
                if (clauseAddress != 0)
                    cfSize = std::min<uint32_t>(cfSize, clauseAddress * 12);
                cfs.push_back(c);
            }
            scan += 3;
            at += 12;
        }

        auto isClause = [](ControlFlowOpcode op)
        {
            return op == ControlFlowOpcode::Exec || op == ControlFlowOpcode::CondExec ||
                   op == ControlFlowOpcode::CondExecPred || op == ControlFlowOpcode::CondExecPredClean ||
                   op == ControlFlowOpcode::Nop || op == ControlFlowOpcode::Alloc ||
                   op == ControlFlowOpcode::MarkVsFetchDone;
        };
        auto isEnd = [](ControlFlowOpcode op)
        {
            return op == ControlFlowOpcode::ExecEnd || op == ControlFlowOpcode::CondExecEnd ||
                   op == ControlFlowOpcode::CondExecPredEnd || op == ControlFlowOpcode::CondExecPredCleanEnd ||
                   op == ControlFlowOpcode::Return;
        };

        const uint32_t n = uint32_t(cfs.size());
        struct Sub
        {
            uint32_t start = 0;
            uint32_t ret = 0;
        };
        std::map<uint32_t, Sub> subs;
        bool ok = true;
        for (uint32_t i = 0; i < n && ok; i++)
        {
            if (cfs[i].instr.opcode != ControlFlowOpcode::CondCall)
                continue;
            const auto& cc = cfs[i].instr.condCall;
            const uint32_t t = cc.address;
            if (!cc.isUnconditional || cc.isPredicated || t == 0 || t >= n)
                ok = false;
            else
                subs[t].start = t;
        }
        if (subs.empty())
            ok = false;
        uint32_t mainEnd = n;
        for (auto& [start, sub] : subs)
        {
            if (!ok)
                break;
            uint32_t r = start;
            while (r < n && cfs[r].instr.opcode != ControlFlowOpcode::Return)
            {
                if (!isClause(cfs[r].instr.opcode))
                {
                    ok = false;
                    break;
                }
                ++r;
            }
            if (!ok || r >= n || !isEnd(cfs[start - 1].instr.opcode))
            {
                ok = false;
                break;
            }
            sub.ret = r;
            mainEnd = std::min(mainEnd, start);
        }
        if (ok)
        {
            // Past the main region only subroutine bodies, their returns and padding may remain.
            for (uint32_t j = mainEnd; j < n && ok; j++)
            {
                bool covered = cfs[j].instr.opcode == ControlFlowOpcode::Nop;
                for (const auto& [start, sub] : subs)
                    covered = covered || (j >= start && j <= sub.ret);
                if (!covered)
                    ok = false;
            }
            // No transfer in the main region may reach into a body.
            for (uint32_t i = 0; i < mainEnd && ok; i++)
            {
                const auto& in = cfs[i].instr;
                uint32_t t = 0;
                bool hasTarget = false;
                switch (in.opcode)
                {
                case ControlFlowOpcode::CondJmp: t = in.condJmp.address; hasTarget = true; break;
                case ControlFlowOpcode::LoopStart: t = in.loopStart.address; hasTarget = true; break;
                case ControlFlowOpcode::LoopEnd: t = in.loopEnd.address; hasTarget = true; break;
                case ControlFlowOpcode::Return: ok = false; break;
                default: break;
                }
                if (hasTarget && t >= mainEnd)
                    ok = false;
            }
        }
        if (ok)
        {
            std::vector<CfWord> seq;
            std::vector<uint32_t> newIndex(mainEnd + 1, 0);
            struct Fixup
            {
                uint32_t seqIndex;
                uint32_t oldTarget;
            };
            std::vector<Fixup> fixups;
            for (uint32_t i = 0; i < mainEnd; i++)
            {
                newIndex[i] = uint32_t(seq.size());
                const auto& in = cfs[i].instr;
                if (in.opcode == ControlFlowOpcode::CondCall)
                {
                    const Sub& sub = subs[in.condCall.address];
                    for (uint32_t j = sub.start; j < sub.ret; j++)
                        seq.push_back(cfs[j]);
                    continue;
                }
                if (in.opcode == ControlFlowOpcode::CondJmp)
                    fixups.push_back({ uint32_t(seq.size()), in.condJmp.address });
                else if (in.opcode == ControlFlowOpcode::LoopStart)
                    fixups.push_back({ uint32_t(seq.size()), in.loopStart.address });
                else if (in.opcode == ControlFlowOpcode::LoopEnd)
                    fixups.push_back({ uint32_t(seq.size()), in.loopEnd.address });
                seq.push_back(cfs[i]);
            }
            newIndex[mainEnd] = uint32_t(seq.size());
            if (seq.size() > n)
                ok = false;
            for (const Fixup& f : fixups)
            {
                if (!ok)
                    break;
                if (f.oldTarget > mainEnd)
                {
                    ok = false;
                    break;
                }
                CfWord& c = seq[f.seqIndex];
                const uint32_t t = newIndex[f.oldTarget];
                switch (c.instr.opcode)
                {
                case ControlFlowOpcode::CondJmp: c.instr.condJmp.address = t; break;
                case ControlFlowOpcode::LoopStart: c.instr.loopStart.address = t; break;
                case ControlFlowOpcode::LoopEnd: c.instr.loopEnd.address = t; break;
                default: break;
                }
                uint32_t words[2];
                memcpy(words, &c.instr, sizeof(words));
                c.lo = words[0];
                c.hi = words[1];
            }
            if (ok)
            {
                if (seq.size() % 2 != 0)
                    seq.push_back(CfWord{});
                for (size_t i = 0; i < seq.size(); i += 2)
                {
                    const CfWord& a = seq[i];
                    const CfWord& b = seq[i + 1];
                    const uint32_t w[3] = { a.lo, (a.hi & 0xFFFF) | (b.lo << 16), (b.lo >> 16) | (b.hi << 16) };
                    for (uint32_t word : w)
                    {
                        be<uint32_t> v;
                        v.value = byteSwap(word);
                        inlinedCf.push_back(v);
                    }
                }
            }
        }
    }
    const be<uint32_t>* cfStream = inlinedCf.empty() ? code : inlinedCf.data();
    if (!inlinedCf.empty())
        instrSize = uint32_t(inlinedCf.size() * sizeof(uint32_t));
    auto controlFlowCode = cfStream;

    bool simpleControlFlow = true;
    // Programs whose transfers all go strictly forward (no loops, calls or returns) get guard
    // emission (`if (pc <= i)` blocks, jumps raise pc) instead of the while/switch pc machine.
    // The dispatch loop turns every block into a dynamic-jump target and keeps all registers live,
    // while a forward-only guard chain stays a plain branch DAG.
    bool forwardOnlyFlow = true;
    uint32_t scanPc = 0;

    // Per-instruction control flow summary for the structurizer below.
    enum class CfKind : uint8_t { None, Cond, Uncond, LoopStart, LoopEnd };
    struct CfSummary
    {
        CfKind kind = CfKind::None;
        uint32_t target = 0;
    };
    std::vector<CfSummary> cfSummaries;
    std::vector<uint32_t> loopStarts;
    std::unordered_map<uint32_t, uint32_t> loopEndFor;

    while (instrAddress < instrSize)
    {
        code0 = controlFlowCode[0];
        code1 = controlFlowCode[1] & 0xFFFF;
        code2 = (controlFlowCode[1] >> 16) | (controlFlowCode[2] << 16);
        code3 = controlFlowCode[2] >> 16;

        for (auto& cfInstr : controlFlow)
        {
            uint32_t address = 0;

            switch (cfInstr.opcode)
            {
            case ControlFlowOpcode::Exec:
            case ControlFlowOpcode::ExecEnd:
                address = cfInstr.exec.address;
                break;

            case ControlFlowOpcode::CondExec:
            case ControlFlowOpcode::CondExecEnd:
            case ControlFlowOpcode::CondExecPredClean:
            case ControlFlowOpcode::CondExecPredCleanEnd:
                address = cfInstr.condExec.address;
                break;

            case ControlFlowOpcode::CondExecPred:
            case ControlFlowOpcode::CondExecPredEnd:
                address = cfInstr.condExecPred.address;
                // The whole clause executes under per-pixel p0.
                divergentPcs.insert(scanPc);
                break;

            // Loop-back transfers rule out guard emission; the structurizer still emits real for-loops.
            case ControlFlowOpcode::LoopStart:
                loopStarts.push_back(scanPc);
                cfSummaries.push_back({ CfKind::LoopStart, 0 });
                forwardOnlyFlow = false;
                break;

            case ControlFlowOpcode::LoopEnd:
                if (loopStarts.empty())
                    simpleControlFlow = false;
                else
                {
                    loopEndFor[loopStarts.back()] = scanPc;
                    loopStarts.pop_back();
                }
                cfSummaries.push_back({ CfKind::LoopEnd, 0 });
                forwardOnlyFlow = false;
                break;

            case ControlFlowOpcode::CondJmp:
            {
                // A backward or degenerate target defeats both the structurizer and guard emission.
                if (cfInstr.condJmp.direction || cfInstr.condJmp.address <= scanPc)
                {
                    simpleControlFlow = false;
                    forwardOnlyFlow = false;
                }

                // A p0-conditional jump splits the quad per-pixel: everything
                // between the jump and its (forward) target runs divergent
                // until reconvergence. Backward/degenerate targets: mark the
                // whole program (conservative, not seen in practice).
                if (cfInstr.condJmp.isPredicated && !cfInstr.condJmp.isUnconditional)
                {
                    if (cfInstr.condJmp.direction || cfInstr.condJmp.address <= scanPc)
                        allPcsDivergent = true;
                    else
                        for (uint32_t d = scanPc + 1; d < cfInstr.condJmp.address; d++)
                            divergentPcs.insert(d);
                }

                cfSummaries.push_back({
                    cfInstr.condJmp.isUnconditional ? CfKind::Uncond : CfKind::Cond,
                    uint32_t(cfInstr.condJmp.address) });
                break;
            }

            case ControlFlowOpcode::CondCall:
                // Subroutines need the pc-dispatch loop (the call target and the
                // return site are arbitrary cf indices). NOTE: condCall.address
                // is a CF index, not an ALU/fetch clause address - it must not
                // shrink instrSize.
                simpleControlFlow = false;
                forwardOnlyFlow = false;
                // A p0-conditional call runs its whole subroutine divergent,
                // and the body is reachable from arbitrary sites: mark the
                // whole program (conservative).
                if (cfInstr.condCall.isPredicated && !cfInstr.condCall.isUnconditional)
                    allPcsDivergent = true;
                break;

            case ControlFlowOpcode::Return:
                simpleControlFlow = false;
                forwardOnlyFlow = false;
                break;
            }

            if (cfSummaries.size() == scanPc)
                cfSummaries.push_back({});

            if (address != 0)
                instrSize = std::min<uint32_t>(instrSize, address * 12);

            ++scanPc;
        }

        controlFlowCode += 3;
        instrAddress += 12;
    }

    if (!loopStarts.empty())
        simpleControlFlow = false;

    // An unconditional jump inside divergent code sends only some lanes to its target, so the
    // skipped span (an else branch) is divergent too.
    for (uint32_t i = 0; i < cfSummaries.size(); i++)
    {
        if (cfSummaries[i].kind == CfKind::Uncond && cfSummaries[i].target > i &&
            (allPcsDivergent || divergentPcs.count(i) != 0))
        {
            for (uint32_t d = i + 1; d < cfSummaries[i].target; d++)
                divergentPcs.insert(d);
        }
    }

    if (simpleControlFlow)
    {
        // Recursive-descent structurizer for forward-jump if/else diamonds (then-exit jumps may chain to an enclosing
        // merge) and LoopStart/LoopEnd pairs over a region [lo, hi) with continuation cont; other shapes fall back.
        const uint32_t instrCount = (instrSize / 12) * 2;
        if (cfSummaries.size() > instrCount)
            cfSummaries.resize(instrCount);

        std::function<bool(uint32_t, uint32_t, uint32_t)> structure =
            [&](uint32_t lo, uint32_t hi, uint32_t cont) -> bool
        {
            uint32_t i = lo;
            while (i < hi)
            {
                const CfSummary& summary = cfSummaries[i];
                switch (summary.kind)
                {
                case CfKind::Uncond:
                {
                    if (i + 1 == hi && (summary.target == cont || summary.target == hi))
                        i = hi;
                    else
                        return false;
                    break;
                }
                case CfKind::Cond:
                {
                    const uint32_t t = summary.target;
                    if (t > hi)
                    {
                        if (t != cont)
                            return false;
                        ++ifEndLabels[hi];
                        if (!structure(i + 1, hi, cont))
                            return false;
                        i = hi;
                    }
                    // A then-exit jump targeting the cond target itself is a
                    // no-op (empty else); fall through to the plain-if path,
                    // whose trailing-jump rule deletes it.
                    else if (t < hi && t - 1 > i && cfSummaries[t - 1].kind == CfKind::Uncond &&
                             cfSummaries[t - 1].target != t)
                    {
                        const uint32_t merge = cfSummaries[t - 1].target;
                        if (merge == cont || merge == hi)
                        {
                            elseLabels.insert(t);
                            ++ifEndLabels[hi];
                            if (!structure(i + 1, t - 1, merge == cont ? cont : hi))
                                return false;
                            if (!structure(t, hi, cont))
                                return false;
                            i = hi;
                        }
                        else if (merge < hi)
                        {
                            elseLabels.insert(t);
                            ++ifEndLabels[merge];
                            if (!structure(i + 1, t - 1, merge))
                                return false;
                            if (!structure(t, merge, merge))
                                return false;
                            i = merge;
                        }
                        else
                        {
                            return false;
                        }
                    }
                    else
                    {
                        ++ifEndLabels[t];
                        if (!structure(i + 1, t, t == hi ? cont : t))
                            return false;
                        i = t;
                    }
                    break;
                }
                case CfKind::LoopStart:
                {
                    auto loopEnd = loopEndFor.find(i);
                    if (loopEnd == loopEndFor.end() || loopEnd->second >= hi)
                        return false;
                    if (!structure(i + 1, loopEnd->second, loopEnd->second))
                        return false;
                    i = loopEnd->second + 1;
                    break;
                }
                case CfKind::LoopEnd:
                    return false; // reached without its LoopStart

                default:
                    ++i;
                    break;
                }
            }
            return true;
        };

        simpleControlFlow = structure(0, instrCount, instrCount);
    }

    if (!simpleControlFlow)
    {
        ifEndLabels.clear();
        elseLabels.clear();
    }

    const bool guardFlow = !simpleControlFlow && forwardOnlyFlow;

    if (simpleControlFlow)
    {
        out += '\n';
        indentation = 1;
    }
    else if (guardFlow)
    {
        // Forward-only: pc is a rising skip watermark, no dispatch loop.
        out += "\n\tuint pc = 0;\n";
    }
    else
    {
        rexPcMachine = true;
        out += "\n\tuint pc = 0;\n";
        out += "\twhile (true)\n";
        out += "\t{\n";
        out += "\t\tswitch (pc)\n";
        out += "\t\t{\n";
    }

    controlFlowCode = cfStream;
    instrAddress = 0;
    uint32_t pc = 0;

    while (instrAddress < instrSize)
    {
        code0 = controlFlowCode[0];
        code1 = controlFlowCode[1] & 0xFFFF;
        code2 = (controlFlowCode[1] >> 16) | (controlFlowCode[2] << 16);
        code3 = controlFlowCode[2] >> 16;

        for (auto& cfInstr : controlFlow)
        {
            if (!simpleControlFlow)
            {
                indentation = 3;
                if (guardFlow)
                {
                    if (pc != 0)
                        out += "\t\t}\n";
                    println("\t\tif (pc <= {}u)", pc);
                    out += "\t\t{\n";
                }
                else
                    println("\t\tcase {}:", pc);
            }
            else
            {
                auto findResult = ifEndLabels.find(pc);
                if (findResult != ifEndLabels.end())
                {
                    for (uint32_t i = 0; i < findResult->second; i++)
                    {
                        --indentation;
                        indent();
                        out += "}\n";
                    }
                }
                // Inner regions close first (validated nesting), then the
                // enclosing if's then-branch flips to its else-branch.
                if (elseLabels.count(pc) != 0)
                {
                    --indentation;
                    indent();
                    out += "}\n";
                    indent();
                    out += "else\n";
                    indent();
                    out += "{\n";
                    ++indentation;
                }
            }

            ++pc;

            uint32_t address = 0;
            uint32_t count = 0;
            uint32_t sequence = 0;
            bool shouldReturn = false;
            bool shouldCloseCurlyBracket = false;
            bool clauseGuardOpen = false;

            switch (cfInstr.opcode)
            {
            case ControlFlowOpcode::Exec:
            case ControlFlowOpcode::ExecEnd:
                address = cfInstr.exec.address;
                count = cfInstr.exec.count;
                sequence = cfInstr.exec.sequence;
                shouldReturn = (cfInstr.opcode == ControlFlowOpcode::ExecEnd);
                break;

            case ControlFlowOpcode::CondExec:
            case ControlFlowOpcode::CondExecEnd:
            case ControlFlowOpcode::CondExecPredClean:
            case ControlFlowOpcode::CondExecPredCleanEnd:
            {
                address = cfInstr.condExec.address;
                count = cfInstr.condExec.count;
                sequence = cfInstr.condExec.sequence;
                shouldReturn = (cfInstr.opcode == ControlFlowOpcode::CondExecEnd ||
                                cfInstr.opcode == ControlFlowOpcode::CondExecPredCleanEnd);

                // The clause runs only when the bool constant matches the condition, and bools are uniform
                // across the draw, so the guard adds no divergent fetches (dxbc_translator.cpp:1561).
                // The shader end after a Cond*End clause stays unconditional; the guard closes before it.
                indent();
                println("if ({})", boolCondition(cfInstr.condExec.boolAddress, cfInstr.condExec.condition));
                indent();
                out += "{\n";
                ++indentation;
                clauseGuardOpen = true;
                break;
            }

            case ControlFlowOpcode::CondExecPred:
            case ControlFlowOpcode::CondExecPredEnd:
                address = cfInstr.condExecPred.address;
                count = cfInstr.condExecPred.count;
                sequence = cfInstr.condExecPred.sequence;
                shouldReturn = (cfInstr.opcode == ControlFlowOpcode::CondExecPredEnd);

                // The clause runs only when p0 matches at its start; predicated
                // instructions inside recheck p0 (dxbc_translator.cpp:1580).
                indent();
                println("if ({}p0)", cfInstr.condExecPred.condition ? "" : "!");
                indent();
                out += "{\n";
                ++indentation;
                clauseGuardOpen = true;
                break;

            case ControlFlowOpcode::LoopStart:
                if (simpleControlFlow)
                {
                    indent();
                #ifdef UNLEASHED_RECOMP
                    print("[unroll] ");
                #endif
                    println("for (aL = 0; aL < i{}.x; aL++)", uint32_t(cfInstr.loopStart.loopId));
                    indent();
                    out += "{\n";
                    ++indentation;
                }
                else 
                {
                    out += "\t\t\taL = 0;\n";
                }
                break;

            case ControlFlowOpcode::LoopEnd:
                if (cfInstr.loopEnd.isPredicatedBreak)
                    throw std::runtime_error("unimplemented control flow: loop_end with predicated break");
                if (simpleControlFlow)
                {
                    --indentation;
                    indent();
                    out += "}\n";
                }
                else
                {
                    out += "\t\t\t++aL;\n";
                    println("\t\t\tif (aL < i{}.x)", uint32_t(cfInstr.loopEnd.loopId));
                    out += "\t\t\t{\n";
                    println("\t\t\t\tpc = {};", uint32_t(cfInstr.loopEnd.address));
                    out += "\t\t\t\tcontinue;\n";
                    out += "\t\t\t}\n";
                }
                break;

            case ControlFlowOpcode::CondJmp:
            {
                if (cfInstr.condJmp.isUnconditional)
                {
                    // Structured mode: the then-branch exit jump was consumed by
                    // the "} else {" emitted at the next instruction.
                    if (!simpleControlFlow)
                    {
                        println("\t\t\tpc = {}u;", uint32_t(cfInstr.condJmp.address));
                        if (!guardFlow)
                            out += "\t\t\tcontinue;\n";
                    }
                }
                else
                {
                    indent();
                    if (cfInstr.condJmp.isPredicated)
                    {
                        println("if ({}p0)", cfInstr.condJmp.condition ^ simpleControlFlow ? "" : "!");
                    }
                    else
                    {
                        println("if ({})", boolCondition(cfInstr.condJmp.boolAddress, cfInstr.condJmp.condition ^ simpleControlFlow));
                    }

                    if (simpleControlFlow)
                    {
                        indent();
                        out += "{\n";
                        ++indentation;
                    }
                    else
                    {
                        out += "\t\t\t{\n";
                        println("\t\t\t\tpc = {}u;", uint32_t(cfInstr.condJmp.address));
                        if (!guardFlow)
                            out += "\t\t\t\tcontinue;\n";
                        out += "\t\t\t}\n";
                    }
                }
                break;
            }

            case ControlFlowOpcode::CondCall:
            {
                // Xenos has a 4-deep call stack; depth 1 is implemented,
                // matching the runtime translator (nested calls have not been
                // seen in titles). pc was already incremented, so it is the
                // return site's cf index.
                assert(!simpleControlFlow);
                if (cfInstr.condCall.isUnconditional)
                {
                    println("\t\t\treturnPc = {};", pc);
                    println("\t\t\tpc = {};", uint32_t(cfInstr.condCall.address));
                    out += "\t\t\tcontinue;\n";
                }
                else
                {
                    if (cfInstr.condCall.isPredicated)
                    {
                        println("\t\t\tif ({}p0)", cfInstr.condCall.condition ? "" : "!");
                    }
                    else
                    {
                        println("\t\t\tif ({})", boolCondition(cfInstr.condCall.boolAddress, cfInstr.condCall.condition));
                    }
                    out += "\t\t\t{\n";
                    println("\t\t\t\treturnPc = {};", pc);
                    println("\t\t\t\tpc = {};", uint32_t(cfInstr.condCall.address));
                    out += "\t\t\t\tcontinue;\n";
                    out += "\t\t\t}\n";
                }
                break;
            }

            case ControlFlowOpcode::Return:
            {
                assert(!simpleControlFlow);
                out += "\t\t\tpc = returnPc;\n";
                out += "\t\t\tcontinue;\n";
                break;
            }
            }

            // pc was already incremented: the current cf index is pc - 1.
            inDivergentFlow = allPcsDivergent || divergentPcs.count(pc - 1) != 0;

            auto instructionCode = code + address * 3;

            for (uint32_t i = 0; i < count; i++)
            {
                union
                {
                    VertexFetchInstruction vertexFetch;
                    TextureFetchInstruction textureFetch;
                    AluInstruction alu;
                    struct
                    {
                        uint32_t code0;
                        uint32_t code1;
                        uint32_t code2;
                    };
                };
            
                code0 = instructionCode[0];
                code1 = instructionCode[1];
                code2 = instructionCode[2];
            
                if ((sequence & 0x1) != 0)
                {
                    if (vertexFetch.opcode == FetchOpcode::VertexFetch)
                    {
                        recompile(vertexFetch, address + i);
                    }
                    else
                    {
                    #ifdef UNLEASHED_RECOMP
                        if (textureFetch.constIndex == 10) // g_GISampler
                        {
                            specConstantsMask |= SPEC_CONSTANT_BICUBIC_GI_FILTER;

                            indent();
                            out += "if (g_SpecConstants() & SPEC_CONSTANT_BICUBIC_GI_FILTER)";
                            indent();
                            out += '{';

                            ++indentation;
                            recompile(textureFetch, true);
                            --indentation;

                            indent();
                            out += "}";
                            indent();
                            out += "else";
                            indent();
                            out += '{';

                            ++indentation;
                            recompile(textureFetch, false);
                            --indentation;

                            indent();
                            out += '}';
                        }
                        else
                    #endif
                        {
                            recompile(textureFetch, false);
                        }
                    }
                }
                else
                {
                    recompile(alu);
                }
            
                sequence >>= 2;
                instructionCode += 3;
            }

            if (clauseGuardOpen)
            {
                --indentation;
                indent();
                out += "}\n";
            }

            if (shouldReturn)
            {
                if (rexglueMode)
                {
                    if (isPixelShader)
                    {
                        // Mirrors CompletePixelShader: alpha test on guest
                        // (pre-bias) alpha, then per-RT exponent bias.
                        if (psOutputsMask & PIXEL_SHADER_OUTPUT_COLOR0)
                        {
                            indent();
                            out += "xe_alpha_test(oC0.w);\n";
                        }
                        static constexpr char BIAS_COMPONENTS[] = { 'x', 'y', 'z', 'w' };
                        for (uint32_t i = 0; i < 4; i++)
                        {
                            if (psOutputsMask & (1u << i))
                            {
                                indent();
                                println("oC{} *= xe_color_exp_bias.{};", i, BIAS_COMPONENTS[i]);
                            }
                        }
                    }
                    else
                    {
                        indent();
                        out += "oPos = xe_apply_position(oPos);\n";
                    }
                }
                else if (isPixelShader)
                {
                    specConstantsMask |= SPEC_CONSTANT_ALPHA_TEST;

                    indent();
                    out += "[branch] if (g_SpecConstants() & SPEC_CONSTANT_ALPHA_TEST)";
                    indent();
                    out += '{';

                    indent();
                    out += "\tclip(oC0.w - g_AlphaThreshold);\n";

                    indent();
                    out += "}";

                #ifdef UNLEASHED_RECOMP
                    specConstantsMask |= SPEC_CONSTANT_ALPHA_TO_COVERAGE;

                    indent();
                    out += "else if (g_SpecConstants() & SPEC_CONSTANT_ALPHA_TO_COVERAGE)";
                    indent();
                    out += '{';

                    indent();
                    out += "\toC0.w *= 1.0 + computeMipLevel(pixelCoord) * 0.25;\n";
                    indent();
                    out += "\toC0.w = 0.5 + (oC0.w - g_AlphaThreshold) / max(fwidth(oC0.w), 1e-6);\n";

                    indent();
                    out += '}';
                #endif
                }
                else
                {
                #ifdef UNLEASHED_RECOMP
                    if (!hasMtxProjection)
                #endif
                    {
                        out += "\toPos.xy += g_HalfPixelOffset * oPos.w;\n";
                    }
                }

                if (simpleControlFlow)
                {
                    indent();
                #ifdef UNLEASHED_RECOMP
                    if (hasMtxProjection)
                    {
                        out += "continue;\n";
                    }
                    else
                #endif
                    {
                        out += "return;\n";
                    }
                }
                else if (guardFlow)
                {
                    // No dispatch loop to break out of, end the shader.
                    out += "\t\t\treturn;\n";
                }
                else
                {
                    out += "\t\t\tbreak;\n";
                }
            }

            if (shouldCloseCurlyBracket)
            {
                --indentation;
                indent();
                out += "}\n";
            }
        }

        controlFlowCode += 3;
        instrAddress += 12;
    }

    if (guardFlow)
    {
        out += "\t\t}\n";  // close the last guard block
    }
    else if (!simpleControlFlow)
    {
        out += "\t\t\tbreak;\n";
        out += "\t\t}\n";
        out += "\t\tbreak;\n";
        out += "\t}\n";
    }
    else
    {
        // Regions closing one past the last instruction never get visited by
        // the loop above; balance their braces here.
        auto findResult = ifEndLabels.find(pc);
        if (findResult != ifEndLabels.end())
        {
            for (uint32_t i = 0; i < findResult->second; i++)
            {
                --indentation;
                indent();
                out += "}\n";
            }
        }
    }

    // setTexLOD / setGradientH / setGradientV state (the translator's
    // grad_h_lod and grad_v temporaries), declared only when used.
    std::string fetchState;
    if (usesRegisterLod)
        fetchState += "\tfloat xe_lod = 0.0;\n";
    if (usesRegisterGradients)
        fetchState += "\tfloat3 xe_grad_h = 0.0;\n\tfloat3 xe_grad_v = 0.0;\n";
    out.insert(fetchStatePos, fetchState);

#ifdef UNLEASHED_RECOMP
    if (hasMtxProjection)
        out += "\t}\n";

    if (!isPixelShader && hasMtxProjection)
        out += "\toPos.xy += g_HalfPixelOffset * oPos.w;\n";
#endif

    out += "}";
}
