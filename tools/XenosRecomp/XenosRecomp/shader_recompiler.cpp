#include "shader_recompiler.h"
#include "shader_common.h"
#include "ucode_fingerprint.h"

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
    "uint4", // NORMAL
    "float4", // PSIZE
    "float4", // TEXCOORD
    "uint4", // TANGENT
    "uint4", // BINORMAL
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

// NOTE: These are specialized Vulkan locations for Unleashed Recompiled. Change as necessary. Likely not going to work with other games.
static constexpr DeclUsageLocation USAGE_LOCATIONS[] =
{
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

// Words per vertex format (xenos::VertexFormat values).
static uint32_t rexglueVfetchWordCount(uint32_t format)
{
    switch (format)
    {
    case 26: case 32: case 34: case 37: return 2; // k_16_16_16_16, k_16_16_16_16_FLOAT, k_32_32, k_32_32_FLOAT
    case 57: return 3;                                // k_32_32_32_FLOAT
    case 35: case 38: return 4;                       // k_32_32_32_32, k_32_32_32_32_FLOAT
    default: return 1;
    }
}

// Emits the format decode expression producing a float4 (missing components
// zero). Semantics ported from DxbcShaderTranslator's vertex fetch unpack
// (dxbc_translator_fetch.cpp:266-450).
static std::string rexglueVfetchDecode(const std::string& w, const VertexFetchInstruction& instr)
{
    uint32_t format = uint32_t(instr.format);
    bool isSigned = instr.formatCompAll != 0;
    bool isInteger = instr.numFormatAll != 0;
    bool rfNoZero = instr.signedRfModeAll != 0; // SignedRepeatingFractionMode::kNoZero

    // Packed integer formats: per-component (width, offset, source word).
    struct Packed { uint32_t width, offset, word; };
    Packed packed[4] = {};
    uint32_t packedCount = 0;

    switch (format)
    {
    case 6: // k_8_8_8_8
        packed[0] = { 8, 0, 0 }; packed[1] = { 8, 8, 0 }; packed[2] = { 8, 16, 0 }; packed[3] = { 8, 24, 0 };
        packedCount = 4;
        break;
    case 7: // k_2_10_10_10
        packed[0] = { 10, 0, 0 }; packed[1] = { 10, 10, 0 }; packed[2] = { 10, 20, 0 }; packed[3] = { 2, 30, 0 };
        packedCount = 4;
        break;
    case 16: // k_10_11_11 (x 11 bits, y 11 bits, z 10 bits)
        packed[0] = { 11, 0, 0 }; packed[1] = { 11, 11, 0 }; packed[2] = { 10, 22, 0 };
        packedCount = 3;
        break;
    case 17: // k_11_11_10
        packed[0] = { 10, 0, 0 }; packed[1] = { 11, 10, 0 }; packed[2] = { 11, 21, 0 };
        packedCount = 3;
        break;
    case 25: // k_16_16
        packed[0] = { 16, 0, 0 }; packed[1] = { 16, 16, 0 };
        packedCount = 2;
        break;
    case 26: // k_16_16_16_16
        packed[0] = { 16, 0, 0 }; packed[1] = { 16, 16, 0 }; packed[2] = { 16, 0, 1 }; packed[3] = { 16, 16, 1 };
        packedCount = 4;
        break;
    default:
        break;
    }

    auto wordRef = [&](uint32_t word)
        {
            return rexglueVfetchWordCount(format) == 1 ? w : fmt::format("{}.{}", w, "xyzw"[word]);
        };

    if (packedCount != 0)
    {
        std::string comps[4] = { "0.0", "0.0", "0.0", "0.0" };
        for (uint32_t i = 0; i < packedCount; i++)
        {
            auto& p = packed[i];
            std::string bits;
            if (isSigned)
            {
                // Sign-extending bitfield extract.
                bits = fmt::format("float(asint({} << {}) >> {})", wordRef(p.word), 32 - p.offset - p.width, 32 - p.width);
            }
            else
            {
                bits = fmt::format("float(({} >> {}) & {}u)", wordRef(p.word), p.offset, (1u << p.width) - 1u);
            }

            if (!isInteger)
            {
                if (isSigned)
                {
                    if (rfNoZero)
                        bits = fmt::format("({} * {} + {})", bits, 2.0 / double((1u << p.width) - 1u), 1.0 / double((1u << p.width) - 1u));
                    else if (p.width > 2)
                        bits = fmt::format("max({} * {}, -1.0)", bits, 1.0 / double((1u << (p.width - 1)) - 1u));
                    else
                        bits = fmt::format("max({}, -1.0)", bits);
                }
                else if (p.width > 1)
                {
                    bits = fmt::format("({} * {})", bits, 1.0 / double((1u << p.width) - 1u));
                }
            }

            comps[i] = bits;
        }
        return fmt::format("float4({}, {}, {}, {})", comps[0], comps[1], comps[2], comps[3]);
    }

    switch (format)
    {
    case 31: // k_16_16_FLOAT
        return fmt::format("float4(f16tof32({0} & 0xFFFFu), f16tof32({0} >> 16), 0.0, 0.0)", w);
    case 32: // k_16_16_16_16_FLOAT
        return fmt::format("float4(f16tof32({0}.x & 0xFFFFu), f16tof32({0}.x >> 16), f16tof32({0}.y & 0xFFFFu), f16tof32({0}.y >> 16))", w);
    case 33: case 34: case 35: // k_32, k_32_32, k_32_32_32_32
    {
        uint32_t count = rexglueVfetchWordCount(format);
        std::string comps[4] = { "0.0", "0.0", "0.0", "0.0" };
        for (uint32_t i = 0; i < count; i++)
        {
            std::string bits = isSigned ? fmt::format("float(asint({}))", wordRef(i))
                                        : fmt::format("float({})", wordRef(i));
            if (!isInteger)
            {
                if (isSigned)
                {
                    if (rfNoZero)
                        bits = fmt::format("({} * {} + {})", bits, 1.0 / 2147483647.5, 0.5 / 2147483647.5);
                    else
                        bits = fmt::format("({} * {})", bits, 1.0 / 2147483647.0);
                }
                else
                {
                    bits = fmt::format("({} * {})", bits, 1.0 / 4294967295.0);
                }
            }
            comps[i] = bits;
        }
        return fmt::format("float4({}, {}, {}, {})", comps[0], comps[1], comps[2], comps[3]);
    }
    case 36: // k_32_FLOAT
        return fmt::format("float4(asfloat({}), 0.0, 0.0, 0.0)", w);
    case 37: // k_32_32_FLOAT
        return fmt::format("float4(asfloat({}), 0.0, 0.0)", w);
    case 57: // k_32_32_32_FLOAT
        return fmt::format("float4(asfloat({}), 0.0)", w);
    case 38: // k_32_32_32_32_FLOAT
        return fmt::format("asfloat({})", w);
    default:
        return fmt::format("float4(0.0, 0.0, 0.0, 0.0) /* unsupported vfetch format {} */", format);
    }
}

void ShaderRecompiler::recompileRexglueVfetch(const VertexFetchInstruction& instr)
{
    if (instr.isPredicated)
    {
        indent();
        println("if ({}p0)", instr.predicateCondition ? "" : "!");

        indent();
        out += "{\n";
        ++indentation;
    }

    uint32_t n = rexFetchCounter++;

    // Scope the fetch locals: in complex-control-flow shaders this code sits
    // under a switch case label, where unscoped initializations are illegal.
    indent();
    out += "{\n";
    ++indentation;

    uint32_t fetchDwordIndex;
    uint32_t stride;
    std::string indexExpr;
    if (instr.isMiniFetch)
    {
        // Mini fetches inherit the fetch constant, index and stride from the
        // preceding full fetch; only offset/format/destination are their own.
        fetchDwordIndex = rexLastVfetchDwordIndex;
        stride = rexLastVfetchStride;
        indexExpr = rexLastVfetchIndex;
    }
    else
    {
        fetchDwordIndex = uint32_t(instr.constIndex) * 6 + uint32_t(instr.constIndexSelect) * 2;
        stride = instr.stride;
        indexExpr = fmt::format("r{}.{}", uint32_t(instr.srcRegister), SWIZZLES[instr.srcSwizzle & 0x3]);
        rexLastVfetchDwordIndex = fetchDwordIndex;
        rexLastVfetchStride = stride;
        rexLastVfetchIndex = indexExpr;
    }

    indent();
    println("uint xe_a{} = xe_vfetch_address({}u, {}, {}u, {});", n, fetchDwordIndex, indexExpr, stride,
        instr.isIndexRounded ? "true" : "false");

    uint32_t wordCount = rexglueVfetchWordCount(uint32_t(instr.format));
    int32_t offsetBytes = int32_t(instr.offset) * 4;

    static const char* WORD_TYPES[] = { "uint", "uint2", "uint3", "uint4" };
    indent();
    println("{} xe_w{} = xe_vfetch_load{}(uint(int(xe_a{}) + {}), {}u);", WORD_TYPES[wordCount - 1], n, wordCount,
        n, offsetBytes, fetchDwordIndex);

    indent();
    println("float4 xe_v{} = {};", n, rexglueVfetchDecode(fmt::format("xe_w{}", n), instr));

    if (instr.expAdjust != 0)
    {
        float scale = std::ldexp(1.0f, instr.expAdjust);
        uint32_t scaleBits;
        memcpy(&scaleBits, &scale, sizeof(scaleBits));
        indent();
        println("xe_v{} *= asfloat(0x{:08X}u);", n, scaleBits);
    }

    indent();
    print("r{}.", uint32_t(instr.dstRegister));
    printDstSwizzle(instr.dstSwizzle, false);
    print(" = xe_v{}.", n);
    printDstSwizzle(instr.dstSwizzle, true);
    out += ";\n";

    printDstSwizzle01(instr.dstRegister, instr.dstSwizzle);

    --indentation;
    indent();
    out += "}\n";

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
        recompileRexglueVfetch(instr);
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

    print("{}", findResult->second.name);

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

void ShaderRecompiler::recompile(const TextureFetchInstruction& instr, bool bicubic)
{
    if (instr.opcode != FetchOpcode::TextureFetch && instr.opcode != FetchOpcode::GetTextureWeights)
        return;

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

    // Divergent-flow / VS fetches use the explicit-LOD helper variants:
    // implicit-derivative Sample under per-pixel flow deadlocks Intel EUs,
    // and vertex shaders have no derivatives at all.
    if (rexglueMode && instr.opcode == FetchOpcode::TextureFetch &&
        (inDivergentFlow || instr.isPredicated || !isPixelShader))
        out += "_lod0";

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

    switch (instr.dimension)
    {
    case TextureDimension::Texture2D:
        print(", float2({}, {})", instr.offsetX * 0.5f, instr.offsetY * 0.5f);
        break;
    case TextureDimension::TextureCube:
        out += ", cubeMapData";
        break;
    }

    out += ").";

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

void ShaderRecompiler::recompile(const AluInstruction& instr)
{
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

    auto op = [&](size_t operand)
        {
            size_t reg = 0;
            size_t swizzle = 0;
            bool select = true;
            bool negate = false;
            bool abs = false;

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

            if (select)
            {
                regFormatted = fmt::format("r{}", reg);
            }
            else if (rexglueMode && (recordRexFloatConstant(reg, instr.const0Relative || instr.const1Relative),
                                     rexAbsoluteFloatFile))
            {
                // Container-less: absolute register file (relative reads
                // mirror the named multi-register indexing below).
                regFormatted = fmt::format("xe_fcfile_read({}{})", reg,
                    instr.const0Relative ? (instr.constAddressRegisterRelative ? " + a0" : " + aL") : "");
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
                                reg - findResult->second->registerIndex, instr.const0Relative ? (instr.constAddressRegisterRelative ? " + a0" : " + aL") : "");
                        }
                    }
                    else
                    {
                        assert(!instr.const0Relative && !instr.const1Relative);
                        regFormatted = constantName;
                    }
                }
                else
                {
                    assert(!instr.const0Relative && !instr.const1Relative);
                    regFormatted = fmt::format("c{}", reg);
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

                switch (instr.vectorOpcode)
                {
                case AluVectorOpcode::Dp2Add:
                    mask = (operand == VECTOR_2) ? 0b1 : 0b11;
                    break;

                case AluVectorOpcode::Dp3:
                    mask = 0b111;
                    break;

                case AluVectorOpcode::Dp4:
                case AluVectorOpcode::Max4:
                    mask = 0b1111;
                    break;

                default:
                    mask = instr.vectorWriteMask != 0 ? instr.vectorWriteMask : 0b1;
                    break;
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

    switch (instr.vectorOpcode)
    {
    case AluVectorOpcode::KillEq:
        indent();
        println("clip(any({} == {}) ? -1 : 1);", op(VECTOR_0), op(VECTOR_1));
        break;
    
    case AluVectorOpcode::KillGt:
        indent();
        println("clip(any({} > {}) ? -1 : 1);", op(VECTOR_0), op(VECTOR_1));
        break;
    
    case AluVectorOpcode::KillGe:
        indent();
        println("clip(any({} >= {}) ? -1 : 1);", op(VECTOR_0), op(VECTOR_1));
        break;
    
    case AluVectorOpcode::KillNe:
        indent();
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
                    throw CoverageError(fmt::format("assert: unmapped VS export register {}", uint32_t(instr.vectorDest)));
#endif
                assert(findResult != interpolators.end());
                exportRegister = findResult->second;
                break;
            }
            }
        }
    }

    if (instr.vectorOpcode >= AluVectorOpcode::SetpEqPush && instr.vectorOpcode <= AluVectorOpcode::SetpGePush)
    {
        indent();
        print("p0 = {} == 0.0 && {} ", op(VECTOR_0), op(VECTOR_1));

        switch (instr.vectorOpcode)
        {
        case AluVectorOpcode::SetpEqPush:
            out += "==";
            break;
        case AluVectorOpcode::SetpNePush:
            out += "!=";
            break;
        case AluVectorOpcode::SetpGtPush:
            out += ">";
            break;
        case AluVectorOpcode::SetpGePush:
            out += ">=";
            break;
        }

        out += " 0.0;\n";
    }
    else if (instr.vectorOpcode >= AluVectorOpcode::MaxA)
    {
        indent();
        println("a0 = (int)clamp(floor(({}).w + 0.5), -256.0, 255.0);", op(VECTOR_0));
    }

    uint32_t vectorWriteMask = instr.vectorWriteMask;
    if (instr.exportData)
        vectorWriteMask &= ~instr.scalarWriteMask;

    if (vectorWriteMask != 0)
    {
        indent();
        if (!exportRegister.empty())
        {
            out += exportRegister;
            out += '.';
        }
        else
        {
            print("r{}.", instr.vectorDest);
        }

        for (size_t i = 0; i < 4; i++)
        {
            if ((vectorWriteMask >> i) & 0x1)
                out += SWIZZLES[i];
        }

        out += " = ";

        if (instr.vectorSaturate)
            out += "saturate(";

        switch (instr.vectorOpcode)
        {
        case AluVectorOpcode::Add:
            print("{} + {}", op(VECTOR_0), op(VECTOR_1));
            break;

        case AluVectorOpcode::Mul:
            print("{} * {}", op(VECTOR_0), op(VECTOR_1));
            break;

        case AluVectorOpcode::Max:
        case AluVectorOpcode::MaxA:
            print("max({}, {})", op(VECTOR_0), op(VECTOR_1));
            break;

        case AluVectorOpcode::Min:
            print("min({}, {})", op(VECTOR_0), op(VECTOR_1));
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
            print("{} * {} + {}", op(VECTOR_0), op(VECTOR_1), op(VECTOR_2));
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
            print("dot({}, {})", op(VECTOR_0), op(VECTOR_1));
            break;

        case AluVectorOpcode::Dp2Add:
            print("dot({}, {}) + {}", op(VECTOR_0), op(VECTOR_1), op(VECTOR_2));
            break;

        case AluVectorOpcode::Cube:
            print("cube(r{}, cubeMapData)", instr.src1Register);
            break;

        case AluVectorOpcode::Max4:
            print("max4({})", op(VECTOR_0));
            break;

        case AluVectorOpcode::SetpEqPush:
        case AluVectorOpcode::SetpNePush:
        case AluVectorOpcode::SetpGtPush:
        case AluVectorOpcode::SetpGePush:
            print("p0 ? 0.0 : {} + 1.0", op(VECTOR_0));
            break;

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

    if (instr.scalarOpcode != AluScalarOpcode::RetainPrev)
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
            print("{} * {}", op(SCALAR_0), op(SCALAR_1));
            break;

        case AluScalarOpcode::MulsPrev:
        case AluScalarOpcode::MulsPrev2:
            print("{} * ps", op(SCALAR_0));
            break;

        case AluScalarOpcode::Maxs:
        case AluScalarOpcode::MaxAs:
        case AluScalarOpcode::MaxAsf:
            print("max({}, {})", op(SCALAR_0), op(SCALAR_1));
            break;

        case AluScalarOpcode::Mins:
            print("min({}, {})", op(SCALAR_0), op(SCALAR_1));
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
            print("{0} == 0.0 ? 1.0 : {0}", op(SCALAR_0));
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
            print("{} * {}", op(SCALAR_CONSTANT_0), op(SCALAR_CONSTANT_1));
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

    // Float constants at b1. The runtime uploads only the registers the
    // shader reads, packed in ascending order, unless any read is
    // register-relative, in which case the full 256-register file is
    // uploaded at absolute offsets (see the command processor's
    // float_bitmap gather). Pass 2 with a static-usage shader gets
    // per-register variables at their compacted ranks; the discovery pass
    // and dynamic shaders get the named absolute layout.
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
    for (const ConstantInfo* constantInfo : floatInfos)
    {
        const char* constantName = reinterpret_cast<const char*>(constantTableData + constantInfo->name);

        print("\tfloat4 {}", constantName);
        if (constantInfo->registerCount > 1)
            print("[{}]", constantInfo->registerCount.get());
        println(" : packoffset(c{});", constantInfo->registerIndex.get());

        for (uint16_t j = 0; j < constantInfo->registerCount; j++)
            float4Constants.emplace(constantInfo->registerIndex + j, constantInfo);
    }

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
}

// Mirrors FindOrAddSamplerBinding's key normalization and slot allocation.
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
        // Anisotropic filtering is only used with computed LOD.
        aniso = instr.useCompLod ? uint32_t(instr.anisoFilter) : 0u;
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
        rexBindings.push_back({ uint32_t(rexBindings.size()), true, uint32_t(instr.constIndex), 0, false });
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

    out += include;
    out += '\n';

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
            
            for (uint16_t j = 0; j < constantInfo->registerCount; j++)
                float4Constants.emplace(constantInfo->registerIndex + j, constantInfo);

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

    out += "\n#else\n\n";

    println("cbuffer {}ShaderConstants : register(b{}, space4)", isPixelShader ? "Pixel" : "Vertex", isPixelShader ? 1 : 0);
    out += "{\n";

    for (uint32_t i = 0; i < constantTableContainer->constantTable.constants; i++)
    {
        const auto constantInfo = reinterpret_cast<const ConstantInfo*>(
            constantTableData + constantTableContainer->constantTable.constantInfo + i * sizeof(ConstantInfo));

        if (constantInfo->registerSet == RegisterSet::Float4)
        {
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
            // Xenos wires VS exports to PS input registers BY INDEX (o{i} ->
            // interpolator entry i), regardless of the D3DX semantic names in
            // the containers, which may differ between the paired shaders.
            //
            // Declare only the registers the container's interpolator table
            // maps (the read set). D3D12 links PS inputs by semantic against
            // the VS's full oVar0-15 signature, so a sparse subset is legal,
            // and every input the ucode never reads stops costing attribute
            // setup/interpolation on every pixel of every draw.
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
                for (uint32_t i = 0; i < 16; i++)
                {
                    if (readMask & (1u << i))
                        println("\tin float4 iVar{0} : TEXCOORD{0},", i);
                }
            }
        }
        else
        for (auto& [usage, usageIndex] : INTERPOLATORS)
            println("\tin float4 i{0}{1} : {2}{1},", USAGE_VARIABLES[uint32_t(usage)], usageIndex, USAGE_SEMANTICS[uint32_t(usage)]);

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
                print(",\n\tout float4 oVar{0} : TEXCOORD{0}", i);
            // Point size (export register 63, x component), declared on
            // Every rexglue VS so the whole pack shares one output signature
            // and the runtime's point_expand GS links against any of them.
            out += ",\n\tout float4 oPts : TEXCOORD16";
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
            auto value = reinterpret_cast<const be<uint32_t>*>(shaderData + shaderContainer->virtualSize + definition->physicalOffset);
            for (uint16_t i = 0; i < (definition->count + 3) / 4; i++)
            {
                println("\tfloat4 c{} = asfloat(uint4(0x{:X}, 0x{:X}, 0x{:X}, 0x{:X}));",
                    definition->registerIndex + i - (isPixelShader ? 256 : 0), value[0].get(), value[1].get(), value[2].get(), value[3].get());

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
    #ifdef UNLEASHED_RECOMP
        if (!hasMtxProjection)
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
                if (rexglueMode)
                {
                    // Guest pixel position register: xy = pixel coords
                    // (centers at .5), w sign = facing.
                    out += "float4(iPos.xy, 0.0, iFace ? 1.0 : -1.0);\n";
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
    if (isPixelShader)
    {
#ifdef UNLEASHED_RECOMP
        out += "\tfloat2 pixelCoord = 0.0;\n";
#endif
        out += "\tCubeMapData cubeMapData = (CubeMapData)0;\n";
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

    auto controlFlowCode = code;
    uint32_t instrAddress = 0;
    uint32_t instrSize = rexCodeOverride != nullptr ? rexCodeOverrideSize : shader->size;
    bool simpleControlFlow = true;
    // All transfers strictly forward, no loops/calls/returns: such programs
    // get GUARD emission (`if (pc <= i)` blocks; jumps just raise pc)
    // instead of the while/switch pc machine. The dispatch loop makes every
    // block a dynamic-jump target, the compiler cannot flatten flow and
    // all registers stay live across every case, which is far slower.
    // A forward-only guard chain is a plain branch DAG instead.
    bool forwardOnlyFlow = true;
    uint32_t scanPc = 0;

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

            case ControlFlowOpcode::CondJmp:
            {
                if (cfInstr.condJmp.isUnconditional || cfInstr.condJmp.direction)
                    simpleControlFlow = false;
                else
                    ++ifEndLabels[cfInstr.condJmp.address];

                // Backward or degenerate target: guard emission cannot
                // represent it (pc only ever increases there).
                if (cfInstr.condJmp.direction || cfInstr.condJmp.address <= scanPc)
                    forwardOnlyFlow = false;

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
                break;
            }

            case ControlFlowOpcode::LoopStart:
            case ControlFlowOpcode::LoopEnd:
                // Loop-back transfer: guard emission cannot represent it.
                // (Simple mode still emits real for-loops when nothing else
                // broke structure; this only demotes guard -> pc machine.)
                forwardOnlyFlow = false;
                break;

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

            if (address != 0)
                instrSize = std::min<uint32_t>(instrSize, address * 12);

            ++scanPc;
        }

        controlFlowCode += 3;
        instrAddress += 12;
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
        out += "\n\tuint pc = 0;\n";
        out += "\twhile (true)\n";
        out += "\t{\n";
        out += "\t\tswitch (pc)\n";
        out += "\t\t{\n";
    }

    controlFlowCode = code;
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

                // The clause only runs when the bool constant matches the
                // condition; emitting it unguarded lets a disabled clause
                // clobber live registers. Bool conditions are uniform across
                // the draw, so the guard introduces no divergent fetches. The
                // shader end after a Cond*End clause stays unconditional; the
                // guard closes before it.
                uint32_t boolAddress = cfInstr.condExec.boolAddress;
                if (isPixelShader && boolAddress >= 128)
                    boolAddress -= 128;

                indent();
                auto findResult = boolConstants.find(boolAddress);
                if (findResult != boolConstants.end())
                    println("if ((g_Booleans & {}) {}= 0)", findResult->second, cfInstr.condExec.condition ? "!" : "=");
                else if (rexglueMode)
                    println("if ((g_Booleans & (1 << {})) {}= 0)", boolAddress + (isPixelShader ? 16 : 0), cfInstr.condExec.condition ? "!" : "=");
                else
                    println("if (b{} {}= 0)", uint32_t(cfInstr.condExec.boolAddress), cfInstr.condExec.condition ? "!" : "=");
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
                    assert(!simpleControlFlow);
                    println("\t\t\tpc = {}u;", uint32_t(cfInstr.condJmp.address));
                    if (!guardFlow)
                        out += "\t\t\tcontinue;\n";
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
                        // Pixel shader bool constants live at hardware slots 128-255;
                        // the constant table indexes them 0-based.
                        uint32_t boolAddress = cfInstr.condJmp.boolAddress;
                        if (isPixelShader && boolAddress >= 128)
                            boolAddress -= 128;

                        auto findResult = boolConstants.find(boolAddress);
                        if (findResult != boolConstants.end())
                            println("if ((g_Booleans & {}) {}= 0)", findResult->second, cfInstr.condJmp.condition ^ simpleControlFlow ? "!" : "=");
                        else if (rexglueMode)
                            // No constant table (container-less generation):
                            // g_Booleans bit position directly (PS regs at 16+).
                            println("if ((g_Booleans & (1 << {})) {}= 0)", boolAddress + (isPixelShader ? 16 : 0), cfInstr.condJmp.condition ^ simpleControlFlow ? "!" : "=");
                        else
                            println("if (b{} {}= 0)", uint32_t(cfInstr.condJmp.boolAddress), cfInstr.condJmp.condition ^ simpleControlFlow ? "!" : "=");
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
                        // Pixel shader bool constants live at hardware slots 128-255;
                        // the constant table indexes them 0-based.
                        uint32_t boolAddress = cfInstr.condCall.boolAddress;
                        if (isPixelShader && boolAddress >= 128)
                            boolAddress -= 128;

                        auto findResult = boolConstants.find(boolAddress);
                        if (findResult != boolConstants.end())
                            println("\t\t\tif ((g_Booleans & {}) {}= 0)", findResult->second, cfInstr.condCall.condition ? "!" : "=");
                        else if (rexglueMode)
                            println("\t\t\tif ((g_Booleans & (1 << {})) {}= 0)", boolAddress + (isPixelShader ? 16 : 0), cfInstr.condCall.condition ? "!" : "=");
                        else
                            println("\t\t\tif (b{} {}= 0)", uint32_t(cfInstr.condCall.boolAddress), cfInstr.condCall.condition ? "!" : "=");
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

#ifdef UNLEASHED_RECOMP
    if (hasMtxProjection)
        out += "\t}\n";

    if (!isPixelShader && hasMtxProjection)
        out += "\toPos.xy += g_HalfPixelOffset * oPos.w;\n";
#endif

    out += "}";
}
