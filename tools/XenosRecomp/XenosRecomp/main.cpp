#include "shader.h"
#include "shader_recompiler.h"
#include "dxc_compiler.h"
#include "ucode_fingerprint.h"

#include <cstring>

static std::unique_ptr<uint8_t[]> readAllBytes(const char* filePath, size_t& fileSize)
{
    FILE* file = fopen(filePath, "rb");
    if (file == nullptr)
        throw std::runtime_error(std::string("cannot open ") + filePath);
    fseek(file, 0, SEEK_END);
    fileSize = ftell(file);
    fseek(file, 0, SEEK_SET);
    auto data = std::make_unique<uint8_t[]>(fileSize);
    fread(data.get(), 1, fileSize, file);
    fclose(file);
    return data;
}

static void writeAllBytes(const char* filePath, const void* data, size_t dataSize)
{
    FILE* file = fopen(filePath, "wb");
    fwrite(data, 1, dataSize, file);
    fclose(file);
}

struct RecompiledShader
{
    uint8_t* data = nullptr;
    IDxcBlob* dxil = nullptr;
    std::vector<uint8_t> spirv;
    uint32_t specConstantsMask = 0;
};

// --coverage mode: recompile every container found under the input path and
// write a per-shader CSV report instead of a shader cache. Nothing aborts;
// failures are caught (see the assert override in pch.h) and cataloged.

struct CoverageRow
{
    const uint8_t* data = nullptr;
    XXH64_hash_t containerHash = 0;
    uint64_t runtimeUcodeHash = 0; // XXH3 of LE-swapped ucode; matches rexglue --dump_shaders file names
    uint64_t fingerprint = 0;      // vfetch-insensitive hash; matches bind-time-patched runtime ucode
    bool isPixelShader = false;
    uint32_t ucodeSize = 0;
    bool hlslOk = false;
    bool dxilOk = false;
    bool spirvOk = false;
    std::string reason;
};

static const char* validateContainer(const uint8_t* base)
{
    auto container = reinterpret_cast<const ShaderContainer*>(base);

    if (container->constantTableOffset == 0 ||
        container->constantTableOffset + sizeof(ConstantTableContainer) > container->virtualSize)
        return "constant table offset out of bounds";

    if (container->shaderOffset == 0 ||
        container->shaderOffset + sizeof(Shader) > container->virtualSize)
        return "shader offset out of bounds";

    auto shader = reinterpret_cast<const Shader*>(base + container->shaderOffset);

    if (shader->size == 0 || (shader->size & 3) != 0)
        return "bad ucode size";

    if (shader->physicalOffset + shader->size > container->physicalSize)
        return "ucode out of bounds";

    return nullptr;
}

static uint64_t computeRuntimeUcodeHash(const uint8_t* base)
{
    auto container = reinterpret_cast<const ShaderContainer*>(base);
    auto shader = reinterpret_cast<const Shader*>(base + container->shaderOffset);

    // The runtime (Shader::ucode_data_hash, used for --dump_shaders file
    // names and pipeline-cache keys) hashes the ucode in GUEST byte order,
    // which is exactly how the container stores it. (Dump file CONTENTS are
    // host-order words; hashing those gives a different value.)
    return XXH3_64bits(base + container->virtualSize + shader->physicalOffset, shader->size);
}

static std::string classifyReason(const std::string& reason)
{
    if (reason.empty())
        return "ok";
    if (reason.find("vertexElements") != std::string::npos)
        return "vfetch address not in vertex declaration";
    if (reason.find("interpolators") != std::string::npos)
        return "unmapped export register (interpolator/memexport)";
    if (reason.find("const0Relative") != std::string::npos || reason.find("const1Relative") != std::string::npos)
        return "dynamic constant indexing";
    if (reason.find("simpleControlFlow") != std::string::npos)
        return "unsupported control flow";
    if (reason.rfind("assert: ", 0) == 0)
        return reason;
    return std::string("dxc: ") + reason.substr(0, reason.find('\n'));
}

static int runCoverage(const char* input, const char* reportPath, std::string_view include)
{
    std::vector<std::unique_ptr<uint8_t[]>> files;
    std::map<XXH64_hash_t, CoverageRow> shaders;

    auto scanFile = [&](const std::string& path)
        {
            size_t fileSize = 0;
            auto fileData = readAllBytes(path.c_str(), fileSize);
            bool foundAny = false;

            for (size_t i = 0; fileSize > sizeof(ShaderContainer) && i < fileSize - sizeof(ShaderContainer) - 1;)
            {
                auto shaderContainer = reinterpret_cast<const ShaderContainer*>(fileData.get() + i);
                size_t dataSize = shaderContainer->virtualSize + shaderContainer->physicalSize;

                if ((shaderContainer->flags & 0xFFFFFF00) == 0x102A1100 &&
                    dataSize <= (fileSize - i) &&
                    shaderContainer->field1C == 0 &&
                    shaderContainer->field20 == 0)
                {
                    XXH64_hash_t hash = XXH3_64bits(fileData.get() + i, dataSize);
                    auto shader = shaders.try_emplace(hash);
                    if (shader.second)
                    {
                        auto& row = shader.first->second;
                        row.data = fileData.get() + i;
                        row.containerHash = hash;
                        row.isPixelShader = (shaderContainer->flags & 0x1) == 0;
                        foundAny = true;
                    }

                    i += dataSize;
                }
                else
                {
                    // Byte stride, not dword: shader containers can sit at
                    // unaligned offsets.
                    i += 1;
                }
            }

            if (foundAny)
                files.emplace_back(std::move(fileData));
        };

    if (std::filesystem::is_directory(input))
    {
        for (auto& file : std::filesystem::recursive_directory_iterator(input))
        {
            if (!std::filesystem::is_directory(file))
                scanFile(file.path().string());
        }
    }
    else
    {
        scanFile(input);
    }

    fmt::println("Found {} unique shader containers.", shaders.size());

    std::atomic<uint32_t> progress = 0;

    std::for_each(std::execution::par_unseq, shaders.begin(), shaders.end(), [&](auto& hashShaderPair)
        {
            auto& row = hashShaderPair.second;

            try
            {
                if (const char* invalid = validateContainer(row.data))
                {
                    row.reason = std::string("assert: ") + invalid;
                }
                else
                {
                    auto container = reinterpret_cast<const ShaderContainer*>(row.data);
                    auto shader = reinterpret_cast<const Shader*>(row.data + container->shaderOffset);
                    row.ucodeSize = shader->size;
                    row.runtimeUcodeHash = computeRuntimeUcodeHash(row.data);
                    row.fingerprint = ucodeFingerprint(
                        reinterpret_cast<const uint32_t*>(row.data + container->virtualSize + shader->physicalOffset),
                        shader->size / sizeof(uint32_t), true);

                    thread_local ShaderRecompiler recompiler;
                    recompiler = {};
                    recompiler.recompile(row.data, include);
                    row.hlslOk = true;

                    thread_local DxcCompiler dxcCompiler;

#ifdef XENOS_RECOMP_DXIL
                    IDxcBlob* dxil = dxcCompiler.compile(recompiler.out, recompiler.isPixelShader, recompiler.specConstantsMask != 0, false);
                    if (dxil != nullptr)
                    {
                        row.dxilOk = true;
                        dxil->Release();
                    }
                    else
                    {
                        row.reason = dxcCompiler.lastError;
                    }
#endif

                    IDxcBlob* spirv = dxcCompiler.compile(recompiler.out, recompiler.isPixelShader, false, true);
                    if (spirv != nullptr)
                    {
                        row.spirvOk = true;
                        spirv->Release();
                    }
                    else if (row.reason.empty())
                    {
                        row.reason = dxcCompiler.lastError;
                    }
                }
            }
            catch (const std::exception& e)
            {
                row.reason = e.what();
            }

            size_t currentProgress = ++progress;
            if ((currentProgress % 50) == 0)
                fmt::println("Analyzing shaders... {}/{}", currentProgress, shaders.size());
        });

    StringBuffer csv;
    csv.println("runtime_ucode_hash,fingerprint,container_hash,type,ucode_bytes,hlsl,dxil,spirv,category,reason");

    std::map<std::string, uint32_t> vsCategories, psCategories;
    uint32_t vsTotal = 0, psTotal = 0, vsOk = 0, psOk = 0;

    for (auto& [hash, row] : shaders)
    {
        bool ok = row.hlslOk &&
#ifdef XENOS_RECOMP_DXIL
            row.dxilOk &&
#endif
            row.spirvOk;

        (row.isPixelShader ? psTotal : vsTotal)++;
        if (ok)
            (row.isPixelShader ? psOk : vsOk)++;

        std::string category = ok ? "ok" : classifyReason(row.reason);
        if (!ok)
            (row.isPixelShader ? psCategories : vsCategories)[category]++;

        std::string reason = row.reason;
        for (auto& c : reason)
        {
            if (c == ',' || c == '\n' || c == '\r')
                c = ' ';
        }

        csv.println("{:016X},{:016X},{:016X},{},{},{},{},{},{},{}",
            row.runtimeUcodeHash, row.fingerprint, row.containerHash, row.isPixelShader ? "ps" : "vs",
            row.ucodeSize, int(row.hlslOk), int(row.dxilOk), int(row.spirvOk), category,
            reason.substr(0, 200));
    }

    writeAllBytes(reportPath, csv.out.data(), csv.out.size());

    fmt::println("");
    fmt::println("=== XenosRecomp coverage ===");
    fmt::println("vertex shaders: {}/{} ok ({:.1f}%)", vsOk, vsTotal, vsTotal ? vsOk * 100.0 / vsTotal : 0.0);
    fmt::println("pixel  shaders: {}/{} ok ({:.1f}%)", psOk, psTotal, psTotal ? psOk * 100.0 / psTotal : 0.0);
    fmt::println("total:          {}/{} ok ({:.1f}%)", vsOk + psOk, shaders.size(),
        !shaders.empty() ? (vsOk + psOk) * 100.0 / shaders.size() : 0.0);

    auto printCategories = [](const char* label, const std::map<std::string, uint32_t>& categories)
        {
            if (!categories.empty())
            {
                fmt::println("{} failure categories:", label);
                for (auto& [category, count] : categories)
                    fmt::println("  {:5d}  {}", count, category);
            }
        };

    printCategories("vs", vsCategories);
    printCategories("ps", psCategories);

    fmt::println("report: {}", reportPath);
    return 0;
}

// Minimal in-memory ShaderContainer for runtime shaders with no matching
// container in the corpus (XDK runtime-built / composited shaders). In
// rexglue mode the container contributes metadata only: an empty constant
// table makes codegen fall back to xe_fc{reg}/s{n} names and raw g_Booleans
// bits; PS interpolators are identity-wired (r{i} = iVar{i}, the rexglue
// index-wired varying contract); the PS outputs mask is scanned from ucode.
// No definition table, embedded def constants read b1 like any register.
static std::vector<uint8_t> buildSyntheticContainer(bool isPixelShader, uint32_t psExportMask)
{
    constexpr uint32_t containerSize = sizeof(ShaderContainer);           // 36
    constexpr uint32_t constantTableOffset = containerSize;
    constexpr uint32_t constantTableSize = sizeof(ConstantTableContainer); // 32
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
        // svPos register 0xFF: never matches a GPR (no vPos convention known
        // without a container). 16 identity interpolators: reg in bits 8-11.
        put32(shaderOffset + offsetof(Shader, fieldC), 0xFF << 8);
        put32(shaderOffset + offsetof(Shader, interpolatorInfo), psInterpolatorCount << 5);
        put32(shaderOffset + offsetof(PixelShader, outputs), psExportMask);
        for (uint32_t i = 0; i < psInterpolatorCount; i++)
            put32(shaderOffset + sizeof(PixelShader) + i * sizeof(uint32_t), i << 8);
    }

    return data;
}

// Records, for every mapped runtime dump, the word-level difference between
// the container ucode and the bind-time-patched runtime bytes (the guest's
// vfetch patch). The output CSV replaces the dumps directory as pack input:
// --rexglue-pack-deltas rebuilds byte-identical code from image + CSV alone.
static int runMakeDeltas(const char* containersPath, const char* dumpsPath, const char* mapPath,
    const char* outPath)
{
    std::vector<std::unique_ptr<uint8_t[]>> files;
    std::map<uint64_t, const uint8_t*> containers;

    {
        size_t fileSize = 0;
        auto fileData = readAllBytes(containersPath, fileSize);
        for (size_t i = 0; fileSize > sizeof(ShaderContainer) && i < fileSize - sizeof(ShaderContainer) - 1;)
        {
            auto shaderContainer = reinterpret_cast<const ShaderContainer*>(fileData.get() + i);
            size_t dataSize = shaderContainer->virtualSize + shaderContainer->physicalSize;
            if ((shaderContainer->flags & 0xFFFFFF00) == 0x102A1100 &&
                dataSize <= (fileSize - i) &&
                shaderContainer->field1C == 0 &&
                shaderContainer->field20 == 0 &&
                validateContainer(fileData.get() + i) == nullptr)
            {
                containers.try_emplace(XXH3_64bits(fileData.get() + i, dataSize), fileData.get() + i);
                i += dataSize;
            }
            else
            {
                i += 1;
            }
        }
        files.emplace_back(std::move(fileData));
    }

    size_t mapSize = 0;
    auto mapData = readAllBytes(mapPath, mapSize);
    std::string_view map(reinterpret_cast<const char*>(mapData.get()), mapSize);

    StringBuffer out;
    out.println("runtime_ucode_hash,type,match,container_hash,fingerprint,recompiles_ok,patches");

    size_t rows = 0, totalPatched = 0, maxPatched = 0;
    size_t pos = 0;
    bool header = true;
    while (pos < map.size())
    {
        size_t eol = map.find('\n', pos);
        std::string_view line = map.substr(pos, (eol == std::string_view::npos ? map.size() : eol) - pos);
        pos = (eol == std::string_view::npos) ? map.size() : eol + 1;
        if (header) { header = false; continue; }
        if (!line.empty() && line.back() == '\r')
            line.remove_suffix(1);
        if (line.empty())
            continue;

        std::string cols[6];
        size_t colStart = 0;
        for (uint32_t c = 0; c < 6 && colStart <= line.size(); c++)
        {
            size_t comma = line.find(',', colStart);
            cols[c] = std::string(line.substr(colStart, (comma == std::string_view::npos ? line.size() : comma) - colStart));
            colStart = (comma == std::string_view::npos) ? line.size() + 1 : comma + 1;
        }

        if (cols[5] != "1" || cols[3].empty())
        {
            fprintf(stderr, "[deltas] %s has no recompilable container, cannot reconstruct\n", cols[0].c_str());
            return 1;
        }
        auto it = containers.find(strtoull(cols[3].c_str(), nullptr, 16));
        if (it == containers.end())
        {
            fprintf(stderr, "[deltas] container %s not found in image\n", cols[3].c_str());
            return 1;
        }

        const bool isPixel = cols[1] == "ps";
        std::string dumpFile = fmt::format("{}/shader_{}.ucode.bin.{}", dumpsPath, cols[0],
            isPixel ? "frag" : "vert");
        size_t dumpSize = 0;
        auto dumpData = readAllBytes(dumpFile.c_str(), dumpSize);

        auto c = reinterpret_cast<const ShaderContainer*>(it->second);
        auto sh = reinterpret_cast<const Shader*>(it->second + c->shaderOffset);
        if (dumpSize != sh->size)
        {
            fprintf(stderr, "[deltas] %s size mismatch: dump %zu vs container %u\n",
                cols[0].c_str(), dumpSize, uint32_t(sh->size));
            return 1;
        }

        auto containerWords = reinterpret_cast<const uint32_t*>(it->second + c->virtualSize + sh->physicalOffset);
        auto dumpWords = reinterpret_cast<const uint32_t*>(dumpData.get());

        std::string patches;
        size_t patched = 0;
        for (size_t i = 0; i < dumpSize / sizeof(uint32_t); i++)
        {
            const uint32_t runtimeWord = __builtin_bswap32(dumpWords[i]); // LE dump -> guest-order memory word
            if (runtimeWord != containerWords[i])
            {
                if (!patches.empty())
                    patches += ';';
                patches += fmt::format("{}:{:08X}", i, runtimeWord);
                patched++;
            }
        }

        rows++;
        totalPatched += patched;
        if (patched > maxPatched)
            maxPatched = patched;
        out.println("{},{},{},{},{},{},{}", cols[0], cols[1], cols[2], cols[3], cols[4], cols[5], patches);
    }

    writeAllBytes(outPath, out.out.data(), out.out.size());
    fmt::println("deltas: {} shaders, {} patched words total (max {} in one shader) -> {}",
        rows, totalPatched, maxPatched, outPath);
    return 0;
}

// Builds the per-title native shader pack: for every runtime-seen shader in
// the map CSV that resolves to a recompilable container, generates HLSL from
// (container metadata + bind-time-patched runtime ucode), compiles DXIL, and
// writes <hash>.<vs|ps>.dxil plus a manifest the runtime consumes.
static int runRexgluePack(const char* containersPath, const char* dumpsPath, const char* mapPath,
    const char* outPath, std::string_view include, bool deltasMode = false)
{
    fprintf(stderr, "[pack] enter\n"); fflush(stderr);

    // Containers by XXH3 hash (matches the map CSV's container_hash column).
    std::vector<std::unique_ptr<uint8_t[]>> files;
    std::map<uint64_t, const uint8_t*> containers;

    auto scanFile = [&](const std::string& path)
        {
            size_t fileSize = 0;
            auto fileData = readAllBytes(path.c_str(), fileSize);
            bool foundAny = false;

            for (size_t i = 0; fileSize > sizeof(ShaderContainer) && i < fileSize - sizeof(ShaderContainer) - 1;)
            {
                auto shaderContainer = reinterpret_cast<const ShaderContainer*>(fileData.get() + i);
                size_t dataSize = shaderContainer->virtualSize + shaderContainer->physicalSize;

                if ((shaderContainer->flags & 0xFFFFFF00) == 0x102A1100 &&
                    dataSize <= (fileSize - i) &&
                    shaderContainer->field1C == 0 &&
                    shaderContainer->field20 == 0)
                {
                    if (validateContainer(fileData.get() + i) == nullptr)
                    {
                        containers.try_emplace(XXH3_64bits(fileData.get() + i, dataSize), fileData.get() + i);
                        foundAny = true;
                    }
                    i += dataSize;
                }
                else
                {
                    i += 1;
                }
            }

            if (foundAny)
                files.emplace_back(std::move(fileData));
        };

    if (std::filesystem::is_directory(containersPath))
    {
        for (auto& file : std::filesystem::recursive_directory_iterator(containersPath))
        {
            if (!std::filesystem::is_directory(file))
                scanFile(file.path().string());
        }
    }
    else
    {
        scanFile(containersPath);
    }

    fprintf(stderr, "[pack] %zu containers indexed\n", containers.size()); fflush(stderr);

    struct PackJob
    {
        std::string runtimeHash;
        std::string fingerprint;
        bool isPixelShader = false;
        const uint8_t* container = nullptr; // null = container-less generation
        std::vector<uint8_t> synthetic;     // backing store for the null case
        // deltas mode: word index -> value overrides applied to the container
        // ucode (the bind-time vfetch patches recorded by --make-deltas)
        std::vector<std::pair<uint32_t, uint32_t>> patches;
        // results
        bool ok = false;
        std::string reason;
        std::string bindings;
        std::string floatBitmap; // 4x16 hex chars, matches ConstantRegisterMap::float_bitmap
        std::vector<uint8_t> dxil;
        // SPIR-V twin for the Vulkan backend ({hash}.{type}.spirv). Empty on
        // spirv-compile failure, non-fatal, the D3D12 path is unaffected and
        // the runtime falls back per shader.
        std::vector<uint8_t> spirv;
        std::vector<uint8_t> spirvTrim;
        std::string hlsl;
        // VS trim: same body with unwritten oVar/oPts outputs removed
        // from the signature. Loaded as {hash}.vst.dxil; the runtime prefers
        // it except when an expansion GS (which links the full signature) is
        // attached. Empty when the shader writes everything or trim failed.
        std::vector<uint8_t> dxilTrim;
        std::string hlslTrim;
    };

    // Remove unwritten output declarations + their zero-inits from a rexglue
    // VS. The body never references removed vars (the mask says unwritten),
    // and each parameter carries its own leading comma, so the signature
    // stays well-formed for any subset.
    auto trimVsOutputs = [](const std::string& hlsl, uint32_t writtenMask, bool wroteOPts)
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
            removeAll(fmt::format(",\n\tout float4 oVar{0} : TEXCOORD{0}", i));
            removeAll(fmt::format("\toVar{} = 0.0;\n", i));
        }
        if (!wroteOPts)
        {
            removeAll(",\n\tout float4 oPts : TEXCOORD16");
            removeAll("\toPts = 0.0;\n");
        }
        return s;
    };

    std::vector<PackJob> jobs;
    std::set<std::string> mappedHashes;
    {
        size_t mapSize = 0;
        auto mapData = readAllBytes(mapPath, mapSize);
        std::string_view map(reinterpret_cast<const char*>(mapData.get()), mapSize);

        size_t pos = 0;
        bool header = true;
        while (pos < map.size())
        {
            size_t eol = map.find('\n', pos);
            std::string_view line = map.substr(pos, (eol == std::string_view::npos ? map.size() : eol) - pos);
            pos = (eol == std::string_view::npos) ? map.size() : eol + 1;

            if (header) { header = false; continue; }
            if (!line.empty() && line.back() == '\r')
                line.remove_suffix(1);
            if (line.empty())
                continue;

            // runtime_ucode_hash,type,match,container_hash,fingerprint,recompiles_ok[,patches]
            std::string cols[7];
            size_t colStart = 0;
            for (uint32_t c = 0; c < 7 && colStart <= line.size(); c++)
            {
                size_t comma = line.find(',', colStart);
                cols[c] = std::string(line.substr(colStart, (comma == std::string_view::npos ? line.size() : comma) - colStart));
                colStart = (comma == std::string_view::npos) ? line.size() + 1 : comma + 1;
            }

            PackJob job;
            job.runtimeHash = cols[0];
            job.fingerprint = cols[4];
            job.isPixelShader = (cols[1] == "ps");

            // Rows without a recompilable container (match=none, composited
            // VS, patched-beyond-vfetch PS, non-recompiling containers) fall
            // back to container-less generation from the raw runtime ucode.
            if (cols[5] == "1" && !cols[3].empty())
            {
                auto it = containers.find(strtoull(cols[3].c_str(), nullptr, 16));
                if (it != containers.end())
                    job.container = it->second;
            }

            if (deltasMode)
            {
                // patches column: semicolon list of wordIndex:HEXWORD
                size_t p = 0;
                const std::string& s = cols[6];
                while (p < s.size())
                {
                    size_t colon = s.find(':', p);
                    size_t semi = s.find(';', p);
                    if (semi == std::string::npos)
                        semi = s.size();
                    if (colon == std::string::npos || colon > semi)
                        break;
                    job.patches.emplace_back(
                        uint32_t(strtoul(s.substr(p, colon - p).c_str(), nullptr, 10)),
                        uint32_t(strtoul(s.substr(colon + 1, semi - colon - 1).c_str(), nullptr, 16)));
                    p = semi + 1;
                }
            }

            mappedHashes.insert(job.runtimeHash);
            jobs.emplace_back(std::move(job));
        }
    }

    // Runtime dumps the map has never seen (e.g. native-video-layer pack-miss
    // dumps collected after the corpus capture): container-less jobs too.
    if (!deltasMode)
    for (auto& file : std::filesystem::directory_iterator(dumpsPath))
    {
        if (std::filesystem::is_directory(file))
            continue;
        std::string name = file.path().filename().string();
        size_t marker = name.find(".ucode.bin.");
        if (name.rfind("shader_", 0) != 0 || marker == std::string::npos)
            continue;

        PackJob job;
        job.runtimeHash = name.substr(7, marker - 7);
        job.isPixelShader = name.compare(name.size() - 5, 5, ".frag") == 0;
        if (job.runtimeHash.size() != 16 || !mappedHashes.insert(job.runtimeHash).second)
            continue;
        jobs.emplace_back(std::move(job));
    }

    size_t containerless = 0;
    for (auto& job : jobs)
        containerless += job.container == nullptr;
    fprintf(stderr, "[pack] %zu shaders to generate (%zu container-less)\n", jobs.size(), containerless); fflush(stderr);
    std::filesystem::create_directories(outPath);

    std::atomic<uint32_t> progress = 0;

    std::for_each(std::execution::par_unseq, jobs.begin(), jobs.end(), [&](PackJob& job)
        {
            try
            {
                size_t dumpSize = 0;
                std::unique_ptr<uint8_t[]> dumpData;
                std::unique_ptr<uint8_t[]> swapped;
                const uint32_t* src = nullptr;

                if (deltasMode)
                {
                    // Code source = container ucode + recorded bind-time
                    // patches; byte-identical to the runtime dump the deltas
                    // were extracted from.
                    if (job.container == nullptr)
                        throw std::runtime_error("deltas row without container");
                    auto c = reinterpret_cast<const ShaderContainer*>(job.container);
                    auto sh = reinterpret_cast<const Shader*>(job.container + c->shaderOffset);
                    dumpSize = sh->size;
                    swapped = std::make_unique<uint8_t[]>(dumpSize);
                    std::memcpy(swapped.get(), job.container + c->virtualSize + sh->physicalOffset, dumpSize);
                    auto words = reinterpret_cast<uint32_t*>(swapped.get());
                    for (auto& [idx, value] : job.patches)
                    {
                        if (idx >= dumpSize / sizeof(uint32_t))
                            throw std::runtime_error("patch index out of range");
                        words[idx] = value;
                    }
                }
                else
                {
                    std::string dumpFile = fmt::format("{}/shader_{}.ucode.bin.{}", dumpsPath, job.runtimeHash,
                        job.isPixelShader ? "frag" : "vert");

                    dumpData = readAllBytes(dumpFile.c_str(), dumpSize);

                    // Runtime dumps are LE words; the recompiler parses BE.
                    swapped = std::make_unique<uint8_t[]>(dumpSize);
                    src = reinterpret_cast<const uint32_t*>(dumpData.get());
                    auto dst = reinterpret_cast<uint32_t*>(swapped.get());
                    for (size_t i = 0; i < dumpSize / sizeof(uint32_t); i++)
                        dst[i] = __builtin_bswap32(src[i]);
                }

                if (job.container == nullptr)
                {
                    const uint32_t exportMask = job.isPixelShader
                        ? ucodePsExportMask(src, dumpSize / sizeof(uint32_t), false)
                        : 0;
                    job.synthetic = buildSyntheticContainer(job.isPixelShader, exportMask);
                    job.container = job.synthetic.data();
                }
                if (job.fingerprint.empty() && src != nullptr)
                    job.fingerprint = fmt::format("{:016X}",
                        ucodeFingerprint(src, dumpSize / sizeof(uint32_t), false));

                thread_local ShaderRecompiler recompiler;

                // Pass 1: discover float-constant usage (output discarded).
                recompiler = {};
                recompiler.rexglueMode = true;
                recompiler.rexAbsoluteFloatFile = !job.synthetic.empty();
                recompiler.rexCodeOverride = swapped.get();
                recompiler.rexCodeOverrideSize = uint32_t(dumpSize);
                recompiler.recompile(job.container, include);

                std::set<uint32_t> usedFloats = std::move(recompiler.rexUsedFloatConstants);
                bool floatsDynamic = recompiler.rexFloatsDynamic;

                {
                    uint64_t bitmap[4] = {};
                    if (floatsDynamic)
                    {
                        bitmap[0] = bitmap[1] = bitmap[2] = bitmap[3] = ~0ull;
                    }
                    else
                    {
                        for (uint32_t reg : usedFloats)
                            bitmap[reg >> 6] |= 1ull << (reg & 63);
                    }
                    job.floatBitmap = fmt::format("{:016X}{:016X}{:016X}{:016X}",
                        bitmap[3], bitmap[2], bitmap[1], bitmap[0]);
                }

                // Pass 2: final layout. Static-usage shaders read their float
                // constants at the compacted ranks the runtime uploads;
                // dynamically-addressed shaders keep the absolute 256-register
                // layout (the runtime uploads the full file for those).
                recompiler = {};
                recompiler.rexglueMode = true;
                recompiler.rexAbsoluteFloatFile = !job.synthetic.empty();
                recompiler.rexCodeOverride = swapped.get();
                recompiler.rexCodeOverrideSize = uint32_t(dumpSize);
                if (!floatsDynamic && job.synthetic.empty())
                {
                    // Absolute layout for static shaders too: the runtime
                    // forces float_dynamic_addressing (identity uploads)
                    // whenever native_shaders is enabled, so compacted ranks
                    // would read the wrong registers.
                    for (uint32_t reg : usedFloats)
                        recompiler.rexFloatRank.emplace(reg, reg);
                }
                recompiler.recompile(job.container, include);

                job.hlsl = recompiler.out;

                for (auto& binding : recompiler.rexBindings)
                {
                    if (!job.bindings.empty())
                        job.bindings += ';';
                    job.bindings += fmt::format("{}:{}:{}:{}", binding.isSampler ? 's' : 't',
                        binding.fetchConstant, binding.dimension, int(binding.isSigned));
                }

                thread_local DxcCompiler dxcCompiler;
                IDxcBlob* dxil = dxcCompiler.compile(recompiler.out, recompiler.isPixelShader, false, false);
                if (dxil != nullptr)
                {
                    job.dxil.assign(reinterpret_cast<uint8_t*>(dxil->GetBufferPointer()),
                        reinterpret_cast<uint8_t*>(dxil->GetBufferPointer()) + dxil->GetBufferSize());
                    dxil->Release();
                    job.ok = true;

                    if (IDxcBlob* spirv = dxcCompiler.compile(recompiler.out, recompiler.isPixelShader, false, true))
                    {
                        job.spirv.assign(reinterpret_cast<uint8_t*>(spirv->GetBufferPointer()),
                            reinterpret_cast<uint8_t*>(spirv->GetBufferPointer()) + spirv->GetBufferSize());
                        spirv->Release();
                    }

                    // Trimmed variant: only when the trim removes something.
                    // A trim-compile failure is non-fatal, the
                    // full-signature dxil above remains the shader.
                    // Container-less (synthetic) jobs have no interpolator
                    // table, so trim safety cannot be established, skip.
                    if (!recompiler.isPixelShader && job.synthetic.empty() &&
                        (recompiler.rexWrittenOVarMask != 0xFFFFu || !recompiler.rexWroteOPts))
                    {
                        job.hlslTrim = trimVsOutputs(recompiler.out,
                            recompiler.rexWrittenOVarMask, recompiler.rexWroteOPts);
                        IDxcBlob* dxilTrim = dxcCompiler.compile(job.hlslTrim, false, false, false);
                        if (dxilTrim != nullptr)
                        {
                            job.dxilTrim.assign(reinterpret_cast<uint8_t*>(dxilTrim->GetBufferPointer()),
                                reinterpret_cast<uint8_t*>(dxilTrim->GetBufferPointer()) + dxilTrim->GetBufferSize());
                            dxilTrim->Release();
                            if (IDxcBlob* spirvTrim = dxcCompiler.compile(job.hlslTrim, false, false, true))
                            {
                                job.spirvTrim.assign(reinterpret_cast<uint8_t*>(spirvTrim->GetBufferPointer()),
                                    reinterpret_cast<uint8_t*>(spirvTrim->GetBufferPointer()) + spirvTrim->GetBufferSize());
                                spirvTrim->Release();
                            }
                        }
                        else
                        {
                            job.hlslTrim.clear();
                        }
                    }
                }
                else
                {
                    job.reason = dxcCompiler.lastError.substr(0, dxcCompiler.lastError.find('\n'));
                }
            }
            catch (const std::exception& e)
            {
                job.reason = e.what();
            }

            uint32_t currentProgress = ++progress;
            if ((currentProgress % 50) == 0)
                fmt::println("Generating... {}/{}", currentProgress, jobs.size());
        });

    StringBuffer manifest;
    manifest.println("runtime_ucode_hash,type,fingerprint,dxil,bindings,float_bitmap,reason");

    uint32_t okCount = 0;
    std::map<std::string, uint32_t> failures;

    for (auto& job : jobs)
    {
        const char* type = job.isPixelShader ? "ps" : "vs";
        std::string dxilName;

        if (job.ok)
        {
            okCount++;
            dxilName = fmt::format("{}.{}.dxil", job.runtimeHash, type);
            writeAllBytes(fmt::format("{}/{}", outPath, dxilName).c_str(), job.dxil.data(), job.dxil.size());
            writeAllBytes(fmt::format("{}/{}.{}.hlsl", outPath, job.runtimeHash, type).c_str(),
                job.hlsl.data(), job.hlsl.size());
            if (!job.spirv.empty())
                writeAllBytes(fmt::format("{}/{}.{}.spirv", outPath, job.runtimeHash, type).c_str(),
                    job.spirv.data(), job.spirv.size());
            if (!job.dxilTrim.empty())
            {
                writeAllBytes(fmt::format("{}/{}.vst.dxil", outPath, job.runtimeHash).c_str(),
                    job.dxilTrim.data(), job.dxilTrim.size());
                writeAllBytes(fmt::format("{}/{}.vst.hlsl", outPath, job.runtimeHash).c_str(),
                    job.hlslTrim.data(), job.hlslTrim.size());
                if (!job.spirvTrim.empty())
                    writeAllBytes(fmt::format("{}/{}.vst.spirv", outPath, job.runtimeHash).c_str(),
                        job.spirvTrim.data(), job.spirvTrim.size());
            }
        }
        else
        {
            failures[job.reason.substr(0, 120)]++;
        }

        std::string reason = job.reason;
        for (auto& c : reason)
        {
            if (c == ',' || c == '\n' || c == '\r')
                c = ' ';
        }

        manifest.println("{},{},{},{},{},{},{}", job.runtimeHash, type, job.fingerprint, dxilName,
            job.bindings, job.floatBitmap, reason.substr(0, 160));
    }

    writeAllBytes(fmt::format("{}/manifest.csv", outPath).c_str(), manifest.out.data(), manifest.out.size());

    // Rectangle-list expansion geometry shader. Xenos RECTLIST primitives
    // carry three corners of a rectangle; hardware derives the 4th vertex by
    // parallelogram completion of the shader OUTPUTS. Every rexglue native
    // VS/PS pair uses the same interpolator signature (SV_Position + 16
    // float4 TEXCOORDs), so one geometry shader serves every pipeline; the
    // runtime attaches it to RECTLIST draws.
    {
        static const char RECT_GS[] = R"(
struct XeRectVertex
{
    float4 pos : SV_Position;
    float4 var[16] : TEXCOORD0;
};

[maxvertexcount(4)]
void main(triangle XeRectVertex v[3], inout TriangleStream<XeRectVertex> stream)
{
    // The right-angle corner is the vertex not on the longest (hypotenuse)
    // edge; the 4th vertex mirrors it across the hypotenuse midpoint:
    // v3 = hypA + hypB - corner, for position and every interpolator.
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

    // Strip (corner, hypA, hypB, v3) = the original triangle plus the
    // mirrored half. Rectangles are never culled on Xenos; the runtime
    // disables culling on rect pipelines, so winding does not matter.
    stream.Append(v[corner]);
    stream.Append(v[hypA]);
    stream.Append(v[hypB]);
    stream.Append(v3);
}
)";
        DxcCompiler gsCompiler;
        IDxcBlob* gs = gsCompiler.compile(RECT_GS, false, false, false, L"-T gs_6_0");
        if (gs != nullptr)
        {
            writeAllBytes(fmt::format("{}/rect_expand.gs.dxil", outPath).c_str(),
                gs->GetBufferPointer(), gs->GetBufferSize());
            gs->Release();
        }
        else
        {
            fmt::println("rect_expand GS compile failed: {}", gsCompiler.lastError);
        }
        if (IDxcBlob* gsv = gsCompiler.compile(RECT_GS, false, false, true, L"-T gs_6_0"))
        {
            writeAllBytes(fmt::format("{}/rect_expand.gs.spirv", outPath).c_str(),
                gsv->GetBufferPointer(), gsv->GetBufferSize());
            gsv->Release();
        }
        else
        {
            fmt::println("rect_expand GS spirv compile failed: {}", gsCompiler.lastError);
        }
    }

    // User-clip-plane geometry shader. Xenos clips against up to 6 planes
    // (PA_CL_UCP, programmed by D3DDevice_SetClipPlane) by dotting the RAW
    // clip-space position with each plane. The pack VS applies the b0
    // ndc_scale/ndc_offset affine to its output position, so the RUNTIME
    // pre-transforms the planes by that affine's inverse-transpose before
    // writing them to b0 c2..c7 (xe_user_clip_planes), this GS is then a
    // pure passthrough that emits the six dots as SV_ClipDistance. Disabled
    // plane slots are zero: dot = 0, and D3D only clips distances < 0.
    // Attached by the runtime only when the PA_CL_CLIP_CNTL UCP mask is
    // nonzero (and no rect/point expansion GS is needed).
    {
        static const char CLIP_GS[] = R"(
#ifdef __spirv__
struct XePushConstants { uint64_t System; uint64_t FloatsVs; uint64_t FloatsPs; uint64_t BoolLoop; uint64_t Fetch; uint64_t IdxVs; uint64_t IdxPs; uint64_t SharedMem; };
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
        DxcCompiler gsCompiler;
        IDxcBlob* gs = gsCompiler.compile(CLIP_GS, false, false, false, L"-T gs_6_0");
        if (gs != nullptr)
        {
            writeAllBytes(fmt::format("{}/clip_planes.gs.dxil", outPath).c_str(),
                gs->GetBufferPointer(), gs->GetBufferSize());
            gs->Release();
        }
        else
        {
            fmt::println("clip_planes GS compile failed: {}", gsCompiler.lastError);
        }
        if (IDxcBlob* gsv = gsCompiler.compile(CLIP_GS, false, false, true, L"-T gs_6_0"))
        {
            writeAllBytes(fmt::format("{}/clip_planes.gs.spirv", outPath).c_str(),
                gsv->GetBufferPointer(), gsv->GetBufferSize());
            gsv->Release();
        }
        else
        {
            fmt::println("clip_planes GS spirv compile failed: {}", gsCompiler.lastError);
        }
    }

    // Point-sprite expansion geometry shader. Xenos POINTLIST draws expand
    // each point into a screen-aligned square sized by the VS point-size
    // export (register 63 -> the pack-wide oPts : TEXCOORD16 output, size in
    // pixels). Interpolators are copied flat (the sprite-light pixel shaders
    // key off SV_Position + per-light interpolators, not sprite UVs). The
    // pixel->NDC conversion reuses the system CB's half-pixel offsets
    // (c9.xy = ±1/viewport), whose magnitude is exactly 1 px in NDC/2.
    {
        static const char POINT_GS[] = R"(
#ifdef __spirv__
struct XePushConstants { uint64_t System; uint64_t FloatsVs; uint64_t FloatsPs; uint64_t BoolLoop; uint64_t Fetch; uint64_t IdxVs; uint64_t IdxPs; uint64_t SharedMem; };
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
    // oPts.x = vertex point diameter in pixels, clamped to PA_SU_POINT_MINMAX
    // (c10.zw); a VS that does not export a size leaves oPts at 0 and the
    // PA_SU_POINT_SIZE constant diameters (c10.xy) apply instead, mirroring
    // the ring backend's vertex-vs-constant point size selection. Half-extent
    // in NDC (pre-divide clip units need the *w) = (size/2) * (2/viewport)
    // = size * |c9.xy|.
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
        DxcCompiler gsCompiler;
        IDxcBlob* gs = gsCompiler.compile(POINT_GS, false, false, false, L"-T gs_6_0");
        if (gs != nullptr)
        {
            writeAllBytes(fmt::format("{}/point_expand.gs.dxil", outPath).c_str(),
                gs->GetBufferPointer(), gs->GetBufferSize());
            gs->Release();
        }
        else
        {
            fmt::println("point_expand GS compile failed: {}", gsCompiler.lastError);
        }
        if (IDxcBlob* gsv = gsCompiler.compile(POINT_GS, false, false, true, L"-T gs_6_0"))
        {
            writeAllBytes(fmt::format("{}/point_expand.gs.spirv", outPath).c_str(),
                gsv->GetBufferPointer(), gsv->GetBufferSize());
            gsv->Release();
        }
        else
        {
            fmt::println("point_expand GS spirv compile failed: {}", gsCompiler.lastError);
        }
    }

    // Fullscreen blit pair (frontbuffer-texture -> swapchain composite in the
    // native video layer's Swap). Standalone shaders that only declare the
    // slices of the rexglue root signature they use: b4 descriptor indices,
    // bindless 2D textures (t0 space1), bindless samplers (s0 space0).
    {
        static const char BLIT_VS[] = R"(
void main(in uint vid : SV_VertexID, out float4 pos : SV_Position, out float2 uv : TEXCOORD0)
{
    // Fullscreen triangle.
    uv = float2((vid << 1) & 2, vid & 2);
    pos = float4(uv * float2(2.0, -2.0) + float2(-1.0, 1.0), 0.0, 1.0);
}
)";
        static const char BLIT_PS[] = R"(
#ifdef __spirv__
struct XePushConstants { uint64_t System; uint64_t FloatsVs; uint64_t FloatsPs; uint64_t BoolLoop; uint64_t Fetch; uint64_t IdxVs; uint64_t IdxPs; uint64_t SharedMem; };
[[vk::push_constant]] ConstantBuffer<XePushConstants> xe_push;
#define XE_IDX(i) vk::RawBufferLoad<uint>(xe_push.IdxPs + (i) * 4)
#else
cbuffer XeDescriptorIndices : register(b4, space0)
{
    uint4 xe_descriptor_indices[8];
};
#define XE_IDX(i) (xe_descriptor_indices[(i) >> 2][(i) & 3])
#endif
// Texture2DArray to match the runtime's forced 2D-array SRVs (the pack-wide
// xe_textures_2d contract), a plain Texture2D declaration over an array
// descriptor is undefined in D3D12.
Texture2DArray<float4> xe_textures2d[] : register(t0, space1);
SamplerState xe_samplers[] : register(s0, space0);

void main(in float4 pos : SV_Position, in float2 uv : TEXCOORD0, out float4 color : SV_Target0)
{
    color = xe_textures2d[XE_IDX(0)]
        .SampleLevel(xe_samplers[0], float3(uv, 0.0), 0.0);
    // Display gamma ramp (D3DDevice_SetGammaRamp / DC_LUT emulation): b4
    // [0].y carries a 256x3 R16_UNORM LUT descriptor (rows = R,G,B curves),
    // 0 = identity/no ramp. The ring backend applies the guest ramp in its
    // present pass (apply_gamma); without it the native output misses the
    // game's display contrast curve.
    uint lut = XE_IDX(1);
    if (lut != 0)
    {
        uint3 idx = uint3(saturate(color.rgb) * 255.0 + 0.5);
        color.r = xe_textures2d[(lut)].Load(int4(idx.r, 0, 0, 0)).x;
        color.g = xe_textures2d[(lut)].Load(int4(idx.g, 1, 0, 0)).x;
        color.b = xe_textures2d[(lut)].Load(int4(idx.b, 2, 0, 0)).x;
    }
    // Optional user warmth grade (b4 [0].z = strength float as bits, 0 = off):
    // a present-time warm push toward the 360 look. Preserves luma roughly by
    // trading blue for red/green rather than scaling up.
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
        DxcCompiler blitCompiler;
        IDxcBlob *vs = blitCompiler.compile(BLIT_VS, false, false, false);
        IDxcBlob *ps = vs != nullptr ? blitCompiler.compile(BLIT_PS, true, false, false) : nullptr;
        if (vs != nullptr && ps != nullptr)
        {
            writeAllBytes(fmt::format("{}/blit.vs.dxil", outPath).c_str(), vs->GetBufferPointer(), vs->GetBufferSize());
            writeAllBytes(fmt::format("{}/blit.ps.dxil", outPath).c_str(), ps->GetBufferPointer(), ps->GetBufferSize());
            if (IDxcBlob* vsv = blitCompiler.compile(BLIT_VS, false, false, true))
            {
                writeAllBytes(fmt::format("{}/blit.vs.spirv", outPath).c_str(), vsv->GetBufferPointer(), vsv->GetBufferSize());
                vsv->Release();
            }
            if (IDxcBlob* psv = blitCompiler.compile(BLIT_PS, true, false, true))
            {
                writeAllBytes(fmt::format("{}/blit.ps.spirv", outPath).c_str(), psv->GetBufferPointer(), psv->GetBufferSize());
                psv->Release();
            }
        }
        else
        {
            fmt::println("blit shader compile failed: {}", blitCompiler.lastError);
        }
        if (vs != nullptr) vs->Release();
        if (ps != nullptr) ps->Release();
    }

    fmt::println("");
    fmt::println("=== rexglue pack ===");
    fmt::println("generated: {}/{}", okCount, jobs.size());
    for (auto& [reason, count] : failures)
        fmt::println("  {:4d}  {}", count, reason);
    fmt::println("pack: {}", outPath);
    return okCount == jobs.size() ? 0 : 1;
}

int main(int argc, char** argv)
{
    if (argc >= 5 && strcmp(argv[1], "--coverage") == 0)
    {
        size_t includeSize = 0;
        auto includeData = readAllBytes(argv[4], includeSize);
        return runCoverage(argv[2], argv[3],
            std::string_view(reinterpret_cast<const char*>(includeData.get()), includeSize));
    }

    // Fingerprint raw runtime ucode dumps (rexglue --dump_shaders *.ucode.bin.*,
    // LE words): emits "name,ucode_hash,fingerprint" CSV for matching against
    // the coverage report's fingerprint column.
    if (argc >= 4 && strcmp(argv[1], "--fingerprint") == 0)
    {
        StringBuffer csv;
        csv.println("name,runtime_ucode_hash,fingerprint");

        for (auto& file : std::filesystem::recursive_directory_iterator(argv[2]))
        {
            if (std::filesystem::is_directory(file))
                continue;

            std::string name = file.path().filename().string();
            if (name.find(".ucode.bin.") == std::string::npos)
                continue;

            size_t fileSize = 0;
            auto fileData = readAllBytes(file.path().string().c_str(), fileSize);

            // The runtime hash is the file NAME (shader_<hash>.ucode.bin.*):
            // XXH3 of the guest-order ucode. File contents are host-order.
            uint64_t hash = strtoull(name.c_str() + name.find('_') + 1, nullptr, 16);
            uint64_t fingerprint = ucodeFingerprint(
                reinterpret_cast<const uint32_t*>(fileData.get()), fileSize / sizeof(uint32_t), false);

            csv.println("{},{:016X},{:016X}", name, hash, fingerprint);
        }

        writeAllBytes(argv[3], csv.out.data(), csv.out.size());
        fmt::println("fingerprints -> {}", argv[3]);
        return 0;
    }

    if (argc >= 6 && strcmp(argv[1], "--make-deltas") == 0)
    {
        try
        {
            return runMakeDeltas(argv[2], argv[3], argv[4], argv[5]);
        }
        catch (const std::exception& e)
        {
            fprintf(stderr, "make-deltas failed: %s\n", e.what());
            return 1;
        }
    }

    if (argc >= 6 && strcmp(argv[1], "--rexglue-pack-deltas") == 0)
    {
        try
        {
            size_t includeSize = 0;
            auto includeData = readAllBytes(argv[5], includeSize);
            return runRexgluePack(argv[2], nullptr, argv[3], argv[4],
                std::string_view(reinterpret_cast<const char*>(includeData.get()), includeSize), true);
        }
        catch (const std::exception& e)
        {
            fprintf(stderr, "rexglue-pack-deltas failed: %s\n", e.what());
            return 1;
        }
    }

    if (argc >= 7 && strcmp(argv[1], "--rexglue-pack") == 0)
    {
        try
        {
            size_t includeSize = 0;
            auto includeData = readAllBytes(argv[6], includeSize);
            return runRexgluePack(argv[2], argv[3], argv[4], argv[5],
                std::string_view(reinterpret_cast<const char*>(includeData.get()), includeSize));
        }
        catch (const std::exception& e)
        {
            fprintf(stderr, "rexglue-pack failed: %s\n", e.what());
            fflush(stderr);
            return 1;
        }
    }

    // Histogram vertex-fetch formats/strides across raw runtime dumps,
    // sizes the format-decode helper set for the REXGLUE codegen mode.
    if (argc >= 5 && strcmp(argv[1], "--compile") == 0)
    {
        // --compile <in.hlsl> <out> <profile e.g. ps_6_0> [spirv]
        // Standalone HLSL compile with the same flag set as the pack shaders
        // (-HV 2021, -Gis, and for spirv: -fvk-use-dx-layout [+invert-y for
        // vs]). Used to build the runtime's internal shader headers
        // (EASU/RCAS/FPS overlay) so both backends bind identically.
        try
        {
            size_t srcSize = 0;
            auto src = readAllBytes(argv[2], srcSize);
            const bool spirv = argc >= 6 && strcmp(argv[5], "spirv") == 0;
            std::string profile = argv[4];
            std::wstring targetArg = L"-T " + std::wstring(profile.begin(), profile.end());
            const bool isPixel = profile.rfind("ps", 0) == 0;
            // Pixel/vertex profiles ride the default selection so the
            // vs-only -fvk-invert-y rule applies; other profiles override.
            const wchar_t* over =
                (isPixel || profile.rfind("vs", 0) == 0) ? nullptr : targetArg.c_str();
            DxcCompiler compiler;
            IDxcBlob* blob = compiler.compile(
                std::string(reinterpret_cast<const char*>(src.get()), srcSize),
                isPixel, false, spirv, over);
            if (blob == nullptr)
            {
                fprintf(stderr, "compile failed: %s\n", compiler.lastError.c_str());
                return 1;
            }
            writeAllBytes(argv[3], blob->GetBufferPointer(), blob->GetBufferSize());
            blob->Release();
            fmt::println("{} -> {} ({}, {})", argv[2], argv[3], profile, spirv ? "spirv" : "dxil");
            return 0;
        }
        catch (const std::exception& e)
        {
            fprintf(stderr, "compile failed: %s\n", e.what());
            return 1;
        }
    }

    if (argc >= 3 && strcmp(argv[1], "--vfetchstats") == 0)
    {
        struct VfetchCounters
        {
            std::map<std::tuple<uint32_t, uint32_t, uint32_t>, uint32_t> formats; // (format, signed, integer) -> count
            std::map<uint32_t, uint32_t> strides;
        } counters;

        for (auto& file : std::filesystem::recursive_directory_iterator(argv[2]))
        {
            if (std::filesystem::is_directory(file))
                continue;
            std::string name = file.path().filename().string();
            if (name.find(".ucode.bin.") == std::string::npos)
                continue;

            size_t fileSize = 0;
            auto fileData = readAllBytes(file.path().string().c_str(), fileSize);

            ucodeVisitVfetches(reinterpret_cast<const uint32_t*>(fileData.get()), fileSize / sizeof(uint32_t), false,
                [](const VfetchInfo& info, void* ctx)
                {
                    auto& c = *static_cast<VfetchCounters*>(ctx);
                    ++c.formats[{ info.format, info.isSigned, info.isInteger }];
                    ++c.strides[info.stride];
                }, &counters);
        }

        fmt::println("vfetch formats (format, signed, integer) -> count:");
        for (auto& [key, count] : counters.formats)
            fmt::println("  fmt={:2d} signed={} integer={} : {}", std::get<0>(key), std::get<1>(key), std::get<2>(key), count);
        fmt::println("strides (dwords) -> count:");
        for (auto& [stride, count] : counters.strides)
            fmt::println("  {:3d} : {}", stride, count);
        return 0;
    }

#ifndef XENOS_RECOMP_INPUT
    if (argc < 4)
    {
        printf("Usage: XenosRecomp [input path] [output path] [shader common header file path]\n"
               "       XenosRecomp --coverage [input path] [report csv path] [shader common header file path]");
        return 0;
    }
#endif

    const char* input =
#ifdef XENOS_RECOMP_INPUT 
        XENOS_RECOMP_INPUT
#else
        argv[1]
#endif
    ;

    const char* output =
#ifdef XENOS_RECOMP_OUTPUT 
        XENOS_RECOMP_OUTPUT
#else
        argv[2]
#endif
        ;
    
    const char* includeInput =
#ifdef XENOS_RECOMP_INCLUDE_INPUT
        XENOS_RECOMP_INCLUDE_INPUT
#else
        argv[3]
#endif
        ;

    size_t includeSize = 0;
    auto includeData = readAllBytes(includeInput, includeSize);
    std::string_view include(reinterpret_cast<const char*>(includeData.get()), includeSize);

    if (std::filesystem::is_directory(input))
    {
        std::vector<std::unique_ptr<uint8_t[]>> files;
        std::map<XXH64_hash_t, RecompiledShader> shaders;

        for (auto& file : std::filesystem::recursive_directory_iterator(input))
        {
            if (std::filesystem::is_directory(file))
            {
                continue;
            }
            
            size_t fileSize = 0;
            auto fileData = readAllBytes(file.path().string().c_str(), fileSize);
            bool foundAny = false;

            for (size_t i = 0; fileSize > sizeof(ShaderContainer) && i < fileSize - sizeof(ShaderContainer) - 1;)
            {
                auto shaderContainer = reinterpret_cast<const ShaderContainer*>(fileData.get() + i);
                size_t dataSize = shaderContainer->virtualSize + shaderContainer->physicalSize;

                if ((shaderContainer->flags & 0xFFFFFF00) == 0x102A1100 &&
                    dataSize <= (fileSize - i) &&
                    shaderContainer->field1C == 0 &&
                    shaderContainer->field20 == 0)
                {
                    XXH64_hash_t hash = XXH3_64bits(shaderContainer, dataSize);
                    auto shader = shaders.try_emplace(hash);
                    if (shader.second)
                    {
                        shader.first->second.data = fileData.get() + i;
                        foundAny = true;
                    }

                    i += dataSize;
                }
                else
                {
                    i += sizeof(uint32_t);
                }
            }

            if (foundAny)
                files.emplace_back(std::move(fileData));
        }

        std::atomic<uint32_t> progress = 0;

        std::for_each(std::execution::par_unseq, shaders.begin(), shaders.end(), [&](auto& hashShaderPair)
            {
                auto& shader = hashShaderPair.second;

                thread_local ShaderRecompiler recompiler;
                recompiler = {};
                recompiler.recompile(shader.data, include);

                shader.specConstantsMask = recompiler.specConstantsMask;

                thread_local DxcCompiler dxcCompiler;

#ifdef XENOS_RECOMP_DXIL
                shader.dxil = dxcCompiler.compile(recompiler.out, recompiler.isPixelShader, recompiler.specConstantsMask != 0, false);
                assert(shader.dxil != nullptr);
                assert(*(reinterpret_cast<uint32_t *>(shader.dxil->GetBufferPointer()) + 1) != 0 && "DXIL was not signed properly!");
#endif

                IDxcBlob* spirv = dxcCompiler.compile(recompiler.out, recompiler.isPixelShader, false, true);
                assert(spirv != nullptr);

                bool result = smolv::Encode(spirv->GetBufferPointer(), spirv->GetBufferSize(), shader.spirv, smolv::kEncodeFlagStripDebugInfo);
                assert(result);

                spirv->Release();

                size_t currentProgress = ++progress;
                if ((currentProgress % 10) == 0 || (currentProgress == shaders.size() - 1))
                    fmt::println("Recompiling shaders... {}%", currentProgress / float(shaders.size()) * 100.0f);
            });

        fmt::println("Creating shader cache...");

        StringBuffer f;
        f.println("#include \"shader_cache.h\"");
        f.println("ShaderCacheEntry g_shaderCacheEntries[] = {{");

        std::vector<uint8_t> dxil;
        std::vector<uint8_t> spirv;

        for (auto& [hash, shader] : shaders)
        {
            f.println("\t{{ 0x{:X}, {}, {}, {}, {}, {} }},",
                hash, dxil.size(), (shader.dxil != nullptr) ? shader.dxil->GetBufferSize() : 0, spirv.size(), shader.spirv.size(), shader.specConstantsMask);

            if (shader.dxil != nullptr)
            {
                dxil.insert(dxil.end(), reinterpret_cast<uint8_t *>(shader.dxil->GetBufferPointer()),
                    reinterpret_cast<uint8_t *>(shader.dxil->GetBufferPointer()) + shader.dxil->GetBufferSize());
            }
            
            spirv.insert(spirv.end(), shader.spirv.begin(), shader.spirv.end());
        }

        f.println("}};");

        fmt::println("Compressing DXIL cache...");

        int level = ZSTD_maxCLevel();

#ifdef XENOS_RECOMP_DXIL
        std::vector<uint8_t> dxilCompressed(ZSTD_compressBound(dxil.size()));
        dxilCompressed.resize(ZSTD_compress(dxilCompressed.data(), dxilCompressed.size(), dxil.data(), dxil.size(), level));

        f.print("const uint8_t g_compressedDxilCache[] = {{");

        for (auto data : dxilCompressed)
            f.print("{},", data);

        f.println("}};");
        f.println("const size_t g_dxilCacheCompressedSize = {};", dxilCompressed.size());
        f.println("const size_t g_dxilCacheDecompressedSize = {};", dxil.size());
#endif

        fmt::println("Compressing SPIRV cache...");

        std::vector<uint8_t> spirvCompressed(ZSTD_compressBound(spirv.size()));
        spirvCompressed.resize(ZSTD_compress(spirvCompressed.data(), spirvCompressed.size(), spirv.data(), spirv.size(), level));

        f.print("const uint8_t g_compressedSpirvCache[] = {{");

        for (auto data : spirvCompressed)
            f.print("{},", data);

        f.println("}};");

        f.println("const size_t g_spirvCacheCompressedSize = {};", spirvCompressed.size());
        f.println("const size_t g_spirvCacheDecompressedSize = {};", spirv.size());
        f.println("const size_t g_shaderCacheEntryCount = {};", shaders.size());

        writeAllBytes(output, f.out.data(), f.out.size());
    }
    else
    {
        ShaderRecompiler recompiler;
        size_t fileSize;
        recompiler.recompile(readAllBytes(input, fileSize).get(), include);
        writeAllBytes(output, recompiler.out.data(), recompiler.out.size());
    }

    return 0;
}
