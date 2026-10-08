#include "shader.h"
#include "shader_recompiler.h"
#include "dxc_compiler.h"
#include "ucode_fingerprint.h"
#include "xenosrecomp.h"

#include <chrono>
#include <cstring>

#include <mutex>
#include <vector>

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
    std::string sourceName;
};

// Per-shader recompile failures, collected from the parallel loop and reported before exit.
struct ShaderFailure
{
    XXH64_hash_t hash;
    std::string reason;
};

static std::mutex g_failureMutex;
static std::vector<ShaderFailure> g_failures;

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

// Every container in a file or directory, keyed by XXH3 of its bytes (the map CSV's container_hash column).
// Containers sit at unaligned offsets in the CC2 precache, so the scan steps by one byte.
static void scanContainers(const char* path, std::vector<std::unique_ptr<uint8_t[]>>& files,
    std::map<uint64_t, std::pair<const uint8_t*, size_t>>& containers)
{
    auto scanFile = [&](const std::string& filePath)
        {
            size_t fileSize = 0;
            auto fileData = readAllBytes(filePath.c_str(), fileSize);
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
                        containers.try_emplace(XXH3_64bits(fileData.get() + i, dataSize),
                            std::make_pair(fileData.get() + i, dataSize));
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

    if (std::filesystem::is_directory(path))
    {
        for (auto& file : std::filesystem::recursive_directory_iterator(path))
        {
            if (!std::filesystem::is_directory(file))
                scanFile(file.path().string());
        }
    }
    else
    {
        scanFile(path);
    }
}

// Splits a CSV line into at most maxColumns fields.
static std::vector<std::string> splitCsv(std::string_view line, size_t maxColumns)
{
    std::vector<std::string> cols;
    size_t start = 0;
    while (cols.size() < maxColumns && start <= line.size())
    {
        size_t comma = line.find(',', start);
        if (comma == std::string_view::npos || cols.size() + 1 == maxColumns)
            comma = line.size();
        cols.emplace_back(line.substr(start, comma - start));
        start = comma + 1;
    }
    return cols;
}

static std::vector<std::vector<std::string>> readCsv(const char* path, size_t maxColumns)
{
    size_t size = 0;
    auto data = readAllBytes(path, size);
    std::string_view text(reinterpret_cast<const char*>(data.get()), size);
    std::vector<std::vector<std::string>> rows;
    size_t pos = 0;
    bool header = true;
    while (pos < text.size())
    {
        size_t eol = text.find('\n', pos);
        std::string_view line = text.substr(pos, (eol == std::string_view::npos ? text.size() : eol) - pos);
        pos = (eol == std::string_view::npos) ? text.size() : eol + 1;
        if (!line.empty() && line.back() == '\r')
            line.remove_suffix(1);
        if (header)
        {
            header = false;
            continue;
        }
        if (!line.empty())
            rows.push_back(splitCsv(line, maxColumns));
    }
    return rows;
}

static std::string bindingsText(const xenosrecomp::ReflectionView& view)
{
    std::string text;
    for (auto& b : view.bindings)
    {
        if (!text.empty())
            text += ';';
        text += fmt::format("{}:{}:{}:{}", b.kind ? 's' : 't', b.fetch_constant, b.dimension, b.is_signed);
    }
    return text;
}

static std::string floatBitmapText(const xenosrecomp::ReflectionView& view)
{
    // Four u64 printed high to low, the old manifest's float_bitmap column.
    std::string text;
    for (int i = 3; i >= 0; i--)
    {
        uint64_t v = uint64_t(view.header->float_bitmap[i * 2]) | (uint64_t(view.header->float_bitmap[i * 2 + 1]) << 32);
        text += fmt::format("{:016X}", v);
    }
    return text;
}

// --cc2-pack <corpus> <map.csv> <out.pack>: builds the X4 single-file pack keyed by (stage, fp2), one entry per
// runtime shader the map lists. A containerless row translates its rebuilt code the way the runtime fallback does.
static int runCc2Pack(const char* corpusPath, const char* mapPath, const char* outPath)
{
    using namespace xenosrecomp;
    const auto startTime = std::chrono::steady_clock::now();

    std::vector<std::unique_ptr<uint8_t[]>> files;
    std::map<uint64_t, std::pair<const uint8_t*, size_t>> containers;
    scanContainers(corpusPath, files, containers);
    fmt::println("[pack] {} containers indexed", containers.size());

    std::vector<std::string> errors;
    auto fail = [&](std::string message) { errors.push_back(std::move(message)); };

    struct Row
    {
        uint64_t runtimeHash = 0;
        bool isPixel = false;
        uint64_t containerHash = 0;
        std::span<const uint8_t> container;
        std::vector<uint32_t> code;   // runtime ucode, guest byte order
        bool patched = false;
        bool containerless = false;
        uint64_t fp2 = 0;
    };
    std::vector<Row> rows;

    for (auto& cols : readCsv(mapPath, 7))
    {
        if (cols.size() < 6)
        {
            fail("map row with fewer than 6 columns");
            continue;
        }
        Row row;
        row.runtimeHash = strtoull(cols[0].c_str(), nullptr, 16);
        row.isPixel = cols[1] == "ps";
        row.containerless = cols[2] == "containerless";
        row.containerHash = strtoull(cols[3].c_str(), nullptr, 16);
        auto it = cols[3].empty() ? containers.end() : containers.find(row.containerHash);
        if (it == containers.end())
        {
            fail(fmt::format("FAILED {}.{}: no container {} in the corpus", cols[0], cols[1], cols[3]));
            continue;
        }
        row.container = { it->second.first, it->second.second };
        auto c = reinterpret_cast<const ShaderContainer*>(row.container.data());
        auto sh = reinterpret_cast<const Shader*>(row.container.data() + c->shaderOffset);
        const uint8_t* codeBytes = row.container.data() + c->virtualSize + sh->physicalOffset;
        row.code.resize(sh->size / sizeof(uint32_t));
        memcpy(row.code.data(), codeBytes, row.code.size() * sizeof(uint32_t));

        // Patches are wordIndex:HEXWORD in guest order; they rebuild the runtime bytes the hash names.
        if (cols.size() >= 7 && !cols[6].empty())
        {
            const std::string& s = cols[6];
            size_t p = 0;
            while (p < s.size())
            {
                size_t colon = s.find(':', p);
                size_t semi = s.find(';', p);
                if (semi == std::string::npos)
                    semi = s.size();
                if (colon == std::string::npos || colon > semi)
                    break;
                uint32_t index = uint32_t(strtoul(s.substr(p, colon - p).c_str(), nullptr, 10));
                uint32_t value = uint32_t(strtoul(s.substr(colon + 1, semi - colon - 1).c_str(), nullptr, 16));
                if (index < row.code.size())
                    row.code[index] = value;
                p = semi + 1;
            }
            row.patched = true;
        }
        if (row.patched || row.isPixel || row.containerless)
        {
            uint64_t hash = XXH3_64bits(row.code.data(), row.code.size() * sizeof(uint32_t));
            if (hash != row.runtimeHash)
            {
                fail(fmt::format("FAILED {}.{}: rebuilt runtime ucode hashes to {:016X}", cols[0], cols[1], hash));
                continue;
            }
        }

        // fp2 covers the patched declaration fields, so a container serves one entry per declaration variant.
        row.fp2 = Fingerprint(row.code, true);
        if (row.containerless)
            row.container = {};
        rows.push_back(std::move(row));
    }

    // One full translation per (key, container), plus HLSL-only checks for every other input that shares the key.
    struct Job
    {
        uint8_t stage = 0;
        uint64_t fp2 = 0;
        uint64_t containerHash = 0;
        std::span<const uint8_t> container;
        std::vector<uint32_t> code;
        std::string label;
        bool full = false;
        bool containerless = false;
        size_t fullJob = SIZE_MAX;     // HLSL-only jobs compare against this one
        TranslateResult result;
    };
    std::vector<Job> jobs;
    std::map<std::tuple<uint8_t, uint64_t, uint64_t>, size_t> fullJobs;
    std::map<std::pair<uint8_t, uint64_t>, std::vector<size_t>> groups;   // key -> row indices

    for (size_t i = 0; i < rows.size(); i++)
    {
        const Row& row = rows[i];
        const uint8_t stage = row.isPixel ? kPackPS : kPackVS;
        groups[{ stage, row.fp2 }].push_back(i);
        // A containerless key has no container, so its translations never depend on the byte source.
        auto key = std::make_tuple(stage, row.fp2, row.containerless ? 0 : row.containerHash);
        auto label = fmt::format("{:016X}.{}", row.runtimeHash, row.isPixel ? "ps" : "vs");
        auto found = fullJobs.find(key);
        if (found == fullJobs.end())
        {
            Job job;
            job.stage = stage;
            job.fp2 = row.fp2;
            job.containerHash = row.containerHash;
            job.container = row.container;
            job.code = row.code;
            job.label = label;
            job.full = true;
            job.containerless = row.containerless;
            fullJobs.emplace(key, jobs.size());
            jobs.push_back(std::move(job));
        }
        else
        {
            Job job;
            job.stage = stage;
            job.fp2 = row.fp2;
            job.containerHash = row.containerHash;
            job.container = row.container;
            job.code = row.code;
            job.label = label;
            job.fullJob = found->second;
            jobs.push_back(std::move(job));
        }
    }
    // The container's own ucode must translate like the patched runtime ucode of the same key.
    for (auto& [key, index] : std::map(fullJobs))
    {
        const Job& full = jobs[index];
        if (full.containerless)
            continue;
        auto c = reinterpret_cast<const ShaderContainer*>(full.container.data());
        auto sh = reinterpret_cast<const Shader*>(full.container.data() + c->shaderOffset);
        const auto* words = reinterpret_cast<const uint32_t*>(full.container.data() + c->virtualSize + sh->physicalOffset);
        if (std::equal(full.code.begin(), full.code.end(), words))
            continue;
        // A container whose own declaration fields differ from the key's is another variant, not a cross-check.
        if (Fingerprint({ words, sh->size / sizeof(uint32_t) }, true) != full.fp2)
            continue;
        Job job;
        job.stage = full.stage;
        job.fp2 = full.fp2;
        job.containerHash = full.containerHash;
        job.container = full.container;
        job.code.assign(words, words + sh->size / sizeof(uint32_t));
        job.label = fmt::format("container {:016X}", full.containerHash);
        job.fullJob = index;
        jobs.push_back(std::move(job));
    }

    size_t fullCount = 0;
    for (auto& job : jobs)
        fullCount += job.full;
    fmt::println("[pack] {} runtime rows, {} keys, {} full translations, {} cross-check translations",
        rows.size(), groups.size(), fullCount, jobs.size() - fullCount);

    std::atomic<uint32_t> progress = 0;
    std::for_each(std::execution::par_unseq, jobs.begin(), jobs.end(), [&](Job& job)
        {
            TranslateInput in;
            in.stage = job.stage == kPackPS ? Stage::kPixel : Stage::kVertex;
            in.ucode = job.code;
            in.container = job.container;
            in.targets = job.full ? (kDxil | kSpirv | kTrim) : kTrim;
            in.keep_hlsl = true;
            job.result = Translate(in);
            uint32_t done = ++progress;
            if ((done % 200) == 0)
                fmt::println("[pack] translated {}/{}", done, jobs.size());
        });

    size_t crossChecks = 0;
    for (auto& job : jobs)
    {
        if (!job.result.ok)
        {
            fail(fmt::format("FAILED {}: {}", job.label, job.result.error.substr(0, job.result.error.find('\n'))));
            continue;
        }
        if (job.result.fp2 != job.fp2)
            fail(fmt::format("FAILED {}: translator fp2 {:016X} differs from the key {:016X}", job.label, job.result.fp2, job.fp2));
        if (!job.full)
        {
            const Job& full = jobs[job.fullJob];
            if (full.result.ok && (job.result.hlsl != full.result.hlsl || job.result.reflection != full.result.reflection))
            {
                fail(fmt::format("FAILED {}: same key and container as {} but a different translation", job.label, full.label));
                // Both texts go next to the pack for diffing.
                const auto dir = std::filesystem::path(outPath).parent_path() / "mismatch";
                std::filesystem::create_directories(dir);
                const std::string stem = fmt::format("{:016X}", job.fp2);
                writeAllBytes((dir / (stem + ".a.hlsl")).string().c_str(), full.result.hlsl.data(), full.result.hlsl.size());
                writeAllBytes((dir / (stem + ".b.hlsl")).string().c_str(), job.result.hlsl.data(), job.result.hlsl.size());
                writeAllBytes((dir / (stem + ".a.refl")).string().c_str(), full.result.reflection.data(), full.result.reflection.size());
                writeAllBytes((dir / (stem + ".b.refl")).string().c_str(), job.result.reflection.data(), job.result.reflection.size());
            }
            crossChecks++;
        }
    }

    // Layout records as the renderer will read them from each runtime VS, checked against the reflection.
    size_t recordCount = 0, unhandledRecords = 0, rowsWithUnhandled = 0;
    std::map<uint32_t, uint32_t> recordFormats;
    for (auto& row : rows)
    {
        if (row.isPixel)
            continue;
        const Job& full = jobs[fullJobs.at(std::make_tuple(uint8_t(kPackVS), row.fp2, row.containerless ? 0 : row.containerHash))];
        ReflectionView view;
        if (!full.result.ok || !ParseReflection(full.result.reflection, view))
            continue;
        XeVfetchRecord records[kMaxVfetch * 2];
        const uint32_t count = ReadVfetchRecords(row.code, true, records);
        if (count != view.header->vfetch_count)
        {
            fail(fmt::format("FAILED {:016X}.vs: {} layout records for {} translated vfetches", row.runtimeHash, count,
                view.header->vfetch_count));
            continue;
        }
        bool unhandled = false;
        for (uint32_t i = 0; i < count; i++)
        {
            const uint32_t format = (records[i].word0 >> 8) & 0x3F;
            recordFormats[format]++;
            static constexpr uint32_t kHandled[] = { 6, 7, 16, 17, 25, 26, 31, 32, 33, 34, 35, 36, 37, 38, 57 };
            if (std::find(std::begin(kHandled), std::end(kHandled), format) == std::end(kHandled))
            {
                unhandledRecords++;
                unhandled = true;
            }
        }
        recordCount += count;
        rowsWithUnhandled += unhandled;
    }

    // Keys spanning several containers must still agree on everything the renderer consumes.
    size_t multiContainerKeys = 0, benignCollisions = 0, literalCollisions = 0;
    // Literal values come from the shader object at draw time, so they may differ under one key.
    auto withoutLiteralValues = [](const std::vector<uint8_t>& reflection)
        {
            std::vector<uint8_t> copy = reflection;
            ReflectionView view;
            if (ParseReflection(copy, view))
            {
                for (auto& literal : view.literals)
                    memset(const_cast<uint32_t*>(literal.value), 0, sizeof(literal.value));
            }
            return copy;
        };
    std::vector<PackInput> inputs;
    for (auto& [key, rowIndices] : groups)
    {
        std::vector<size_t> keyJobs;
        for (auto& [jobKey, index] : fullJobs)
        {
            if (std::get<0>(jobKey) == key.first && std::get<1>(jobKey) == key.second)
                keyJobs.push_back(index);
        }
        const Job& first = jobs[keyJobs.front()];
        if (keyJobs.size() > 1)
        {
            multiContainerKeys++;
            for (size_t k = 1; k < keyJobs.size(); k++)
            {
                const Job& other = jobs[keyJobs[k]];
                if (!first.result.ok || !other.result.ok)
                    continue;
                const bool sameBlobs = first.result.dxil == other.result.dxil && first.result.spirv == other.result.spirv &&
                    first.result.dxil_trim == other.result.dxil_trim && first.result.spirv_trim == other.result.spirv_trim;
                const bool sameReflection = first.result.reflection == other.result.reflection;
                if (sameBlobs && !sameReflection &&
                    withoutLiteralValues(first.result.reflection) == withoutLiteralValues(other.result.reflection))
                {
                    literalCollisions++;
                    continue;
                }
                if (first.result.hlsl == other.result.hlsl && sameReflection)
                    continue;
                if (sameBlobs && sameReflection)
                {
                    benignCollisions++;
                    continue;
                }
                fail(fmt::format("FAILED fp2 collision {:016X}: containers {:016X} and {:016X} translate differently",
                    key.second, first.containerHash, other.containerHash));
            }
        }
        if (!first.result.ok)
            continue;

        PackInput input;
        input.stage = key.first;
        input.fp2 = key.second;
        ReflectionView view;
        ParseReflection(first.result.reflection, view);
        if (view.header->flags & kReflHasTrim)
            input.flags |= kEntryHasTrim;
        input.reflection = first.result.reflection;
        input.dxil = first.result.dxil;
        input.spirv = first.result.spirv;
        input.dxil_trim = first.result.dxil_trim;
        input.spirv_trim = first.result.spirv_trim;
        for (size_t r : rowIndices)
            input.aliases.push_back(rows[r].runtimeHash);
        inputs.push_back(std::move(input));
    }

    // Built-ins compile with the pack's DXC setup; any failure fails the build.
    for (auto& builtin : Builtins())
    {
        PackInput input;
        input.stage = builtin.stage;
        input.fp2 = BuiltinKey(builtin.name);
        XeReflHeader h{};
        h.stage = builtin.stage;
        h.pixel_pos_reg = 0xFF;
        input.reflection.assign(reinterpret_cast<const uint8_t*>(&h), reinterpret_cast<const uint8_t*>(&h) + sizeof(h));
        const bool isGs = builtin.stage == kPackGS;
        const bool isPixel = builtin.stage == kPackBlitPS;
        DxcCompiler dxc;
        for (bool spirv : { false, true })
        {
            IDxcBlob* blob = dxc.compile(builtin.source, isPixel, false, spirv, isGs ? L"-T gs_6_0" : nullptr);
            if (blob == nullptr)
            {
                fail(fmt::format("FAILED builtin {} ({}): {}", builtin.name, spirv ? "SPIR-V" : "DXIL",
                    dxc.lastError.substr(0, dxc.lastError.find('\n'))));
                continue;
            }
            auto* p = reinterpret_cast<const uint8_t*>(blob->GetBufferPointer());
            (spirv ? input.spirv : input.dxil).assign(p, p + blob->GetBufferSize());
            blob->Release();
        }
        inputs.push_back(std::move(input));
    }

    std::error_code ec;
    if (!errors.empty())
    {
        for (auto& e : errors)
            fmt::println("{}", e);
        fmt::println("[pack] {} error(s), no pack written", errors.size());
        std::filesystem::remove(outPath, ec);
        return 1;
    }

    size_t vsCount = 0, psCount = 0, trimCount = 0;
    size_t dxilBytes[2] = {}, spirvBytes[2] = {};
    for (auto& input : inputs)
    {
        vsCount += input.stage == kPackVS;
        psCount += input.stage == kPackPS;
        trimCount += !input.dxil_trim.empty();
        const int ps = input.stage == kPackPS ? 1 : 0;
        dxilBytes[ps] += input.dxil.size() + input.dxil_trim.size();
        spirvBytes[ps] += input.spirv.size() + input.spirv_trim.size();
    }

    // Debug listing next to the pack; the runtime never reads it.
    StringBuffer manifest;
    manifest.println("runtime_ucode_hash,type,fp2,container_hash,bindings,float_bitmap,flags");
    for (auto& row : rows)
    {
        const uint8_t stage = row.isPixel ? kPackPS : kPackVS;
        const Job& full = jobs[fullJobs.at(std::make_tuple(stage, row.fp2, row.containerless ? 0 : row.containerHash))];
        ReflectionView view;
        ParseReflection(full.result.reflection, view);
        manifest.println("{:016X},{},{:016X},{:016X},{},{},{:04X}", row.runtimeHash, row.isPixel ? "ps" : "vs", row.fp2,
            row.containerHash, bindingsText(view), floatBitmapText(view), view.header->flags);
    }

    const size_t inputCount = inputs.size();
    std::string error;
    std::vector<uint8_t> pack = BuildPack(std::move(inputs), 19, &error);
    if (pack.empty())
    {
        fmt::println("[pack] write failed: {}", error);
        std::filesystem::remove(outPath, ec);
        return 1;
    }

    const std::filesystem::path out(outPath);
    std::filesystem::create_directories(out.parent_path(), ec);
    const std::filesystem::path temp = out.string() + ".tmp";
    writeAllBytes(temp.string().c_str(), pack.data(), pack.size());
    std::filesystem::rename(temp, out, ec);
    if (ec)
    {
        fmt::println("[pack] cannot rename {} into place", temp.string());
        return 1;
    }
    writeAllBytes((out.parent_path() / "manifest.csv").string().c_str(), manifest.out.data(), manifest.out.size());

    const double seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - startTime).count();
    fmt::println("");
    fmt::println("=== cc2 pack ===");
    fmt::println("runtime rows:        {}", rows.size());
    fmt::println("entries:             {} ({} VS, {} PS, {} built-in), {} VS with trim", inputCount, vsCount, psCount,
        inputCount - vsCount - psCount, trimCount);
    fmt::println("keys over >1 container: {} ({} differ only in HLSL text, {} only in def literal values, 0 conflicting)",
        multiContainerKeys, benignCollisions, literalCollisions);
    fmt::println("cross-checked inputs: {} (identical translation per key)", crossChecks);
    std::string formats;
    for (auto& [format, count] : recordFormats)
        formats += fmt::format(" {}:{}", format, count);
    fmt::println("runtime layout records: {}, unhandled format {} in {} VS; formats{}", recordCount, unhandledRecords,
        rowsWithUnhandled, formats);
    fmt::println("raw DXIL VS {:.1f} MB PS {:.1f} MB, raw SPIR-V VS {:.1f} MB PS {:.1f} MB", dxilBytes[0] / 1048576.0,
        dxilBytes[1] / 1048576.0, spirvBytes[0] / 1048576.0, spirvBytes[1] / 1048576.0);
    fmt::println("pack {} bytes ({:.2f} MB), abi {:016X}, {:.1f} s", pack.size(), pack.size() / 1048576.0, AbiHash(), seconds);
    fmt::println("pack: {}", out.string());
    return 0;
}

// --add-misses <corpus> <misses dir> <map.csv>: appends a map row for every runtime miss dump (<rt>.<vs|ps>.ucode,
// guest byte order), with the patches that rebuild its bytes from a corpus container. Code with a same-fp2 container
// gets an ordinary row; the rest becomes containerless, sourced from the same-size container with the fewest diffs.
static int runAddMisses(const char* corpusPath, const char* missesPath, const char* mapPath)
{
    using namespace xenosrecomp;
    std::vector<std::unique_ptr<uint8_t[]>> files;
    std::map<uint64_t, std::pair<const uint8_t*, size_t>> containers;
    scanContainers(corpusPath, files, containers);

    // (fp2, dword count) -> container hash; the first container in hash order wins, as fp2 keys never span two.
    std::map<std::pair<uint64_t, size_t>, uint64_t> byKey;
    std::multimap<size_t, uint64_t> bySize;
    for (auto& [hash, span] : containers)
    {
        auto c = reinterpret_cast<const ShaderContainer*>(span.first);
        auto sh = reinterpret_cast<const Shader*>(span.first + c->shaderOffset);
        const auto* words = reinterpret_cast<const uint32_t*>(span.first + c->virtualSize + sh->physicalOffset);
        const size_t count = sh->size / sizeof(uint32_t);
        byKey.try_emplace({ Fingerprint({ words, count }, true), count }, hash);
        bySize.emplace(count, hash);
    }
    auto codeOf = [&](uint64_t hash)
        {
            const auto& span = containers.at(hash);
            auto c = reinterpret_cast<const ShaderContainer*>(span.first);
            auto sh = reinterpret_cast<const Shader*>(span.first + c->shaderOffset);
            return reinterpret_cast<const uint32_t*>(span.first + c->virtualSize + sh->physicalOffset);
        };

    std::set<std::pair<uint64_t, bool>> known;
    for (auto& cols : readCsv(mapPath, 7))
    {
        if (cols.size() >= 2)
            known.insert({ strtoull(cols[0].c_str(), nullptr, 16), cols[1] == "ps" });
    }

    StringBuffer out;
    size_t added = 0, unmatched = 0;
    for (auto& file : std::filesystem::directory_iterator(missesPath))
    {
        const std::string name = file.path().filename().string();
        const bool isPixel = name.ends_with(".ps.ucode");
        if (!isPixel && !name.ends_with(".vs.ucode"))
            continue;
        size_t size = 0;
        auto data = readAllBytes(file.path().string().c_str(), size);
        std::vector<uint32_t> code(size / sizeof(uint32_t));
        memcpy(code.data(), data.get(), code.size() * sizeof(uint32_t));
        const uint64_t runtimeHash = XXH3_64bits(code.data(), code.size() * sizeof(uint32_t));
        if (known.contains({ runtimeHash, isPixel }))
            continue;
        const uint64_t fp2 = Fingerprint(code, true);
        auto it = byKey.find({ fp2, code.size() });
        uint64_t source = 0;
        bool containerless = false;
        if (it != byKey.end())
        {
            source = it->second;
        }
        else
        {
            size_t best = SIZE_MAX;
            for (auto [s, e] = bySize.equal_range(code.size()); s != e; ++s)
            {
                const uint32_t* words = codeOf(s->second);
                size_t diffs = 0;
                for (size_t i = 0; i < code.size() && diffs < best; i++)
                    diffs += words[i] != code[i];
                if (diffs < best)
                    best = diffs, source = s->second;
            }
            if (best == SIZE_MAX)
            {
                fmt::println("[misses] {}: no corpus container of {} dwords", name, code.size());
                unmatched++;
                continue;
            }
            containerless = true;
            fmt::println("[misses] {}: fp2 {:016X} has no container; containerless, {} of {} words patched from {:016X}",
                name, fp2, best, code.size(), source);
        }
        const uint32_t* words = codeOf(source);
        std::string patches;
        for (size_t i = 0; i < code.size(); i++)
        {
            // Patch values are the words as a host load reads guest memory, like the deltas mode writes them.
            const uint32_t word = code[i];
            if (word != words[i])
                patches += fmt::format("{}{}:{:08X}", patches.empty() ? "" : ";", i, word);
        }
        out.println("{:016X},{},{},{:016X},{:016X},1,{}", runtimeHash, isPixel ? "ps" : "vs",
            containerless ? "containerless" : patches.empty() ? "exact" : "fingerprint", source,
            ucodeFingerprint(code.data(), code.size(), true), patches);
        known.insert({ runtimeHash, isPixel });
        added++;
    }

    if (added != 0)
    {
        size_t mapSize = 0;
        auto mapData = readAllBytes(mapPath, mapSize);
        std::string text(reinterpret_cast<const char*>(mapData.get()), mapSize);
        if (!text.empty() && text.back() != '\n')
            text += '\n';
        text.append(out.out.data(), out.out.size());
        writeAllBytes(mapPath, text.data(), text.size());
    }
    fmt::println("[misses] {} rows added to {}, {} without a corpus container", added, mapPath, unmatched);
    return unmatched == 0 ? 0 : 1;
}

// --translate <corpus> <container hash> <out prefix> [nocontainer]: one shader through the library, every output
// on disk, then a disk cache round trip. nocontainer takes the container-less path the fallback uses for unknown code.
static int runTranslateOne(const char* corpusPath, const char* containerHash, const char* outPrefix, bool withContainer)
{
    using namespace xenosrecomp;
    std::vector<std::unique_ptr<uint8_t[]>> files;
    std::map<uint64_t, std::pair<const uint8_t*, size_t>> containers;
    scanContainers(corpusPath, files, containers);
    auto it = containers.find(strtoull(containerHash, nullptr, 16));
    if (it == containers.end())
    {
        fmt::println("container {} not found", containerHash);
        return 1;
    }
    std::span<const uint8_t> container(it->second.first, it->second.second);
    auto c = reinterpret_cast<const ShaderContainer*>(container.data());
    auto sh = reinterpret_cast<const Shader*>(container.data() + c->shaderOffset);
    TranslateInput in;
    in.stage = (c->flags & 0x1) == 0 ? Stage::kPixel : Stage::kVertex;
    in.ucode = { reinterpret_cast<const uint32_t*>(container.data() + c->virtualSize + sh->physicalOffset), sh->size / 4u };
    if (withContainer)
        in.container = container;
    in.keep_hlsl = true;
    TranslateResult r = Translate(in);
    if (!r.ok)
    {
        fmt::println("FAILED: {}", r.error);
        return 1;
    }
    const std::string prefix = outPrefix;
    const char* type = in.stage == Stage::kPixel ? "ps" : "vs";
    writeAllBytes((prefix + "." + type + ".hlsl").c_str(), r.hlsl.data(), r.hlsl.size());
    writeAllBytes((prefix + "." + type + ".dxil").c_str(), r.dxil.data(), r.dxil.size());
    writeAllBytes((prefix + "." + type + ".spirv").c_str(), r.spirv.data(), r.spirv.size());
    writeAllBytes((prefix + "." + type + ".refl").c_str(), r.reflection.data(), r.reflection.size());
    if (!r.hlsl_trim.empty())
    {
        writeAllBytes((prefix + ".vst.hlsl").c_str(), r.hlsl_trim.data(), r.hlsl_trim.size());
        writeAllBytes((prefix + ".vst.dxil").c_str(), r.dxil_trim.data(), r.dxil_trim.size());
        writeAllBytes((prefix + ".vst.spirv").c_str(), r.spirv_trim.data(), r.spirv_trim.size());
    }
    fmt::println("fp2 {:016X}, dxil {} B, spirv {} B, trim {} / {} B", r.fp2, r.dxil.size(), r.spirv.size(),
        r.dxil_trim.size(), r.spirv_trim.size());

    // Disk cache round trip through the same loader the pack uses.
    const std::filesystem::path cacheDir = CacheDirectory(prefix + "_cache");
    std::string error;
    Pack cached;
    if (!CacheStore(cacheDir, in.stage, r, &error) || !CacheLoad(cacheDir, in.stage, r.fp2, cached, &error))
    {
        fmt::println("cache round trip failed: {}", error);
        return 1;
    }
    const PackEntry* entry = cached.Find(uint8_t(in.stage), r.fp2);
    std::vector<uint8_t> dxil, spirv, dxilTrim, spirvTrim;
    const bool same = entry != nullptr && cached.Decompress(*entry, BlobKind::kDxil, dxil, &error) &&
        cached.Decompress(*entry, BlobKind::kSpirv, spirv, &error) &&
        cached.Decompress(*entry, BlobKind::kDxilTrim, dxilTrim, &error) &&
        cached.Decompress(*entry, BlobKind::kSpirvTrim, spirvTrim, &error) &&
        dxil == r.dxil && spirv == r.spirv && dxilTrim == r.dxil_trim && spirvTrim == r.spirv_trim &&
        std::ranges::equal(cached.Reflection(*entry), r.reflection) && cached.FindAlias(r.runtime_ucode_hash) == entry;
    fmt::println("cache round trip {}: {}", same ? "ok" : "MISMATCH", CacheFile(cacheDir, in.stage, r.fp2).string());
    return same ? 0 : 1;
}

static bool validDxil(const std::vector<uint8_t>& blob)
{
    if (blob.size() < 32 || memcmp(blob.data(), "DXBC", 4) != 0)
        return false;
    uint32_t size;
    memcpy(&size, blob.data() + 24, sizeof(size));
    bool signedDigest = false;
    for (size_t i = 4; i < 20; i++)
        signedDigest |= blob[i] != 0;
    return size == blob.size() && signedDigest;
}

static bool validSpirv(const std::vector<uint8_t>& blob)
{
    if (blob.size() < 20 || (blob.size() & 3) != 0)
        return false;
    uint32_t magic, version;
    memcpy(&magic, blob.data(), 4);
    memcpy(&version, blob.data() + 4, 4);
    return magic == 0x07230203 && (version >> 16) == 1;
}

// --pack-check <pack> <old manifest.csv> [N]: loads the pack as the renderer would, resolves N manifest rows
// (0 = all) through the alias table and the key index, decompresses every blob and validates its header.
static int runPackCheck(const char* packPath, const char* manifestPath, size_t limit)
{
    using namespace xenosrecomp;
    Pack pack;
    std::string error;
    if (!pack.OpenFile(packPath, &error))
    {
        fmt::println("pack refused: {}", error);
        return 1;
    }
    const PackHeader& h = pack.header();
    fmt::println("pack v{} fp{} abi {:016X}: {} entries ({} built-in), {} aliases, refl {} B, blobs {} B",
        h.version, h.fingerprint_ver, h.abi_hash, h.entry_count, h.builtin_count, h.alias_count, h.refl_size, h.blob_size);

    size_t looked = 0, found = 0, missing = 0, bad = 0, blobsChecked = 0, bindingDiffs = 0, bitmapDiffs = 0;
    auto rows = readCsv(manifestPath, 7);
    for (auto& cols : rows)
    {
        if (limit != 0 && looked >= limit)
            break;
        if (cols.size() < 2)
            continue;
        looked++;
        const uint64_t hash = strtoull(cols[0].c_str(), nullptr, 16);
        const uint8_t stage = cols[1] == "ps" ? kPackPS : kPackVS;
        const PackEntry* entry = pack.FindAlias(hash);
        if (entry == nullptr)
        {
            missing++;
            continue;
        }
        found++;
        bool ok = entry->stage == stage && pack.Find(entry->stage, entry->fp2) == entry;

        ReflectionView view;
        ok &= ParseReflection(pack.Reflection(*entry), view) && view.header->stage == stage;
        if (ok && cols.size() >= 6)
        {
            // Old manifest columns: runtime_ucode_hash,type,fingerprint,dxil,bindings,float_bitmap,reason.
            bindingDiffs += bindingsText(view) != cols[4];
            bitmapDiffs += floatBitmapText(view) != cols[5];
        }

        std::vector<uint8_t> blob;
        for (BlobKind kind : { BlobKind::kDxil, BlobKind::kSpirv, BlobKind::kDxilTrim, BlobKind::kSpirvTrim })
        {
            if (!pack.Decompress(*entry, kind, blob, &error))
            {
                ok = false;
                continue;
            }
            const bool trim = kind == BlobKind::kDxilTrim || kind == BlobKind::kSpirvTrim;
            if (blob.empty())
            {
                ok &= trim && (view.header == nullptr || !(view.header->flags & kReflHasTrim));
                continue;
            }
            blobsChecked++;
            ok &= (kind == BlobKind::kDxil || kind == BlobKind::kDxilTrim) ? validDxil(blob) : validSpirv(blob);
        }
        if (!ok)
        {
            bad++;
            fmt::println("BAD {}.{}", cols[0], cols[1]);
        }
    }

    size_t builtinsOk = 0;
    for (auto& builtin : Builtins())
    {
        const PackEntry* entry = pack.FindBuiltin(builtin.stage, builtin.name);
        std::vector<uint8_t> dxil, spirv;
        if (entry != nullptr && pack.Decompress(*entry, BlobKind::kDxil, dxil, &error) &&
            pack.Decompress(*entry, BlobKind::kSpirv, spirv, &error) && validDxil(dxil) && validSpirv(spirv))
            builtinsOk++;
        else
            bad++;
    }

    fmt::println("manifest rows looked up: {}, found {}, not in pack set {}, bad {}", looked, found, missing, bad);
    fmt::println("blobs decompressed and validated: {}, built-ins ok {}/{}", blobsChecked, builtinsOk, Builtins().size());
    fmt::println("reflection vs old manifest: {} binding lists differ, {} float bitmaps differ", bindingDiffs, bitmapDiffs);

    size_t literalShaders = 0, literals = 0, literalsRead = 0, dynamicFloats = 0;
    for (auto& entry : pack.entries())
    {
        ReflectionView view;
        if (entry.stage > kPackPS || !ParseReflection(pack.Reflection(entry), view))
            continue;
        literalShaders += !view.literals.empty();
        literals += view.literals.size();
        for (auto& l : view.literals)
            literalsRead += (l.flags & kLiteralRead) != 0;
        dynamicFloats += view.header->float_mode;
    }
    fmt::println("def literals: {} in {} shaders, {} read by the shader; {} shaders with a dynamic float file",
        literals, literalShaders, literalsRead, dynamicFloats);
    return bad == 0 && found != 0 ? 0 : 1;
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

    // Fingerprints raw runtime ucode dumps (LE words) into "name,runtime_ucode_hash,fingerprint,fp2" rows.
    // fp2 is the pack key, fingerprint the v1 hash of the coverage report.
    if (argc >= 4 && strcmp(argv[1], "--fingerprint") == 0)
    {
        StringBuffer csv;
        csv.println("name,runtime_ucode_hash,fingerprint,fp2");

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

            uint64_t fp2 = ucodeFingerprint2(
                reinterpret_cast<const uint32_t*>(fileData.get()), fileSize / sizeof(uint32_t), false);

            csv.println("{},{:016X},{:016X},{:016X}", name, hash, fingerprint, fp2);
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

    if (argc >= 5 && strcmp(argv[1], "--cc2-pack") == 0)
    {
        try
        {
            return runCc2Pack(argv[2], argv[3], argv[4]);
        }
        catch (const std::exception& e)
        {
            fprintf(stderr, "cc2-pack failed: %s\n", e.what());
            return 1;
        }
    }

    if (argc >= 5 && strcmp(argv[1], "--add-misses") == 0)
    {
        try
        {
            return runAddMisses(argv[2], argv[3], argv[4]);
        }
        catch (const std::exception& e)
        {
            fprintf(stderr, "add-misses failed: %s\n", e.what());
            return 1;
        }
    }

    if (argc >= 5 && strcmp(argv[1], "--translate") == 0)
    {
        try
        {
            return runTranslateOne(argv[2], argv[3], argv[4], !(argc >= 6 && strcmp(argv[5], "nocontainer") == 0));
        }
        catch (const std::exception& e)
        {
            fprintf(stderr, "translate failed: %s\n", e.what());
            return 1;
        }
    }

    if (argc >= 4 && strcmp(argv[1], "--pack-check") == 0)
    {
        try
        {
            return runPackCheck(argv[2], argv[3], argc >= 5 ? strtoull(argv[4], nullptr, 10) : 0);
        }
        catch (const std::exception& e)
        {
            fprintf(stderr, "pack-check failed: %s\n", e.what());
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
        printf("Usage: XenosRecomp [input path] [output path] [shader common header file path] [optional: HLSL dump dir]\n"
               "       XenosRecomp --coverage [input path] [report csv path] [shader common header file path]\n"
               "       XenosRecomp --cc2-pack [corpus] [map csv] [out .pack]\n"
               "       XenosRecomp --add-misses [corpus] [misses dir] [map csv]\n"
               "       XenosRecomp --pack-check [.pack] [manifest csv] [optional: row count]\n");
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

    std::string hlslDumpDir;
    if (argc > 4)
        hlslDumpDir = argv[4];
    else if (const char* env = std::getenv("XENOS_RECOMP_HLSL_DUMP"))
        hlslDumpDir = env;

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
            int containerIndex = 0;

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
                        std::string stem = file.path().stem().string();
                        if (containerIndex > 0)
                            stem += fmt::format(".{}", containerIndex);
                        shader.first->second.sourceName = std::move(stem);
                        foundAny = true;
                    }

                    ++containerIndex;
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

        if (!hlslDumpDir.empty())
            std::filesystem::create_directories(hlslDumpDir);

        std::atomic<uint32_t> progress = 0;

        std::for_each(std::execution::par_unseq, shaders.begin(), shaders.end(), [&](auto& hashShaderPair)
            {
                auto& shader = hashShaderPair.second;
                const XXH64_hash_t hash = hashShaderPair.first;

                auto recordFailure = [hash](std::string reason)
                {
                    std::lock_guard<std::mutex> lock(g_failureMutex);
                    g_failures.push_back({hash, std::move(reason)});
                };

                thread_local ShaderRecompiler recompiler;
                recompiler = {};
                try
                {
                    recompiler.recompile(shader.data, include);
                }
                catch (const std::exception& e)
                {
                    // Unsupported ops throw with their name; report them with the other failures.
                    recordFailure(e.what());
                    return;
                }

                shader.specConstantsMask = recompiler.specConstantsMask;

                if (!hlslDumpDir.empty())
                {
                    auto path = std::filesystem::path(hlslDumpDir) /
                        (shader.sourceName.empty()
                            ? fmt::format("{}_{:016X}", recompiler.isPixelShader ? "ps" : "vs", hash)
                            : shader.sourceName);
                    path += ".hlsl";

                    std::string contents = fmt::format(
                        "// {} shader  hash=0x{:016X}  specConstants=0x{:X}\n",
                        recompiler.isPixelShader ? "pixel" : "vertex",
                        hash, recompiler.specConstantsMask);
                    contents.append(recompiler.out);
                    writeAllBytes(path.string().c_str(), contents.data(), contents.size());
                }

                thread_local DxcCompiler dxcCompiler;

#ifdef XENOS_RECOMP_DXIL
                shader.dxil = dxcCompiler.compile(recompiler.out, recompiler.isPixelShader, recompiler.specConstantsMask != 0, false);
                if (shader.dxil == nullptr)
                {
                    recordFailure("dxc-dxil-compile-failed");
                    return;
                }
                if (*(reinterpret_cast<uint32_t*>(shader.dxil->GetBufferPointer()) + 1) == 0)
                {
                    recordFailure("dxil-not-signed");
                    return;
                }
#endif

                IDxcBlob* spirv = dxcCompiler.compile(recompiler.out, recompiler.isPixelShader, false, true);
                if (spirv == nullptr)
                {
                    recordFailure("dxc-spirv-compile-failed");
                    return;
                }

                if (!smolv::Encode(spirv->GetBufferPointer(), spirv->GetBufferSize(), shader.spirv, smolv::kEncodeFlagStripDebugInfo))
                {
                    spirv->Release();
                    recordFailure("smolv-encode-failed");
                    return;
                }

                spirv->Release();

                size_t currentProgress = ++progress;
                if ((currentProgress % 10) == 0 || (currentProgress == shaders.size() - 1))
                    fmt::println("Recompiling shaders... {}%", currentProgress / float(shaders.size()) * 100.0f);
            });

        if (!g_failures.empty())
        {
            fmt::println(stderr, "Recompile failures ({}):", g_failures.size());
            for (const auto& failure : g_failures)
                fmt::println(stderr, "  hash=0x{:016X} reason={}", failure.hash, failure.reason);
            return 2;
        }

        fmt::println("Creating shader cache...");

        StringBuffer f;
#ifdef REBLUE_RECOMP
        f.println("#include \"gpu/shaders/shader_cache.h\"");
#else
        f.println("#include \"shader_cache.h\"");
#endif
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
