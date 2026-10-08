#pragma once

#include <string>
#include <vector>

struct DxcCompiler
{
    IDxcCompiler3* dxcCompiler = nullptr;
    std::string lastError;

    DxcCompiler();
    ~DxcCompiler();

    // targetOverride (e.g. L"-T gs_6_0") wins over the vs/ps/lib selection.
    IDxcBlob* compile(const std::string& shaderSource, bool compilePixelShader, bool compileLibrary, bool compileSpirv,
        const wchar_t* targetOverride = nullptr);

    // The exact argument list compile() passes, so the pack ABI hash can cover it.
    static std::vector<const wchar_t*> arguments(bool compilePixelShader, bool compileLibrary, bool compileSpirv,
        const wchar_t* targetOverride);

    // "major.minor+commits.hash" of the loaded dxcompiler, part of the pack ABI hash.
    std::string version() const;
};
