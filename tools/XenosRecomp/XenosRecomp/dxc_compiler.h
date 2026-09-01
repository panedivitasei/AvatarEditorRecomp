#pragma once

struct DxcCompiler
{
    IDxcCompiler3* dxcCompiler = nullptr;
    std::string lastError;

    DxcCompiler();
    ~DxcCompiler();

    // targetOverride (e.g. L"-T gs_6_0") wins over the vs/ps/lib selection.
    IDxcBlob* compile(const std::string& shaderSource, bool compilePixelShader, bool compileLibrary, bool compileSpirv,
        const wchar_t* targetOverride = nullptr);
};
