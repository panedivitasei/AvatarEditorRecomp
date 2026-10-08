#include "dxc_compiler.h"

DxcCompiler::DxcCompiler()
{
    HRESULT hr = DxcCreateInstance(CLSID_DxcCompiler, IID_PPV_ARGS(&dxcCompiler));
    assert(SUCCEEDED(hr));
}

DxcCompiler::~DxcCompiler()
{
    if (dxcCompiler != nullptr)
        dxcCompiler->Release();
}

std::vector<const wchar_t*> DxcCompiler::arguments(bool compilePixelShader, bool compileLibrary, bool compileSpirv,
    const wchar_t* targetOverride)
{
    std::vector<const wchar_t*> args;

    const wchar_t* target = targetOverride;
    if (target == nullptr)
    {
        if (compileLibrary)
        {
            assert(!compileSpirv);
            target = L"-T lib_6_3";
        }
        else
        {
            target = compilePixelShader ? L"-T ps_6_0" : L"-T vs_6_0";
        }
    }

    args.push_back(target);
    args.push_back(L"-HV 2021");
    args.push_back(L"-all-resources-bound");
    // Guest shaders rely on inf/NaN semantics (clamped log/rcp feeding pow chains) that fast-math may elide.
    // DXC rejects -Gis together with -spirv, so SPIR-V relies on the backend's default float rules.
    if (!compileSpirv)
        args.push_back(L"-Gis");

    if (compileSpirv)
    {
        args.push_back(L"-spirv");
        args.push_back(L"-fvk-use-dx-layout");

        // Y inversion belongs in the vertex stage only, because a GS copies the already-inverted position as data.
        // Inverting again at the GS write would un-flip every GS-attached draw.
        if (!compilePixelShader && targetOverride == nullptr)
            args.push_back(L"-fvk-invert-y");
    }
    else
    {
        args.push_back(L"-Wno-ignored-attributes");
        args.push_back(L"-Qstrip_reflect");
    }

    args.push_back(L"-Qstrip_debug");

    // Keep the HLSL preprocessor in step with how this tool was built so shader_common.h picks the matching layout.
#ifdef REBLUE_RECOMP
    args.push_back(L"-DREBLUE_RECOMP");
#elif defined(UNLEASHED_RECOMP)
    args.push_back(L"-DUNLEASHED_RECOMP");
#endif

    return args;
}

std::string DxcCompiler::version() const
{
    std::string text = "unknown";
    IDxcVersionInfo* info = nullptr;
    if (dxcCompiler != nullptr && SUCCEEDED(dxcCompiler->QueryInterface(IID_PPV_ARGS(&info))))
    {
        UINT32 major = 0, minor = 0;
        info->GetVersion(&major, &minor);
        text = fmt::format("{}.{}", major, minor);

        IDxcVersionInfo2* info2 = nullptr;
        if (SUCCEEDED(info->QueryInterface(IID_PPV_ARGS(&info2))))
        {
            UINT32 commitCount = 0;
            char* commitHash = nullptr;
            if (SUCCEEDED(info2->GetCommitInfo(&commitCount, &commitHash)))
            {
                text += fmt::format("+{}.{}", commitCount, commitHash != nullptr ? commitHash : "");
                if (commitHash != nullptr)
                    CoTaskMemFree(commitHash);
            }
            info2->Release();
        }
        info->Release();
    }
    return text;
}

IDxcBlob* DxcCompiler::compile(const std::string& shaderSource, bool compilePixelShader, bool compileLibrary, bool compileSpirv,
    const wchar_t* targetOverride)
{
    DxcBuffer source{};
    source.Ptr = shaderSource.c_str();
    source.Size = shaderSource.size();

    std::vector<const wchar_t*> args = arguments(compilePixelShader, compileLibrary, compileSpirv, targetOverride);

    lastError.clear();

    IDxcResult* result = nullptr;
    HRESULT hr = dxcCompiler->Compile(&source, args.data(), uint32_t(args.size()), nullptr, IID_PPV_ARGS(&result));

    IDxcBlob* object = nullptr;
    if (SUCCEEDED(hr))
    {
        assert(result != nullptr);

        HRESULT status;
        hr = result->GetStatus(&status);
        assert(SUCCEEDED(hr));

        if (FAILED(status))
        {
            lastError = "DXC compile failed";

            if (result->HasOutput(DXC_OUT_ERRORS))
            {
                IDxcBlobUtf8* errors = nullptr;
                hr = result->GetOutput(DXC_OUT_ERRORS, IID_PPV_ARGS(&errors), nullptr);
                assert(SUCCEEDED(hr) && errors != nullptr);

                lastError = errors->GetStringPointer();
                errors->Release();
            }
        }
        else
        {
            hr = result->GetOutput(DXC_OUT_OBJECT, IID_PPV_ARGS(&object), nullptr);
            assert(SUCCEEDED(hr) && object != nullptr);
        }

        result->Release();
    }
    else
    {
        lastError = fmt::format("DxcCompiler::Compile HRESULT 0x{:08X}", static_cast<uint32_t>(hr));
        assert(result == nullptr);
    }

    return object;
}
