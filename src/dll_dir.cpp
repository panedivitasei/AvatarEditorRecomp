// Delay-load hook: rexruntime.dll, dxcompiler.dll and dxil.dll live in bin\ under the exe, so the loader is handed
// them from there instead of the exe folder.

#if defined(_WIN32)

#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#include <delayimp.h>

#include <string>

namespace {

std::wstring BinDir() {
  wchar_t path[MAX_PATH];
  const DWORD n = GetModuleFileNameW(nullptr, path, MAX_PATH);
  std::wstring dir(path, n);
  const size_t cut = dir.find_last_of(L"\\/");
  return (cut == std::wstring::npos ? L"." : dir.substr(0, cut)) + L"\\bin\\";
}

HMODULE FromBin(const char* name) {
  std::wstring wide(name, name + strlen(name));
  const std::wstring bin = BinDir();
  // dxcompiler asks for dxil.dll by bare name later, which the loader satisfies from the module already loaded here.
  if (_stricmp(name, "dxcompiler.dll") == 0) LoadLibraryExW((bin + L"dxil.dll").c_str(), nullptr, LOAD_WITH_ALTERED_SEARCH_PATH);
  return LoadLibraryExW((bin + wide).c_str(), nullptr, LOAD_WITH_ALTERED_SEARCH_PATH);
}

FARPROC WINAPI DelayHook(unsigned reason, DelayLoadInfo* info) {
  if (reason != dliNotePreLoadLibrary || !info || !info->szDll) return nullptr;
  return reinterpret_cast<FARPROC>(FromBin(info->szDll));
}

}  // namespace

// A constant initializer, so the hook is in place before any static initializer touches rexruntime.
extern "C" const PfnDliHook __pfnDliNotifyHook2 = DelayHook;

#endif
