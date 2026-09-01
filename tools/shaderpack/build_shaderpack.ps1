# Builds shaderpack/ from your own game files: recompiles the Avatar
# Editor's shaders (extracted from the xex image) to DXIL + SPIR-V.
# Run AFTER the SDK, codegen, and title build steps (see BUILDING.md).
$ErrorActionPreference = 'Continue'
if (-not (Get-Command cmake -ErrorAction SilentlyContinue) -or
    -not (Get-Command ninja -ErrorAction SilentlyContinue) -or
    -not (Get-Command clang -ErrorAction SilentlyContinue)) {
    # Any 2022 edition will do, as long as it has the C++ workload.
    $vs = Get-ChildItem 'C:\Program Files\Microsoft Visual Studio\2022' -Directory -ErrorAction SilentlyContinue |
          Where-Object { Test-Path "$($_.FullName)\VC\Tools\Llvm\x64\bin\clang.exe" } |
          Select-Object -First 1 -ExpandProperty FullName
    if (-not $vs) { 'Visual Studio 2022 with the C++ workload not found; run this from a Developer PowerShell'; exit 1 }
    $env:PATH = "$vs\VC\Tools\Llvm\x64\bin;$vs\Common7\IDE\CommonExtensions\Microsoft\CMake\CMake\bin;$vs\Common7\IDE\CommonExtensions\Microsoft\CMake\Ninja;$env:PATH"
}
$root = (Resolve-Path "$PSScriptRoot\..\..").Path
$title = "$root\out\build\win-amd64-relwithdebinfo\avatareditor.exe"
$xrDir = "$root\tools\XenosRecomp"
$xrBuild = "$root\out\build\xenosrecomp"
$xr = "$xrBuild\XenosRecomp\XenosRecomp.exe"
$image = "$root\out\ae_image.bin"
$pack = "$root\shaderpack"

if (-not (Test-Path $title)) { "avatareditor.exe missing - build the title first"; exit 1 }

# The decrypted image only exists once the runtime has mapped it; the title
# writes it out and quits before launching any guest code.
"=== image dump (avatareditor --dump_image_path)"
Remove-Item $image -ErrorAction SilentlyContinue
$p = Start-Process -FilePath $title -ArgumentList "--dump_image_path=$image" -PassThru -WindowStyle Minimized
if (-not $p.WaitForExit(60000)) { Stop-Process -Id $p.Id -Force; "image dump timed out"; exit 1 }
if (-not (Test-Path $image)) { "image dump failed"; exit 1 }

if (-not (Test-Path $xr)) {
    "=== XenosRecomp build"
    cmake -S $xrDir -B $xrBuild -G Ninja -DCMAKE_BUILD_TYPE=Release | Select-Object -Last 2
    if ($LASTEXITCODE -ne 0) { "XenosRecomp configure failed"; exit 1 }
    cmake --build $xrBuild | Select-Object -Last 2
    if ($LASTEXITCODE -ne 0) { "XenosRecomp build failed"; exit 1 }
}

"=== shader pack"
if (Test-Path $pack) { Remove-Item -Recurse -Force $pack }
& $xr --rexglue-pack-deltas $image "$root\tools\shaderpack\ae_shader_deltas.csv" $pack "$xrDir\XenosRecomp\rexglue_shader_common.h"
if ($LASTEXITCODE -ne 0) { "pack generation failed"; exit 1 }

py -3 "$root\tools\shaderpack\apply_face_fix.py" $pack $xr
if ($LASTEXITCODE -ne 0) { "face fix failed"; exit 1 }

Remove-Item $image -ErrorAction SilentlyContinue
$n = (Get-ChildItem $pack -Filter *.dxil).Count
"shaderpack ready: $n DXIL shaders"
