# Builds out/shaderpack/ae.pack from your own game files: the shaders embedded in the Avatar Editor's xex image,
# translated to DXIL (docs/research/pack_contract.md). Run after building the title (see BUILDING.md).
param(
    [string]$Misses = '',  # the runtime's gpu_shader_cache/<abi>/misses, appended to the deltas CSV first
    [string]$BuildDir = ''  # title build folder whose avatareditor.exe dumps the image
)
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
if (-not $BuildDir) { $BuildDir = "$root\out\build\win-amd64-release" }
$BuildDir = (Resolve-Path $BuildDir).Path
$title = "$BuildDir\avatareditor.exe"
$xrDir = "$root\tools\XenosRecomp"
$xrBuild = "$root\out\build\xenosrecomp"
$xr = "$xrBuild\XenosRecomp\XenosRecomp.exe"
$image = "$root\out\ae_image.bin"
$deltas = "$root\tools\shaderpack\ae_shader_deltas.csv"
$pack = "$root\out\shaderpack\ae.pack"

if (-not (Test-Path $title)) { "avatareditor.exe missing in $BuildDir - build the title first"; exit 1 }

# The decrypted image only exists once the runtime has mapped it; the title writes it out and quits before any
# guest code runs. The image is the whole corpus: the editor has no shader precache.
"=== image dump (avatareditor --dump_image_path)"
Remove-Item $image -ErrorAction SilentlyContinue
$p = Start-Process -FilePath $title -ArgumentList "--dump_image_path=$image" -WorkingDirectory $BuildDir -PassThru -WindowStyle Minimized
if (-not $p.WaitForExit(60000)) { Stop-Process -Id $p.Id -Force; "image dump timed out"; exit 1 }
if (-not (Test-Path $image)) { "image dump failed"; exit 1 }

"=== XenosRecomp build"
# The pack's ABI hash covers the translator sources, so the tool is rebuilt whenever they change.
if (-not (Test-Path "$xrBuild\build.ninja")) {
    cmake -S $xrDir -B $xrBuild -G Ninja -DCMAKE_BUILD_TYPE=Release -DCMAKE_C_COMPILER=clang -DCMAKE_CXX_COMPILER=clang++ | Select-Object -Last 2
    if ($LASTEXITCODE -ne 0) { "XenosRecomp configure failed"; exit 1 }
}
cmake --build $xrBuild | Select-Object -Last 2
if ($LASTEXITCODE -ne 0) { "XenosRecomp build failed"; exit 1 }

"=== shader pack"
if ($Misses) {
    & $xr --add-misses $image $Misses $deltas
    if ($LASTEXITCODE -ne 0) { "adding the pack misses failed"; exit 1 }
}
New-Item -ItemType Directory -Force (Split-Path $pack) | Out-Null
# The CSV lists the runtime shaders; the patches column rebuilds each runtime ucode from its container.
& $xr --cc2-pack $image $deltas $pack
if ($LASTEXITCODE -ne 0) { "pack generation failed"; exit 1 }

Remove-Item $image -ErrorAction SilentlyContinue
"shader pack ready: $pack"
