# Building

## Requirements

- Windows 10 or 11
- Visual Studio 2022 with C++, Clang, CMake and Ninja
- Python 3.10 or newer
- The [ReXGlue SDK](https://github.com/rexglue/rexglue-sdk)
- Your own copy of the Avatar Editor and assets

## Assets

This project was built and tested against a specific version of the Avatar Editor:

| File | Title ID | Version |
|------|----------|---------|
| `avatareditor.xex` | `584D07D1` | 2.0.12581.0 |

This build is the second-generation editor from the Fall 2010 Kinect dashboard refresh, distributed through the Xbox Live Preview Program in late September 2010. 
This was a public beta about a month ahead of the 2.0.12611.0 retail release.

The following files are required and obtained via a dump of the 2.0.12581.0 dash. 

```
assets/
├── avatareditor.xex
├── fonts/
│   ├── XenonCLatin.xtt
│   └── XenonJKLatin.xtt
├── Avatar6400.Avatar
├── Avatar6404.Avatar
├── Avatar6411.Avatar
├── Cheer.AvatarAnimation
├── Gift.AvatarAnimation
├── IdleOffScreen.AvatarAnimation
├── Look.AvatarAnimation
├── RunIn.AvatarAnimation
├── RunOutLong.AvatarAnimation
├── RunOutMedium.AvatarAnimation
├── RunOutShort.AvatarAnimation
├── RunOutStandard.AvatarAnimation
├── Salute.AvatarAnimation
├── AvatarAssetPack.toc
└── AvatarAssetPackLegacyV1.toc
```

## Steps

Builds with CMake against the ReXGlue SDK source tree (`-DREXSDK_DIR=<path>`).

```powershell
cmake --preset win-amd64-release -DREXSDK_DIR=<path>
cmake --build --preset win-amd64-release
tools/shaderpack/build_shaderpack.ps1
```

Run the build twice on a fresh tree.
