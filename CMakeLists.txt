# avatareditor - ReXGlue Recompiled Project
#
# SDK boilerplate lives in generated/rexglue.cmake.

cmake_minimum_required(VERSION 3.25)
project(avatareditor LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

include(generated/rexglue.cmake)

# sources.cmake is committed so a fresh tree knows the generated file names
# up front; watching it makes a build reconfigure if codegen reshapes the list.
set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS
    "${CMAKE_CURRENT_SOURCE_DIR}/generated/ae/sources.cmake")

# Sources: the app shell plus the title-owned kernel subsystems (avatar
# library, XAM avatar/user surface, HTTP and XLiveBase services). These
# compile into the exe so their __imp__ definitions override the runtime's
# at link time.
file(GLOB AVATAR_LIB_SOURCES src/kernel/xam/avatars/*.cpp)
set(AVATAREDITOR_SOURCES
    src/main.cpp
    src/patches.cpp
    src/xui_hooks.cpp
    src/search_state.cpp
    src/input_hooks.cpp
    src/catalog_search.cpp
    src/notify_hooks.cpp
    src/gpu/edge.cpp
    src/gpu/gpu_capture.cpp
    src/gpu/gpu_cmdbuf.cpp
    src/gpu/gpu_draw.cpp
    src/gpu/gpu_formats.cpp
    src/gpu/gpu_memory.cpp
    src/gpu/gpu_renderer.cpp
    src/gpu/gpu_shader_pack.cpp
    src/gpu/gpu_texture_decode.cpp
    src/gpu/gpu_tracker.cpp
    src/fault_diag.cpp
    src/dll_dir.cpp
    src/kernel/xam/xam_avatar.cpp
    src/kernel/xam/xam_user.cpp
    src/kernel/xam/xam_loader.cpp
    src/kernel/xam/xam_ui.cpp
    src/kernel/xam/xhttp.cpp
    src/kernel/xam/marketplace_key.cpp
    src/kernel/xam/apps/xlivebase_app.cpp
    ${AVATAR_LIB_SOURCES}
)

# Title audio backend: XAudio2 on Windows, selected via
# RuntimeConfig::audio_factory in ae_app.h OnPreSetup.
if(WIN32)
    list(APPEND AVATAREDITOR_SOURCES
        src/audio/xaudio2/xaudio2_audio_driver.cpp
        src/audio/xaudio2/xaudio2_audio_system.cpp
    )
endif()

if(WIN32)
    add_executable(avatareditor WIN32 ${AVATAREDITOR_SOURCES})
    # Windows resources: app icon (MAINICON, loaded by the runtime's window
    # class) plus the version block (Task Manager display name). Linked as a
    # precompiled .res rather than built through CMake's clang RC pipeline;
    # regenerate with:
    #   rc.exe /nologo /fo src/ae.res src/ae.rc
    target_link_options(avatareditor PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/src/ae.res")
else()
    add_executable(avatareditor ${AVATAREDITOR_SOURCES})
endif()

# Title-owned headers (rex/kernel avatar surface) resolve before the
# installed SDK headers; glm serves the avatar library.
target_include_directories(avatareditor BEFORE PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/src
    ${CMAKE_CURRENT_SOURCE_DIR}/thirdparty/glm
)

# ImGui headers for the SDK ui headers the app and xam_ui pull in. Include
# dirs only: the objects live in the runtime DLL, which exports them.
target_include_directories(avatareditor PRIVATE
    $<TARGET_PROPERTY:imgui,INTERFACE_INCLUDE_DIRECTORIES>)

# plume RHI for the native renderer (src/gpu), D3D12 only.
add_subdirectory("${CMAKE_CURRENT_SOURCE_DIR}/thirdparty/plume" plume)
target_link_libraries(avatareditor PRIVATE plume)

# The XenosRecomp fork's translator library (tools/XenosRecomp): the pack reader, the fp2 key and the vfetch layout
# table. DXC is delay-loaded, so the title only needs its DLLs once a runtime translation runs.
enable_language(C)
set(XENOS_RECOMP_THIRDPARTY_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/tools/XenosRecomp/thirdparty")
set(XENOS_RECOMP_BUILD_CLI OFF CACHE BOOL "" FORCE)
set(ZSTD_BUILD_PROGRAMS OFF CACHE BOOL "" FORCE)
set(ZSTD_BUILD_SHARED OFF CACHE BOOL "" FORCE)
set(ZSTD_BUILD_TESTS OFF CACHE BOOL "" FORCE)
add_subdirectory("${XENOS_RECOMP_THIRDPARTY_ROOT}" xenosrecomp_thirdparty EXCLUDE_FROM_ALL)
add_subdirectory("${CMAKE_CURRENT_SOURCE_DIR}/tools/XenosRecomp/XenosRecomp" xenosrecomp EXCLUDE_FROM_ALL)
target_link_libraries(avatareditor PRIVATE xenosrecomp_core xxHash::xxhash)
if(WIN32)
    target_link_options(avatareditor PRIVATE "LINKER:/DELAYLOAD:rexruntime.dll" "LINKER:/DELAYLOAD:dxcompiler.dll" "LINKER:/DELAYLOAD:dxil.dll")
    target_link_libraries(avatareditor PRIVATE delayimp)
    add_custom_command(TARGET avatareditor POST_BUILD COMMAND ${CMAKE_COMMAND} -E copy_if_different
        "${XENOS_RECOMP_THIRDPARTY_ROOT}/dxc-bin/bin/x64/dxcompiler.dll"
        "${XENOS_RECOMP_THIRDPARTY_ROOT}/dxc-bin/bin/x64/dxil.dll"
        "$<TARGET_FILE_DIR:avatareditor>")
endif()
# The shader pack loader finds out/shaderpack/ae.pack relative to the source tree.
target_compile_definitions(avatareditor PRIVATE AE_SOURCE_DIR="${CMAKE_CURRENT_SOURCE_DIR}")

# XAudio2 system library for the title audio backend.
if(WIN32)
    target_link_libraries(avatareditor PRIVATE xaudio2)
endif()

# No GPU plugin: the native renderer owns the window (see ae_app.h).
rexglue_setup_target(avatareditor)

# The config is read from the executable root.
if(NOT EXISTS "${CMAKE_BINARY_DIR}/avatareditor.toml")
    configure_file("${CMAKE_CURRENT_SOURCE_DIR}/avatareditor.toml"
                   "${CMAKE_BINARY_DIR}/avatareditor.toml" COPYONLY)
endif()
