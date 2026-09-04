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
    src/video_hooks.cpp
    src/input_hooks.cpp
    src/catalog_search.cpp
    src/notify_hooks.cpp
    src/kernel/xam/xam_avatar.cpp
    src/kernel/xam/xam_user.cpp
    src/kernel/xam/xam_loader.cpp
    src/kernel/xam/xam_ui.cpp
    src/kernel/xam/xhttp.cpp
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

# Native video layer: plume RHI + rexvideonative, both title-owned
# (hook table in src/video_hooks.cpp).
add_subdirectory("${CMAKE_CURRENT_SOURCE_DIR}/thirdparty/plume" plume)
add_subdirectory("${CMAKE_CURRENT_SOURCE_DIR}/src/videonative" videonative)
target_link_libraries(avatareditor PRIVATE rex::videonative)

# XAudio2 system library for the title audio backend.
if(WIN32)
    target_link_libraries(avatareditor PRIVATE xaudio2)
endif()

rexglue_setup_target(avatareditor GPU_PLUGINS xenos)

# The config is read from the executable root.
if(NOT EXISTS "${CMAKE_BINARY_DIR}/avatareditor.toml")
    configure_file("${CMAKE_CURRENT_SOURCE_DIR}/avatareditor.toml"
                   "${CMAKE_BINARY_DIR}/avatareditor.toml" COPYONLY)
endif()
