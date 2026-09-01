# rexvideonative: native implementation of the guest-facing XDK D3D API,
# ported from UnleashedRecomp (hedge-dev, GPL-3.0) onto the plume RHI.
#
# Licensing: everything under src/videonative is GPL-3.0 (see LICENSE in
# this directory). It is statically linked into the app, which makes the
# resulting binary distribution subject to GPL-3.0.
# Namespace: rex::videonative

set(REXVIDEONATIVE_SOURCES
    render_queue.cpp
    renderer.cpp
    renderer.h
    renderer_fps.cpp
    renderer_fps.h
    semantics/sampler_semantics.cpp
    semantics/sampler_semantics.h
    shader_cache.cpp
    shader_cache.h
    video_native.cpp
    video_native.h
    video_native_internal.h
)

add_library(rexvideonative STATIC ${REXVIDEONATIVE_SOURCES})
add_library(rex::videonative ALIAS rexvideonative)

target_include_directories(rexvideonative PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
# xxHash in header-only mode (XXH_INLINE_ALL), no link dependency; take it
# from wherever REXSDK_DIR points instead of assuming sdk/ in the repo root.
get_filename_component(REXSDK_ABS "${REXSDK_DIR}" ABSOLUTE BASE_DIR "${CMAKE_SOURCE_DIR}")
target_include_directories(rexvideonative PRIVATE "${REXSDK_ABS}/thirdparty/xxHash")

target_link_libraries(rexvideonative PUBLIC plume rex::runtime)

set_target_properties(rexvideonative PROPERTIES FOLDER "src")
