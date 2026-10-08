project(XenosRecomp)

if (WIN32)
    option(XENOS_RECOMP_DXIL "Generate DXIL shader cache" ON)
endif()

# The game links only xenosrecomp_core and turns the CLI off.
option(XENOS_RECOMP_BUILD_CLI "Build the XenosRecomp command-line tool" ON)

set(SMOLV_SOURCE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../thirdparty/smol-v/source")

# ABI inputs baked into xenosrecomp_core: the shader header and BASE bytes plus a hash of the translator sources.
# The pack header carries a hash over them, so a pack built from other sources is refused at load.
set(XR_ABI_SOURCES
    constant_table.h
    dxc_compiler.cpp
    dxc_compiler.h
    pch.h
    shader.h
    shader_code.h
    shader_common.h
    shader_recompiler.cpp
    shader_recompiler.h
    ucode_fingerprint.cpp
    ucode_fingerprint.h
    xenosrecomp.h
    xenosrecomp_core.cpp)
set(XR_COMMON_HEADER "${CMAKE_CURRENT_SOURCE_DIR}/rexglue_shader_common.h")
set(XR_BASE_FILE "${CMAKE_CURRENT_SOURCE_DIR}/../BASE")

set(_xr_sources_text "")
foreach(_xr_file ${XR_ABI_SOURCES})
    file(READ "${CMAKE_CURRENT_SOURCE_DIR}/${_xr_file}" _xr_content HEX)
    string(APPEND _xr_sources_text "${_xr_file}:${_xr_content};")
    set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/${_xr_file}")
endforeach()
string(SHA256 _xr_source_hash "${_xr_sources_text}")
string(SUBSTRING "${_xr_source_hash}" 0 16 _xr_source_hash)
set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS "${XR_COMMON_HEADER}" "${XR_BASE_FILE}")

function(xr_hex_array out_var path)
    file(READ "${path}" _hex HEX)
    string(REGEX REPLACE "([0-9a-f][0-9a-f])" "0x\\1," _hex "${_hex}")
    set(${out_var} "${_hex}" PARENT_SCOPE)
endfunction()
xr_hex_array(_xr_header_bytes "${XR_COMMON_HEADER}")
xr_hex_array(_xr_base_bytes "${XR_BASE_FILE}")

set(XR_GEN_DIR "${CMAKE_CURRENT_BINARY_DIR}/generated")
set(_xr_gen_text "#pragma once\n\n#include <cstdint>\n\n")
string(APPEND _xr_gen_text "inline constexpr uint64_t kXrTranslatorSourceHash = 0x${_xr_source_hash}ull;\n")
string(APPEND _xr_gen_text "inline constexpr unsigned char kXrCommonHeader[] = { ${_xr_header_bytes} };\n")
string(APPEND _xr_gen_text "inline constexpr unsigned char kXrBaseFile[] = { ${_xr_base_bytes} };\n")
file(CONFIGURE OUTPUT "${XR_GEN_DIR}/xenosrecomp_abi_gen.h" CONTENT "${_xr_gen_text}")

add_library(xenosrecomp_core STATIC
    constant_table.h
    dxc_compiler.cpp
    dxc_compiler.h
    pch.h
    shader.h
    shader_code.h
    shader_common.h
    shader_recompiler.cpp
    shader_recompiler.h
    ucode_fingerprint.cpp
    ucode_fingerprint.h
    xenosrecomp.h
    xenosrecomp_core.cpp
    "${XR_GEN_DIR}/xenosrecomp_abi_gen.h")

target_compile_features(xenosrecomp_core PUBLIC cxx_std_20)
target_include_directories(xenosrecomp_core PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}" PRIVATE "${XR_GEN_DIR}" "${SMOLV_SOURCE_DIR}")
target_link_libraries(xenosrecomp_core PUBLIC
    Microsoft::DirectXShaderCompiler
    xxHash::xxhash
    libzstd_static
    fmt::fmt)
target_precompile_headers(xenosrecomp_core PRIVATE pch.h)
# Asserts throw instead of aborting, so a bad shader becomes a translation error.
target_compile_definitions(xenosrecomp_core PRIVATE XENOS_COVERAGE)

if (CMAKE_CXX_COMPILER_ID STREQUAL "Clang" OR CMAKE_CXX_COMPILER_ID STREQUAL "AppleClang")
    target_compile_options(xenosrecomp_core PRIVATE -Wno-switch -Wno-unused-variable -Wno-null-arithmetic -fms-extensions)

    include(CheckCXXSymbolExists)
    check_cxx_symbol_exists(_LIBCPP_VERSION version LIBCPP)
    if(LIBCPP)
        # Allows using std::execution
        target_compile_options(xenosrecomp_core PRIVATE -fexperimental-library)
    endif()
endif()

if (WIN32)
    target_compile_definitions(xenosrecomp_core PRIVATE _CRT_SECURE_NO_WARNINGS NOMINMAX)
endif()

if (XENOS_RECOMP_DXIL)
    target_compile_definitions(xenosrecomp_core PRIVATE XENOS_RECOMP_DXIL)
    target_link_libraries(xenosrecomp_core PUBLIC Microsoft::DXIL)
endif()

if (XENOS_RECOMP_BUILD_CLI)
    add_executable(XenosRecomp
        main.cpp
        "${SMOLV_SOURCE_DIR}/smolv.cpp")

    target_link_libraries(XenosRecomp PRIVATE xenosrecomp_core)
    target_include_directories(XenosRecomp PRIVATE ${SMOLV_SOURCE_DIR})
    target_precompile_headers(XenosRecomp PRIVATE pch.h)
    target_compile_definitions(XenosRecomp PRIVATE XENOS_COVERAGE)

    if (CMAKE_CXX_COMPILER_ID STREQUAL "Clang" OR CMAKE_CXX_COMPILER_ID STREQUAL "AppleClang")
        target_compile_options(XenosRecomp PRIVATE -Wno-switch -Wno-unused-variable -Wno-null-arithmetic -fms-extensions)
        if(LIBCPP)
            target_compile_options(XenosRecomp PRIVATE -fexperimental-library)
        endif()
    endif()

    if (WIN32)
        target_compile_definitions(XenosRecomp PRIVATE _CRT_SECURE_NO_WARNINGS NOMINMAX)
        add_custom_command(TARGET XenosRecomp POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E copy $<TARGET_RUNTIME_DLLS:XenosRecomp> $<TARGET_FILE_DIR:XenosRecomp>
            COMMAND_EXPAND_LISTS
        )
    endif()

    if (XENOS_RECOMP_DXIL)
        target_compile_definitions(XenosRecomp PRIVATE XENOS_RECOMP_DXIL)
    endif()
endif()
