# avatarextract: standalone CLI that extracts an Xbox 360 avatar (marketplace
# STFS "LIVE" package) into .obj + .mtl + .png. Reuses the avatar decode pipeline
# from src/kernel/xam/avatars without linking the heavy runtime/graphics stack.

set(AVATAR_DIR "${REXGLUE_ROOT}/src/kernel/xam/avatars")

add_executable(avatarextract
    main.cpp
    avatar_export.cpp   # --avatar: saved-avatar export (resolve + bake + JSON)
    refit.cpp           # --refit-female: add a female hiding shape to a male item

    # Avatar decode pipeline (compiled directly into the tool).
    "${AVATAR_DIR}/model.cpp"
    "${AVATAR_DIR}/texture.cpp"
    "${AVATAR_DIR}/strb.cpp"
    "${AVATAR_DIR}/compression.cpp"
    "${AVATAR_DIR}/bit_stream.cpp"
    "${AVATAR_DIR}/asset_pack.cpp"
    "${AVATAR_DIR}/blend_shape.cpp"
    "${AVATAR_DIR}/blend_shape_apply.cpp"
    "${AVATAR_DIR}/closet.cpp"
    "${AVATAR_DIR}/skeleton.cpp"
    "${AVATAR_DIR}/skeleton_data.cpp"
    "${AVATAR_DIR}/skeleton_scaling.cpp"
    "${AVATAR_DIR}/animation.cpp"
    "${AVATAR_DIR}/prop.cpp"

    # LZX decompression used by compression.cpp.
    "${REXGLUE_ROOT}/src/system/lzx.cpp"
)

target_include_directories(avatarextract PRIVATE
    "${CMAKE_CURRENT_SOURCE_DIR}"   # main.cpp + stb_image_write.h
    "${AVATAR_DIR}"                 # same-dir quoted includes ("model.h", ...)
    "${REXGLUE_ROOT}/thirdparty/glm" # header-only glm (skeleton quaternion math)
)

target_link_libraries(avatarextract PRIVATE
    rexcore     # logging (REXKRNL_*), types/math/string + fmt/spdlog transitively
    fmt::fmt    # fmt::format (explicit; also transitive via rexcore)
    mspack      # lzx.cpp -> libmspack lzxd
)

set_target_properties(avatarextract PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/tools/avatarextract"
)
