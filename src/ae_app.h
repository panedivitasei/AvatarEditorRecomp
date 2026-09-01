// avatareditor - ReXGlue Recompiled Project
//
// Customize your app by overriding virtual hooks from rex::ReXApp.

#pragma once

#include <cstdio>
#include <cstring>
#include <filesystem>
#include <memory>
#include <system_error>

#include <rex/cvar.h>
#include <rex/filesystem.h>
#include <rex/filesystem/devices/host_path_device.h>
#include <rex/kernel/xam/apps/xlivebase_app.h>  // InitializeTitleKernel
#include <rex/logging.h>
#include <rex/platform.h>
#include <rex/rex_app.h>
#include <rex/system/function_dispatcher.h>
#include <rex/system/xmemory.h>
#include <rex/ui/window.h>
#include <rex/ui/window_listener.h>
#include <video_native.h>

#include "input_hooks.h"

#if REX_PLATFORM_WIN32
#include <rex/audio/xaudio2/xaudio2_audio_system.h>
#endif

REXCVAR_DECLARE(std::string, user_data_root);

// Shader pack generation input: the pack builder recompiles the shaders
// embedded in the decrypted image, and this is the only place the decrypted
// bytes exist.
REXCVAR_DEFINE_STRING(dump_image_path, "", "AE",
                      "Write the decrypted guest image to this path and exit.");

class AvatareditorApp : public rex::ReXApp {
 public:
  using rex::ReXApp::ReXApp;

  static std::unique_ptr<rex::ui::WindowedApp> Create(
      rex::ui::WindowedAppContext& ctx) {
    return std::unique_ptr<AvatareditorApp>(new AvatareditorApp(ctx, "avatareditor",
        ShadowTableImageInfo()));
  }

  // The SDK allocates its function table at image_base + image_size in the
  // title heap, which for a 0x92000000 system exe is out of range and kills
  // Setup at boot. So hand Setup shadow bounds inside the title heap and
  // mirror the table over the real location before launch; anything
  // registered late misses the mirror, reads zero, and falls back to the
  // dispatcher.
  static constexpr uint32_t kShadowTableBase = 0x8F000000u;
  static constexpr uint32_t kDispatchTableBase =
      uint32_t(REX_IMAGE_BASE + REX_IMAGE_SIZE);
  static constexpr uint32_t kDispatchTableSize =
      uint32_t((REX_CODE_SIZE + REX_THUNK_RESERVE_SIZE) * 2);
  static_assert(REX_THUNK_RESERVE_SIZE ==
                    rex::runtime::FunctionDispatcher::kThunkReserveSize,
                "generated thunk reserve must match the SDK per-module pool");

  static rex::PPCImageInfo ShadowTableImageInfo() {
    rex::PPCImageInfo info = PPCImageConfig;
    info.image_base = kShadowTableBase;
    info.image_size = 0;
    return info;
  }

  // The entry module is AvatarEditor.xex, a 0x92000000-based system executable
  // from the NXE Kinect dashboard, rather than the usual default.xex.
  void OnLoadXexImage(std::string& xex_image) override {
    xex_image = "game:\\AvatarEditor.xex";
  }

  // Config lives next to the exe; PathConfig::config_path carries the
  // location. The SDK default is the same value; set it explicitly so the
  // location survives default changes.
  void OnConfigurePaths(rex::PathConfig& paths) override {
    paths.config_path = rex::filesystem::GetExecutableFolder() / "avatareditor.toml";
    // The runtime resolves paths from cvars before it loads the config file,
    // so a toml-set game_data_root would arrive too late. Load the config
    // here and re-derive the paths; the runtime's own load right after is a
    // harmless reapply.
    if (std::filesystem::exists(paths.config_path)) {
      rex::cvar::LoadConfig(paths.config_path);
    }
    std::string game_root = REXCVAR_GET(game_data_root);
    if (!game_root.empty()) {
      std::filesystem::path p = game_root;
      if (p.is_relative()) {
        p = rex::filesystem::GetExecutableFolder() / p;
      }
      paths.game_data_root = p.lexically_normal();
    }
    // Profiles, saves, and the avatar manifest are shared with the other
    // recomps through one userdata folder.
    if (REXCVAR_GET(user_data_root).empty()) {
      paths.user_data_root =
          rex::filesystem::GetUserFolder() / "ReXGlue" / "userdata";
    }
  }

  void OnPreSetup(rex::RuntimeConfig& config) override {
#if REX_PLATFORM_WIN32
    // Title XAudio2 backend in place of the SDK's SDL default.
    config.audio_factory = REX_AUDIO_BACKEND(xaudio2::XAudio2AudioSystem);
#endif
    // Title kernel init: registers the title's XLiveBaseApp for app id 0xFC
    // (the SDK default would register its built-in one; see xlivebase_app.h),
    // then mounts the guest font device. kernel_init runs inside
    // Runtime::Setup, the earliest point where file_system() and the resolved
    // game_data_root are both live.
    config.kernel_init = [this](rex::Runtime* runtime,
                                rex::system::KernelState* kernel_state) {
      rex::kernel::xam::apps::InitializeTitleKernel(runtime, kernel_state);
      MountFontDevice(runtime);
    };
  }

  // Retitle the window the base created. With native video on, detach the
  // GPU plugin's presenter from the window: the plugin stays loaded so the
  // command processor consumes the ring and delivers fences and vblanks,
  // but the native renderer's swapchain owns the window surface.
  bool SetupPresentation() override {
    if (!rex::ReXApp::SetupPresentation()) {
      return false;
    }
    window()->SetTitle("Avatar Editor");
    if (rex::videonative::Enabled()) {
      window()->SetPresenter(nullptr);
    }
    window()->AddInputListener(&key_feed_, 0);
    return true;
  }

  // Separate input listener feeding the keystroke synthesizer
  // (input_hooks.cpp); the base app's own listener keeps the overlay binds.
  class KeyFeed : public rex::ui::WindowInputListener {
    void OnKeyDown(rex::ui::KeyEvent& e) override {
      ae_input::OnHostKey(e.virtual_key(), true, e.is_shift_pressed(),
                          e.is_ctrl_pressed(), e.is_alt_pressed());
    }
    void OnKeyUp(rex::ui::KeyEvent& e) override {
      ae_input::OnHostKey(e.virtual_key(), false, e.is_shift_pressed(),
                          e.is_ctrl_pressed(), e.is_alt_pressed());
    }
  };
  KeyFeed key_feed_;

  // Image dump for the shader pack build (tools/shaderpack): write the
  // mapped, decrypted image and quit before any guest code runs.
  void OnPostLoadXexImage() override {
    const std::string out = REXCVAR_GET(dump_image_path);
    if (out.empty()) {
      return;
    }
    const auto* bytes = runtime()->memory()->TranslateVirtual<const uint8_t*>(
        PPCImageConfig.image_base);
    FILE* f = std::fopen(out.c_str(), "wb");
    if (f && bytes) {
      std::fwrite(bytes, 1, PPCImageConfig.image_size, f);
    }
    if (f) {
      std::fclose(f);
    }
    REXLOG_INFO("Dumped guest image to {} ({} bytes)", out,
                uint32_t(PPCImageConfig.image_size));
    app_context().QuitFromUIThread();
  }

  // AvatarEditor is a system executable that addresses xam.xex and xboxkrnl
  // internals through hardcoded pointers (both load at fixed addresses on
  // hardware; xam at 0x8E000000). Neither module is loaded here, so commit
  // zeroed shim pages over the ranges the binary touches: a zero read is the
  // benign "flag not set" answer. First hits: the editor reads xam state
  // flags at 0x8E038634 in main() and writes an init-once status at
  // 0x80300004 on its exit path.
  void OnPreLaunchModule() override {
    MirrorDispatchTable();
    static constexpr struct {
      uint32_t base;
      uint32_t size;
    } kSystemShimRegions[] = {
        {0x8E000000u, 0x00600000u},  // xam.xex image/data range
        {0x80300000u, 0x00010000u},  // xboxkrnl data page referenced by CRT
    };
    auto* memory = runtime()->memory();
    for (const auto& region : kSystemShimRegions) {
      auto* heap = memory->LookupHeap(region.base);
      if (heap &&
          heap->AllocFixed(region.base, region.size, 0x10000,
                           rex::memory::kMemoryAllocationReserve |
                               rex::memory::kMemoryAllocationCommit,
                           rex::memory::kMemoryProtectRead |
                               rex::memory::kMemoryProtectWrite)) {
        memory->Zero(region.base, region.size);
        REXLOG_INFO("System-exe shim region committed at {:08X}-{:08X}", region.base,
                    region.base + region.size);
      } else {
        REXLOG_WARN("System-exe shim region at {:08X} could not be committed", region.base);
      }
    }
  }

 private:
  // Commit the table the generated code reads and copy the shadow table
  // into it, so everything registered before launch stays on the fast path.
  void MirrorDispatchTable() {
    auto* memory = runtime()->memory();
    auto* heap = memory->LookupHeap(kDispatchTableBase);
    if (!heap ||
        !heap->AllocFixed(kDispatchTableBase, kDispatchTableSize, 0x1000,
                          rex::memory::kMemoryAllocationReserve |
                              rex::memory::kMemoryAllocationCommit,
                          rex::memory::kMemoryProtectRead |
                              rex::memory::kMemoryProtectWrite)) {
      REXLOG_ERROR("Dispatch table at {:08X} could not be committed",
                   kDispatchTableBase);
      return;
    }
    std::memcpy(memory->TranslateVirtual<uint8_t*>(kDispatchTableBase),
                memory->TranslateVirtual<const uint8_t*>(kShadowTableBase),
                kDispatchTableSize);
    REXLOG_INFO("Dispatch table mirrored at {:08X}-{:08X}", kDispatchTableBase,
                kDispatchTableBase + kDispatchTableSize);
  }

  // Mounts <game_data_root>/fonts as \Device\Flash\Fonts with a font: symlink
  // when the directory exists. System executables load the console's .xtt
  // fonts through this device (it maps into the flash filesystem on
  // hardware). Registration order against the game mount does not matter;
  // the paths do not overlap.
  void MountFontDevice(rex::Runtime* runtime) {
    std::error_code ec;
    const auto fonts_root =
        std::filesystem::absolute(runtime->game_data_root(), ec) / "fonts";
    if (ec || !std::filesystem::is_directory(fonts_root, ec)) {
      return;
    }
    constexpr const char* kFontMount = "\\Device\\Flash\\Fonts";
    auto device = std::make_unique<rex::filesystem::HostPathDevice>(
        kFontMount, fonts_root, /*read_only=*/true);
    if (device->Initialize() &&
        runtime->file_system()->RegisterDevice(std::move(device))) {
      runtime->file_system()->RegisterSymbolicLink("font:", kFontMount);
      REXLOG_INFO("Mounted {} at font:", fonts_root.string());
    } else {
      REXLOG_WARN("font: mount failed for {}", fonts_root.string());
    }
  }
};
