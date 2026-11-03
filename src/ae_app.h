// avatareditor - ReXGlue Recompiled Project
//
// Customize your app by overriding virtual hooks from rex::ReXApp.

#pragma once

#include <atomic>
#include <cstdio>
#include <cstdlib>
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
#include <rex/input/input_system.h>
#include <rex/system/xmemory.h>
#include <rex/ui/imgui_dialog.h>
#include <rex/ui/imgui_drawer.h>
#include <rex/ui/overlay/achievement_notification.h>
#include <rex/ui/presenter.h>
#include <rex/ui/window.h>
#include <rex/ui/window_listener.h>

#include <jmlive/jmlive.h>
#include <rexglue_guide/guide.h>

#include "ae_achievements.h"
#include "catalog_search.h"
#include "fault_diag.h"
#include "gpu/gpu.h"
#include "gpu/gpu_overlay.h"
#include "input_hooks.h"

#if REX_PLATFORM_WIN32
#include <rex/audio/xaudio2/xaudio2_audio_system.h>
#endif

REXCVAR_DECLARE(std::string, user_data_root);
REXCVAR_DECLARE(std::string, log_file);
REXCVAR_DECLARE(std::string, user_name);

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
    // A toml that names the log file gets one fresh log per launch, no rotation history.
    const std::string log = REXCVAR_GET(log_file);
    if (!log.empty()) {
      std::error_code ec;
      std::filesystem::remove(log, ec);
      const std::filesystem::path rotated = std::filesystem::path(log).replace_extension(".1.log");
      std::filesystem::remove(rotated, ec);
    }
    std::string game_root = REXCVAR_GET(game_data_root);
    if (!game_root.empty()) {
      std::filesystem::path p = game_root;
      if (p.is_relative()) {
        p = rex::filesystem::GetExecutableFolder() / p;
      }
      paths.game_data_root = p.lexically_normal();
    }
    // JMinstaller's layout, which dev builds default to: user data in userdata\Avatar Editor, the avatar shared in
    // userdata\Avatar\Manifest.
    const std::filesystem::path userdata = rex::filesystem::GetUserFolder() / "JMstudios" / "userdata";
    const std::string user_root = REXCVAR_GET(user_data_root);
    paths.user_data_root = user_root.empty() ? userdata / "Avatar Editor" : std::filesystem::path(user_root);
    if (rex::cvar::GetFlagSource("avatar_data_dir") == rex::cvar::Source::kDefault) {
      rex::cvar::SetFlagByName("avatar_data_dir", (userdata / "Avatar" / "Manifest").string());
    }
  }

  void OnPreSetup(rex::RuntimeConfig& config) override {
    // No emulated GPU: D3D runs recompiled against the title's edge layer and nothing reads the ring.
    config.graphics = nullptr;
    config.gpu_plugin.clear();
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
      ae::diag::InstallFaultReporter();
    };
  }

  // Retitle the window the base created and hook up the key listeners.
  bool SetupPresentation() override {
    if (!rex::ReXApp::SetupPresentation()) {
      return false;
    }
    window()->SetTitle("Avatar Editor");
    key_feed_.window = window();
    window()->AddInputListener(&key_feed_, 0);
    ae_search::Get().Attach(window());
    // Each present asks the UI thread for the next overlay frame; one request is queued at a time.
    ae::gpu::SetOverlayFrameCallback([this] {
      if (overlay_queued_.exchange(true)) return;
      if (!app_context().CallInUIThread([this] {
            overlay_queued_ = false;
            DrawOverlay();
          })) {
        overlay_queued_ = false;
      }
    });
    // The device comes up here, before any guest thread exists, so the first Swap already has a swap chain.
    ae::gpu::Initialize(window());
    return true;
  }

  // Detached mode: ImGui draws through the native renderer (gpu_overlay.h), for JMlive's Guide and the SDK overlays.
  std::unique_ptr<rex::ui::ImmediateDrawer> OnCreateImmediateDrawer() override {
    return ae::gpu::CreateOverlayDrawer();
  }

  void OnConfigureFonts(ImFontAtlas* atlas) override {
    rex::ReXApp::OnConfigureFonts(atlas);
    rexglue_guide::ConfigureFonts(atlas);
  }

  // JMlive's Guide (F10): the local profile from the start, Game Jolt sign-in from inside it.
  void OnCreateDialogs(rex::ui::ImGuiDrawer* drawer) override {
    rex::ReXApp::OnCreateDialogs(drawer);
    // Installed builds (JMinstaller sets user_data_root) have no debug (F3) or settings (F4) overlay.
    if (!REXCVAR_GET(user_data_root).empty()) {
      for (const char* bind : {"bind_debug_overlay", "bind_settings"}) {
        if (rex::cvar::GetFlagSource(bind) == rex::cvar::Source::kDefault) rex::cvar::SetFlagByName(bind, "");
      }
    }
    jmlive::Configure({.gamertag = REXCVAR_GET(user_name)});
    rexglue_guide::Install(this, drawer,
                           {.title_name = "Avatar Editor",
                            .data_dir = rex::filesystem::GetUserFolder() / "JMstudios" / "userdata" / "JMlive",
                            .on_quit = [this] { OnWindowCloseRequested(); },
                            .immediate_drawer = immediate_drawer(),
                            .config_file = rex::filesystem::GetExecutableFolder() / "avatareditor.toml"});
  }

  // Game input pauses while the Guide is up, and while ImGui holds the mouse as the SDK's own callback did.
  void OnPostSetup() override {
    auto* input = static_cast<rex::input::InputSystem*>(runtime()->input_system());
    if (input && imgui_drawer()) {
      input->SetActiveCallback(
          [this] { return !rexglue_guide::IsOpen() && !imgui_drawer()->GetIO().WantCaptureMouse; });
    }
  }

  // UI thread: one ImGui frame into the overlay the renderer draws at its next present.
  void DrawOverlay() {
    if (!imgui_drawer() || !window()) return;
    rex::ui::AppUIDrawContext context(window()->GetActualPhysicalWidth(), window()->GetActualPhysicalHeight());
    ae::gpu::BeginOverlayFrame();
    imgui_drawer()->Draw(context);
    ae::gpu::EndOverlayFrame();
    // The Guide starts listening for unlocks on its first frame, so Hello waits for that.
    if (!hello_unlocked_) {
      hello_unlocked_ = true;
      ae::achievements::Unlock(ae::achievements::kHello);
    }
  }
  std::atomic<bool> overlay_queued_{false};
  bool hello_unlocked_ = false;

  void OnWindowPixelSizeChanged(uint32_t, uint32_t) override { ae::gpu::RequestResize(); }

  // SDL quit callbacks hold the event-watch lock, so the renderer teardown and the final close wait for the UI loop.
  bool OnWindowCloseRequested() override {
    if (!close_queued_) {
      close_queued_ = true;
      if (!app_context().CallInUIThreadDeferred([this] {
            ae::gpu::SetOverlayFrameCallback(nullptr);
            ae::gpu::Shutdown();
            window()->RequestClose();
          })) {
        close_queued_ = false;
        REXLOG_ERROR("Could not queue window close");
      }
    }
    return false;
  }

  void OnShutdown() override {
    rexglue_guide::Shutdown();
    ae::gpu::SetOverlayFrameCallback(nullptr);
    ae::gpu::Shutdown();
  }

  // The Guide shows unlocks and the achievements page; the SDK's ImGui ones would double them.
  std::unique_ptr<rex::ui::AchievementNotificationDialog> CreateAchievementNotificationDialog() override {
    return nullptr;
  }
  std::unique_ptr<rex::ui::ImGuiDialog> CreateAchievementsOverlay() override { return nullptr; }

  // Separate input listener feeding the keystroke synthesizer
  // (input_hooks.cpp); the base app's own listener keeps the overlay binds.
  class KeyFeed : public rex::ui::WindowInputListener {
   public:
    rex::ui::Window* window = nullptr;

   private:
    // F11 flips the window and the fullscreen cvar together, so the settings page and the toml stay in step.
    void OnKeyDown(rex::ui::KeyEvent& e) override {
      if (e.virtual_key() == rex::ui::VirtualKey::kF11 && window) {
        const bool on = !window->IsFullscreen();
        rex::cvar::SetFlagByName("fullscreen", on ? "true" : "false");
        window->SetFullscreen(on);
        e.set_handled(true);
        return;
      }

      // Typing into the search box is not navigation.
      if (ae_search::Get().open()) return;
      ae_input::OnHostKey(e.virtual_key(), true, e.is_shift_pressed(),
                          e.is_ctrl_pressed(), e.is_alt_pressed());
    }
    void OnKeyUp(rex::ui::KeyEvent& e) override {
      if (ae_search::Get().open()) return;
      ae_input::OnHostKey(e.virtual_key(), false, e.is_shift_pressed(),
                          e.is_ctrl_pressed(), e.is_alt_pressed());
    }
  };
  KeyFeed key_feed_;

  // Registers the editor's achievements, then the image dump for the shader pack build
  // (tools/shaderpack): write the mapped, decrypted image and quit before any guest code runs.
  void OnPostLoadXexImage() override {
    ae::achievements::Register();
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
  bool close_queued_ = false;

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
