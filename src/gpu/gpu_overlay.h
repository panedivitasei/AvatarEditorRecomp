// Dear ImGui over the native renderer: the SDK's ImGui drawer records each UI frame on the UI thread through the
// drawer made here, and the render thread draws the newest one over the swap chain image before present.
#pragma once

#include <functional>
#include <memory>

namespace rex::ui {
class ImmediateDrawer;
}  // namespace rex::ui

namespace ae::gpu {

// The detached-mode drawer for ReXApp::OnCreateImmediateDrawer; its textures live until the frames using them retire.
std::unique_ptr<rex::ui::ImmediateDrawer> CreateOverlayDrawer();

// Render thread, after each present: the app draws its next UI frame from here (on its UI thread). Null clears it.
void SetOverlayFrameCallback(std::function<void()> callback);

// UI thread, around ImGuiDrawer::Draw: a pass that drew nothing clears the overlay, so closed dialogs disappear.
void BeginOverlayFrame();
void EndOverlayFrame();

}  // namespace ae::gpu
