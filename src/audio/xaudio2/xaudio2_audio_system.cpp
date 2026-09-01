// =====================================================================
//  xaudio2_audio_system.cpp: XAudio2 (native Windows) audio system.
//  Ported from masterspike52/reDAHM (redahm_engine/audio).
//
//  Difference from the reference: device creation is lazy. The reference
//  creates the XAudio2 device + mastering voice in Initialize(), which the
//  base AudioSystem calls from the audio worker thread during early boot,
//  putting the device open on the boot-critical path. Here Initialize() stays
//  lightweight and EnsureEngine() creates the device on the first
//  CreateDriver() call (when the game registers an audio client), the same
//  point the SDL backend opens its device.
// =====================================================================
#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <windows.h>

#include <rex/audio/xaudio2/xaudio2_audio_system.h>
#include <rex/audio/xaudio2/xaudio2_audio_driver.h>

#include <rex/logging.h>
#include <rex/types.h>

namespace xaudio2 {

// The X_STATUS_* macros expand to a cast to (unqualified) X_STATUS.
using rex::X_STATUS;

XAudio2AudioSystem::XAudio2AudioSystem(rex::runtime::FunctionDispatcher* function_dispatcher)
    : AudioSystem(function_dispatcher) {}

XAudio2AudioSystem::~XAudio2AudioSystem() {
  if (mastering_voice_) {
    mastering_voice_->DestroyVoice();
    mastering_voice_ = nullptr;
  }
  xaudio2_.Reset();
  if (com_initialized_) {
    CoUninitialize();
    com_initialized_ = false;
  }
}

std::unique_ptr<rex::system::IAudioSystem> XAudio2AudioSystem::Create(
    rex::runtime::FunctionDispatcher* dispatcher) {
  return std::make_unique<XAudio2AudioSystem>(dispatcher);
}

void XAudio2AudioSystem::Initialize() {
  // Intentionally lightweight: no COM or XAudio2 device creation here. This
  // runs on the audio worker thread during early boot; deferring device
  // creation to EnsureEngine()/CreateDriver() keeps boot off the device-open
  // path (see file header).
  AudioSystem::Initialize();
}

bool XAudio2AudioSystem::EnsureEngine() {
  if (xaudio2_) {
    return true;
  }

  // Modern XAudio2 (2.8+/2.9) does not strictly require COM, but initialize it
  // defensively on the calling thread. RPC_E_CHANGED_MODE means COM was already
  // initialized with a different model on this thread, which is fine.
  HRESULT hr = CoInitializeEx(nullptr, COINIT_MULTITHREADED);
  if (FAILED(hr) && hr != RPC_E_CHANGED_MODE) {
    REXAPU_ERROR("XAudio2: CoInitializeEx failed: {:08X}", static_cast<uint32_t>(hr));
    return false;
  }
  // Only CoUninitialize when this call is the one that initialized COM (S_OK).
  com_initialized_ = (hr == S_OK);

  hr = XAudio2Create(xaudio2_.GetAddressOf(), 0, XAUDIO2_DEFAULT_PROCESSOR);
  if (FAILED(hr)) {
    REXAPU_ERROR("XAudio2Create failed: {:08X}", static_cast<uint32_t>(hr));
    xaudio2_.Reset();
    return false;
  }

  hr = xaudio2_->CreateMasteringVoice(&mastering_voice_);
  if (FAILED(hr)) {
    REXAPU_ERROR("XAudio2: CreateMasteringVoice failed: {:08X}", static_cast<uint32_t>(hr));
    xaudio2_.Reset();
    return false;
  }

  REXAPU_INFO("XAudio2: device + mastering voice created (lazy)");
  return true;
}

rex::X_STATUS XAudio2AudioSystem::CreateDriver(size_t index, rex::thread::Semaphore* semaphore,
                                               rex::audio::AudioDriver** out_driver) {
  if (!EnsureEngine()) {
    return X_STATUS_UNSUCCESSFUL;
  }

  uint8_t device_channels = 6;
  if (mastering_voice_) {
    XAUDIO2_VOICE_DETAILS details = {};
    mastering_voice_->GetVoiceDetails(&details);
    if (details.InputChannels < 6) {
      device_channels = 2;
    }
  }

  auto on_completed = [semaphore]() { semaphore->Release(1, nullptr); };

  auto* driver = new XAudio2AudioDriver(memory_, xaudio2_.Get(), device_channels,
                                        std::move(on_completed));
  if (!driver->Initialize()) {
    delete driver;
    return X_STATUS_UNSUCCESSFUL;
  }

  *out_driver = driver;
  return X_STATUS_SUCCESS;
}

void XAudio2AudioSystem::DestroyDriver(rex::audio::AudioDriver* driver) {
  auto* d = static_cast<XAudio2AudioDriver*>(driver);
  d->Shutdown();
  delete d;
}

}  // namespace xaudio2
