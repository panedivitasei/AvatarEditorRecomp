// =====================================================================
//  xaudio2_audio_system.h: XAudio2 (native Windows) audio system.
//
//  Ported from masterspike52/reDAHM (redahm_engine/audio). Owns the
//  IXAudio2 device + mastering voice and creates XAudio2AudioDriver
//  clients. Selected via RuntimeConfig::audio_factory:
//    config.audio_factory = [](rex::runtime::FunctionDispatcher* fd) {
//        return xaudio2::XAudio2AudioSystem::Create(fd);
//    };
//
//  Note: device creation is lazy (EnsureEngine, on first CreateDriver),
//  not in Initialize(). AudioSystem::Initialize() runs on the audio
//  worker thread during early boot, and opening the audio device there
//  puts it on the boot-critical path. Deferring to CreateDriver matches
//  the SDL backend's timing.
// =====================================================================
#pragma once

#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <windows.h>
#include <xaudio2.h>
#include <wrl/client.h>

#include <rex/audio/audio_system.h>

namespace xaudio2 {

class XAudio2AudioSystem : public rex::audio::AudioSystem {
 public:
  explicit XAudio2AudioSystem(rex::runtime::FunctionDispatcher* function_dispatcher);
  ~XAudio2AudioSystem() override;

  static std::unique_ptr<rex::system::IAudioSystem> Create(
      rex::runtime::FunctionDispatcher* dispatcher);

 protected:
  void Initialize() override;

  rex::X_STATUS CreateDriver(size_t index, rex::thread::Semaphore* semaphore,
                             rex::audio::AudioDriver** out_driver) override;
  void DestroyDriver(rex::audio::AudioDriver* driver) override;

 private:
  // Lazily create the XAudio2 device + mastering voice on first use. Returns
  // false if the device could not be created. Safe to call repeatedly.
  bool EnsureEngine();

  Microsoft::WRL::ComPtr<IXAudio2> xaudio2_;
  IXAudio2MasteringVoice* mastering_voice_ = nullptr;
  bool com_initialized_ = false;
};

}  // namespace xaudio2
