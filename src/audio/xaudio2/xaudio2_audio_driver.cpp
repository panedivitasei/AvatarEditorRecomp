// =====================================================================
//  xaudio2_audio_driver.cpp: XAudio2 (native Windows) audio backend.
//  Ported from masterspike52/reDAHM (redahm_engine/audio).
// =====================================================================
#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <windows.h>

#include <mmreg.h>  // WAVEFORMATEXTENSIBLE

#include <algorithm>
#include <cmath>
#include <cstring>

#include <rex/audio/xaudio2/xaudio2_audio_driver.h>

#include <rex/assert.h>
#include <rex/audio/conversion.h>
#include <rex/audio/downmix.h>
#include <rex/chrono/clock.h>
#include <rex/cvar.h>
#include <rex/logging.h>

// The `audio_mute` cvar is defined by the SDL backend
// (sdl_audio_driver.cpp); reuse it here rather than redefining.
REXCVAR_DECLARE(bool, audio_mute);

namespace xaudio2 {


XAudio2AudioDriver::XAudio2AudioDriver(rex::memory::Memory* memory, IXAudio2* xaudio2,
                                       uint8_t device_channels,
                                       std::function<void()> on_buffer_completed)
    : AudioDriver(memory),
      xaudio2_(xaudio2),
      device_channels_(device_channels),
      on_buffer_completed_(std::move(on_buffer_completed)) {}

XAudio2AudioDriver::~XAudio2AudioDriver() {
  assert_true(frames_queued_.empty());
  assert_true(frames_unused_.empty());
}

bool XAudio2AudioDriver::Initialize() {
  // WAVEFORMATEXTENSIBLE with an explicit speaker mask (FM2 conversion guide,
  // step 1): plain WAVEFORMATEX leaves the >2ch channel->speaker mapping to
  // the audio driver's guess.
  static constexpr GUID kIeeeFloatSubformat = {
      0x00000003, 0x0000, 0x0010, {0x80, 0x00, 0x00, 0xaa, 0x00, 0x38, 0x9b, 0x71}};
  WAVEFORMATEXTENSIBLE fmt = {};
  fmt.Format.wFormatTag = WAVE_FORMAT_EXTENSIBLE;
  fmt.Format.nChannels = device_channels_;
  fmt.Format.nSamplesPerSec = frame_frequency_;
  fmt.Format.wBitsPerSample = 32;
  fmt.Format.nBlockAlign = fmt.Format.nChannels * (fmt.Format.wBitsPerSample / 8);
  fmt.Format.nAvgBytesPerSec = fmt.Format.nSamplesPerSec * fmt.Format.nBlockAlign;
  fmt.Format.cbSize = sizeof(WAVEFORMATEXTENSIBLE) - sizeof(WAVEFORMATEX);
  fmt.Samples.wValidBitsPerSample = 32;
  // SPEAKER_FRONT_LEFT..._BACK_RIGHT (5.1) = 0x3F; FL|FR (stereo) = 0x3.
  fmt.dwChannelMask = device_channels_ == 6 ? 0x3Fu : 0x3u;
  fmt.SubFormat = kIeeeFloatSubformat;


  HRESULT hr = xaudio2_->CreateSourceVoice(&source_voice_, &fmt.Format, 0,
                                           XAUDIO2_DEFAULT_FREQ_RATIO, this);
  if (FAILED(hr)) {
    REXAPU_ERROR("XAudio2: CreateSourceVoice failed: {:08X}", static_cast<uint32_t>(hr));
    return false;
  }

  // Track guest clock scaling (FM2 guide, step 1.5). The scalar is 1.0 in
  // normal operation (src/core/clock.cpp) but a scaled guest clock must pitch
  // the source voice with it or the queue drifts into under/overrun.
  source_voice_->SetFrequencyRatio(
      static_cast<float>(rex::chrono::Clock::guest_time_scalar()));

  hr = source_voice_->Start(0);
  if (FAILED(hr)) {
    REXAPU_ERROR("XAudio2: Start failed: {:08X}", static_cast<uint32_t>(hr));
    source_voice_->DestroyVoice();
    source_voice_ = nullptr;
    return false;
  }

  return true;
}

void XAudio2AudioDriver::Shutdown() {
  if (source_voice_) {
    source_voice_->Stop(0);
    source_voice_->FlushSourceBuffers();
    source_voice_->DestroyVoice();
    source_voice_ = nullptr;
  }
  std::unique_lock<std::mutex> guard(frames_mutex_);
  while (!frames_unused_.empty()) {
    delete[] frames_unused_.top();
    frames_unused_.pop();
  }
  while (!frames_queued_.empty()) {
    delete[] frames_queued_.front();
    frames_queued_.pop();
  }
}

void XAudio2AudioDriver::SubmitFrame(uint32_t frame_ptr) {
  const auto input_frame = memory_->TranslateVirtual<float*>(frame_ptr);

  float* converted;
  {
    std::unique_lock<std::mutex> guard(frames_mutex_);
    if (frames_unused_.empty()) {
      converted = new float[frame_samples_];
    } else {
      converted = frames_unused_.top();
      frames_unused_.pop();
    }
  }

  if (REXCVAR_GET(audio_mute)) {
    std::memset(converted, 0, frame_samples_ * sizeof(float));
  } else {
    // Snapshot once. A change mid-frame would split it across two mixes.
    const rex::audio::StereoFold fold = rex::audio::GetStereoFold();
    const rex::audio::SurroundMix mix = rex::audio::GetSurroundMix();
    const float gain = rex::audio::GetOutputGain();
    switch (device_channels_) {
      case 2:
        rex::audio::conversion::sequential_6_BE_to_interleaved_2_LE(converted, input_frame,
                                                                    channel_samples_, fold, gain);
        break;
      case 6:
        rex::audio::conversion::sequential_6_BE_to_interleaved_6_LE(converted, input_frame,
                                                                    channel_samples_, mix, gain);
        break;
      default:
        assert_unhandled_case(device_channels_);
        break;
    }
  }

  {
    std::unique_lock<std::mutex> guard(frames_mutex_);
    frames_queued_.push(converted);
  }

  XAUDIO2_BUFFER buffer = {};
  buffer.AudioBytes = static_cast<UINT32>(channel_samples_ * device_channels_ * sizeof(float));
  buffer.pAudioData = reinterpret_cast<const BYTE*>(converted);
  buffer.pContext = converted;

  if (source_voice_) {
    HRESULT hr = source_voice_->SubmitSourceBuffer(&buffer);
    if (FAILED(hr)) {
      REXAPU_ERROR("XAudio2: SubmitSourceBuffer failed: {:08X}", static_cast<uint32_t>(hr));
      std::unique_lock<std::mutex> guard(frames_mutex_);
      frames_queued_.pop();
      frames_unused_.push(converted);
    }
  }
}

STDMETHODIMP_(void) XAudio2AudioDriver::OnBufferEnd(void* buffer_context) {
  auto* buffer = static_cast<float*>(buffer_context);
  {
    std::unique_lock<std::mutex> guard(frames_mutex_);
    if (!frames_queued_.empty() && frames_queued_.front() == buffer) {
      frames_queued_.pop();
    }
    frames_unused_.push(buffer);
  }
  if (on_buffer_completed_) {
    on_buffer_completed_();
  }
}

}  // namespace xaudio2
