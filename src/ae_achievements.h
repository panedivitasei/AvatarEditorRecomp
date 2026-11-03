// The Avatar Editor's own achievements, registered with the SDK and unlocked from the hooks that see each event.
#pragma once

#include <cstddef>
#include <cstdint>

namespace rex::avatars {
struct X_AVATAR_METADATA;
}

namespace ae::achievements {

// High range so the ids never meet XDBF ones.
enum Id : uint32_t {
  kHello = 0x10001,
  kFreshFit,
  kWindowShopper,
  kTreatYourself,
  kGlowUp,
  kClosetHoarder,
  kStrikeAPose,
};

// Registers the catalog; call once the XEX is loaded.
void Register();

// First-time unlock with its toast; a no-op when already unlocked.
void Unlock(Id id);

// Compares a saved manifest against the one it replaces for Fresh Fit and Glow Up.
void OnManifestSaved(const rex::avatars::X_AVATAR_METADATA& before,
                     const rex::avatars::X_AVATAR_METADATA& after);

// Closet Hoarder at 25 items.
void OnClosetCount(size_t count);

}  // namespace ae::achievements
