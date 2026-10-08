#pragma once

// Public interface of xenosrecomp_core: the translator, the CC2 shader pack and the disk cache
// (docs/research/pack_contract.md). The pack build and the renderer link the same library.

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <memory>
#include <span>
#include <string>
#include <string_view>
#include <vector>

namespace xenosrecomp {

enum class Stage : uint8_t { kVertex = 0, kPixel = 1 };
enum TargetBits : uint32_t { kDxil = 1, kSpirv = 2, kTrim = 4 };

// Pack entry stages; 0 and 1 match Stage.
enum PackStage : uint8_t { kPackVS = 0, kPackPS = 1, kPackGS = 2, kPackBlitVS = 3, kPackBlitPS = 4 };

// ABI constants compiled into every translated shader, checked field by field at pack load.
constexpr uint32_t kB0Size = 272;
constexpr uint32_t kPushSlots = 9;
constexpr uint32_t kTextureHeap = 16384;
constexpr uint32_t kSamplerHeap = 1024;
constexpr uint32_t kSharedMemSize = 0x8000000;
constexpr uint32_t kMaxVfetch = 32;

constexpr uint16_t kPackVersion = 1;
constexpr uint16_t kFingerprintVersion = 2;

// ---- Vfetch layout table (pack_contract.md 2.4) ----

struct XeVfetchRecord {
  uint32_t word0;          // bits 0-7 fetch dword index, 8-13 format, 14 signed, 15 integer, 16 rf no-zero, 31 valid
  uint32_t stride_dwords;
  int32_t offset_dwords;
  uint32_t dst_swizzle;    // 3 bits per component: 0-3 xyzw, 4 = 0, 5 = 1, 7 = keep
};
static_assert(sizeof(XeVfetchRecord) == 16);

constexpr uint32_t kVfetchSigned = 1u << 14;
constexpr uint32_t kVfetchInteger = 1u << 15;
constexpr uint32_t kVfetchRfNoZero = 1u << 16;
constexpr uint32_t kVfetchValid = 1u << 31;

// ---- Reflection record (pack_contract.md 2.3) ----

enum ReflFlags : uint16_t {
  kReflHasTrim = 1 << 0,
  kReflWritesPointSize = 1 << 1,
  kReflWritesDepth = 1 << 2,
  kReflUsesKill = 1 << 3,
  kReflUsesPixelPos = 1 << 4,
  kReflUsesFace = 1 << 5,
  kReflContainerless = 1 << 6,
  kReflPcMachineFlow = 1 << 7,
};

struct XeReflHeader {
  uint8_t stage;               // 0 = VS, 1 = PS; built-ins carry their PackStage
  uint8_t float_mode;          // 0 = static bitmap, 1 = dynamic (upload all 256)
  uint16_t flags;              // ReflFlags
  uint32_t float_bitmap[8];    // bit N = register N read; all ones when float_mode = 1
  uint32_t bool_mask;          // g_Booleans bits tested (VS bools 0-31, PS bools 128-143 in bits 16-31)
  uint16_t vs_written_ovar;    // VS: oVar mask kept by the .vst trim
  uint16_t ps_read_ivar;       // PS: iVar registers declared
  uint8_t ps_outputs;          // COLOR0-3 bits 0-3, DEPTH bit 4
  uint8_t pixel_pos_reg;       // PS: guest register given the pixel position, 0xFF if none
  uint8_t vfetch_count;
  uint8_t binding_count;
  uint8_t literal_count;
  uint8_t interp_count;
  uint8_t _pad[14];
};
static_assert(sizeof(XeReflHeader) == 64);

struct XeReflVfetch {
  uint16_t slot_address;       // clause instruction slot, the container VertexElement key
  uint8_t usage;               // DeclUsage, 0xFF if the container names none
  uint8_t usage_index;
};
static_assert(sizeof(XeReflVfetch) == 4);

struct XeReflBinding {
  uint8_t kind;                // 0 = texture, 1 = sampler
  uint8_t fetch_constant;
  uint8_t dimension;           // texture: 1 = 2D/1D, 2 = 3D, 3 = cube; sampler: 0
  uint8_t is_signed;
  uint8_t mag, min, mip, aniso;  // sampler override key; 0xFF = take it from the fetch constant; textures 0
};
static_assert(sizeof(XeReflBinding) == 8);

struct XeReflInterp {
  uint8_t reg, usage, usage_index, _pad;
};
static_assert(sizeof(XeReflInterp) == 4);

// Literal values belong to the shader object: two containers with one microcode can differ only here, and every
// literal register is read from b1. The renderer overlays per-object values (ReadLiterals) unless V1 shows the
// device shadow already holds them (pack_contract.md O-1).
enum ReflLiteralFlags : uint16_t { kLiteralRead = 1 << 0, kLiteralNoValue = 1 << 1 };

struct XeReflLiteral {
  uint16_t reg;                // stage-local register (PS registers 0-255)
  uint16_t flags;              // ReflLiteralFlags
  uint32_t value[4];           // raw bits
};
static_assert(sizeof(XeReflLiteral) == 20);

// Arrays follow the header in this order: vfetch, binding, interp, literal.
struct ReflectionView {
  const XeReflHeader* header = nullptr;
  std::span<const XeReflVfetch> vfetches;
  std::span<const XeReflBinding> bindings;
  std::span<const XeReflInterp> interps;
  std::span<const XeReflLiteral> literals;
};

size_t ReflectionSize(const XeReflHeader& header);
bool ParseReflection(std::span<const uint8_t> bytes, ReflectionView& out);

// ---- Pack file layout (pack_contract.md 4.2) ----

struct PackBlob {
  uint32_t offset;             // relative to blob_offset
  uint32_t csize;              // compressed size, 0 = absent
};

struct PackHeader {
  char magic[8];               // "CC2XPACK"
  uint16_t version;
  uint16_t fingerprint_ver;
  uint32_t flags;              // bit0 has_dxil, bit1 has_spirv, bit2 has_trim
  uint64_t abi_hash;
  uint32_t b0_size;
  uint32_t push_slots;
  uint32_t texture_heap;
  uint32_t sampler_heap;
  uint32_t shared_mem_size;
  uint32_t max_vfetch;
  uint32_t entry_count;
  uint32_t alias_count;
  uint64_t index_offset;
  uint64_t alias_offset;
  uint64_t refl_offset;
  uint64_t refl_size;
  uint64_t blob_offset;
  uint64_t blob_size;
  uint64_t dict_offset;
  uint32_t dict_size;
  uint32_t builtin_count;
  uint64_t file_xxh3;          // XXH3 of bytes [0x80, end)
};
static_assert(sizeof(PackHeader) == 128);

enum PackFlags : uint32_t { kPackHasDxil = 1, kPackHasSpirv = 2, kPackHasTrim = 4 };
enum EntryFlags : uint8_t { kEntryHasTrim = 1, kEntryContainerless = 2 };

struct PackEntry {
  uint64_t fp2;                // built-ins: XXH3 of the name
  uint8_t stage;               // PackStage
  uint8_t flags;               // EntryFlags
  uint16_t _pad;
  uint32_t refl_offset;        // relative to the header's refl_offset
  PackBlob dxil, spirv, dxil_trim, spirv_trim;
};
static_assert(sizeof(PackEntry) == 48);

struct PackAlias {
  uint64_t runtime_ucode_hash;
  uint32_t entry_index;
  uint8_t stage;
  uint8_t _pad[3];
};
static_assert(sizeof(PackAlias) == 16);

enum class BlobKind { kDxil, kSpirv, kDxilTrim, kSpirvTrim };

// Read-only view of a pack or a single-entry .xsc cache file.
class Pack {
 public:
  Pack() = default;
  Pack(const Pack&) = delete;
  Pack& operator=(const Pack&) = delete;
  Pack(Pack&&) = default;
  Pack& operator=(Pack&&) = default;

  // Validates magic, versions, the ABI constants, abi_hash (unless check_abi is false) and file_xxh3.
  // The bytes must outlive the Pack; error names the first failing field.
  bool Open(std::span<const uint8_t> bytes, std::string* error, bool check_abi = true);
  bool OpenFile(const std::filesystem::path& path, std::string* error, bool check_abi = true);

  const PackHeader& header() const { return *header_; }
  std::span<const PackEntry> entries() const { return entries_; }
  std::span<const PackAlias> aliases() const { return aliases_; }

  const PackEntry* Find(uint8_t stage, uint64_t fp2) const;
  const PackEntry* FindAlias(uint64_t runtime_ucode_hash) const;
  const PackEntry* FindBuiltin(uint8_t stage, std::string_view name) const;

  std::span<const uint8_t> Reflection(const PackEntry& entry) const;
  // Decompresses one blob; an absent blob gives an empty vector and true.
  bool Decompress(const PackEntry& entry, BlobKind kind, std::vector<uint8_t>& out, std::string* error) const;

 private:
  std::vector<uint8_t> owned_;
  std::span<const uint8_t> bytes_;
  const PackHeader* header_ = nullptr;
  std::span<const PackEntry> entries_;
  std::span<const PackAlias> aliases_;
};

// One shader for the pack writer, blobs uncompressed.
struct PackInput {
  uint8_t stage = 0;
  uint64_t fp2 = 0;
  uint8_t flags = 0;
  std::vector<uint8_t> reflection;
  std::vector<uint8_t> dxil, spirv, dxil_trim, spirv_trim;
  std::vector<uint64_t> aliases;   // runtime_ucode_hash values that resolve to this entry
};

// Serializes a pack (entries sorted by (stage, fp2), aliases by hash), one zstd frame per blob.
std::vector<uint8_t> BuildPack(std::vector<PackInput> inputs, int zstd_level, std::string* error);

// ---- Translation (pack_contract.md 5.1) ----

struct TranslateInput {
  Stage stage = Stage::kVertex;
  std::span<const uint32_t> ucode;       // guest byte order, as it sits in guest memory
  std::span<const uint8_t> container;    // the object's embedded container (metadata only); empty = container-less
  std::span<const uint8_t> physical;     // the object's physical data (literal bits); empty = after the container
  uint32_t targets = kDxil | kSpirv | kTrim;  // TargetBits; 0 = HLSL and reflection only
  bool keep_hlsl = false;
};

struct TranslateResult {
  bool ok = false;
  std::string error;                     // first failure, same text the pack build prints
  uint64_t fp2 = 0;
  uint64_t runtime_ucode_hash = 0;
  std::vector<uint8_t> dxil, spirv, dxil_trim, spirv_trim;
  std::vector<uint8_t> reflection;       // serialized 2.3 record
  std::string hlsl, hlsl_trim;           // kept only when keep_hlsl is set
};

uint64_t AbiHash();                                                         // the value the pack header must match
uint64_t Fingerprint(std::span<const uint32_t> ucode, bool big_endian);     // fp2
uint32_t ReadVfetchRecords(std::span<const uint32_t> patched_ucode, bool big_endian,
                           std::span<XeVfetchRecord> out);                  // returns the vfetch count
TranslateResult Translate(const TranslateInput& in);                        // thread-safe, one DXC per thread

// Def literals of a shader object from its container's definition table and its physical data (empty = the
// physical part follows the container). Literals outside the data come back flagged kLiteralNoValue.
std::vector<XeReflLiteral> ReadLiterals(Stage stage, std::span<const uint8_t> container, std::span<const uint8_t> physical);

// The embedded rexglue_shader_common.h every translation includes.
std::string_view CommonHeader();

// Built-in GS and blit shaders (pack_contract.md 1.10), compiled with the same DXC setup.
struct BuiltinShader {
  const char* name;
  uint8_t stage;               // PackStage
  const char* source;
};
std::span<const BuiltinShader> Builtins();
uint64_t BuiltinKey(std::string_view name);

// ---- Disk cache (pack_contract.md 5.3) ----

// <root>/<abi_hash:016X>; root is <user data dir>/gpu_shader_cache or the cvar override.
std::filesystem::path CacheDirectory(const std::filesystem::path& root);
// <dir>/<vs|ps>_<fp2:016X>.xsc
std::filesystem::path CacheFile(const std::filesystem::path& dir, Stage stage, uint64_t fp2);
// Writes a single-entry pack through a temp name and a rename, so a crash never leaves a torn file.
bool CacheStore(const std::filesystem::path& dir, Stage stage, const TranslateResult& result, std::string* error);
// Opens <dir>/<stage>_<fp2>.xsc with the full pack checks; false on a miss or a stale file.
bool CacheLoad(const std::filesystem::path& dir, Stage stage, uint64_t fp2, Pack& out, std::string* error);
// Writes <dir>/misses/<rt>.<vs|ps>.ucode and .container, the inputs a pack rebuild needs.
bool CacheWriteMiss(const std::filesystem::path& dir, Stage stage, uint64_t runtime_ucode_hash,
                    std::span<const uint32_t> ucode, std::span<const uint8_t> container, std::string* error);

}  // namespace xenosrecomp
