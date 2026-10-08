#pragma once

#include <cstddef>
#include <cstdint>

// v1 hash of Xenos ucode with every vfetch's d0 bits 19-31, d1 and d2 cleared, the fields the D3D runtime
// patches at bind time and then some. bigEndian selects the input word order (containers BE, dumps LE).
uint64_t ucodeFingerprint(const uint32_t* code, size_t dwordCount, bool bigEndian);

// v2 hash, the pack key (pack_contract.md 3.2): clears exactly the declaration-dependent vfetch fields.
// Container ucode and the bind-time-patched runtime ucode of the same shader give the same value.
uint64_t ucodeFingerprint2(const uint32_t* code, size_t dwordCount, bool bigEndian);

// Declaration-dependent vfetch bits, cleared by ucodeFingerprint2 and never read by the X3 translator.
constexpr uint32_t kVfetchKeepMask0 = 0xC00FFFFFu;
constexpr uint32_t kVfetchKeepMask1 = 0xBFC08000u;
constexpr uint32_t kVfetchKeepMask2 = 0x80000000u;

struct VfetchInfo
{
    uint32_t format;     // dword1 bits 16-21
    uint32_t stride;     // dword2 bits 0-7 (dwords)
    uint32_t isSigned;   // formatCompAll
    uint32_t isInteger;  // numFormatAll
};

// Invokes visit for every vertex-fetch instruction found by the same walk the fingerprint uses.
void ucodeVisitVfetches(const uint32_t* code, size_t dwordCount, bool bigEndian,
    void (*visit)(const VfetchInfo&, void*), void* context);

// Invokes visit with the slot index (instruction address) and the 3 native-order dwords of every vertex fetch,
// in control-flow walk order. This order defines the vfetch ordinals of the X3 layout table.
void ucodeVisitVfetchSlots(const uint32_t* code, size_t dwordCount, bool bigEndian,
    void (*visit)(uint32_t slot, const uint32_t* instructionDwords, void*), void* context);

// Invokes visit with a pointer to the 3 native-order dwords of every fetch instruction slot, in control-flow order.
// Check the opcode in the low 5 bits of dword 0 before reinterpreting the pointer as a fetch instruction.
void ucodeVisitFetchSlots(const uint32_t* code, size_t dwordCount, bool bigEndian,
    void (*visit)(const uint32_t* instructionDwords, void*), void* context);

// PIXEL_SHADER_OUTPUT_* mask (COLOR0-3 = bits 0-3, DEPTH = bit 4) scanned from a pixel shader's exports.
// Container-less generation uses it in place of the container's outputs field.
uint32_t ucodePsExportMask(const uint32_t* code, size_t dwordCount, bool bigEndian);
