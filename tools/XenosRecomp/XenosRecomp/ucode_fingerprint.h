#pragma once

// Hash of Xenos ucode that is invariant to the vertex-fetch fields the Xbox
// 360 D3D runtime patches at shader-bind time (fetch constant index, format,
// signedness, stride, offset). A runtime-captured shader and the container it
// was created from produce the same fingerprint; byte hashes never match for
// vertex shaders because of that patching.
//
// Works on raw ucode; no container metadata needed. bigEndian selects the
// word order of the input (containers store BE, runtime dumps are LE).
uint64_t ucodeFingerprint(const uint32_t* code, size_t dwordCount, bool bigEndian);

struct VfetchInfo
{
    uint32_t format;     // dword1 bits 16-21
    uint32_t stride;     // dword2 bits 0-7 (dwords)
    uint32_t isSigned;   // formatCompAll
    uint32_t isInteger;  // numFormatAll
};

// Invokes visit for every vertex-fetch instruction found by the same walk
// the fingerprint uses.
void ucodeVisitVfetches(const uint32_t* code, size_t dwordCount, bool bigEndian,
    void (*visit)(const VfetchInfo&, void*), void* context);

// Invokes visit with a pointer to the 3 native-order dwords of every fetch
// instruction slot (vertex and texture fetches alike), in control-flow order.
// Reinterpret the pointer as VertexFetchInstruction/TextureFetchInstruction
// after checking the opcode in the low 5 bits of dword 0.
void ucodeVisitFetchSlots(const uint32_t* code, size_t dwordCount, bool bigEndian,
    void (*visit)(const uint32_t* instructionDwords, void*), void* context);

// Scans a pixel shader's ALU exports and returns the PIXEL_SHADER_OUTPUT_*
// mask (COLOR0-3 = bits 0-3, DEPTH = bit 4), the containers' PixelShader
// outputs field, derived from raw ucode for container-less generation.
uint32_t ucodePsExportMask(const uint32_t* code, size_t dwordCount, bool bigEndian);
