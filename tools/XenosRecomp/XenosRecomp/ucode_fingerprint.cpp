#include "ucode_fingerprint.h"
#include "shader_code.h"

#include <vector>

// Mirrors the control-flow walk in ShaderRecompiler::recompile: CF
// instructions come in pairs packed into 3 dwords; exec-type CF entries
// address clauses of 3-dword instruction slots whose fetch/alu kind is
// given by the low bit of each 2-bit sequence field.
//
// Calls onSlot(slotDwordIndex, isFetch) for every instruction slot in every
// exec clause (fetch and ALU alike).
template<typename OnSlot>
static void walkExecSlots(std::vector<uint32_t>& words, const OnSlot& onSlot)
{
    union
    {
        ControlFlowInstruction controlFlow[2];
        struct
        {
            uint32_t code0;
            uint32_t code1;
            uint32_t code2;
            uint32_t code3;
        };
    };

    // Pass 1: find where the control-flow region ends (first clause address).
    // The bound shrinks while walking, like the recompiler's loop, so clause
    // data is never parsed as control flow.
    uint32_t instrSizeDwords = uint32_t(words.size());
    {
        uint32_t addr = 0;
        while (addr + 3 <= instrSizeDwords)
        {
            code0 = words[addr];
            code1 = words[addr + 1] & 0xFFFF;
            code2 = (words[addr + 1] >> 16) | (words[addr + 2] << 16);
            code3 = words[addr + 2] >> 16;

            for (auto& cfInstr : controlFlow)
            {
                uint32_t clause = 0;
                switch (cfInstr.opcode)
                {
                case ControlFlowOpcode::Exec:
                case ControlFlowOpcode::ExecEnd:
                    clause = cfInstr.exec.address;
                    break;
                case ControlFlowOpcode::CondExec:
                case ControlFlowOpcode::CondExecEnd:
                case ControlFlowOpcode::CondExecPredClean:
                case ControlFlowOpcode::CondExecPredCleanEnd:
                    clause = cfInstr.condExec.address;
                    break;
                case ControlFlowOpcode::CondExecPred:
                case ControlFlowOpcode::CondExecPredEnd:
                    clause = cfInstr.condExecPred.address;
                    break;
                }
                if (clause != 0 && clause * 3 < instrSizeDwords)
                    instrSizeDwords = clause * 3;
            }
            addr += 3;
        }
    }

    // Pass 2: visit every instruction slot inside every exec clause.
    {
        uint32_t addr = 0;
        while (addr + 3 <= instrSizeDwords)
        {
            code0 = words[addr];
            code1 = words[addr + 1] & 0xFFFF;
            code2 = (words[addr + 1] >> 16) | (words[addr + 2] << 16);
            code3 = words[addr + 2] >> 16;

            for (auto& cfInstr : controlFlow)
            {
                uint32_t clause = 0, count = 0, sequence = 0;
                switch (cfInstr.opcode)
                {
                case ControlFlowOpcode::Exec:
                case ControlFlowOpcode::ExecEnd:
                    clause = cfInstr.exec.address;
                    count = cfInstr.exec.count;
                    sequence = cfInstr.exec.sequence;
                    break;
                case ControlFlowOpcode::CondExec:
                case ControlFlowOpcode::CondExecEnd:
                case ControlFlowOpcode::CondExecPredClean:
                case ControlFlowOpcode::CondExecPredCleanEnd:
                    clause = cfInstr.condExec.address;
                    count = cfInstr.condExec.count;
                    sequence = cfInstr.condExec.sequence;
                    break;
                case ControlFlowOpcode::CondExecPred:
                case ControlFlowOpcode::CondExecPredEnd:
                    clause = cfInstr.condExecPred.address;
                    count = cfInstr.condExecPred.count;
                    sequence = cfInstr.condExecPred.sequence;
                    break;
                }

                uint32_t seq = sequence;
                for (uint32_t i = 0; i < count; i++, seq >>= 2)
                {
                    size_t slot = (size_t(clause) + i) * 3;
                    if (slot + 3 > words.size())
                        break;

                    onSlot(slot, (seq & 0x1) != 0);
                }
            }
            addr += 3;
        }
    }
}

static std::vector<uint32_t> loadWords(const uint32_t* code, size_t dwordCount, bool bigEndian)
{
    std::vector<uint32_t> words(dwordCount);
    for (size_t i = 0; i < dwordCount; i++)
        words[i] = bigEndian ? __builtin_bswap32(code[i]) : code[i];
    return words;
}

uint64_t ucodeFingerprint(const uint32_t* code, size_t dwordCount, bool bigEndian)
{
    auto words = loadWords(code, dwordCount, bigEndian);

    walkExecSlots(words, [&](size_t slot, bool isFetch)
        {
            if (!isFetch || (words[slot] & 0x1F) != uint32_t(FetchOpcode::VertexFetch))
                return;
            // Keep opcode + src/dst register routing (bits 0-18 of dword0);
            // drop the fetch constant index and everything in dwords 1-2
            // (format, signedness, stride, offset), the fields the Xbox 360
            // D3D runtime patches at shader-bind time.
            words[slot] &= 0x0007FFFF;
            words[slot + 1] = 0;
            words[slot + 2] = 0;
        });

    return XXH3_64bits(words.data(), words.size() * sizeof(uint32_t));
}

void ucodeVisitVfetches(const uint32_t* code, size_t dwordCount, bool bigEndian,
    void (*visit)(const VfetchInfo&, void*), void* context)
{
    auto words = loadWords(code, dwordCount, bigEndian);

    walkExecSlots(words, [&](size_t slot, bool isFetch)
        {
            if (!isFetch || (words[slot] & 0x1F) != uint32_t(FetchOpcode::VertexFetch))
                return;
            VfetchInfo info{};
            info.format = (words[slot + 1] >> 16) & 0x3F;
            info.stride = words[slot + 2] & 0xFF;
            info.isSigned = (words[slot + 1] >> 12) & 0x1;
            info.isInteger = (words[slot + 1] >> 13) & 0x1;
            visit(info, context);
        });
}

void ucodeVisitFetchSlots(const uint32_t* code, size_t dwordCount, bool bigEndian,
    void (*visit)(const uint32_t* instructionDwords, void*), void* context)
{
    auto words = loadWords(code, dwordCount, bigEndian);

    walkExecSlots(words, [&](size_t slot, bool isFetch)
        {
            if (isFetch)
                visit(&words[slot], context);
        });
}

uint32_t ucodePsExportMask(const uint32_t* code, size_t dwordCount, bool bigEndian)
{
    auto words = loadWords(code, dwordCount, bigEndian);
    uint32_t mask = 0;

    walkExecSlots(words, [&](size_t slot, bool isFetch)
        {
            if (isFetch)
                return;

            union
            {
                AluInstruction alu;
                struct { uint32_t d0, d1, d2; };
            };
            d0 = words[slot];
            d1 = words[slot + 1];
            d2 = words[slot + 2];

            // Vector and scalar results of an export instruction both write
            // the export register selected by vectorDest (see the
            // recompiler's exportRegister handling).
            if (!alu.exportData)
                return;
            if (alu.vectorDest <= 3) // PSColor0-3 -> PIXEL_SHADER_OUTPUT_COLOR0-3
                mask |= 1u << alu.vectorDest;
            else if (ExportRegister(uint32_t(alu.vectorDest)) == ExportRegister::PSDepth)
                mask |= 0x10; // PIXEL_SHADER_OUTPUT_DEPTH
        });

    return mask;
}
