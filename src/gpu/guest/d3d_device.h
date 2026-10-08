// Guest-memory layout of the XDK D3D software structs the Avatar Editor's statically linked runtime uses.
// Public offsets follow XDK d3d9.h / d3d9gpu.h / d3d9types.h; private ones were read off the AE IDB and each carries its evidence.
#pragma once

#include <cstddef>
#include <cstdint>

#include <rex/graphics/registers.h>
#include <rex/graphics/xenos.h>
#include <rex/types.h>

namespace ae::gpu::guest {

namespace reg = rex::graphics::reg;
namespace xenos = rex::graphics::xenos;

using GuestPtr = rex::be<uint32_t>;
using GuestFn = rex::be<uint32_t>;

// Array bounds from the XDK headers, spelled out so the verifier can read them.
inline constexpr uint32_t kRenderStateCalls = 101;   // D3DRS_MAX / 4
inline constexpr uint32_t kSamplerStateCalls = 20;   // D3DSAMP_MAX / 4
inline constexpr uint32_t kFetchConstants = 32;      // GPU_FETCH_CONSTANTS
inline constexpr uint32_t kTextureFetchCount = 26;   // GPU_D3D_TEXTURE_FETCH_CONSTANT_COUNT, also D3DSAMP_MAXSAMPLERS
inline constexpr uint32_t kVertexFetchBase = 26;     // GPU_D3D_VERTEX_FETCH_CONSTANT_BASE
inline constexpr uint32_t kVertexFetchCount = 18;    // 3 * GPU_D3D_VERTEX_FETCH_CONSTANT_COUNT
inline constexpr uint32_t kFloatConstants = 256;     // GPU_D3D_{VERTEX,PIXEL}_CONSTANTF_COUNT
inline constexpr uint32_t kBoolConstantWords = 4;    // GPU_D3D_{VERTEX,PIXEL}_CONSTANTB_COUNT / 32
inline constexpr uint32_t kIntConstants = 16;        // GPU_D3D_{VERTEX,PIXEL}_CONSTANTI_COUNT
inline constexpr uint32_t kClipPlanes = 6;           // D3DMAXUSERCLIPPLANES
inline constexpr uint32_t kRenderTargets = 4;
inline constexpr uint32_t kStreams = 16;             // D3DMAXSTREAMS

// A GPU register held big-endian in guest memory, decoded through the SDK's register union.
template <typename R>
struct Reg {
  rex::be<uint32_t> raw;
  R get() const { R r; r.value = raw; return r; }
  void set(R r) { raw = r.value; }
};

// D3DTAGCOLLECTION: dirty bits for every GPU context register group.
struct TagCollection {
  rex::be<uint64_t> m_Mask[5];
};

// One GPUFETCH_CONSTANT slot: a texture fetch, or three vertex fetches of two dwords each.
struct FetchConstant {
  rex::be<uint32_t> dword[6];
  xenos::xe_gpu_fetch_group_t get() const { xenos::xe_gpu_fetch_group_t f; for (int i = 0; i < 6; ++i) (&f.dword_0)[i] = dword[i]; return f; }
};

// D3DConstants, the shadow of every GPU constant.
struct D3DConstants {
  FetchConstant Fetch[kFetchConstants];
  rex::be<float> VertexShaderF[kFloatConstants][4];
  rex::be<float> PixelShaderF[kFloatConstants][4];
  rex::be<uint32_t> VertexShaderB[kBoolConstantWords];
  rex::be<uint32_t> PixelShaderB[kBoolConstantWords];
  rex::be<uint32_t> VertexShaderI[kIntConstants];
  rex::be<uint32_t> PixelShaderI[kIntConstants];
};

// Vertex fetch i (0..17) lives in Fetch[26..31], two dwords per fetch.
inline const rex::be<uint32_t>* VertexFetch(const D3DConstants& c, uint32_t i) { return &c.Fetch[kVertexFetchBase + i / 3].dword[(i % 3) * 2]; }

// Register shadow packets, in d3d9gpu.h order.
struct DestinationPacket {
  Reg<reg::RB_SURFACE_INFO> SurfaceInfo;
  Reg<reg::RB_COLOR_INFO> Color0Info;
  Reg<reg::RB_DEPTH_INFO> DepthInfo;
  Reg<reg::RB_COLOR_INFO> Color1Info;
  Reg<reg::RB_COLOR_INFO> Color2Info;
  Reg<reg::RB_COLOR_INFO> Color3Info;
  rex::be<uint32_t> CoherDestBase[8];
  Reg<reg::PA_SC_SCREEN_SCISSOR_TL> ScreenScissorTL;
  Reg<reg::PA_SC_SCREEN_SCISSOR_BR> ScreenScissorBR;
};

struct WindowPacket {
  Reg<reg::PA_SC_WINDOW_OFFSET> WindowOffset;
  Reg<reg::PA_SC_WINDOW_SCISSOR_TL> WindowScissorTL;
  Reg<reg::PA_SC_WINDOW_SCISSOR_BR> WindowScissorBR;
};

struct ValuesPacket {
  Reg<reg::VGT_MAX_VTX_INDX> MaxVtxIndx;
  Reg<reg::VGT_MIN_VTX_INDX> MinVtxIndx;
  Reg<reg::VGT_INDX_OFFSET> IndxOffset;
  Reg<reg::VGT_MULTI_PRIM_IB_RESET_INDX> MultiPrimIbResetIndx;
  Reg<reg::RB_COLOR_MASK> ColorMask;
  rex::be<float> BlendRed;
  rex::be<float> BlendGreen;
  rex::be<float> BlendBlue;
  rex::be<float> BlendAlpha;
  rex::be<uint32_t> Unused[3];
  Reg<reg::RB_STENCILREFMASK> StencilRefMaskBF;
  Reg<reg::RB_STENCILREFMASK> StencilRefMask;
  rex::be<float> AlphaRef;
  rex::be<float> VportXScale;
  rex::be<float> VportXOffset;
  rex::be<float> VportYScale;
  rex::be<float> VportYOffset;
  rex::be<float> VportZScale;
  rex::be<float> VportZOffset;
};

struct ProgramPacket {
  Reg<reg::SQ_PROGRAM_CNTL> ProgramControl;
  Reg<reg::SQ_CONTEXT_MISC> ContextMisc;
  Reg<reg::SQ_INTERPOLATOR_CNTL> InterpolatorControl;
  rex::be<uint32_t> Wrapping0;  // SQ_WRAPPING_0
  rex::be<uint32_t> Wrapping1;  // SQ_WRAPPING_1
};

struct ControlPacket {
  Reg<reg::RB_DEPTHCONTROL> DepthControl;
  Reg<reg::RB_BLENDCONTROL> BlendControl0;
  Reg<reg::RB_COLORCONTROL> ColorControl;
  rex::be<uint32_t> HiControl;  // RB_TILECONTROL
  Reg<reg::PA_CL_CLIP_CNTL> ClipControl;
  Reg<reg::PA_SU_SC_MODE_CNTL> ModeControl;
  Reg<reg::PA_CL_VTE_CNTL> VteControl;
  rex::be<uint32_t> Unused;
  Reg<reg::RB_MODECONTROL> EdramModeControl;
  Reg<reg::RB_BLENDCONTROL> BlendControl1;
  Reg<reg::RB_BLENDCONTROL> BlendControl2;
  Reg<reg::RB_BLENDCONTROL> BlendControl3;
};

struct TessellatorPacket {
  Reg<reg::PA_SU_POINT_SIZE> PointSize;
  Reg<reg::PA_SU_POINT_MINMAX> PointMinMax;
  rex::be<uint32_t> LineControl;  // PA_SU_LINE_CNTL
  rex::be<uint32_t> Unused1;
  Reg<reg::VGT_OUTPUT_PATH_CNTL> OutputPathControl;
  Reg<reg::VGT_HOS_CNTL> HosControl;
  rex::be<float> HosMaxTessLevel;
  rex::be<float> HosMinTessLevel;
  rex::be<uint32_t> HosReuseDepth;
  rex::be<uint32_t> GroupPrimType;
  rex::be<uint32_t> GroupFirstDecr;
  rex::be<uint32_t> GroupDecr;
  rex::be<uint32_t> GroupVect0Control;
  rex::be<uint32_t> GroupVect1Control;
  rex::be<uint32_t> GroupVect0FmtControl;
  rex::be<uint32_t> GroupVect1FmtControl;
  rex::be<uint32_t> Unused2[2];
  Reg<reg::PA_SC_MPASS_PS_CNTL> MPassPsControl;
  Reg<reg::PA_SC_VIZ_QUERY> VizQuery;
  rex::be<uint32_t> Enhance;  // VGT_ENHANCE
};

struct MiscPacket {
  rex::be<uint32_t> ScLineControl;  // PA_SC_LINE_CNTL
  rex::be<uint32_t> AaConfig;       // PA_SC_AA_CONFIG
  Reg<reg::PA_SU_VTX_CNTL> VtxControl;
  rex::be<float> GbVertClipAdj;
  rex::be<float> GbVertDiscAdj;
  rex::be<float> GbHorzClipAdj;
  rex::be<float> GbHorzDiscAdj;
  Reg<reg::SQ_VS_CONST> VsConst;
  Reg<reg::SQ_PS_CONST> PsConst;
  rex::be<uint32_t> DebugMisc0;
  rex::be<uint32_t> DebugMisc1;
  rex::be<uint32_t> Unused1[5];
  rex::be<uint32_t> Unused2[2];
  rex::be<uint32_t> AaMask;  // PA_SC_AA_MASK
  rex::be<uint32_t> Unused3[3];
  rex::be<uint32_t> VertexReuseBlockControl;
  rex::be<uint32_t> OutDeallocControl;
  Reg<reg::RB_COPY_CONTROL> CopyControl;
  rex::be<uint32_t> CopyDestBase;
  Reg<reg::RB_COPY_DEST_PITCH> CopyDestPitch;
  Reg<reg::RB_COPY_DEST_INFO> CopyDestInfo;
  rex::be<uint32_t> HiClear;  // RB_TILE_CLEAR
  rex::be<uint32_t> DepthClear;
  rex::be<uint32_t> ColorClear;
  rex::be<uint32_t> ColorClearLo;
  rex::be<uint32_t> CopyFunc;
  rex::be<uint32_t> CopyRef;
  rex::be<uint32_t> CopyMask;
  rex::be<uint32_t> CopySurfaceSlice;
  rex::be<uint32_t> SampleCountControl;
  rex::be<uint32_t> SampleCountAddress;
};

struct PointPacket {
  rex::be<float> PolyOffsetFrontScale;
  rex::be<float> PolyOffsetFrontOffset;
  rex::be<float> PolyOffsetBackScale;
  rex::be<float> PolyOffsetBackOffset;
  rex::be<float> PointXRad;
  rex::be<float> PointYRad;
  rex::be<float> PointConstantSize;
  rex::be<float> PointCullRad;
};

// Viewport as the SetViewport body 0x921192D0 stores it, all floats.
struct GuestViewport {
  rex::be<float> X;
  rex::be<float> Y;
  rex::be<float> Width;
  rex::be<float> Height;
  rex::be<float> MinZ;
  rex::be<float> MaxZ;
};

// RECT as SetScissorRect 0x92118AB8 copies it in.
struct GuestRect {
  rex::be<int32_t> left;
  rex::be<int32_t> top;
  rex::be<int32_t> right;
  rex::be<int32_t> bottom;
};

// X_RTL_CRITICAL_SECTION: dispatcher header, lock count, recursion count, owning thread.
struct GuestCriticalSection {
  rex::be<uint32_t> dword[7];
};

// Fields without an XDK name are private to the runtime and carry a one-line evidence comment (AE addresses).
struct D3DDevice {
  TagCollection m_Pending;
  rex::be<uint64_t> m_Predicated_PendingMask2;
  GuestPtr m_pRing;
  GuestPtr m_pRingLimit;
  GuestPtr m_pRingGuarantee;
  rex::be<uint32_t> m_ReferenceCount;
  GuestFn m_SetRenderStateCall[kRenderStateCalls];
  GuestFn m_SetSamplerStateCall[kSamplerStateCalls];
  GuestFn m_GetRenderStateCall[kRenderStateCalls];
  GuestFn m_GetSamplerStateCall[kSamplerStateCalls];
  uint8_t _pad_0408[0x78];
  D3DConstants m_Constants;
  rex::be<float> m_ClipPlanes[kClipPlanes][4];
  DestinationPacket m_DestinationPacket;
  WindowPacket m_WindowPacket;
  ValuesPacket m_ValuesPacket;
  ProgramPacket m_ProgramPacket;
  ControlPacket m_ControlPacket;
  TessellatorPacket m_TessellatorPacket;
  MiscPacket m_MiscPacket;
  PointPacket m_PointPacket;
  uint8_t _pad_2A70[0x18];
  rex::be<uint32_t> thread_owner;           // private: Acquire/ReleaseThreadOwnership 0x92116010/0x92116050
  uint8_t _pad_2A8C[0x4];
  GuestPtr writeback_page;                  // private: SetFence 0x92125D90 completes into words 0-1, BlockOnPrimaryRange 0x92124C18 reads +0x3C
  uint8_t _pad_2A94[0x8];
  rex::be<uint32_t> fence_next;             // private: SetFence issues it and advances by 2 at 0x92125EC4; BlockOnFence 0x921252C8 compares against it
  uint8_t _pad_2AA0[0x1C];
  uint8_t device_flags0;                    // private: dev+0x2ABC, bit 0x20 = tiling bracket open (BeginTiling 0x9211F940)
  uint8_t device_flags1;                    // private: dev+0x2ABD, bit 2 = CPU completes fences (SetFence 0x92125EA4)
  uint8_t shader_flags;                     // private: dev+0x2ABE, bit 0x80 = VS variant 1 bound (SetVertexShader 0x9211C938)
  uint8_t tiling_flags;                     // private: dev+0x2ABF, bits 0x10/0x20 from BeginTiling's predicated path
  uint8_t predication_flags;                // private: dev+0x2AC0, bit 0x80 = pending predication marker (SetFence 0x92125DC8)
  uint8_t _pad_2AC1[0xB];
  rex::be<uint32_t> primary_write_index;    // private: AddCallsToPrimaryBuffer 0x92125808 stores it at 0x92125B70, then CP_RB_WPTR 0x7FC80714
  uint8_t _pad_2AD0[0x34];
  rex::be<uint32_t> queued_chunks;          // private: outstanding indirect buffers, added by QueueIndirectBuffer 0x92125ED0 at 0x92125FAC/0x9212602C/0x92126058; BlockUntilIdle 0x921267B8 spins on it
  rex::be<uint32_t> queued_chunks_lock;     // private: KfAcquireSpinLock target around the chunk list
  uint8_t _pad_2B0C[0x3CC];
  GuestPtr vertex_declaration;              // private: SetVertexDeclaration 0x9211CB50
  uint8_t _pad_2EDC[0x268];
  GuestPtr index_buffer;                    // private: SetIndices 0x92118DA8
  GuestPtr render_targets[kRenderTargets];  // private: SetRenderTarget 0x92119668 (dev+12616)
  GuestPtr depth_stencil;                   // private: SetDepthStencilSurface 0x921199B8 (dev+12632)
  GuestPtr stream_sources[kStreams];        // private: SetStreamSource 0x92118C00 (dev+12636)
  uint8_t _pad_319C[0x4];
  uint8_t stream_stride_dwords[kStreams];   // private: stride >> 2, SetStreamSource (dev+12704), DirectShaderPatch 0x92136BB0 hands it to the patcher
  GuestPtr textures[kTextureFetchCount];    // private: SetTexture 0x9211BA20 (dev+12720)
  GuestViewport viewport;                   // private: SetViewport body 0x921192D0 (dev+12824)
  uint8_t _pad_3230[0x4];
  GuestRect scissor_rect;                   // private: SetScissorRect 0x92118AB8 (dev+12852)
  GuestPtr pixel_shader;                    // private: SetPixelShader 0x9211C730 (dev+12868)
  GuestPtr vertex_shader;                   // private: SetVertexShader 0x9211C938 (dev+12872)
  uint8_t _pad_324C[0x8];
  rex::be<uint32_t> predication_select;     // private: dev+12884, SetPredication 0x9211F7D0
  uint8_t _pad_3258[0x4];
  rex::be<uint32_t> callback_base;          // private: dev+12892, the InsertCallback body 0x92126080 ors it into its register block
  uint8_t _pad_3260[0x4];
  GuestPtr tiling_target;                   // private: dev+12900, the surface BeginTiling 0x9211F940 tiles (RT0 or the depth buffer)
  GuestPtr tiling_saved_targets[5];         // private: dev+12904, RT0-3 and depth at BeginTiling
  rex::be<uint32_t> tile_count;             // private: dev+12924, BeginTiling
  GuestRect tile_rects[16];                 // private: dev+12928, BeginTiling copies the D3DRECTs
  uint8_t _pad_3380[0xAC];
  rex::be<uint32_t> tile_max_width;         // private: dev+13356, BeginTiling
  rex::be<uint32_t> tile_max_height;        // private: dev+13360
  rex::be<uint32_t> tiling_flags_arg;       // private: dev+13364, BeginTiling's Flags
  uint8_t _pad_3438[0x8];
  rex::be<float> tiling_clear_color[4];     // private: dev+13376, the clear colour EndTiling 0x9211FCB0 hands each non-final tile's Resolve
  rex::be<float> tiling_clear_z;            // private: dev+13392
  rex::be<uint32_t> tiling_clear_stencil;   // private: dev+13396
  uint8_t _pad_3458[0x8];
  rex::be<uint32_t> command_buffer_recording;  // private: dev+13408, non-zero while a command buffer records (QueueIndirectBuffer 0x92125EDC)
  uint8_t _pad_3464[0x2A9C];
};

// The runtime allocates 0x5F00 bytes per device (Direct3D_CreateDevice 0x92116428); the struct spans all of it.
static_assert(sizeof(D3DDevice) == 0x5F00, "D3DDevice span");


static_assert(sizeof(TagCollection) == 0x28, "D3DTAGCOLLECTION");
static_assert(sizeof(FetchConstant) == 0x18, "GPUFETCH_CONSTANT");
static_assert(sizeof(FetchConstant) == sizeof(xenos::xe_gpu_fetch_group_t), "fetch group size matches the SDK");
static_assert(sizeof(D3DConstants) == 0x23A0, "D3DConstants");
static_assert(sizeof(DestinationPacket) == 0x40, "GPU_DESTINATIONPACKET");
static_assert(sizeof(WindowPacket) == 0x0C, "GPU_WINDOWPACKET");
static_assert(sizeof(ValuesPacket) == 0x54, "GPU_VALUESPACKET");
static_assert(sizeof(ProgramPacket) == 0x14, "GPU_PROGRAMPACKET");
static_assert(sizeof(ControlPacket) == 0x30, "GPU_CONTROLPACKET");
static_assert(sizeof(TessellatorPacket) == 0x54, "GPU_TESSELLATORPACKET");
static_assert(sizeof(MiscPacket) == 0x98, "GPU_MISCPACKET");
static_assert(sizeof(PointPacket) == 0x20, "GPU_POINTPACKET");
static_assert(sizeof(GuestViewport) == 0x18, "viewport shadow");
static_assert(sizeof(GuestRect) == 0x10, "RECT");
static_assert(sizeof(GuestCriticalSection) == 0x1C, "X_RTL_CRITICAL_SECTION");

static_assert(offsetof(TagCollection, m_Mask) == 0x0, "TagCollection::m_Mask");
static_assert(offsetof(FetchConstant, dword) == 0x0, "FetchConstant::dword");
static_assert(offsetof(D3DConstants, Fetch) == 0x0, "D3DConstants::Fetch");
static_assert(offsetof(D3DConstants, VertexShaderF) == 0x300, "D3DConstants::VertexShaderF");
static_assert(offsetof(D3DConstants, PixelShaderF) == 0x1300, "D3DConstants::PixelShaderF");
static_assert(offsetof(D3DConstants, VertexShaderB) == 0x2300, "D3DConstants::VertexShaderB");
static_assert(offsetof(D3DConstants, PixelShaderB) == 0x2310, "D3DConstants::PixelShaderB");
static_assert(offsetof(D3DConstants, VertexShaderI) == 0x2320, "D3DConstants::VertexShaderI");
static_assert(offsetof(D3DConstants, PixelShaderI) == 0x2360, "D3DConstants::PixelShaderI");
static_assert(offsetof(DestinationPacket, SurfaceInfo) == 0x0, "DestinationPacket::SurfaceInfo");
static_assert(offsetof(DestinationPacket, Color0Info) == 0x4, "DestinationPacket::Color0Info");
static_assert(offsetof(DestinationPacket, DepthInfo) == 0x8, "DestinationPacket::DepthInfo");
static_assert(offsetof(DestinationPacket, Color1Info) == 0xC, "DestinationPacket::Color1Info");
static_assert(offsetof(DestinationPacket, Color2Info) == 0x10, "DestinationPacket::Color2Info");
static_assert(offsetof(DestinationPacket, Color3Info) == 0x14, "DestinationPacket::Color3Info");
static_assert(offsetof(DestinationPacket, CoherDestBase) == 0x18, "DestinationPacket::CoherDestBase");
static_assert(offsetof(DestinationPacket, ScreenScissorTL) == 0x38, "DestinationPacket::ScreenScissorTL");
static_assert(offsetof(DestinationPacket, ScreenScissorBR) == 0x3C, "DestinationPacket::ScreenScissorBR");
static_assert(offsetof(WindowPacket, WindowOffset) == 0x0, "WindowPacket::WindowOffset");
static_assert(offsetof(WindowPacket, WindowScissorTL) == 0x4, "WindowPacket::WindowScissorTL");
static_assert(offsetof(WindowPacket, WindowScissorBR) == 0x8, "WindowPacket::WindowScissorBR");
static_assert(offsetof(ValuesPacket, MaxVtxIndx) == 0x0, "ValuesPacket::MaxVtxIndx");
static_assert(offsetof(ValuesPacket, MinVtxIndx) == 0x4, "ValuesPacket::MinVtxIndx");
static_assert(offsetof(ValuesPacket, IndxOffset) == 0x8, "ValuesPacket::IndxOffset");
static_assert(offsetof(ValuesPacket, MultiPrimIbResetIndx) == 0xC, "ValuesPacket::MultiPrimIbResetIndx");
static_assert(offsetof(ValuesPacket, ColorMask) == 0x10, "ValuesPacket::ColorMask");
static_assert(offsetof(ValuesPacket, BlendRed) == 0x14, "ValuesPacket::BlendRed");
static_assert(offsetof(ValuesPacket, BlendGreen) == 0x18, "ValuesPacket::BlendGreen");
static_assert(offsetof(ValuesPacket, BlendBlue) == 0x1C, "ValuesPacket::BlendBlue");
static_assert(offsetof(ValuesPacket, BlendAlpha) == 0x20, "ValuesPacket::BlendAlpha");
static_assert(offsetof(ValuesPacket, Unused) == 0x24, "ValuesPacket::Unused");
static_assert(offsetof(ValuesPacket, StencilRefMaskBF) == 0x30, "ValuesPacket::StencilRefMaskBF");
static_assert(offsetof(ValuesPacket, StencilRefMask) == 0x34, "ValuesPacket::StencilRefMask");
static_assert(offsetof(ValuesPacket, AlphaRef) == 0x38, "ValuesPacket::AlphaRef");
static_assert(offsetof(ValuesPacket, VportXScale) == 0x3C, "ValuesPacket::VportXScale");
static_assert(offsetof(ValuesPacket, VportXOffset) == 0x40, "ValuesPacket::VportXOffset");
static_assert(offsetof(ValuesPacket, VportYScale) == 0x44, "ValuesPacket::VportYScale");
static_assert(offsetof(ValuesPacket, VportYOffset) == 0x48, "ValuesPacket::VportYOffset");
static_assert(offsetof(ValuesPacket, VportZScale) == 0x4C, "ValuesPacket::VportZScale");
static_assert(offsetof(ValuesPacket, VportZOffset) == 0x50, "ValuesPacket::VportZOffset");
static_assert(offsetof(ProgramPacket, ProgramControl) == 0x0, "ProgramPacket::ProgramControl");
static_assert(offsetof(ProgramPacket, ContextMisc) == 0x4, "ProgramPacket::ContextMisc");
static_assert(offsetof(ProgramPacket, InterpolatorControl) == 0x8, "ProgramPacket::InterpolatorControl");
static_assert(offsetof(ProgramPacket, Wrapping0) == 0xC, "ProgramPacket::Wrapping0");
static_assert(offsetof(ProgramPacket, Wrapping1) == 0x10, "ProgramPacket::Wrapping1");
static_assert(offsetof(ControlPacket, DepthControl) == 0x0, "ControlPacket::DepthControl");
static_assert(offsetof(ControlPacket, BlendControl0) == 0x4, "ControlPacket::BlendControl0");
static_assert(offsetof(ControlPacket, ColorControl) == 0x8, "ControlPacket::ColorControl");
static_assert(offsetof(ControlPacket, HiControl) == 0xC, "ControlPacket::HiControl");
static_assert(offsetof(ControlPacket, ClipControl) == 0x10, "ControlPacket::ClipControl");
static_assert(offsetof(ControlPacket, ModeControl) == 0x14, "ControlPacket::ModeControl");
static_assert(offsetof(ControlPacket, VteControl) == 0x18, "ControlPacket::VteControl");
static_assert(offsetof(ControlPacket, Unused) == 0x1C, "ControlPacket::Unused");
static_assert(offsetof(ControlPacket, EdramModeControl) == 0x20, "ControlPacket::EdramModeControl");
static_assert(offsetof(ControlPacket, BlendControl1) == 0x24, "ControlPacket::BlendControl1");
static_assert(offsetof(ControlPacket, BlendControl2) == 0x28, "ControlPacket::BlendControl2");
static_assert(offsetof(ControlPacket, BlendControl3) == 0x2C, "ControlPacket::BlendControl3");
static_assert(offsetof(TessellatorPacket, PointSize) == 0x0, "TessellatorPacket::PointSize");
static_assert(offsetof(TessellatorPacket, PointMinMax) == 0x4, "TessellatorPacket::PointMinMax");
static_assert(offsetof(TessellatorPacket, LineControl) == 0x8, "TessellatorPacket::LineControl");
static_assert(offsetof(TessellatorPacket, Unused1) == 0xC, "TessellatorPacket::Unused1");
static_assert(offsetof(TessellatorPacket, OutputPathControl) == 0x10, "TessellatorPacket::OutputPathControl");
static_assert(offsetof(TessellatorPacket, HosControl) == 0x14, "TessellatorPacket::HosControl");
static_assert(offsetof(TessellatorPacket, HosMaxTessLevel) == 0x18, "TessellatorPacket::HosMaxTessLevel");
static_assert(offsetof(TessellatorPacket, HosMinTessLevel) == 0x1C, "TessellatorPacket::HosMinTessLevel");
static_assert(offsetof(TessellatorPacket, HosReuseDepth) == 0x20, "TessellatorPacket::HosReuseDepth");
static_assert(offsetof(TessellatorPacket, GroupPrimType) == 0x24, "TessellatorPacket::GroupPrimType");
static_assert(offsetof(TessellatorPacket, GroupFirstDecr) == 0x28, "TessellatorPacket::GroupFirstDecr");
static_assert(offsetof(TessellatorPacket, GroupDecr) == 0x2C, "TessellatorPacket::GroupDecr");
static_assert(offsetof(TessellatorPacket, GroupVect0Control) == 0x30, "TessellatorPacket::GroupVect0Control");
static_assert(offsetof(TessellatorPacket, GroupVect1Control) == 0x34, "TessellatorPacket::GroupVect1Control");
static_assert(offsetof(TessellatorPacket, GroupVect0FmtControl) == 0x38, "TessellatorPacket::GroupVect0FmtControl");
static_assert(offsetof(TessellatorPacket, GroupVect1FmtControl) == 0x3C, "TessellatorPacket::GroupVect1FmtControl");
static_assert(offsetof(TessellatorPacket, Unused2) == 0x40, "TessellatorPacket::Unused2");
static_assert(offsetof(TessellatorPacket, MPassPsControl) == 0x48, "TessellatorPacket::MPassPsControl");
static_assert(offsetof(TessellatorPacket, VizQuery) == 0x4C, "TessellatorPacket::VizQuery");
static_assert(offsetof(TessellatorPacket, Enhance) == 0x50, "TessellatorPacket::Enhance");
static_assert(offsetof(MiscPacket, ScLineControl) == 0x0, "MiscPacket::ScLineControl");
static_assert(offsetof(MiscPacket, AaConfig) == 0x4, "MiscPacket::AaConfig");
static_assert(offsetof(MiscPacket, VtxControl) == 0x8, "MiscPacket::VtxControl");
static_assert(offsetof(MiscPacket, GbVertClipAdj) == 0xC, "MiscPacket::GbVertClipAdj");
static_assert(offsetof(MiscPacket, GbVertDiscAdj) == 0x10, "MiscPacket::GbVertDiscAdj");
static_assert(offsetof(MiscPacket, GbHorzClipAdj) == 0x14, "MiscPacket::GbHorzClipAdj");
static_assert(offsetof(MiscPacket, GbHorzDiscAdj) == 0x18, "MiscPacket::GbHorzDiscAdj");
static_assert(offsetof(MiscPacket, VsConst) == 0x1C, "MiscPacket::VsConst");
static_assert(offsetof(MiscPacket, PsConst) == 0x20, "MiscPacket::PsConst");
static_assert(offsetof(MiscPacket, DebugMisc0) == 0x24, "MiscPacket::DebugMisc0");
static_assert(offsetof(MiscPacket, DebugMisc1) == 0x28, "MiscPacket::DebugMisc1");
static_assert(offsetof(MiscPacket, Unused1) == 0x2C, "MiscPacket::Unused1");
static_assert(offsetof(MiscPacket, Unused2) == 0x40, "MiscPacket::Unused2");
static_assert(offsetof(MiscPacket, AaMask) == 0x48, "MiscPacket::AaMask");
static_assert(offsetof(MiscPacket, Unused3) == 0x4C, "MiscPacket::Unused3");
static_assert(offsetof(MiscPacket, VertexReuseBlockControl) == 0x58, "MiscPacket::VertexReuseBlockControl");
static_assert(offsetof(MiscPacket, OutDeallocControl) == 0x5C, "MiscPacket::OutDeallocControl");
static_assert(offsetof(MiscPacket, CopyControl) == 0x60, "MiscPacket::CopyControl");
static_assert(offsetof(MiscPacket, CopyDestBase) == 0x64, "MiscPacket::CopyDestBase");
static_assert(offsetof(MiscPacket, CopyDestPitch) == 0x68, "MiscPacket::CopyDestPitch");
static_assert(offsetof(MiscPacket, CopyDestInfo) == 0x6C, "MiscPacket::CopyDestInfo");
static_assert(offsetof(MiscPacket, HiClear) == 0x70, "MiscPacket::HiClear");
static_assert(offsetof(MiscPacket, DepthClear) == 0x74, "MiscPacket::DepthClear");
static_assert(offsetof(MiscPacket, ColorClear) == 0x78, "MiscPacket::ColorClear");
static_assert(offsetof(MiscPacket, ColorClearLo) == 0x7C, "MiscPacket::ColorClearLo");
static_assert(offsetof(MiscPacket, CopyFunc) == 0x80, "MiscPacket::CopyFunc");
static_assert(offsetof(MiscPacket, CopyRef) == 0x84, "MiscPacket::CopyRef");
static_assert(offsetof(MiscPacket, CopyMask) == 0x88, "MiscPacket::CopyMask");
static_assert(offsetof(MiscPacket, CopySurfaceSlice) == 0x8C, "MiscPacket::CopySurfaceSlice");
static_assert(offsetof(MiscPacket, SampleCountControl) == 0x90, "MiscPacket::SampleCountControl");
static_assert(offsetof(MiscPacket, SampleCountAddress) == 0x94, "MiscPacket::SampleCountAddress");
static_assert(offsetof(PointPacket, PolyOffsetFrontScale) == 0x0, "PointPacket::PolyOffsetFrontScale");
static_assert(offsetof(PointPacket, PolyOffsetFrontOffset) == 0x4, "PointPacket::PolyOffsetFrontOffset");
static_assert(offsetof(PointPacket, PolyOffsetBackScale) == 0x8, "PointPacket::PolyOffsetBackScale");
static_assert(offsetof(PointPacket, PolyOffsetBackOffset) == 0xC, "PointPacket::PolyOffsetBackOffset");
static_assert(offsetof(PointPacket, PointXRad) == 0x10, "PointPacket::PointXRad");
static_assert(offsetof(PointPacket, PointYRad) == 0x14, "PointPacket::PointYRad");
static_assert(offsetof(PointPacket, PointConstantSize) == 0x18, "PointPacket::PointConstantSize");
static_assert(offsetof(PointPacket, PointCullRad) == 0x1C, "PointPacket::PointCullRad");
static_assert(offsetof(GuestViewport, X) == 0x0, "GuestViewport::X");
static_assert(offsetof(GuestViewport, Y) == 0x4, "GuestViewport::Y");
static_assert(offsetof(GuestViewport, Width) == 0x8, "GuestViewport::Width");
static_assert(offsetof(GuestViewport, Height) == 0xC, "GuestViewport::Height");
static_assert(offsetof(GuestViewport, MinZ) == 0x10, "GuestViewport::MinZ");
static_assert(offsetof(GuestViewport, MaxZ) == 0x14, "GuestViewport::MaxZ");
static_assert(offsetof(GuestRect, left) == 0x0, "GuestRect::left");
static_assert(offsetof(GuestRect, top) == 0x4, "GuestRect::top");
static_assert(offsetof(GuestRect, right) == 0x8, "GuestRect::right");
static_assert(offsetof(GuestRect, bottom) == 0xC, "GuestRect::bottom");
static_assert(offsetof(GuestCriticalSection, dword) == 0x0, "GuestCriticalSection::dword");

static_assert(offsetof(D3DDevice, m_Pending) == 0x0, "m_Pending");
static_assert(offsetof(D3DDevice, m_Pending.m_Mask) == 0x0, "m_Pending.m_Mask");
static_assert(offsetof(D3DDevice, m_Predicated_PendingMask2) == 0x28, "m_Predicated_PendingMask2");
static_assert(offsetof(D3DDevice, m_pRing) == 0x30, "m_pRing");
static_assert(offsetof(D3DDevice, m_pRingLimit) == 0x34, "m_pRingLimit");
static_assert(offsetof(D3DDevice, m_pRingGuarantee) == 0x38, "m_pRingGuarantee");
static_assert(offsetof(D3DDevice, m_ReferenceCount) == 0x3C, "m_ReferenceCount");
static_assert(offsetof(D3DDevice, m_SetRenderStateCall) == 0x40, "m_SetRenderStateCall");
static_assert(offsetof(D3DDevice, m_SetSamplerStateCall) == 0x1D4, "m_SetSamplerStateCall");
static_assert(offsetof(D3DDevice, m_GetRenderStateCall) == 0x224, "m_GetRenderStateCall");
static_assert(offsetof(D3DDevice, m_GetSamplerStateCall) == 0x3B8, "m_GetSamplerStateCall");
static_assert(offsetof(D3DDevice, m_Constants) == 0x480, "m_Constants");
static_assert(offsetof(D3DDevice, m_Constants.Fetch) == 0x480, "m_Constants.Fetch");
static_assert(offsetof(D3DDevice, m_Constants.VertexShaderF) == 0x780, "m_Constants.VertexShaderF");
static_assert(offsetof(D3DDevice, m_Constants.PixelShaderF) == 0x1780, "m_Constants.PixelShaderF");
static_assert(offsetof(D3DDevice, m_Constants.VertexShaderB) == 0x2780, "m_Constants.VertexShaderB");
static_assert(offsetof(D3DDevice, m_Constants.PixelShaderB) == 0x2790, "m_Constants.PixelShaderB");
static_assert(offsetof(D3DDevice, m_Constants.VertexShaderI) == 0x27A0, "m_Constants.VertexShaderI");
static_assert(offsetof(D3DDevice, m_Constants.PixelShaderI) == 0x27E0, "m_Constants.PixelShaderI");
static_assert(offsetof(D3DDevice, m_ClipPlanes) == 0x2820, "m_ClipPlanes");
static_assert(offsetof(D3DDevice, m_DestinationPacket) == 0x2880, "m_DestinationPacket");
static_assert(offsetof(D3DDevice, m_DestinationPacket.SurfaceInfo) == 0x2880, "m_DestinationPacket.SurfaceInfo");
static_assert(offsetof(D3DDevice, m_DestinationPacket.Color0Info) == 0x2884, "m_DestinationPacket.Color0Info");
static_assert(offsetof(D3DDevice, m_DestinationPacket.DepthInfo) == 0x2888, "m_DestinationPacket.DepthInfo");
static_assert(offsetof(D3DDevice, m_DestinationPacket.Color1Info) == 0x288C, "m_DestinationPacket.Color1Info");
static_assert(offsetof(D3DDevice, m_DestinationPacket.Color2Info) == 0x2890, "m_DestinationPacket.Color2Info");
static_assert(offsetof(D3DDevice, m_DestinationPacket.Color3Info) == 0x2894, "m_DestinationPacket.Color3Info");
static_assert(offsetof(D3DDevice, m_DestinationPacket.CoherDestBase) == 0x2898, "m_DestinationPacket.CoherDestBase");
static_assert(offsetof(D3DDevice, m_DestinationPacket.ScreenScissorTL) == 0x28B8, "m_DestinationPacket.ScreenScissorTL");
static_assert(offsetof(D3DDevice, m_DestinationPacket.ScreenScissorBR) == 0x28BC, "m_DestinationPacket.ScreenScissorBR");
static_assert(offsetof(D3DDevice, m_WindowPacket) == 0x28C0, "m_WindowPacket");
static_assert(offsetof(D3DDevice, m_WindowPacket.WindowOffset) == 0x28C0, "m_WindowPacket.WindowOffset");
static_assert(offsetof(D3DDevice, m_WindowPacket.WindowScissorTL) == 0x28C4, "m_WindowPacket.WindowScissorTL");
static_assert(offsetof(D3DDevice, m_WindowPacket.WindowScissorBR) == 0x28C8, "m_WindowPacket.WindowScissorBR");
static_assert(offsetof(D3DDevice, m_ValuesPacket) == 0x28CC, "m_ValuesPacket");
static_assert(offsetof(D3DDevice, m_ValuesPacket.MaxVtxIndx) == 0x28CC, "m_ValuesPacket.MaxVtxIndx");
static_assert(offsetof(D3DDevice, m_ValuesPacket.MinVtxIndx) == 0x28D0, "m_ValuesPacket.MinVtxIndx");
static_assert(offsetof(D3DDevice, m_ValuesPacket.IndxOffset) == 0x28D4, "m_ValuesPacket.IndxOffset");
static_assert(offsetof(D3DDevice, m_ValuesPacket.MultiPrimIbResetIndx) == 0x28D8, "m_ValuesPacket.MultiPrimIbResetIndx");
static_assert(offsetof(D3DDevice, m_ValuesPacket.ColorMask) == 0x28DC, "m_ValuesPacket.ColorMask");
static_assert(offsetof(D3DDevice, m_ValuesPacket.BlendRed) == 0x28E0, "m_ValuesPacket.BlendRed");
static_assert(offsetof(D3DDevice, m_ValuesPacket.BlendGreen) == 0x28E4, "m_ValuesPacket.BlendGreen");
static_assert(offsetof(D3DDevice, m_ValuesPacket.BlendBlue) == 0x28E8, "m_ValuesPacket.BlendBlue");
static_assert(offsetof(D3DDevice, m_ValuesPacket.BlendAlpha) == 0x28EC, "m_ValuesPacket.BlendAlpha");
static_assert(offsetof(D3DDevice, m_ValuesPacket.Unused) == 0x28F0, "m_ValuesPacket.Unused");
static_assert(offsetof(D3DDevice, m_ValuesPacket.StencilRefMaskBF) == 0x28FC, "m_ValuesPacket.StencilRefMaskBF");
static_assert(offsetof(D3DDevice, m_ValuesPacket.StencilRefMask) == 0x2900, "m_ValuesPacket.StencilRefMask");
static_assert(offsetof(D3DDevice, m_ValuesPacket.AlphaRef) == 0x2904, "m_ValuesPacket.AlphaRef");
static_assert(offsetof(D3DDevice, m_ValuesPacket.VportXScale) == 0x2908, "m_ValuesPacket.VportXScale");
static_assert(offsetof(D3DDevice, m_ValuesPacket.VportXOffset) == 0x290C, "m_ValuesPacket.VportXOffset");
static_assert(offsetof(D3DDevice, m_ValuesPacket.VportYScale) == 0x2910, "m_ValuesPacket.VportYScale");
static_assert(offsetof(D3DDevice, m_ValuesPacket.VportYOffset) == 0x2914, "m_ValuesPacket.VportYOffset");
static_assert(offsetof(D3DDevice, m_ValuesPacket.VportZScale) == 0x2918, "m_ValuesPacket.VportZScale");
static_assert(offsetof(D3DDevice, m_ValuesPacket.VportZOffset) == 0x291C, "m_ValuesPacket.VportZOffset");
static_assert(offsetof(D3DDevice, m_ProgramPacket) == 0x2920, "m_ProgramPacket");
static_assert(offsetof(D3DDevice, m_ProgramPacket.ProgramControl) == 0x2920, "m_ProgramPacket.ProgramControl");
static_assert(offsetof(D3DDevice, m_ProgramPacket.ContextMisc) == 0x2924, "m_ProgramPacket.ContextMisc");
static_assert(offsetof(D3DDevice, m_ProgramPacket.InterpolatorControl) == 0x2928, "m_ProgramPacket.InterpolatorControl");
static_assert(offsetof(D3DDevice, m_ProgramPacket.Wrapping0) == 0x292C, "m_ProgramPacket.Wrapping0");
static_assert(offsetof(D3DDevice, m_ProgramPacket.Wrapping1) == 0x2930, "m_ProgramPacket.Wrapping1");
static_assert(offsetof(D3DDevice, m_ControlPacket) == 0x2934, "m_ControlPacket");
static_assert(offsetof(D3DDevice, m_ControlPacket.DepthControl) == 0x2934, "m_ControlPacket.DepthControl");
static_assert(offsetof(D3DDevice, m_ControlPacket.BlendControl0) == 0x2938, "m_ControlPacket.BlendControl0");
static_assert(offsetof(D3DDevice, m_ControlPacket.ColorControl) == 0x293C, "m_ControlPacket.ColorControl");
static_assert(offsetof(D3DDevice, m_ControlPacket.HiControl) == 0x2940, "m_ControlPacket.HiControl");
static_assert(offsetof(D3DDevice, m_ControlPacket.ClipControl) == 0x2944, "m_ControlPacket.ClipControl");
static_assert(offsetof(D3DDevice, m_ControlPacket.ModeControl) == 0x2948, "m_ControlPacket.ModeControl");
static_assert(offsetof(D3DDevice, m_ControlPacket.VteControl) == 0x294C, "m_ControlPacket.VteControl");
static_assert(offsetof(D3DDevice, m_ControlPacket.Unused) == 0x2950, "m_ControlPacket.Unused");
static_assert(offsetof(D3DDevice, m_ControlPacket.EdramModeControl) == 0x2954, "m_ControlPacket.EdramModeControl");
static_assert(offsetof(D3DDevice, m_ControlPacket.BlendControl1) == 0x2958, "m_ControlPacket.BlendControl1");
static_assert(offsetof(D3DDevice, m_ControlPacket.BlendControl2) == 0x295C, "m_ControlPacket.BlendControl2");
static_assert(offsetof(D3DDevice, m_ControlPacket.BlendControl3) == 0x2960, "m_ControlPacket.BlendControl3");
static_assert(offsetof(D3DDevice, m_TessellatorPacket) == 0x2964, "m_TessellatorPacket");
static_assert(offsetof(D3DDevice, m_TessellatorPacket.PointSize) == 0x2964, "m_TessellatorPacket.PointSize");
static_assert(offsetof(D3DDevice, m_TessellatorPacket.PointMinMax) == 0x2968, "m_TessellatorPacket.PointMinMax");
static_assert(offsetof(D3DDevice, m_TessellatorPacket.LineControl) == 0x296C, "m_TessellatorPacket.LineControl");
static_assert(offsetof(D3DDevice, m_TessellatorPacket.Unused1) == 0x2970, "m_TessellatorPacket.Unused1");
static_assert(offsetof(D3DDevice, m_TessellatorPacket.OutputPathControl) == 0x2974, "m_TessellatorPacket.OutputPathControl");
static_assert(offsetof(D3DDevice, m_TessellatorPacket.HosControl) == 0x2978, "m_TessellatorPacket.HosControl");
static_assert(offsetof(D3DDevice, m_TessellatorPacket.HosMaxTessLevel) == 0x297C, "m_TessellatorPacket.HosMaxTessLevel");
static_assert(offsetof(D3DDevice, m_TessellatorPacket.HosMinTessLevel) == 0x2980, "m_TessellatorPacket.HosMinTessLevel");
static_assert(offsetof(D3DDevice, m_TessellatorPacket.HosReuseDepth) == 0x2984, "m_TessellatorPacket.HosReuseDepth");
static_assert(offsetof(D3DDevice, m_TessellatorPacket.GroupPrimType) == 0x2988, "m_TessellatorPacket.GroupPrimType");
static_assert(offsetof(D3DDevice, m_TessellatorPacket.GroupFirstDecr) == 0x298C, "m_TessellatorPacket.GroupFirstDecr");
static_assert(offsetof(D3DDevice, m_TessellatorPacket.GroupDecr) == 0x2990, "m_TessellatorPacket.GroupDecr");
static_assert(offsetof(D3DDevice, m_TessellatorPacket.GroupVect0Control) == 0x2994, "m_TessellatorPacket.GroupVect0Control");
static_assert(offsetof(D3DDevice, m_TessellatorPacket.GroupVect1Control) == 0x2998, "m_TessellatorPacket.GroupVect1Control");
static_assert(offsetof(D3DDevice, m_TessellatorPacket.GroupVect0FmtControl) == 0x299C, "m_TessellatorPacket.GroupVect0FmtControl");
static_assert(offsetof(D3DDevice, m_TessellatorPacket.GroupVect1FmtControl) == 0x29A0, "m_TessellatorPacket.GroupVect1FmtControl");
static_assert(offsetof(D3DDevice, m_TessellatorPacket.Unused2) == 0x29A4, "m_TessellatorPacket.Unused2");
static_assert(offsetof(D3DDevice, m_TessellatorPacket.MPassPsControl) == 0x29AC, "m_TessellatorPacket.MPassPsControl");
static_assert(offsetof(D3DDevice, m_TessellatorPacket.VizQuery) == 0x29B0, "m_TessellatorPacket.VizQuery");
static_assert(offsetof(D3DDevice, m_TessellatorPacket.Enhance) == 0x29B4, "m_TessellatorPacket.Enhance");
static_assert(offsetof(D3DDevice, m_MiscPacket) == 0x29B8, "m_MiscPacket");
static_assert(offsetof(D3DDevice, m_MiscPacket.ScLineControl) == 0x29B8, "m_MiscPacket.ScLineControl");
static_assert(offsetof(D3DDevice, m_MiscPacket.AaConfig) == 0x29BC, "m_MiscPacket.AaConfig");
static_assert(offsetof(D3DDevice, m_MiscPacket.VtxControl) == 0x29C0, "m_MiscPacket.VtxControl");
static_assert(offsetof(D3DDevice, m_MiscPacket.GbVertClipAdj) == 0x29C4, "m_MiscPacket.GbVertClipAdj");
static_assert(offsetof(D3DDevice, m_MiscPacket.GbVertDiscAdj) == 0x29C8, "m_MiscPacket.GbVertDiscAdj");
static_assert(offsetof(D3DDevice, m_MiscPacket.GbHorzClipAdj) == 0x29CC, "m_MiscPacket.GbHorzClipAdj");
static_assert(offsetof(D3DDevice, m_MiscPacket.GbHorzDiscAdj) == 0x29D0, "m_MiscPacket.GbHorzDiscAdj");
static_assert(offsetof(D3DDevice, m_MiscPacket.VsConst) == 0x29D4, "m_MiscPacket.VsConst");
static_assert(offsetof(D3DDevice, m_MiscPacket.PsConst) == 0x29D8, "m_MiscPacket.PsConst");
static_assert(offsetof(D3DDevice, m_MiscPacket.DebugMisc0) == 0x29DC, "m_MiscPacket.DebugMisc0");
static_assert(offsetof(D3DDevice, m_MiscPacket.DebugMisc1) == 0x29E0, "m_MiscPacket.DebugMisc1");
static_assert(offsetof(D3DDevice, m_MiscPacket.Unused1) == 0x29E4, "m_MiscPacket.Unused1");
static_assert(offsetof(D3DDevice, m_MiscPacket.Unused2) == 0x29F8, "m_MiscPacket.Unused2");
static_assert(offsetof(D3DDevice, m_MiscPacket.AaMask) == 0x2A00, "m_MiscPacket.AaMask");
static_assert(offsetof(D3DDevice, m_MiscPacket.Unused3) == 0x2A04, "m_MiscPacket.Unused3");
static_assert(offsetof(D3DDevice, m_MiscPacket.VertexReuseBlockControl) == 0x2A10, "m_MiscPacket.VertexReuseBlockControl");
static_assert(offsetof(D3DDevice, m_MiscPacket.OutDeallocControl) == 0x2A14, "m_MiscPacket.OutDeallocControl");
static_assert(offsetof(D3DDevice, m_MiscPacket.CopyControl) == 0x2A18, "m_MiscPacket.CopyControl");
static_assert(offsetof(D3DDevice, m_MiscPacket.CopyDestBase) == 0x2A1C, "m_MiscPacket.CopyDestBase");
static_assert(offsetof(D3DDevice, m_MiscPacket.CopyDestPitch) == 0x2A20, "m_MiscPacket.CopyDestPitch");
static_assert(offsetof(D3DDevice, m_MiscPacket.CopyDestInfo) == 0x2A24, "m_MiscPacket.CopyDestInfo");
static_assert(offsetof(D3DDevice, m_MiscPacket.HiClear) == 0x2A28, "m_MiscPacket.HiClear");
static_assert(offsetof(D3DDevice, m_MiscPacket.DepthClear) == 0x2A2C, "m_MiscPacket.DepthClear");
static_assert(offsetof(D3DDevice, m_MiscPacket.ColorClear) == 0x2A30, "m_MiscPacket.ColorClear");
static_assert(offsetof(D3DDevice, m_MiscPacket.ColorClearLo) == 0x2A34, "m_MiscPacket.ColorClearLo");
static_assert(offsetof(D3DDevice, m_MiscPacket.CopyFunc) == 0x2A38, "m_MiscPacket.CopyFunc");
static_assert(offsetof(D3DDevice, m_MiscPacket.CopyRef) == 0x2A3C, "m_MiscPacket.CopyRef");
static_assert(offsetof(D3DDevice, m_MiscPacket.CopyMask) == 0x2A40, "m_MiscPacket.CopyMask");
static_assert(offsetof(D3DDevice, m_MiscPacket.CopySurfaceSlice) == 0x2A44, "m_MiscPacket.CopySurfaceSlice");
static_assert(offsetof(D3DDevice, m_MiscPacket.SampleCountControl) == 0x2A48, "m_MiscPacket.SampleCountControl");
static_assert(offsetof(D3DDevice, m_MiscPacket.SampleCountAddress) == 0x2A4C, "m_MiscPacket.SampleCountAddress");
static_assert(offsetof(D3DDevice, m_PointPacket) == 0x2A50, "m_PointPacket");
static_assert(offsetof(D3DDevice, m_PointPacket.PolyOffsetFrontScale) == 0x2A50, "m_PointPacket.PolyOffsetFrontScale");
static_assert(offsetof(D3DDevice, m_PointPacket.PolyOffsetFrontOffset) == 0x2A54, "m_PointPacket.PolyOffsetFrontOffset");
static_assert(offsetof(D3DDevice, m_PointPacket.PolyOffsetBackScale) == 0x2A58, "m_PointPacket.PolyOffsetBackScale");
static_assert(offsetof(D3DDevice, m_PointPacket.PolyOffsetBackOffset) == 0x2A5C, "m_PointPacket.PolyOffsetBackOffset");
static_assert(offsetof(D3DDevice, m_PointPacket.PointXRad) == 0x2A60, "m_PointPacket.PointXRad");
static_assert(offsetof(D3DDevice, m_PointPacket.PointYRad) == 0x2A64, "m_PointPacket.PointYRad");
static_assert(offsetof(D3DDevice, m_PointPacket.PointConstantSize) == 0x2A68, "m_PointPacket.PointConstantSize");
static_assert(offsetof(D3DDevice, m_PointPacket.PointCullRad) == 0x2A6C, "m_PointPacket.PointCullRad");

static_assert(offsetof(D3DDevice, thread_owner) == 0x2A88, "thread_owner");
static_assert(offsetof(D3DDevice, writeback_page) == 0x2A90, "writeback_page");
static_assert(offsetof(D3DDevice, fence_next) == 0x2A9C, "fence_next");
static_assert(offsetof(D3DDevice, device_flags0) == 0x2ABC, "device_flags0");
static_assert(offsetof(D3DDevice, predication_flags) == 0x2AC0, "predication_flags");
static_assert(offsetof(D3DDevice, primary_write_index) == 0x2ACC, "primary_write_index");
static_assert(offsetof(D3DDevice, queued_chunks) == 0x2B04, "queued_chunks");
static_assert(offsetof(D3DDevice, queued_chunks_lock) == 0x2B08, "queued_chunks_lock");
static_assert(offsetof(D3DDevice, vertex_declaration) == 0x2ED8, "vertex_declaration");
static_assert(offsetof(D3DDevice, index_buffer) == 0x3144, "index_buffer");
static_assert(offsetof(D3DDevice, render_targets) == 0x3148, "render_targets");
static_assert(offsetof(D3DDevice, depth_stencil) == 0x3158, "depth_stencil");
static_assert(offsetof(D3DDevice, stream_sources) == 0x315C, "stream_sources");
static_assert(offsetof(D3DDevice, stream_stride_dwords) == 0x31A0, "stream_stride_dwords");
static_assert(offsetof(D3DDevice, textures) == 0x31B0, "textures");
static_assert(offsetof(D3DDevice, viewport) == 0x3218, "viewport");
static_assert(offsetof(D3DDevice, scissor_rect) == 0x3234, "scissor_rect");
static_assert(offsetof(D3DDevice, pixel_shader) == 0x3244, "pixel_shader");
static_assert(offsetof(D3DDevice, vertex_shader) == 0x3248, "vertex_shader");
static_assert(offsetof(D3DDevice, predication_select) == 0x3254, "predication_select");
static_assert(offsetof(D3DDevice, callback_base) == 0x325C, "callback_base");
static_assert(offsetof(D3DDevice, tiling_target) == 0x3264, "tiling_target");
static_assert(offsetof(D3DDevice, tiling_saved_targets) == 0x3268, "tiling_saved_targets");
static_assert(offsetof(D3DDevice, tile_count) == 0x327C, "tile_count");
static_assert(offsetof(D3DDevice, tile_rects) == 0x3280, "tile_rects");
static_assert(offsetof(D3DDevice, tile_max_width) == 0x342C, "tile_max_width");
static_assert(offsetof(D3DDevice, tiling_flags_arg) == 0x3434, "tiling_flags_arg");
static_assert(offsetof(D3DDevice, tiling_clear_color) == 0x3440, "tiling_clear_color");
static_assert(offsetof(D3DDevice, tiling_clear_z) == 0x3450, "tiling_clear_z");
static_assert(offsetof(D3DDevice, command_buffer_recording) == 0x3460, "command_buffer_recording");

// Resource headers. Common flag bits and the per-type tails come from d3d9.h.
inline constexpr uint32_t kIndexBufferIndex32 = 0x80000000;         // D3DINDEXBUFFER_INDEX32
inline constexpr uint32_t kIndexBufferEndianMask = 0x60000000;      // D3DINDEXBUFFER_ENDIAN_MASK
inline constexpr uint32_t kIndexBufferEndianShift = 29;
inline constexpr uint32_t kSurfaceEdramAllocated = 0x80000000;      // D3DCOMMON_SURFACE_D3D_EDRAM_ALLOCATED
inline constexpr uint32_t kSurfaceFromTexture = 0x40000000;         // D3DCOMMON_SURFACE_FROM_TEXTURE

struct D3DResource {
  rex::be<uint32_t> Common;
  rex::be<uint32_t> ReferenceCount;
  rex::be<uint32_t> Fence;
  rex::be<uint32_t> ReadFence;
  rex::be<uint32_t> Identifier;
  rex::be<uint32_t> BaseFlush;
};

struct D3DBaseTexture {
  D3DResource resource;
  rex::be<uint32_t> MipFlush;
  FetchConstant Format;
};

struct D3DVertexBuffer {
  D3DResource resource;
  rex::be<uint32_t> Format[2];  // GPUVERTEX_FETCH_CONSTANT
};

struct D3DIndexBuffer {
  D3DResource resource;
  rex::be<uint32_t> Address;
  rex::be<uint32_t> Size;
};

// With kSurfaceFromTexture set in Common, +0x18 is the parent texture and +0x1C holds MipLevel/ArrayIndex.
struct D3DSurface {
  D3DResource resource;
  Reg<reg::RB_SURFACE_INFO> SurfaceInfo;
  rex::be<uint32_t> ColorOrDepthInfo;  // RB_COLOR_INFO or RB_DEPTH_INFO
  rex::be<uint32_t> HiControl;         // RB_TILECONTROL
  rex::be<uint32_t> Dimensions;
  rex::be<uint32_t> Format;            // D3DFORMAT
  rex::be<uint32_t> Size;
  uint32_t parent() const { return SurfaceInfo.raw; }
  // The SetViewport body 0x921192D0 reads Width from bits 31..18 and Height from bits 17..3, adding 1 to each.
  uint32_t width() const { return ((uint32_t(Dimensions) >> 18) & 0x3FFF) + 1; }
  uint32_t height() const { return ((uint32_t(Dimensions) >> 3) & 0x7FFF) + 1; }
};

// Only the D3DResource header is public. The private offsets below were read off the AE SetVertexShader 0x9211C938,
// IncrementalShaderPatchAndLoad 0x92136918, DirectShaderPatch 0x92136BB0 and D3DVertexShader_Bind 0x92137660.
struct D3DVertexShader {
  D3DResource resource;
};

struct D3DPixelShader {
  D3DResource resource;
};

// Microcode header (UCODE_HEADER): +0x00 flags, +0x14 offset of the literal table, then code offset and size.
inline constexpr uint32_t kUcodeLiteralOffset = 0x14;

inline constexpr uint32_t kVsCodeBase = 0x20;        // private: virtual base the code offsets add to
inline constexpr uint32_t kVsHeader = 0x368;         // private: variant 0 microcode header; bit 0x20 of its flags selects variant 1
inline constexpr uint32_t kVsVariantFlag = 0x20;
inline constexpr uint32_t kVsDescriptor = 0x380;     // private: header offset per variant, +8 for variant 1
inline constexpr uint32_t kVsCodeOffset = 0x368;     // private: from vs + descriptor, then the code size at +4
inline constexpr uint32_t kVsPatchRecord = 0x1A0;    // private: per-variant patch cache stride (DirectShaderPatch)

inline constexpr uint32_t kPsCodeBase = 0x18;        // private: virtual base the code offset adds to
inline constexpr uint32_t kPsHeader = 0x28;          // private: embedded microcode header copy
inline constexpr uint32_t kPsShaderOffset = 0x18;    // private: from the header, offset of the code offset/size pair

// Where D3DDevice_BeginVertices 0x92120898 binds the inline data: vertex fetch 95 (group 31, dwords 4-5),
// written straight to the ring without touching m_Constants.
inline constexpr uint32_t kInlineVertexFetch = 95;

static_assert(sizeof(D3DResource) == 0x18, "D3DResource");
static_assert(sizeof(D3DBaseTexture) == 0x34, "D3DBaseTexture");
static_assert(sizeof(D3DVertexBuffer) == 0x20, "D3DVertexBuffer");
static_assert(sizeof(D3DIndexBuffer) == 0x20, "D3DIndexBuffer");
static_assert(sizeof(D3DSurface) == 0x30, "D3DSurface");

static_assert(offsetof(D3DResource, Common) == 0x0, "D3DResource::Common");
static_assert(offsetof(D3DResource, ReferenceCount) == 0x4, "D3DResource::ReferenceCount");
static_assert(offsetof(D3DResource, Fence) == 0x8, "D3DResource::Fence");
static_assert(offsetof(D3DResource, ReadFence) == 0xC, "D3DResource::ReadFence");
static_assert(offsetof(D3DResource, Identifier) == 0x10, "D3DResource::Identifier");
static_assert(offsetof(D3DResource, BaseFlush) == 0x14, "D3DResource::BaseFlush");
static_assert(offsetof(D3DBaseTexture, resource) == 0x0, "D3DBaseTexture::resource");
static_assert(offsetof(D3DBaseTexture, MipFlush) == 0x18, "D3DBaseTexture::MipFlush");
static_assert(offsetof(D3DBaseTexture, Format) == 0x1C, "D3DBaseTexture::Format");
static_assert(offsetof(D3DVertexBuffer, resource) == 0x0, "D3DVertexBuffer::resource");
static_assert(offsetof(D3DVertexBuffer, Format) == 0x18, "D3DVertexBuffer::Format");
static_assert(offsetof(D3DIndexBuffer, resource) == 0x0, "D3DIndexBuffer::resource");
static_assert(offsetof(D3DIndexBuffer, Address) == 0x18, "D3DIndexBuffer::Address");
static_assert(offsetof(D3DIndexBuffer, Size) == 0x1C, "D3DIndexBuffer::Size");
static_assert(offsetof(D3DSurface, resource) == 0x0, "D3DSurface::resource");
static_assert(offsetof(D3DSurface, SurfaceInfo) == 0x18, "D3DSurface::SurfaceInfo");
static_assert(offsetof(D3DSurface, ColorOrDepthInfo) == 0x1C, "D3DSurface::ColorOrDepthInfo");
static_assert(offsetof(D3DSurface, HiControl) == 0x20, "D3DSurface::HiControl");
static_assert(offsetof(D3DSurface, Dimensions) == 0x24, "D3DSurface::Dimensions");
static_assert(offsetof(D3DSurface, Format) == 0x28, "D3DSurface::Format");
static_assert(offsetof(D3DSurface, Size) == 0x2C, "D3DSurface::Size");
static_assert(offsetof(D3DVertexShader, resource) == 0x0, "D3DVertexShader::resource");
static_assert(offsetof(D3DPixelShader, resource) == 0x0, "D3DPixelShader::resource");

}  // namespace ae::gpu::guest
