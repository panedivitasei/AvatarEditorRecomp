// Fullscreen-triangle blit: samples a source rectangle, scales it and writes the target, for resolves and present.
// Rebuild the embedded blobs with tools/gpu/compile_shaders.py after editing.

struct Constants {
  float2 uv_offset;
  float2 uv_scale;
  float4 scale;
  uint linear_filter;
};

ConstantBuffer<Constants> g_constants : register(b0, space1);

Texture2D<float4> g_source : register(t0, space0);
SamplerState g_point : register(s1, space0);
SamplerState g_linear : register(s2, space0);

struct VsOut {
  float4 position : SV_Position;
  float2 uv : TEXCOORD0;
};

VsOut VsMain(uint id : SV_VertexID) {
  VsOut o;
  const float2 corner = float2((id << 1) & 2, id & 2);
  o.position = float4(corner * float2(2.0, -2.0) + float2(-1.0, 1.0), 0.0, 1.0);
  o.uv = g_constants.uv_offset + corner * g_constants.uv_scale;
  return o;
}

float4 PsMain(VsOut i) : SV_Target0 {
  float4 c = g_constants.linear_filter != 0 ? g_source.SampleLevel(g_linear, i.uv, 0.0)
                                            : g_source.SampleLevel(g_point, i.uv, 0.0);
  return c * g_constants.scale;
}
