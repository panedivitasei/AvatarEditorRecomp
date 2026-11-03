// Dear ImGui overlay: textured, vertex-coloured triangles in the UI's coordinate space, alpha-blended over the frame.
// Rebuild the embedded blobs with tools/gpu/compile_shaders.py after editing.

struct Constants {
  float2 scale;
  float2 offset;
  uint linear_filter;
};

ConstantBuffer<Constants> g_constants : register(b0, space1);

Texture2D<float4> g_texture : register(t0, space0);
SamplerState g_point : register(s1, space0);
SamplerState g_linear : register(s2, space0);

struct VsIn {
  float2 position : POSITION;
  float2 uv : TEXCOORD0;
  float4 color : COLOR0;
};

struct VsOut {
  float4 position : SV_Position;
  float2 uv : TEXCOORD0;
  float4 color : COLOR0;
};

VsOut VsMain(VsIn i) {
  VsOut o;
  o.position = float4(i.position * g_constants.scale + g_constants.offset, 0.0, 1.0);
  o.uv = i.uv;
  o.color = i.color;
  return o;
}

float4 PsMain(VsOut i) : SV_Target0 {
  float4 texel = g_constants.linear_filter != 0 ? g_texture.Sample(g_linear, i.uv) : g_texture.Sample(g_point, i.uv);
  return i.color * texel;
}
