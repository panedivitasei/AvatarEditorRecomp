// Multisampled resolve: loads the source samples texel for texel, averages colour or keeps depth sample 0, then
// applies the resolve's exponent scale. Rebuild the embedded blobs with tools/gpu/compile_shaders.py after editing.

struct Constants {
  int2 source_offset;  // source texel = destination pixel + offset
  uint samples;
  uint average;        // 0 = sample 0 only (depth)
  float4 scale;
};

ConstantBuffer<Constants> g_constants : register(b0, space1);

Texture2DMS<float4> g_source : register(t0, space0);

float4 VsMain(uint id : SV_VertexID) : SV_Position {
  const float2 corner = float2((id << 1) & 2, id & 2);
  return float4(corner * float2(2.0, -2.0) + float2(-1.0, 1.0), 0.0, 1.0);
}

float4 PsMain(float4 position : SV_Position) : SV_Target0 {
  const int2 texel = int2(position.xy) + g_constants.source_offset;
  float4 c = g_source.Load(texel, 0);
  if (g_constants.average != 0) {
    for (uint s = 1; s < g_constants.samples; ++s) c += g_source.Load(texel, s);
    c /= float(g_constants.samples);
  }
  return c * g_constants.scale;
}
