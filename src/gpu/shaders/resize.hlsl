// Copies retained render-target contents between resolution scales, including individual MSAA samples.
cbuffer ResizeConstants : register(b0, space1) {
  float2 source_ratio;
  uint stencil_mask;
};

Texture2D<float4> source_single : register(t0);
Texture2DMS<float4> source_multi : register(t0);
Texture2D<uint2> stencil_single : register(t0);
Texture2DMS<uint2> stencil_multi : register(t0);

float4 ColorMain(float4 position : SV_Position) : SV_Target0 {
  return source_single.Load(int3(int2((floor(position.xy) + 0.5) * source_ratio), 0));
}
float4 ColorMsMain(float4 position : SV_Position, uint sample : SV_SampleIndex) : SV_Target0 {
  return source_multi.Load(int2((floor(position.xy) + 0.5) * source_ratio), sample);
}
float DepthMain(float4 position : SV_Position) : SV_Depth {
  return source_single.Load(int3(int2((floor(position.xy) + 0.5) * source_ratio), 0)).x;
}
float DepthMsMain(float4 position : SV_Position, uint sample : SV_SampleIndex) : SV_Depth {
  return source_multi.Load(int2((floor(position.xy) + 0.5) * source_ratio), sample).x;
}
void StencilMain(float4 position : SV_Position) {
  if ((stencil_single.Load(int3(int2((floor(position.xy) + 0.5) * source_ratio), 0)).y & stencil_mask) == 0) discard;
}
void StencilMsMain(float4 position : SV_Position, uint sample : SV_SampleIndex) {
  if ((stencil_multi.Load(int2((floor(position.xy) + 0.5) * source_ratio), sample).y & stencil_mask) == 0) discard;
}
