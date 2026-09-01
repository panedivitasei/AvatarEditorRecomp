// Depth-resolve pixel shader: copies a region of a depth render target into
// an R32_FLOAT resolve destination by rendering instead of copying.
//
// D3D12 forbids partial CopyTextureRegion out of a depth-stencil resource, so
// a copy path has to stage the whole depth plane through a buffer and then
// extract the wanted region: two full-plane trips per resolve. Rendering the
// region removes the staging round-trip entirely.
//
// A depth resource can be read through an R32_FLOAT SRV, so a fullscreen
// triangle over the destination rect reading Load() and writing SV_Target is
// an exact texel-for-texel transfer with no intermediate buffer: one read
// pass and one write pass, both through the texture/RT caches.
//
// Regenerate: dxc -T ps_6_0 -E main -O3 -Fo depth_resolve.ps.dxil
//             depth_resolve.ps.hlsl, then bin2c into renderer_depth_ps.h.
//
// b4 layout (filled by the depth-resolve path in ResolveRegion):
//   [0].x = source depth descriptor index (R32_FLOAT view of the DSV)
//   [1].xy = asint(src_origin - dst_origin): added to SV_Position to reach
//            the source texel, so the shader needs no rect uniforms
//   [2].xy = source extent (w, h) for clamping; a resolve rect may exceed
//            the source when the guest asks for a padded region
cbuffer XeDescriptorIndices : register(b4, space0)
{
    uint4 xe_descriptor_indices[8];
};
Texture2DArray<float4> xe_textures2d[] : register(t0, space1);

void main(in float4 pos : SV_Position, out float depth : SV_Target0)
{
    const uint desc = xe_descriptor_indices[0].x;
    const int2 delta = int2(asint(xe_descriptor_indices[1].x),
                            asint(xe_descriptor_indices[1].y));
    const int2 dim = int2(xe_descriptor_indices[2].xy);
    const int2 sp = clamp(int2(pos.xy) + delta, int2(0, 0), dim - 1);
    depth = xe_textures2d[NonUniformResourceIndex(desc)]
                .Load(int4(sp, 0, 0)).r;
}
