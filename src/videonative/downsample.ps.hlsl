// Write-back downsample pixel shader: box-filters a region of a host-scaled
// resolve texture down to guest resolution so the bytes the guest reads back
// (tile icons, page copies) are right at any native_video_resolution_scale.
//
// Renders a fullscreen triangle over a guest-sized staging target; each
// destination texel averages the source texels its footprint covers, which
// is an exact box filter at integer scales and a fair one in between. Below
// 100% the footprint is under one texel and this degrades to a nearest pick.
//
// Regenerate: dxc -T ps_6_0 -E main -O3 -Fo downsample.ps.dxil
//             downsample.ps.hlsl, then bin2c into renderer_downsample_ps.h.
//
// b4 layout (filled by the write-back snapshot path in ResolveRegion):
//   [0].x  = source descriptor index (the resolve destination texture)
//   [1].xy = source origin of the region, host texels
//   [2].xy = source extent (w, h) for clamping
//   [3].xy = host scale as num/den (host = guest * num / den)
cbuffer XeDescriptorIndices : register(b4, space0)
{
    uint4 xe_descriptor_indices[8];
};
Texture2DArray<float4> xe_textures2d[] : register(t0, space1);

void main(in float4 pos : SV_Position, out float4 color : SV_Target0)
{
    const uint desc = xe_descriptor_indices[0].x;
    const int2 origin = int2(xe_descriptor_indices[1].xy);
    const int2 dim = int2(xe_descriptor_indices[2].xy);
    const int num = int(xe_descriptor_indices[3].x);
    const int den = int(xe_descriptor_indices[3].y);
    const int2 d = int2(pos.xy);
    const int2 lo = origin + (d * num) / den;
    // Exclusive upper bound, at least one texel wide.
    const int2 hi = max(lo + 1, origin + ((d + 1) * num + den - 1) / den);
    float4 sum = 0.0;
    float count = 0.0;
    [loop] for (int y = lo.y; y < hi.y && y < lo.y + 4; ++y)
    {
        [loop] for (int x = lo.x; x < hi.x && x < lo.x + 4; ++x)
        {
            const int2 sp = clamp(int2(x, y), int2(0, 0), dim - 1);
            sum += xe_textures2d[NonUniformResourceIndex(desc)].Load(int4(sp, 0, 0));
            count += 1.0;
        }
    }
    color = sum / max(count, 1.0);
}
