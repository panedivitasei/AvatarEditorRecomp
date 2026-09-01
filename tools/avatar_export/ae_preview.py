"""ae_preview: dependency-free software renderer for Avatar Export.

Renders the converted avatar (any Rig from ae_convert, rest pose or an
animation frame) into an RGBA PNG: orthographic camera, painter's algorithm,
smooth (per-vertex) three-light shading, textures, a contact shadow and the
green radial backdrop. Pure python; a 384px frame of the
~7k-triangle avatar takes a few seconds.
"""
import math
import os
import struct
import zlib

import ae_convert as ac


# ---------------------------------------------------------------------------
# PNG I/O (8-bit RGB/RGBA, non-interlaced)
# ---------------------------------------------------------------------------

def read_png(path):
    with open(path, "rb") as f:
        data = f.read()
    if data[:8] != b"\x89PNG\r\n\x1a\n":
        raise ValueError("not a PNG")
    o = 8
    idat = b""
    w = h = 0
    ctype = 6
    while o + 8 <= len(data):
        ln, tag = struct.unpack_from(">I4s", data, o)
        body = data[o + 8:o + 8 + ln]
        if tag == b"IHDR":
            w, h, depth, ctype, comp, filt, ilace = struct.unpack(">IIBBBBB", body)
            if depth != 8 or ilace:
                raise ValueError("unsupported PNG (depth/interlace)")
        elif tag == b"IDAT":
            idat += body
        elif tag == b"IEND":
            break
        o += 12 + ln
    ch = {0: 1, 2: 3, 4: 2, 6: 4}[ctype]
    raw = zlib.decompress(idat)
    stride = w * ch
    out = bytearray(w * h * 4)
    prev = bytearray(stride)
    pos = 0
    for y in range(h):
        ft = raw[pos]
        pos += 1
        line = bytearray(raw[pos:pos + stride])
        pos += stride
        if ft == 1:
            for i in range(ch, stride):
                line[i] = (line[i] + line[i - ch]) & 0xFF
        elif ft == 2:
            for i in range(stride):
                line[i] = (line[i] + prev[i]) & 0xFF
        elif ft == 3:
            for i in range(stride):
                a = line[i - ch] if i >= ch else 0
                line[i] = (line[i] + ((a + prev[i]) >> 1)) & 0xFF
        elif ft == 4:
            for i in range(stride):
                a = line[i - ch] if i >= ch else 0
                b = prev[i]
                c = prev[i - ch] if i >= ch else 0
                p = a + b - c
                pa, pb, pc = abs(p - a), abs(p - b), abs(p - c)
                pr = a if (pa <= pb and pa <= pc) else (b if pb <= pc else c)
                line[i] = (line[i] + pr) & 0xFF
        prev = line
        row = y * w * 4
        if ch == 4:
            out[row:row + w * 4] = line
        elif ch == 3:
            for x in range(w):
                out[row + 4 * x:row + 4 * x + 3] = line[3 * x:3 * x + 3]
                out[row + 4 * x + 3] = 255
        elif ch == 2:
            for x in range(w):
                g = line[2 * x]
                out[row + 4 * x:row + 4 * x + 4] = bytes((g, g, g, line[2 * x + 1]))
        else:
            for x in range(w):
                g = line[x]
                out[row + 4 * x:row + 4 * x + 4] = bytes((g, g, g, 255))
    return w, h, out


def write_png(path, w, h, rgba):
    raw = bytearray()
    stride = w * 4
    for y in range(h):
        raw.append(0)
        raw += rgba[y * stride:(y + 1) * stride]

    def chunk(tag, body):
        c = struct.pack(">I", len(body)) + tag + body
        return c + struct.pack(">I", zlib.crc32(tag + body) & 0xFFFFFFFF)
    png = b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, 8, 6, 0, 0, 0))
    png += chunk(b"IDAT", zlib.compress(bytes(raw), 6)) + chunk(b"IEND", b"")
    with open(path, "wb") as f:
        f.write(png)


class Texture:
    def __init__(self, path):
        self.w, self.h, self.px = read_png(path)

    def sample(self, u, v):
        x = int(u * self.w) % self.w
        y = int(v * self.h) % self.h
        i = (y * self.w + x) * 4
        return self.px[i], self.px[i + 1], self.px[i + 2], self.px[i + 3]


# ---------------------------------------------------------------------------
# Views, poses
# ---------------------------------------------------------------------------

VIEWS = {
    # name: (right axis, up axis, toward-camera axis) as (index, sign) in the rig frame
    "tu_front": ((0, 1.0), (2, 1.0), (1, -1.0)),    # TU: Z up, camera at -Y
    "tu_side": ((1, -1.0), (2, 1.0), (0, -1.0)),    # TU: camera at -X (avatar's right side)
    "xbox_front": ((0, 1.0), (1, 1.0), (2, 1.0)),   # Xbox: Y up, camera at +Z
    "xbox_side": ((2, -1.0), (1, 1.0), (0, 1.0)),   # Xbox: camera at +X (avatar's left side)
    "source_front": ((1, 1.0), (2, 1.0), (0, 1.0)),  # Source: Z up, camera at +X
}


def peak_frame(anim):
    """The clip's most expressive frame: the one with the largest total joint
    rotation away from the rest pose (a wave at the top, a bow at the bottom)."""
    best, best_score = 0, -1.0
    tracks = anim.get("tracks", [])
    frames = anim.get("frame_count", 0)
    for f in range(frames):
        score = 0.0
        for tr in tracks:
            if f < len(tr["r"]):
                w = min(1.0, abs(tr["r"][f][3]))
                score += 2.0 * math.acos(w)
        if score > best_score:
            best, best_score = f, score
    return best


def skinned_positions(avatar, rig, anim=None, frame=0, tu_mode=False):
    """World positions + normals of every mesh vertex in the rig frame, posed."""
    if anim is None:
        skin = None
    else:
        locs = ac.anim_local_matrices(avatar, anim, frame)
        worlds = ac.fk_world(avatar, locs)
        now = ac.retarget_frame_to_rig(avatar, rig, worlds, tu_mode)
        skin = [ac.mat_mul(now[i], ac.mat_inverse(b.world)) for i, b in enumerate(rig.bones)]
    out = []
    for mesh in avatar.meshes:
        pos = []
        nrm = []
        for k, p in enumerate(mesh.positions):
            q = rig.xform_point(p)
            n = rig.xform_dir(mesh.normals[k])
            if skin is not None:
                infl = rig.vertex_influences(mesh.joints[k], mesh.weights[k])
                acc = [0.0, 0.0, 0.0]
                nacc = [0.0, 0.0, 0.0]
                for b, w in infl:
                    m = skin[b]
                    tp = ac.mat_vec(m, q)
                    tn = ac.mat_vec(m, n, 0.0)
                    acc = [acc[i] + w * tp[i] for i in range(3)]
                    nacc = [nacc[i] + w * tn[i] for i in range(3)]
                q = acc
                n = ac.vnorm(nacc)
            pos.append(q)
            nrm.append(n)
        out.append((pos, nrm))
    return out


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------

# The green radial backdrop.
EDITOR_BG_INNER = (146, 204, 112)
EDITOR_BG_OUTER = (34, 78, 44)

# Lights in screen space (x right, y up, z toward the camera).
KEY_LIGHT = ac.vnorm([-0.45, 0.60, 0.70])
FILL_LIGHT = ac.vnorm([0.70, 0.15, 0.55])
RIM_LIGHT = ac.vnorm([0.30, 0.50, -0.80])


def _shade(n):
    """Smooth three-light shading for a screen-space unit normal."""
    key = max(0.0, n[0] * KEY_LIGHT[0] + n[1] * KEY_LIGHT[1] + n[2] * KEY_LIGHT[2])
    fill = max(0.0, n[0] * FILL_LIGHT[0] + n[1] * FILL_LIGHT[1] + n[2] * FILL_LIGHT[2])
    rim = max(0.0, n[0] * RIM_LIGHT[0] + n[1] * RIM_LIGHT[1] + n[2] * RIM_LIGHT[2])
    # half-lambert on the key keeps the shadow side soft like the console renderer
    key = key * 0.5 + 0.5
    return min(1.25, 0.22 + 0.72 * key * key + 0.20 * fill + 0.18 * rim * rim)


def _fill_background(buf, W, H, style):
    if style != "editor":
        for i in range(W * H):
            buf[4 * i:4 * i + 4] = bytes((40, 40, 48, 255))
        return
    cx, cy = W * 0.5, H * 0.40
    rmax = math.hypot(W * 0.5, H * 0.6)
    ir, ig, ib = EDITOR_BG_INNER
    orr, og, ob = EDITOR_BG_OUTER
    for y in range(H):
        dy = (y - cy)
        for x in range(W):
            t = math.hypot(x - cx, dy) / rmax
            t = min(1.0, t * t * 0.9 + t * 0.1)
            i = (y * W + x) * 4
            buf[i] = int(ir + (orr - ir) * t)
            buf[i + 1] = int(ig + (og - ig) * t)
            buf[i + 2] = int(ib + (ob - ib) * t)
            buf[i + 3] = 255


def _draw_shadow(buf, W, H, cx, cy, rx, ry, strength=0.55):
    x0, x1 = max(0, int(cx - rx)), min(W - 1, int(cx + rx))
    y0, y1 = max(0, int(cy - ry)), min(H - 1, int(cy + ry))
    for y in range(y0, y1 + 1):
        for x in range(x0, x1 + 1):
            d = ((x - cx) / rx) ** 2 + ((y - cy) / ry) ** 2
            if d >= 1.0:
                continue
            a = strength * (1.0 - d) ** 1.5
            i = (y * W + x) * 4
            buf[i] = int(buf[i] * (1 - a))
            buf[i + 1] = int(buf[i + 1] * (1 - a))
            buf[i + 2] = int(buf[i + 2] * (1 - a))


def render(avatar, rig, out_png, size=512, view="tu_front", anim=None, frame=0, tu_mode=False,
           style="editor", textures=True, margin=0.08):
    right, up, toward = VIEWS[view]
    posed = skinned_positions(avatar, rig, anim, frame, tu_mode)
    xs, ys = [], []
    for pos, _ in posed:
        for p in pos:
            xs.append(p[right[0]] * right[1])
            ys.append(p[up[0]] * up[1])
    if not xs:
        return None
    minx, maxx, miny, maxy = min(xs), max(xs), min(ys), max(ys)
    span = max(maxx - minx, maxy - miny) * (1.0 + 2 * margin)
    cx, cy = (minx + maxx) / 2, (miny + maxy) / 2
    scale = size / span
    W = H = size
    buf = bytearray(W * H * 4)
    _fill_background(buf, W, H, style)

    # contact shadow under the lowest point
    feet_y = H / 2 - (miny - cy) * scale
    body_w = (maxx - minx) * scale
    _draw_shadow(buf, W, H, W / 2 + ((minx + maxx) / 2 - cx) * scale, min(H - 2, feet_y + 2),
                 max(12.0, body_w * 0.28), max(4.0, body_w * 0.05))

    texcache = {}
    tris = []
    for mi, mesh in enumerate(avatar.meshes):
        pos, nrm = posed[mi]
        md = avatar.materials[mesh.material]
        tex = None
        if textures:
            path = os.path.join(avatar.dir, md["diffuse"])
            if path not in texcache:
                texcache[path] = Texture(path) if os.path.isfile(path) else None
            tex = texcache[path]
        mask = bool(md.get("alpha_mask"))
        # per-vertex screen data
        sv = []
        for k, p in enumerate(pos):
            sx = (p[right[0]] * right[1] - cx) * scale + W / 2
            sy = H / 2 - (p[up[0]] * up[1] - cy) * scale
            sz = p[toward[0]] * toward[1]
            n = nrm[k]
            ns = (n[right[0]] * right[1], n[up[0]] * up[1], n[toward[0]] * toward[1])
            sv.append((sx, sy, sz, _shade(ns)))
        idx = mesh.indices
        colors = mesh.colors
        for t in range(0, len(idx) - 2, 3):
            a, b, c = idx[t], idx[t + 1], idx[t + 2]
            depth = (sv[a][2] + sv[b][2] + sv[c][2]) / 3
            tris.append((depth, sv[a], sv[b], sv[c], mesh.uvs[a], mesh.uvs[b], mesh.uvs[c], tex, mask,
                         colors[a] if colors else None))
    tris.sort(key=lambda x: x[0])  # far first
    for depth, va, vb, vc, ua, ub, uc, tex, mask, vcol in tris:
        x0, y0, _, s0 = va
        x1, y1, _, s1 = vb
        x2, y2, _, s2 = vc
        area = (x1 - x0) * (y2 - y0) - (x2 - x0) * (y1 - y0)
        if abs(area) < 1e-6:
            continue
        bx0 = max(0, int(math.floor(min(x0, x1, x2))))
        bx1 = min(W - 1, int(math.ceil(max(x0, x1, x2))))
        by0 = max(0, int(math.floor(min(y0, y1, y2))))
        by1 = min(H - 1, int(math.ceil(max(y0, y1, y2))))
        inv = 1.0 / area
        vr = vcol[0] if vcol else 1.0
        vg = vcol[1] if vcol else 1.0
        vb_ = vcol[2] if vcol else 1.0
        for py in range(by0, by1 + 1):
            cyp = py + 0.5
            for px in range(bx0, bx1 + 1):
                cxp = px + 0.5
                w0 = ((x1 - cxp) * (y2 - cyp) - (x2 - cxp) * (y1 - cyp)) * inv
                w1 = ((x2 - cxp) * (y0 - cyp) - (x0 - cxp) * (y2 - cyp)) * inv
                w2 = 1.0 - w0 - w1
                if w0 < -0.002 or w1 < -0.002 or w2 < -0.002:
                    continue
                if tex is not None:
                    u = w0 * ua[0] + w1 * ub[0] + w2 * uc[0]
                    v = w0 * ua[1] + w1 * ub[1] + w2 * uc[1]
                    r, g, b, al = tex.sample(u, v)
                    if mask and al < 128:
                        continue
                else:
                    r, g, b = 200, 200, 200
                sh = w0 * s0 + w1 * s1 + w2 * s2
                i = (py * W + px) * 4
                buf[i] = min(255, int(r * vr * sh))
                buf[i + 1] = min(255, int(g * vg * sh))
                buf[i + 2] = min(255, int(b * vb_ * sh))
                buf[i + 3] = 255
    write_png(out_png, W, H, buf)
    return out_png
