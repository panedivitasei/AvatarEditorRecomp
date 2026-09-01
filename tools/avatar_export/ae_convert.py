"""ae_convert: turn an `avatarextract --avatar` export (avatar.json + baked
PNGs) into model files:

  * Tower Unite character: COLLADA .dae re-skinned onto the Workshop Rig 3.0
    (bone names/orientations from tu_rig_v3.json, centimetres, Z up, facing -Y,
    <Material>_Diffuse.png textures next to the .dae)
  * COLLADA .dae on the Xbox skeleton (metres, Y up), optionally one per clip
  * glTF 2.0 binary (.glb) with skin, vertex colours and every animation clip
  * Wavefront .obj/.mtl (static)
  * Source .smd (reference + one per clip)

Pure python (no numpy). Data frame of avatar.json: right-handed, Y up, the
avatar faces +Z, its left is +X, metres, UV origin top-left.
"""
import json
import math
import os
import shutil
import struct
import xml.sax.saxutils as sx
from datetime import datetime, timezone

# ---------------------------------------------------------------------------
# tiny linear algebra (column vectors, 4x4 as list of 4 rows)
# ---------------------------------------------------------------------------

def mat_identity():
    return [[1.0, 0, 0, 0], [0, 1.0, 0, 0], [0, 0, 1.0, 0], [0, 0, 0, 1.0]]


def mat_mul(a, b):
    return [[sum(a[i][k] * b[k][j] for k in range(4)) for j in range(4)] for i in range(4)]


def mat_vec(m, v, w=1.0):
    x, y, z = v
    return [m[0][0] * x + m[0][1] * y + m[0][2] * z + m[0][3] * w,
            m[1][0] * x + m[1][1] * y + m[1][2] * z + m[1][3] * w,
            m[2][0] * x + m[2][1] * y + m[2][2] * z + m[2][3] * w]


def mat_from_rt(r3, t):
    return [[r3[0][0], r3[0][1], r3[0][2], t[0]],
            [r3[1][0], r3[1][1], r3[1][2], t[1]],
            [r3[2][0], r3[2][1], r3[2][2], t[2]],
            [0.0, 0.0, 0.0, 1.0]]


def mat_rot3(m):
    return [[m[0][0], m[0][1], m[0][2]], [m[1][0], m[1][1], m[1][2]], [m[2][0], m[2][1], m[2][2]]]


def mat_trans(m):
    return [m[0][3], m[1][3], m[2][3]]


def mat_inverse_rigid(m):
    """Inverse of a rotation+translation matrix (orthonormal 3x3)."""
    r = mat_rot3(m)
    rt = [[r[j][i] for j in range(3)] for i in range(3)]
    t = mat_trans(m)
    nt = [-(rt[i][0] * t[0] + rt[i][1] * t[1] + rt[i][2] * t[2]) for i in range(3)]
    return mat_from_rt(rt, nt)


def mat_inverse(m):
    """General 4x4 inverse (Gauss-Jordan)."""
    n = 4
    a = [row[:] + [1.0 if i == j else 0.0 for j in range(n)] for i, row in enumerate(m)]
    for c in range(n):
        piv = max(range(c, n), key=lambda r: abs(a[r][c]))
        if abs(a[piv][c]) < 1e-12:
            return mat_identity()
        a[c], a[piv] = a[piv], a[c]
        p = a[c][c]
        a[c] = [v / p for v in a[c]]
        for r in range(n):
            if r != c and a[r][c] != 0.0:
                f = a[r][c]
                a[r] = [rv - f * cv for rv, cv in zip(a[r], a[c])]
    return [row[n:] for row in a]


def r3_mul(a, b):
    return [[sum(a[i][k] * b[k][j] for k in range(3)) for j in range(3)] for i in range(3)]


def r3_t(a):
    return [[a[j][i] for j in range(3)] for i in range(3)]


def r3_vec(r, v):
    return [r[i][0] * v[0] + r[i][1] * v[1] + r[i][2] * v[2] for i in range(3)]


def r3_normalize_columns(r):
    cols = []
    for j in range(3):
        c = [r[0][j], r[1][j], r[2][j]]
        l = math.sqrt(sum(x * x for x in c)) or 1.0
        cols.append([x / l for x in c])
    return [[cols[j][i] for j in range(3)] for i in range(3)]


def quat_to_r3(q):
    x, y, z, w = q
    n = math.sqrt(x * x + y * y + z * z + w * w) or 1.0
    x, y, z, w = x / n, y / n, z / n, w / n
    return [[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]]


def r3_to_quat(m):
    t = m[0][0] + m[1][1] + m[2][2]
    if t > 0:
        s = math.sqrt(t + 1.0) * 2
        w = 0.25 * s
        x = (m[2][1] - m[1][2]) / s
        y = (m[0][2] - m[2][0]) / s
        z = (m[1][0] - m[0][1]) / s
    elif m[0][0] > m[1][1] and m[0][0] > m[2][2]:
        s = math.sqrt(1.0 + m[0][0] - m[1][1] - m[2][2]) * 2
        w = (m[2][1] - m[1][2]) / s
        x = 0.25 * s
        y = (m[0][1] + m[1][0]) / s
        z = (m[0][2] + m[2][0]) / s
    elif m[1][1] > m[2][2]:
        s = math.sqrt(1.0 + m[1][1] - m[0][0] - m[2][2]) * 2
        w = (m[0][2] - m[2][0]) / s
        x = (m[0][1] + m[1][0]) / s
        y = 0.25 * s
        z = (m[1][2] + m[2][1]) / s
    else:
        s = math.sqrt(1.0 + m[2][2] - m[0][0] - m[1][1]) * 2
        w = (m[1][0] - m[0][1]) / s
        x = (m[0][2] + m[2][0]) / s
        y = (m[1][2] + m[2][1]) / s
        z = 0.25 * s
    n = math.sqrt(x * x + y * y + z * z + w * w) or 1.0
    return [x / n, y / n, z / n, w / n]


def r3_to_euler_xyz(m):
    """Euler XYZ (radians) such that R = Rz * Ry * Rx, the Source SMD order."""
    sy = -m[2][0]
    sy = max(-1.0, min(1.0, sy))
    y = math.asin(sy)
    if abs(sy) < 0.9999999:
        x = math.atan2(m[2][1], m[2][2])
        z = math.atan2(m[1][0], m[0][0])
    else:
        x = math.atan2(-m[1][2], m[1][1])
        z = 0.0
    return [x, y, z]


def vsub(a, b):
    return [a[i] - b[i] for i in range(3)]


def vadd(a, b):
    return [a[i] + b[i] for i in range(3)]


def vscale(a, s):
    return [a[i] * s for i in range(3)]


def vlen(a):
    return math.sqrt(sum(x * x for x in a))


def vnorm(a):
    l = vlen(a)
    return [x / l for x in a] if l > 1e-12 else [0.0, 0.0, 1.0]


def vdot(a, b):
    return sum(a[i] * b[i] for i in range(3))


# Frame conversions (rotation matrices, proper / det +1).
# Xbox export frame: RH, Y up, faces +Z, left +X.
FRAME_XBOX = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
# Tower Unite / Blender DAE: RH, Z up, faces -Y, right -X:  (x,y,z) -> (x, -z, y)
FRAME_TU = [[1, 0, 0], [0, 0, -1], [0, 1, 0]]
# Source engine: Z up, faces +X, left +Y:  (x,y,z) -> (z, x, y)
FRAME_SOURCE = [[0, 0, 1], [1, 0, 0], [0, 1, 0]]


# ---------------------------------------------------------------------------
# avatar.json loading
# ---------------------------------------------------------------------------

class Mesh:
    def __init__(self, d):
        self.name = d["name"]
        self.component = d["component"]
        self.material = d["material"]
        self.is_prop = d.get("is_prop", False)
        n = d["vertex_count"]
        P = d["positions"]
        N = d["normals"]
        U = d["uv"]
        C = d.get("colors")
        J = d["joints"]
        W = d["weights"]
        self.positions = [P[3 * i:3 * i + 3] for i in range(n)]
        self.normals = [N[3 * i:3 * i + 3] for i in range(n)]
        self.uvs = [U[2 * i:2 * i + 2] for i in range(n)]
        self.colors = [[c / 255.0 for c in C[4 * i:4 * i + 4]] for i in range(n)] if C else None
        self.joints = [J[4 * i:4 * i + 4] for i in range(n)]
        self.weights = [W[4 * i:4 * i + 4] for i in range(n)]
        self.indices = d["indices"]


class Avatar:
    def __init__(self, json_path):
        self.path = json_path
        self.dir = os.path.dirname(os.path.abspath(json_path))
        with open(json_path, "r", encoding="utf-8") as f:
            d = json.load(f)
        self.raw = d
        self.info = d["avatar"]
        self.skeleton = d["skeleton"]["joints"]
        self.components = d["components"]
        self.materials = d["materials"]
        self.meshes = [Mesh(m) for m in d["meshes"]]
        self.animations = d.get("animations", [])
        self.prop_skeleton = d.get("prop_skeleton")
        self.face = d.get("face", {})

    def joint_names(self):
        return [j["name"] for j in self.skeleton]

    def joint_world_positions(self):
        return [j["rest_world"] for j in self.skeleton]

    def height_m(self):
        top = 0.0
        bottom = 1e9
        for m in self.meshes:
            if m.is_prop:
                continue
            for p in m.positions:
                top = max(top, p[1])
                bottom = min(bottom, p[1])
        return max(0.01, top - bottom)


# ---------------------------------------------------------------------------
# Skeleton abstraction used by the writers
# ---------------------------------------------------------------------------

class Bone:
    __slots__ = ("name", "parent", "world", "local", "tip", "roll", "collection", "index")

    def __init__(self, name, parent, world, local, tip=None, roll=0.0, collection=""):
        self.name = name
        self.parent = parent  # index or -1
        self.world = world    # 4x4 in the target frame/units
        self.local = local    # 4x4 relative to parent
        self.tip = tip        # armature-space tail offset (for Blender extras)
        self.roll = roll
        self.collection = collection
        self.index = -1


class Rig:
    """A bone hierarchy in a target frame + a per-vertex (joint,weight) remap."""

    def __init__(self, bones, remap, frame, scale):
        self.bones = bones          # list[Bone], parents before children
        self.remap = remap          # source joint index -> bone index (or -1)
        self.frame = frame          # 3x3 applied to source positions/normals
        self.scale = scale          # unit scale applied after the frame
        self.by_name = {b.name: i for i, b in enumerate(bones)}
        for i, b in enumerate(bones):
            b.index = i

    def xform_point(self, p):
        return vscale(r3_vec(self.frame, p), self.scale)

    def xform_dir(self, n):
        return vnorm(r3_vec(self.frame, n))

    def reframed(self, rot3, scale):
        """A copy of this rig rigidly re-expressed in another frame: every bone
        world matrix is rotated by rot3 and its translation scaled; locals are
        rebuilt. Used to hand the Workshop-named rig to glTF (Y up, metres)."""
        bones = []
        for b in self.bones:
            r = r3_mul(rot3, mat_rot3(b.world))
            t = vscale(r3_vec(rot3, mat_trans(b.world)), scale)
            world = mat_from_rt(r, t)
            local = mat_mul(mat_inverse_rigid(bones[b.parent].world), world) if b.parent >= 0 else world
            tip = vscale(r3_vec(rot3, b.tip), scale) if b.tip else None
            bones.append(Bone(b.name, b.parent, world, local, tip, b.roll, b.collection))
        rig = Rig(bones, list(self.remap), r3_mul(rot3, self.frame), self.scale * scale)
        if hasattr(self, "height_ratio"):
            rig.height_ratio = self.height_ratio
        return rig

    def vertex_influences(self, joints, weights):
        acc = {}
        for j, w in zip(joints, weights):
            if w <= 0.0:
                continue
            b = self.remap[j] if 0 <= j < len(self.remap) else -1
            if b < 0:
                continue
            acc[b] = acc.get(b, 0.0) + w
        if not acc:
            return [(0, 1.0)]
        total = sum(acc.values()) or 1.0
        infl = sorted(((b, w / total) for b, w in acc.items()), key=lambda x: -x[1])
        return infl


def build_source_rig(avatar, frame, scale, joints=None):
    """The export's own skeleton (Xbox joint names) in an arbitrary frame."""
    joints = joints if joints is not None else avatar.skeleton
    bones = []
    world_rots = []
    for i, j in enumerate(joints):
        pos = vscale(r3_vec(frame, j["rest_world"]), scale)
        rot = r3_mul(frame, r3_mul(quat_to_r3(j["rest_world_rot"]), r3_t(frame)))
        world = mat_from_rt(rot, pos)
        parent = j["parent"]
        if parent >= 0:
            local = mat_mul(mat_inverse_rigid(bones[parent].world), world)
        else:
            local = world
        bones.append(Bone(j["name"], parent, world, local))
        world_rots.append(rot)
    # tips: first child offset, else a short stub along +Y
    for i, b in enumerate(bones):
        child = next((c for c in bones if c.parent == i), None)
        if child is not None:
            b.tip = vsub(mat_trans(child.world), mat_trans(b.world))
        else:
            b.tip = [0.0, 0.0, 0.03 * scale]
    remap = list(range(len(joints)))
    return Rig(bones, remap, frame, scale)


# ---------------------------------------------------------------------------
# Tower Unite Workshop Rig 3.0 retarget
# ---------------------------------------------------------------------------

# Which Xbox joint supplies the POSITION of each TU bone (d = l/r).
TU_POSITION_SOURCE = {
    "pelvis": "BASE", "spine_01": "SC_BASE", "spine_02": "SC_BACKA", "spine_03": "BACKB",
    "neck_01": "NECK", "head": "HEAD",
    "clavicle_{d}": "{S}_C", "upperarm_{d}": "{S}_S", "upperarm_twist_01_{d}": "{S}_SC_TWIST_S",
    "upperarm_twist_02_{d}": "{S}_E", "lowerarm_{d}": "{S}_E", "lowerarm_twist_01_{d}": "{S}_E",
    "lowerarm_twist_02_{d}": "{S}_E_TWIST", "hand_{d}": "{S}_W",
    "index_01_{d}": "{S}_FINGA", "index_02_{d}": "{S}_FINGA1", "index_03_{d}": "{S}_FINGA2",
    "middle_01_{d}": "{S}_FINGB", "middle_02_{d}": "{S}_FINGB1", "middle_03_{d}": "{S}_FINGB2",
    "ring_01_{d}": "{S}_FINGC", "ring_02_{d}": "{S}_FINGC1", "ring_03_{d}": "{S}_FINGC2",
    "pinky_01_{d}": "{S}_FINGD", "pinky_02_{d}": "{S}_FINGD1", "pinky_03_{d}": "{S}_FINGD2",
    "thumb_01_{d}": "{S}_THUMB", "thumb_02_{d}": "{S}_THUMB1", "thumb_03_{d}": "{S}_THUMB2",
    "gun_{d}": "{S}_PROP",
    "thigh_{d}": "{S}_H", "calf_{d}": "{S}_K", "calf_twist_01_{d}": "{S}_K",
    "calf_twist_02_{d}": "{S}_A", "foot_{d}": "{S}_A", "ball_{d}": "{S}_T",
    "ik_foot_{d}": "{S}_A",
}

# Which TU bone receives each Xbox joint's skin WEIGHTS.
XBOX_WEIGHT_TARGET = {
    "BASE": "pelvis", "BACKA": "spine_01", "SC_BASE": "spine_01", "SC_BACKA": "spine_02",
    "BACKB": "spine_03", "SC_BACKB": "spine_03", "NECK": "neck_01", "SC_NECK": "neck_01",
    "HEAD": "head",
    "{S}_C": "clavicle_{d}", "{S}_S": "upperarm_{d}", "{S}_SC_S": "upperarm_{d}",
    "{S}_SC_TWIST_S": "upperarm_twist_01_{d}", "{S}_E": "lowerarm_{d}", "{S}_SC_E": "lowerarm_{d}",
    "{S}_E_TWIST": "lowerarm_twist_02_{d}", "{S}_W": "hand_{d}",
    "{S}_FINGA": "index_01_{d}", "{S}_FINGA1": "index_02_{d}", "{S}_FINGA2": "index_03_{d}",
    "{S}_FINGB": "middle_01_{d}", "{S}_FINGB1": "middle_02_{d}", "{S}_FINGB2": "middle_03_{d}",
    "{S}_FINGC": "ring_01_{d}", "{S}_FINGC1": "ring_02_{d}", "{S}_FINGC2": "ring_03_{d}",
    "{S}_FINGD": "pinky_01_{d}", "{S}_FINGD1": "pinky_02_{d}", "{S}_FINGD2": "pinky_03_{d}",
    "{S}_THUMB": "thumb_01_{d}", "{S}_THUMB1": "thumb_02_{d}", "{S}_THUMB2": "thumb_03_{d}",
    "{S}_PROP": "gun_{d}", "{S}_SPECIAL": "hand_{d}",
    "{S}_H": "thigh_{d}", "{S}_SC_H": "thigh_{d}", "{S}_K": "calf_{d}", "{S}_SC_K": "calf_{d}",
    "{S}_A": "foot_{d}", "{S}_T": "ball_{d}",
}


def _expand_sides(table):
    out = {}
    for k, v in table.items():
        if "{S}" in k or "{d}" in k:
            for S, d in (("LF", "l"), ("RT", "r")):
                out[k.replace("{S}", S).replace("{d}", d)] = v.replace("{S}", S).replace("{d}", d)
        else:
            out[k] = v
    return out


TU_POSITION_SOURCE_X = _expand_sides(TU_POSITION_SOURCE)
XBOX_WEIGHT_TARGET_X = _expand_sides(XBOX_WEIGHT_TARGET)


def load_tu_template(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def build_tu_rig(avatar, template, include_optional=True):
    """Tower Unite rig: template bone names/orientations, our joint positions.

    Positions: each TU bone sits at its source Xbox joint (scaled, in the TU
    frame, cm). Orientations: the template's LOCAL rotations, composed down the
    hierarchy, so world bone angles equal the Workshop Rig's ("only change the
    bone lengths"). Translations are re-expressed in the template-oriented
    parent frame.
    """
    names = avatar.joint_names()
    jpos = {n: vscale(r3_vec(FRAME_TU, p), 100.0)
            for n, p in zip(names, avatar.joint_world_positions())}

    tbones = template["bones"]
    tindex = {b["name"]: i for i, b in enumerate(tbones)}
    # template local rotations + world rotations
    t_local_rot = []
    t_world_rot = []
    t_world_pos = []
    for b in tbones:
        m = b["local_matrix"]
        M = [m[0:4], m[4:8], m[8:12], m[12:16]]
        r = r3_normalize_columns(mat_rot3(M))
        t_local_rot.append(r)
        p = tindex.get(b["parent"], -1) if b["parent"] else -1
        if p >= 0:
            wr = r3_mul(t_world_rot[p], r)
            wp = vadd(t_world_pos[p], r3_vec(t_world_rot[p], mat_trans(M)))
        else:
            wr = r
            wp = mat_trans(M)
        t_world_rot.append(wr)
        t_world_pos.append(wp)

    # our world positions per TU bone
    pos = {}
    for b in tbones:
        n = b["name"]
        src = TU_POSITION_SOURCE_X.get(n)
        if src and src in jpos:
            pos[n] = jpos[src][:]
    # synthesized bones
    pos["root"] = [0.0, 0.0, 0.0]
    pos["ik_foot_root"] = [0.0, 0.0, 0.0]
    for d in ("l", "r"):
        if f"calf_{d}" in pos:
            pos[f"ik_knee_{d}"] = vadd(pos[f"calf_{d}"], [0.0, -25.0, 0.0])
        if f"lowerarm_{d}" in pos:
            pos[f"ik_elbow_{d}"] = vadd(pos[f"lowerarm_{d}"], [0.0, 25.0, 0.0])
    # anything still missing: place at the parent (zero length)
    order = []
    for b in tbones:
        n = b["name"]
        if n not in pos:
            p = b["parent"]
            pos[n] = pos.get(p, [0.0, 0.0, 0.0])[:]
        order.append(n)

    optional = {"upperarm_twist_01", "upperarm_twist_02", "lowerarm_twist_01", "lowerarm_twist_02",
                "calf_twist_01", "calf_twist_02"}
    keep = []
    for b in tbones:
        base = b["name"][:-2] if b["name"].endswith(("_l", "_r")) else b["name"]
        if not include_optional and base in optional:
            continue
        keep.append(b)

    bones = []
    by_name = {}
    height_ratio = 1.0
    if "head" in pos and "pelvis" in pos:
        ours = pos["head"][2]
        theirs = t_world_pos[tindex["head"]][2] or 1.0
        height_ratio = max(0.2, ours / theirs)
    for b in keep:
        n = b["name"]
        pidx = by_name.get(b["parent"], -1) if b["parent"] else -1
        wr = t_world_rot[tindex[n]]
        wp = pos[n]
        world = mat_from_rt(wr, wp)
        if pidx >= 0:
            pw = bones[pidx].world
            local_t = r3_vec(r3_t(mat_rot3(pw)), vsub(wp, mat_trans(pw)))
            local = mat_from_rt(t_local_rot[tindex[n]], local_t)
        else:
            local = world
        tip = [c * height_ratio for c in b.get("tip", [0, 0, 5])]
        bone = Bone(n, pidx, world, local, tip, b.get("roll", 0.0), b.get("collection", ""))
        by_name[n] = len(bones)
        bones.append(bone)
    # tips toward the matching child when one exists (keeps bone lengths honest)
    for i, b in enumerate(bones):
        children = [c for c in bones if c.parent == i]
        if not children:
            continue
        best, best_dot = None, -2.0
        tdir = vnorm(b.tip)
        for c in children:
            off = vsub(mat_trans(c.world), mat_trans(b.world))
            if vlen(off) < 1e-3:
                continue
            dd = vdot(vnorm(off), tdir)
            if dd > best_dot:
                best, best_dot = off, dd
        if best is not None and best_dot > 0.5:
            b.tip = best

    remap = []
    for n in names:
        t = XBOX_WEIGHT_TARGET_X.get(n)
        remap.append(by_name.get(t, -1) if t else -1)
    rig = Rig(bones, remap, FRAME_TU, 100.0)
    rig.height_ratio = height_ratio
    return rig


# ---------------------------------------------------------------------------
# Animation evaluation (FK on the source skeleton) + retarget helpers
# ---------------------------------------------------------------------------

def anim_local_matrices(avatar, anim, frame_index):
    """Local matrices (source frame, metres) of every source joint at a frame."""
    locs = []
    tracks = anim["tracks"]
    for ji, j in enumerate(avatar.skeleton):
        if ji < len(tracks) and frame_index < len(tracks[ji]["t"]):
            tr = tracks[ji]
            t = tr["t"][frame_index]
            r = quat_to_r3(tr["r"][frame_index])
            s = tr["s"][frame_index]
            r = [[r[i][k] * s[k] for k in range(3)] for i in range(3)]
            locs.append(mat_from_rt(r, t))
        else:
            locs.append(mat_from_rt(quat_to_r3(j["rest_world_rot"]), j["rest_local"]))
    return locs


def fk_world(avatar, locals_):
    worlds = []
    for ji, j in enumerate(avatar.skeleton):
        p = j["parent"]
        worlds.append(mat_mul(worlds[p], locals_[ji]) if p >= 0 else locals_[ji])
    return worlds


def retarget_frame_to_rig(avatar, rig, worlds_src, tu_mode):
    """World matrices for rig bones at an animated frame.

    Source-rig: the source world matrices moved into the rig frame/units.
    TU-rig: each TU bone follows its position-source joint; its orientation is
    the template rest orientation rotated by the source joint's world rotation
    delta (source rest rotations are identity, so the delta is the animated
    world rotation itself).
    """
    out = [None] * len(rig.bones)
    names = avatar.joint_names()
    name_to_world = {}
    for n, w in zip(names, worlds_src):
        pos = vscale(r3_vec(rig.frame, mat_trans(w)), rig.scale)
        rot = r3_mul(rig.frame, r3_mul(r3_normalize_columns(mat_rot3(w)), r3_t(rig.frame)))
        name_to_world[n] = (pos, rot)
    if not tu_mode:
        for i, b in enumerate(rig.bones):
            pos, rot = name_to_world[b.name]
            out[i] = mat_from_rt(rot, pos)
        return out
    for i, b in enumerate(rig.bones):
        src = TU_POSITION_SOURCE_X.get(b.name)
        if src in name_to_world:
            pos, rot = name_to_world[src]
            rest_rot = mat_rot3(b.world)
            out[i] = mat_from_rt(r3_mul(rot, rest_rot), pos)
        else:
            # synthesized bones ride their parent (root stays put)
            if b.parent >= 0:
                parent_rest = rig.bones[b.parent].world
                parent_now = out[b.parent]
                rel = mat_mul(mat_inverse_rigid(parent_rest), b.world)
                out[i] = mat_mul(parent_now, rel)
            else:
                out[i] = b.world
    return out


def rig_locals_from_worlds(rig, worlds):
    locs = []
    for i, b in enumerate(rig.bones):
        if b.parent >= 0:
            locs.append(mat_mul(mat_inverse(worlds[b.parent]), worlds[i]))
        else:
            locs.append(worlds[i])
    return locs


# ---------------------------------------------------------------------------
# COLLADA writer
# ---------------------------------------------------------------------------

def _fmt(v, nd=6):
    s = f"{v:.{nd}f}"
    if "." in s:
        s = s.rstrip("0").rstrip(".")
    return s if s not in ("-0", "") else "0"


def _mat_text(m, nd=6):
    return " ".join(_fmt(m[i][j], nd) for i in range(4) for j in range(4))


def _safe_id(s):
    out = "".join(c if (c.isalnum() or c in "_-.") else "_" for c in s)
    return out or "x"


def write_dae(avatar, rig, out_path, *, unit_meter, up_axis, tu_mode, animation=None,
              texture_dir=None, copy_textures=True, include_colors=True, flip_v=True,
              armature_name="Armature", scene_name="Scene", material_suffix=""):
    """COLLADA 1.4.1 with one skinned geometry per mesh + the rig hierarchy.

    texture_dir: where the <Material>_Diffuse.png files live relative to the dae
    (default: same folder; copies them there when copy_textures).
    """
    out_dir = os.path.dirname(os.path.abspath(out_path))
    os.makedirs(out_dir, exist_ok=True)
    now = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S")
    L = []
    w = L.append
    w('<?xml version="1.0" encoding="utf-8"?>')
    w('<COLLADA xmlns="http://www.collada.org/2005/11/COLLADASchema" version="1.4.1" '
      'xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance">')
    w("  <asset>")
    w("    <contributor>")
    w("      <author>ReXGlue Avatar Export</author>")
    w("      <authoring_tool>ReXGlue Avatar Export (avatarextract --avatar)</authoring_tool>")
    w("    </contributor>")
    w(f"    <created>{now}</created>")
    w(f"    <modified>{now}</modified>")
    uname = "centimeter" if abs(unit_meter - 0.01) < 1e-9 else "meter"
    w(f'    <unit name="{uname}" meter="{_fmt(unit_meter, 4)}"/>')
    w(f"    <up_axis>{up_axis}</up_axis>")
    w("  </asset>")

    # ---- materials / images / effects ----
    used_materials = sorted({m.material for m in avatar.meshes})
    mats = []
    for mi in used_materials:
        md = avatar.materials[mi]
        name = _safe_id(md["name"] + material_suffix)
        src_png = os.path.join(avatar.dir, md["diffuse"])
        dst_name = f"{name}_Diffuse.png"
        rel_dir = texture_dir or ""
        if copy_textures and os.path.isfile(src_png):
            dst_dir = os.path.join(out_dir, rel_dir) if rel_dir else out_dir
            os.makedirs(dst_dir, exist_ok=True)
            dst = os.path.join(dst_dir, dst_name)
            if os.path.abspath(dst) != os.path.abspath(src_png):
                shutil.copyfile(src_png, dst)
        init_from = (rel_dir.replace("\\", "/") + "/" if rel_dir else "") + dst_name
        mats.append((mi, name, init_from))
    w("  <library_images>")
    for mi, name, init_from in mats:
        w(f'    <image id="{name}_png" name="{name}_png">')
        w(f"      <init_from>{sx.escape(init_from)}</init_from>")
        w("    </image>")
    w("  </library_images>")
    w("  <library_effects>")
    for mi, name, init_from in mats:
        w(f'    <effect id="{name}-effect">')
        w("      <profile_COMMON>")
        w(f'        <newparam sid="{name}_png-surface">')
        w('          <surface type="2D">')
        w(f"            <init_from>{name}_png</init_from>")
        w("          </surface>")
        w("        </newparam>")
        w(f'        <newparam sid="{name}_png-sampler">')
        w("          <sampler2D>")
        w(f"            <source>{name}_png-surface</source>")
        w("          </sampler2D>")
        w("        </newparam>")
        w('        <technique sid="common">')
        w("          <lambert>")
        w('            <emission><color sid="emission">0 0 0 1</color></emission>')
        w(f'            <diffuse><texture texture="{name}_png-sampler" texcoord="UVMap"/></diffuse>')
        w('            <index_of_refraction><float sid="ior">1.45</float></index_of_refraction>')
        w("          </lambert>")
        w("        </technique>")
        w("      </profile_COMMON>")
        w("    </effect>")
    w("  </library_effects>")
    w("  <library_materials>")
    for mi, name, init_from in mats:
        w(f'    <material id="{name}-material" name="{name}">')
        w(f'      <instance_effect url="#{name}-effect"/>')
        w("    </material>")
    w("  </library_materials>")
    mat_name_by_index = {mi: name for mi, name, _ in mats}

    # ---- geometries ----
    w("  <library_geometries>")
    mesh_ids = []
    for mesh in avatar.meshes:
        gid = _safe_id(mesh.name)
        mesh_ids.append(gid)
        n = len(mesh.positions)
        pos = [rig.xform_point(p) for p in mesh.positions]
        nrm = [rig.xform_dir(v) for v in mesh.normals]
        w(f'    <geometry id="{gid}-mesh" name="{gid}">')
        w("      <mesh>")
        w(f'        <source id="{gid}-mesh-positions">')
        w(f'          <float_array id="{gid}-mesh-positions-array" count="{3 * n}">'
          + " ".join(_fmt(c, 4) for p in pos for c in p) + "</float_array>")
        w("          <technique_common>")
        w(f'            <accessor source="#{gid}-mesh-positions-array" count="{n}" stride="3">')
        w('              <param name="X" type="float"/><param name="Y" type="float"/><param name="Z" type="float"/>')
        w("            </accessor>")
        w("          </technique_common>")
        w("        </source>")
        w(f'        <source id="{gid}-mesh-normals">')
        w(f'          <float_array id="{gid}-mesh-normals-array" count="{3 * n}">'
          + " ".join(_fmt(c, 4) for v in nrm for c in v) + "</float_array>")
        w("          <technique_common>")
        w(f'            <accessor source="#{gid}-mesh-normals-array" count="{n}" stride="3">')
        w('              <param name="X" type="float"/><param name="Y" type="float"/><param name="Z" type="float"/>')
        w("            </accessor>")
        w("          </technique_common>")
        w("        </source>")
        w(f'        <source id="{gid}-mesh-map-0">')
        uvs = []
        for u, v in mesh.uvs:
            uvs.append(u)
            uvs.append(1.0 - v if flip_v else v)
        w(f'          <float_array id="{gid}-mesh-map-0-array" count="{2 * n}">'
          + " ".join(_fmt(c, 5) for c in uvs) + "</float_array>")
        w("          <technique_common>")
        w(f'            <accessor source="#{gid}-mesh-map-0-array" count="{n}" stride="2">')
        w('              <param name="S" type="float"/><param name="T" type="float"/>')
        w("            </accessor>")
        w("          </technique_common>")
        w("        </source>")
        has_colors = include_colors and mesh.colors is not None
        if has_colors:
            w(f'        <source id="{gid}-mesh-colors-Col">')
            w(f'          <float_array id="{gid}-mesh-colors-Col-array" count="{4 * n}">'
              + " ".join(_fmt(c, 4) for col in mesh.colors for c in col) + "</float_array>")
            w("          <technique_common>")
            w(f'            <accessor source="#{gid}-mesh-colors-Col-array" count="{n}" stride="4">')
            w('              <param name="R" type="float"/><param name="G" type="float"/>'
              '<param name="B" type="float"/><param name="A" type="float"/>')
            w("            </accessor>")
            w("          </technique_common>")
            w("        </source>")
        w(f'        <vertices id="{gid}-mesh-vertices">')
        w(f'          <input semantic="POSITION" source="#{gid}-mesh-positions"/>')
        w("        </vertices>")
        tri_count = len(mesh.indices) // 3
        mname = mat_name_by_index[mesh.material]
        w(f'        <triangles material="{mname}-material" count="{tri_count}">')
        w(f'          <input semantic="VERTEX" source="#{gid}-mesh-vertices" offset="0"/>')
        w(f'          <input semantic="NORMAL" source="#{gid}-mesh-normals" offset="1"/>')
        w(f'          <input semantic="TEXCOORD" source="#{gid}-mesh-map-0" offset="2" set="0"/>')
        if has_colors:
            w(f'          <input semantic="COLOR" source="#{gid}-mesh-colors-Col" offset="3" set="0"/>')
        per = 4 if has_colors else 3
        p = []
        for idx in mesh.indices:
            p.extend([str(idx)] * per)
        w("          <p>" + " ".join(p) + "</p>")
        w("        </triangles>")
        w("      </mesh>")
        w("    </geometry>")
    w("  </library_geometries>")

    # ---- controllers ----
    bone_sids = [b.name for b in rig.bones]
    inv_binds = [mat_inverse(b.world) for b in rig.bones]
    w("  <library_controllers>")
    for mesh, gid in zip(avatar.meshes, mesh_ids):
        cid = f"{armature_name}_{gid}-skin"
        n = len(mesh.positions)
        w(f'    <controller id="{cid}" name="{armature_name}">')
        w(f'      <skin source="#{gid}-mesh">')
        w("        <bind_shape_matrix>1 0 0 0 0 1 0 0 0 0 1 0 0 0 0 1</bind_shape_matrix>")
        w(f'        <source id="{cid}-joints">')
        w(f'          <Name_array id="{cid}-joints-array" count="{len(bone_sids)}">'
          + " ".join(bone_sids) + "</Name_array>")
        w("          <technique_common>")
        w(f'            <accessor source="#{cid}-joints-array" count="{len(bone_sids)}" stride="1">')
        w('              <param name="JOINT" type="name"/>')
        w("            </accessor>")
        w("          </technique_common>")
        w("        </source>")
        w(f'        <source id="{cid}-bind_poses">')
        w(f'          <float_array id="{cid}-bind_poses-array" count="{16 * len(inv_binds)}">'
          + " ".join(_mat_text(m, 6) for m in inv_binds) + "</float_array>")
        w("          <technique_common>")
        w(f'            <accessor source="#{cid}-bind_poses-array" count="{len(inv_binds)}" stride="16">')
        w('              <param name="TRANSFORM" type="float4x4"/>')
        w("            </accessor>")
        w("          </technique_common>")
        w("        </source>")
        weights = []
        vcount = []
        v = []
        for ji, wi in zip(mesh.joints, mesh.weights):
            infl = rig.vertex_influences(ji, wi)
            vcount.append(str(len(infl)))
            for b, wt in infl:
                v.append(str(b))
                v.append(str(len(weights)))
                weights.append(wt)
        w(f'        <source id="{cid}-weights">')
        w(f'          <float_array id="{cid}-weights-array" count="{len(weights)}">'
          + " ".join(_fmt(x, 5) for x in weights) + "</float_array>")
        w("          <technique_common>")
        w(f'            <accessor source="#{cid}-weights-array" count="{len(weights)}" stride="1">')
        w('              <param name="WEIGHT" type="float"/>')
        w("            </accessor>")
        w("          </technique_common>")
        w("        </source>")
        w("        <joints>")
        w(f'          <input semantic="JOINT" source="#{cid}-joints"/>')
        w(f'          <input semantic="INV_BIND_MATRIX" source="#{cid}-bind_poses"/>')
        w("        </joints>")
        w(f'        <vertex_weights count="{n}">')
        w(f'          <input semantic="JOINT" source="#{cid}-joints" offset="0"/>')
        w(f'          <input semantic="WEIGHT" source="#{cid}-weights" offset="1"/>')
        w("          <vcount>" + " ".join(vcount) + "</vcount>")
        w("          <v>" + " ".join(v) + "</v>")
        w("        </vertex_weights>")
        w("      </skin>")
        w("    </controller>")
    w("  </library_controllers>")

    # ---- animation (one clip; matrix channels on every bone) ----
    if animation is not None:
        frames = animation["frame_count"]
        fps = animation["fps"] or 30.0
        times = [f / fps for f in range(frames)]
        # per-bone local matrices per frame
        per_bone = [[] for _ in rig.bones]
        for f in range(frames):
            locs = anim_local_matrices(avatar, animation, f)
            worlds = fk_world(avatar, locs)
            rig_worlds = retarget_frame_to_rig(avatar, rig, worlds, tu_mode)
            rig_locs = rig_locals_from_worlds(rig, rig_worlds)
            for bi, m in enumerate(rig_locs):
                per_bone[bi].append(m)
        aname = _safe_id(animation["name"])
        w("  <library_animations>")
        w(f'    <animation id="action_container-{armature_name}" name="{aname}">')
        for bi, b in enumerate(rig.bones):
            nid = f"{armature_name}_{_safe_id(b.name)}"
            aid = f"{aname}_{nid}_pose_matrix"
            w(f'      <animation id="{aid}" name="{_safe_id(b.name)}">')
            w(f'        <source id="{aid}-input">')
            w(f'          <float_array id="{aid}-input-array" count="{frames}">'
              + " ".join(_fmt(t, 5) for t in times) + "</float_array>")
            w("          <technique_common>")
            w(f'            <accessor source="#{aid}-input-array" count="{frames}" stride="1">')
            w('              <param name="TIME" type="float"/>')
            w("            </accessor>")
            w("          </technique_common>")
            w("        </source>")
            w(f'        <source id="{aid}-output">')
            w(f'          <float_array id="{aid}-output-array" count="{16 * frames}">'
              + " ".join(_mat_text(m, 5) for m in per_bone[bi]) + "</float_array>")
            w("          <technique_common>")
            w(f'            <accessor source="#{aid}-output-array" count="{frames}" stride="16">')
            w('              <param name="TRANSFORM" type="float4x4"/>')
            w("            </accessor>")
            w("          </technique_common>")
            w("        </source>")
            w(f'        <source id="{aid}-interpolation">')
            w(f'          <Name_array id="{aid}-interpolation-array" count="{frames}">'
              + " ".join(["LINEAR"] * frames) + "</Name_array>")
            w("          <technique_common>")
            w(f'            <accessor source="#{aid}-interpolation-array" count="{frames}" stride="1">')
            w('              <param name="INTERPOLATION" type="name"/>')
            w("            </accessor>")
            w("          </technique_common>")
            w("        </source>")
            w(f'        <sampler id="{aid}-sampler">')
            w(f'          <input semantic="INPUT" source="#{aid}-input"/>')
            w(f'          <input semantic="OUTPUT" source="#{aid}-output"/>')
            w(f'          <input semantic="INTERPOLATION" source="#{aid}-interpolation"/>')
            w("        </sampler>")
            w(f'        <channel source="#{aid}-sampler" target="{nid}/transform"/>')
            w("      </animation>")
        w("    </animation>")
        w("  </library_animations>")

    # ---- visual scene ----
    w("  <library_visual_scenes>")
    w(f'    <visual_scene id="{scene_name}" name="{scene_name}">')
    w(f'      <node id="{armature_name}" name="{armature_name}" type="NODE">')
    w('        <matrix sid="transform">1 0 0 0 0 1 0 0 0 0 1 0 0 0 0 1</matrix>')
    children = {}
    for i, b in enumerate(rig.bones):
        children.setdefault(b.parent, []).append(i)

    def emit_bone(i, depth):
        b = rig.bones[i]
        ind = "  " * (4 + depth)
        nid = f"{armature_name}_{_safe_id(b.name)}"
        w(f'{ind}<node id="{nid}" name="{b.name}" sid="{b.name}" type="JOINT">')
        w(f'{ind}  <matrix sid="transform">{_mat_text(b.local, 6)}</matrix>')
        tip = b.tip or [0.0, 0.0, 1.0]
        w(f"{ind}  <extra>")
        w(f'{ind}    <technique profile="blender">')
        if b.collection:
            w(f'{ind}      <collections type="string">{sx.escape(b.collection)}</collections>')
        w(f'{ind}      <connect sid="connect" type="bool">0</connect>')
        w(f'{ind}      <layer sid="layer" type="string">0</layer>')
        w(f'{ind}      <roll sid="roll" type="float">{_fmt(b.roll, 6)}</roll>')
        w(f'{ind}      <tip_x sid="tip_x" type="float">{_fmt(tip[0], 5)}</tip_x>')
        w(f'{ind}      <tip_y sid="tip_y" type="float">{_fmt(tip[1], 5)}</tip_y>')
        w(f'{ind}      <tip_z sid="tip_z" type="float">{_fmt(tip[2], 5)}</tip_z>')
        w(f"{ind}    </technique>")
        w(f"{ind}  </extra>")
        for c in children.get(i, []):
            emit_bone(c, depth + 1)
        w(f"{ind}</node>")

    for r in children.get(-1, []):
        emit_bone(r, 0)
    w("      </node>")
    root_bone = rig.bones[children.get(-1, [0])[0]]
    for mesh, gid in zip(avatar.meshes, mesh_ids):
        cid = f"{armature_name}_{gid}-skin"
        mname = mat_name_by_index[mesh.material]
        w(f'      <node id="{gid}" name="{gid}" type="NODE">')
        w('        <matrix sid="transform">1 0 0 0 0 1 0 0 0 0 1 0 0 0 0 1</matrix>')
        w(f'        <instance_controller url="#{cid}">')
        w(f"          <skeleton>#{armature_name}_{_safe_id(root_bone.name)}</skeleton>")
        w("          <bind_material>")
        w("            <technique_common>")
        w(f'              <instance_material symbol="{mname}-material" target="#{mname}-material">')
        w('                <bind_vertex_input semantic="UVMap" input_semantic="TEXCOORD" input_set="0"/>')
        w("              </instance_material>")
        w("            </technique_common>")
        w("          </bind_material>")
        w("        </instance_controller>")
        w("      </node>")
    w("    </visual_scene>")
    w("  </library_visual_scenes>")
    w("  <scene>")
    w(f'    <instance_visual_scene url="#{scene_name}"/>')
    w("  </scene>")
    w("</COLLADA>")
    with open(out_path, "w", encoding="utf-8", newline="\n") as f:
        f.write("\n".join(L))
        f.write("\n")
    return out_path


# ---------------------------------------------------------------------------
# OBJ writer
# ---------------------------------------------------------------------------

def write_obj(avatar, rig, out_path, copy_textures=True, flip_v=True):
    out_dir = os.path.dirname(os.path.abspath(out_path))
    os.makedirs(out_dir, exist_ok=True)
    stem = os.path.splitext(os.path.basename(out_path))[0]
    mtl_name = stem + ".mtl"
    used = sorted({m.material for m in avatar.meshes})
    with open(os.path.join(out_dir, mtl_name), "w", encoding="utf-8", newline="\n") as mf:
        for mi in used:
            md = avatar.materials[mi]
            name = _safe_id(md["name"])
            dst = f"{name}_Diffuse.png"
            src = os.path.join(avatar.dir, md["diffuse"])
            if copy_textures and os.path.isfile(src):
                d = os.path.join(out_dir, dst)
                if os.path.abspath(d) != os.path.abspath(src):
                    shutil.copyfile(src, d)
            mf.write(f"newmtl {name}\nKa 1 1 1\nKd 1 1 1\nKs 0 0 0\nd 1\nillum 1\nmap_Kd {dst}\n")
            if md.get("has_alpha"):
                mf.write(f"map_d {dst}\n")
            mf.write("\n")
    with open(out_path, "w", encoding="utf-8", newline="\n") as f:
        f.write("# ReXGlue Avatar Export\n")
        f.write(f"mtllib {mtl_name}\n\n")
        base = 1
        for mesh in avatar.meshes:
            f.write(f"o {_safe_id(mesh.name)}\n")
            for p in mesh.positions:
                q = rig.xform_point(p)
                f.write(f"v {_fmt(q[0], 5)} {_fmt(q[1], 5)} {_fmt(q[2], 5)}\n")
            for u, v in mesh.uvs:
                f.write(f"vt {_fmt(u, 5)} {_fmt(1.0 - v if flip_v else v, 5)}\n")
            for nrm in mesh.normals:
                q = rig.xform_dir(nrm)
                f.write(f"vn {_fmt(q[0], 4)} {_fmt(q[1], 4)} {_fmt(q[2], 4)}\n")
            f.write(f"usemtl {_safe_id(avatar.materials[mesh.material]['name'])}\n")
            idx = mesh.indices
            for t in range(0, len(idx) - 2, 3):
                a, b, c = idx[t] + base, idx[t + 1] + base, idx[t + 2] + base
                f.write(f"f {a}/{a}/{a} {b}/{b}/{b} {c}/{c}/{c}\n")
            base += len(mesh.positions)
            f.write("\n")
    return out_path


# ---------------------------------------------------------------------------
# glTF 2.0 (.glb) writer
# ---------------------------------------------------------------------------

class _BinBuilder:
    def __init__(self):
        self.chunks = []
        self.views = []
        self.length = 0

    def add(self, data, target=None):
        pad = (-len(data)) % 4
        self.chunks.append(data + b"\x00" * pad)
        view = {"buffer": 0, "byteOffset": self.length, "byteLength": len(data)}
        if target:
            view["target"] = target
        self.views.append(view)
        self.length += len(data) + pad
        return len(self.views) - 1

    def blob(self):
        return b"".join(self.chunks)


def write_glb(avatar, rig, out_path, include_animations=True, include_colors=True,
              embed_textures=True, mask_alpha=True):
    """glTF: the rig's frame is whatever `rig` says; glTF itself is Y-up RH
    (+Z forward), so pass a source rig with FRAME_XBOX and scale 1 for the
    canonical file. Animations are retargeted onto the rig's bones."""
    out_dir = os.path.dirname(os.path.abspath(out_path))
    os.makedirs(out_dir, exist_ok=True)
    bb = _BinBuilder()
    accessors = []
    gltf = {
        "asset": {"version": "2.0", "generator": "ReXGlue Avatar Export"},
        "scene": 0, "scenes": [{"nodes": []}], "nodes": [], "meshes": [], "materials": [],
        "textures": [], "images": [], "samplers": [{"magFilter": 9729, "minFilter": 9987,
                                                     "wrapS": 10497, "wrapT": 10497}],
        "skins": [], "accessors": accessors, "bufferViews": bb.views, "buffers": [],
        "animations": [],
    }

    def accessor(data_bytes, comp_type, count, atype, target=None, minmax=None, normalized=False):
        vi = bb.add(data_bytes, target)
        acc = {"bufferView": vi, "componentType": comp_type, "count": count, "type": atype}
        if minmax:
            acc["min"], acc["max"] = minmax
        if normalized:
            acc["normalized"] = True
        accessors.append(acc)
        return len(accessors) - 1

    # materials + images
    mat_index = {}
    for mi, md in enumerate(avatar.materials):
        name = _safe_id(md["name"])
        src = os.path.join(avatar.dir, md["diffuse"])
        img_index = None
        if os.path.isfile(src):
            if embed_textures:
                with open(src, "rb") as f:
                    png = f.read()
                vi = bb.add(png)
                gltf["images"].append({"bufferView": vi, "mimeType": "image/png", "name": name})
            else:
                dst = f"{name}_Diffuse.png"
                shutil.copyfile(src, os.path.join(out_dir, dst))
                gltf["images"].append({"uri": dst, "name": name})
            img_index = len(gltf["images"]) - 1
            gltf["textures"].append({"sampler": 0, "source": img_index, "name": name})
        mat = {"name": name, "pbrMetallicRoughness": {"metallicFactor": 0.0, "roughnessFactor": 0.9},
               "doubleSided": bool(md.get("double_sided", False))}
        if img_index is not None:
            mat["pbrMetallicRoughness"]["baseColorTexture"] = {"index": len(gltf["textures"]) - 1}
        if mask_alpha and md.get("alpha_mask"):
            mat["alphaMode"] = "MASK"
            mat["alphaCutoff"] = 0.5
        gltf["materials"].append(mat)
        mat_index[mi] = len(gltf["materials"]) - 1

    # skeleton nodes
    node_base = 0
    joint_nodes = []
    for b in rig.bones:
        t = mat_trans(b.local)
        q = r3_to_quat(r3_normalize_columns(mat_rot3(b.local)))
        node = {"name": b.name, "translation": [float(x) for x in t], "rotation": [float(x) for x in q]}
        gltf["nodes"].append(node)
        joint_nodes.append(len(gltf["nodes"]) - 1)
    for i, b in enumerate(rig.bones):
        if b.parent >= 0:
            gltf["nodes"][joint_nodes[b.parent]].setdefault("children", []).append(joint_nodes[i])
    roots = [joint_nodes[i] for i, b in enumerate(rig.bones) if b.parent < 0]
    # inverse bind matrices
    ibm = b""
    for b in rig.bones:
        inv = mat_inverse(b.world)
        # glTF is column-major
        ibm += struct.pack("<16f", *[inv[r][c] for c in range(4) for r in range(4)])
    ibm_acc = accessor(ibm, 5126, len(rig.bones), "MAT4")
    gltf["skins"].append({"name": "Armature", "joints": joint_nodes, "inverseBindMatrices": ibm_acc,
                          "skeleton": roots[0]})
    armature_node = {"name": "Armature", "children": roots}
    gltf["nodes"].append(armature_node)
    gltf["scenes"][0]["nodes"].append(len(gltf["nodes"]) - 1)

    # meshes
    for mesh in avatar.meshes:
        n = len(mesh.positions)
        pos = [rig.xform_point(p) for p in mesh.positions]
        nrm = [rig.xform_dir(v) for v in mesh.normals]
        pmin = [min(p[i] for p in pos) for i in range(3)]
        pmax = [max(p[i] for p in pos) for i in range(3)]
        pos_acc = accessor(b"".join(struct.pack("<3f", *p) for p in pos), 5126, n, "VEC3", 34962,
                           (pmin, pmax))
        nrm_acc = accessor(b"".join(struct.pack("<3f", *v) for v in nrm), 5126, n, "VEC3", 34962)
        uv_acc = accessor(b"".join(struct.pack("<2f", u, v) for u, v in mesh.uvs), 5126, n, "VEC2",
                          34962)
        jb = b""
        wb = b""
        for ji, wi in zip(mesh.joints, mesh.weights):
            infl = rig.vertex_influences(ji, wi)[:4]
            while len(infl) < 4:
                infl.append((0, 0.0))
            jb += struct.pack("<4H", *[x[0] for x in infl])
            wb += struct.pack("<4f", *[x[1] for x in infl])
        j_acc = accessor(jb, 5123, n, "VEC4", 34962)
        w_acc = accessor(wb, 5126, n, "VEC4", 34962)
        attrs = {"POSITION": pos_acc, "NORMAL": nrm_acc, "TEXCOORD_0": uv_acc,
                 "JOINTS_0": j_acc, "WEIGHTS_0": w_acc}
        if include_colors and mesh.colors is not None:
            cb = b"".join(struct.pack("<4f", *c) for c in mesh.colors)
            attrs["COLOR_0"] = accessor(cb, 5126, n, "VEC4", 34962)
        idx_acc = accessor(struct.pack(f"<{len(mesh.indices)}H", *mesh.indices), 5123,
                           len(mesh.indices), "SCALAR", 34963)
        gltf["meshes"].append({"name": _safe_id(mesh.name), "primitives": [
            {"attributes": attrs, "indices": idx_acc, "material": mat_index[mesh.material], "mode": 4}]})
        gltf["nodes"].append({"name": _safe_id(mesh.name), "mesh": len(gltf["meshes"]) - 1, "skin": 0})
        gltf["scenes"][0]["nodes"].append(len(gltf["nodes"]) - 1)

    # animations
    if include_animations:
        tu_mode = getattr(rig, "height_ratio", None) is not None
        for anim in avatar.animations:
            frames = anim["frame_count"]
            fps = anim["fps"] or 30.0
            if frames <= 0:
                continue
            times = [f / fps for f in range(frames)]
            per_bone_t = [[] for _ in rig.bones]
            per_bone_r = [[] for _ in rig.bones]
            per_bone_s = [[] for _ in rig.bones]
            for f in range(frames):
                locs = anim_local_matrices(avatar, anim, f)
                worlds = fk_world(avatar, locs)
                rig_worlds = retarget_frame_to_rig(avatar, rig, worlds, tu_mode)
                rig_locs = rig_locals_from_worlds(rig, rig_worlds)
                for bi, m in enumerate(rig_locs):
                    r = mat_rot3(m)
                    sc = [vlen([r[0][c], r[1][c], r[2][c]]) for c in range(3)]
                    rn = [[r[i][c] / (sc[c] or 1.0) for c in range(3)] for i in range(3)]
                    per_bone_t[bi].append(mat_trans(m))
                    per_bone_r[bi].append(r3_to_quat(rn))
                    per_bone_s[bi].append(sc)
            t_acc = accessor(struct.pack(f"<{frames}f", *times), 5126, frames, "SCALAR", None,
                             ([times[0]], [times[-1]]))
            channels = []
            samplers = []
            for bi in range(len(rig.bones)):
                # quaternion continuity
                qs = per_bone_r[bi]
                for k in range(1, len(qs)):
                    if sum(a * b for a, b in zip(qs[k], qs[k - 1])) < 0:
                        qs[k] = [-x for x in qs[k]]
                tacc = accessor(b"".join(struct.pack("<3f", *t) for t in per_bone_t[bi]), 5126, frames, "VEC3")
                racc = accessor(b"".join(struct.pack("<4f", *q) for q in qs), 5126, frames, "VEC4")
                sacc = accessor(b"".join(struct.pack("<3f", *s) for s in per_bone_s[bi]), 5126, frames, "VEC3")
                for path, acc in (("translation", tacc), ("rotation", racc), ("scale", sacc)):
                    samplers.append({"input": t_acc, "output": acc, "interpolation": "LINEAR"})
                    channels.append({"sampler": len(samplers) - 1,
                                     "target": {"node": joint_nodes[bi], "path": path}})
            gltf["animations"].append({"name": _safe_id(anim["name"]), "channels": channels,
                                       "samplers": samplers,
                                       "extras": {"face": anim.get("face", {}),
                                                  "fps": fps}})
    if not gltf["animations"]:
        del gltf["animations"]

    # face metadata as scene extras
    gltf["scenes"][0]["extras"] = {"avatar": avatar.info, "face": avatar.face.get("slots", {})}

    blob = bb.blob()
    gltf["buffers"].append({"byteLength": len(blob)})
    js = json.dumps(gltf, separators=(",", ":")).encode("utf-8")
    js += b" " * ((-len(js)) % 4)
    total = 12 + 8 + len(js) + 8 + len(blob)
    with open(out_path, "wb") as f:
        f.write(struct.pack("<4sII", b"glTF", 2, total))
        f.write(struct.pack("<I4s", len(js), b"JSON"))
        f.write(js)
        f.write(struct.pack("<I4s", len(blob), b"BIN\x00"))
        f.write(blob)
    return out_path


# ---------------------------------------------------------------------------
# Source SMD writer
# ---------------------------------------------------------------------------

def _smd_nodes(rig):
    lines = ["nodes"]
    for i, b in enumerate(rig.bones):
        lines.append(f'{i} "{b.name}" {b.parent}')
    lines.append("end")
    return lines


def _smd_skeleton_frame(rig, locals_, time):
    lines = [f"time {time}"]
    for i, m in enumerate(locals_):
        t = mat_trans(m)
        e = r3_to_euler_xyz(r3_normalize_columns(mat_rot3(m)))
        lines.append(f"{i} {_fmt(t[0], 5)} {_fmt(t[1], 5)} {_fmt(t[2], 5)} "
                     f"{_fmt(e[0], 6)} {_fmt(e[1], 6)} {_fmt(e[2], 6)}")
    return lines


def write_smd_reference(avatar, rig, out_path, copy_textures=True):
    out_dir = os.path.dirname(os.path.abspath(out_path))
    os.makedirs(out_dir, exist_ok=True)
    L = ["version 1"]
    L += _smd_nodes(rig)
    L.append("skeleton")
    L += _smd_skeleton_frame(rig, [b.local for b in rig.bones], 0)
    L.append("end")
    L.append("triangles")
    for mesh in avatar.meshes:
        md = avatar.materials[mesh.material]
        mname = _safe_id(md["name"])
        if copy_textures:
            src = os.path.join(avatar.dir, md["diffuse"])
            if os.path.isfile(src):
                shutil.copyfile(src, os.path.join(out_dir, f"{mname}_Diffuse.png"))
        pos = [rig.xform_point(p) for p in mesh.positions]
        nrm = [rig.xform_dir(v) for v in mesh.normals]
        idx = mesh.indices
        for t in range(0, len(idx) - 2, 3):
            L.append(mname)
            for k in (idx[t], idx[t + 1], idx[t + 2]):
                infl = rig.vertex_influences(mesh.joints[k], mesh.weights[k])
                p, n = pos[k], nrm[k]
                u, v = mesh.uvs[k]
                ws = " ".join(f"{b} {_fmt(w, 5)}" for b, w in infl)
                L.append(f"{infl[0][0]} {_fmt(p[0], 5)} {_fmt(p[1], 5)} {_fmt(p[2], 5)} "
                         f"{_fmt(n[0], 4)} {_fmt(n[1], 4)} {_fmt(n[2], 4)} "
                         f"{_fmt(u, 5)} {_fmt(1.0 - v, 5)} {len(infl)} {ws}")
    L.append("end")
    with open(out_path, "w", encoding="utf-8", newline="\n") as f:
        f.write("\n".join(L) + "\n")
    return out_path


def write_smd_animation(avatar, rig, anim, out_path, tu_mode=False):
    out_dir = os.path.dirname(os.path.abspath(out_path))
    os.makedirs(out_dir, exist_ok=True)
    L = ["version 1"]
    L += _smd_nodes(rig)
    L.append("skeleton")
    for f in range(anim["frame_count"]):
        locs = anim_local_matrices(avatar, anim, f)
        worlds = fk_world(avatar, locs)
        rig_worlds = retarget_frame_to_rig(avatar, rig, worlds, tu_mode)
        rig_locs = rig_locals_from_worlds(rig, rig_worlds)
        L += _smd_skeleton_frame(rig, rig_locs, f)
    L.append("end")
    with open(out_path, "w", encoding="utf-8", newline="\n") as f:
        f.write("\n".join(L) + "\n")
    return out_path


# ---------------------------------------------------------------------------
# Face flipbook helpers (copy the tinted layers + frame composites)
# ---------------------------------------------------------------------------

def copy_face_assets(avatar, out_dir):
    face = avatar.face or {}
    written = []
    for key in ("layer_files", "composite_files"):
        for entry in face.get(key, []):
            src = os.path.join(avatar.dir, entry["file"])
            if not os.path.isfile(src):
                continue
            dst = os.path.join(out_dir, entry["file"])
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            shutil.copyfile(src, dst)
            written.append(dst)
    # a compact index of what each frame means
    index = {
        "slots": face.get("slots", {}),
        "layer_names": face.get("layer_names", {}),
        "layers": face.get("layer_files", []),
        "composites": face.get("composite_files", []),
        "animations": [{"name": a["name"], "fps": a["fps"], "face": a.get("face", {})}
                       for a in avatar.animations],
    }
    p = os.path.join(out_dir, "face", "face_index.json")
    os.makedirs(os.path.dirname(p), exist_ok=True)
    with open(p, "w", encoding="utf-8") as f:
        json.dump(index, f, indent=1)
    written.append(p)
    return written
