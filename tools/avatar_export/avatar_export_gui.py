"""Avatar Export: GUI (and CLI) that turns the avatar saved by the Avatar
Editor into model files (.dae rigged on the Workshop skeleton, .glb, .obj,
.smd) plus, from the CLI, the decoded animation clips and face frames.

Pipeline:  avatar_manifest.bin  --(avatarextract --avatar)-->  avatar.json +
baked PNGs  --(ae_convert)-->  model files.

    py avatar_export_gui.py                              GUI
    py avatar_export_gui.py --cli --out <dir> [options]  headless; --help lists them
"""
import json
import os
import shutil
import struct
import subprocess
import sys
import tempfile
import traceback

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

import ae_convert as ac  # noqa: E402

APP_NAME = "Avatar Export"

# key, GUI label, default on, output subfolder
FORMATS = [
    ("dae", ".dae   COLLADA, rigged (Workshop skeleton)", True, "dae"),
    ("glb", ".glb   glTF binary, rigged", True, "glb"),
    ("obj", ".obj   + .mtl, static", True, "obj"),
    ("smd", ".smd   Source reference", False, "smd"),
]
CLI_ONLY_FORMATS = [("dae-xbox", "COLLADA on the Xbox skeleton", "dae_xbox")]
STOCK_TAIL = bytes.fromhex("C1C8F109A19CB2E0")

# Preview poses: (label, stock asset-pack clip ({G} = Male/Female), frame
# fraction that shows the pose best). None = the rest (T) pose. Previews only;
# the export is always the T-pose.
POSES = [
    ("T-Pose", None, None),
    ("Stand", "Generic Stand 0", 0.5),
    ("Wave", "Generic Wave", 0.25),
    ("Clap", "Generic Clap", 0.25),
    ("Celebrate", "Generic Celebration", 0.25),
    ("Laugh", "{G} Laugh", 0.5),
    ("Angry", "{G} Angry", 0.75),
    ("Sad", "{G} Sad Cry", 0.5),
    ("Shocked", "{G} Shocked Surprised", 0.75),
    ("Yawn", "{G} Yawn", 0.25),
]


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

def bundle_dir():
    return getattr(sys, "_MEIPASS", HERE)


def exe_dir():
    if getattr(sys, "frozen", False):
        return os.path.dirname(os.path.abspath(sys.executable))
    return HERE


def expand(p):
    return os.path.expandvars(os.path.expanduser(p.strip().strip('"'))) if p else ""


def find_avatarextract():
    cands = [
        os.path.join(exe_dir(), "avatarextract.exe"),
        os.path.join(bundle_dir(), "avatarextract.exe"),
        os.path.join(HERE, "avatarextract.exe"),
        os.path.normpath(os.path.join(HERE, "..", "..", "sdk", "out", "build", "win-amd64",
                                      "tools", "avatarextract", "RelWithDebInfo", "avatarextract.exe")),
        os.path.normpath(os.path.join(HERE, "..", "..", "sdk", "out", "build", "win-amd64",
                                      "tools", "avatarextract", "Release", "avatarextract.exe")),
    ]
    for c in cands:
        if os.path.isfile(c):
            return c
    return ""


def rig_template_path():
    for d in (exe_dir(), bundle_dir(), HERE):
        p = os.path.join(d, "tu_rig_v3.json")
        if os.path.isfile(p):
            return p
    return ""


USERDATA = r"%USERPROFILE%\Documents\ReXGlue\userdata"


# The saved avatar is the one input with a single canonical location (the
# runtime writes it there), so it alone is pre-filled. The asset pack and
# closet are picked explicitly: a silently guessed pack or closet changes
# what an export contains.
def default_manifest():
    p = USERDATA + r"\avatars\avatar_manifest.bin"
    return p if os.path.isfile(expand(p)) else ""


def _read_manifest(path):
    try:
        with open(path, "rb") as f:
            d = f.read()
    except OSError:
        return b""
    return d if len(d) == 1000 else b""


def manifest_closet_guids(path):
    """GUIDs of worn items that must come from the closet (non-stock tail)."""
    d = _read_manifest(path)
    out = []
    if not d:
        return out
    for off in [0x120, 0x140] + [0x160 + 32 * i for i in range(13)]:
        g = d[off:off + 16]
        if g == b"\x00" * 16 or g[8:16] == STOCK_TAIL:
            continue
        a, b, c = struct.unpack_from(">IHH", g, 0)
        out.append(f"{a:08x}-{b:04x}-{c:04x}-{g[8]:02x}{g[9]:02x}-{g[10:16].hex()}")
    return out


def manifest_gender(path):
    """'Male' / 'Female' from the body component's gender field (c)."""
    d = _read_manifest(path)
    if not d:
        return "Male"
    c = struct.unpack_from(">H", d, 0x120 + 6)[0]
    return "Female" if c == 2 else "Male"


def pose_clip_name(pattern, gender):
    # "Animation " prefix keeps "Male Laugh" from also matching "Female Laugh".
    return "Animation " + pattern.replace("{G}", gender)


def pose_clip_filters(gender):
    """--pack-anim arguments: one "<clip>@<frame fraction>" per pose."""
    return [f"{pose_clip_name(p, gender)}@{frac}" for _, p, frac in POSES if p]


def safe_name(s):
    """Mirror of the C++ SanitizeName (preview file names)."""
    out = "".join(c if (c.isalnum() or c in "_-") else ("_" if c == " " else "") for c in s)
    return out or "item"


def pose_preview_file(label, gender):
    pattern = {l: p for l, p, _ in POSES}.get(label)
    name = "T-Pose" if not pattern else safe_name(pose_clip_name(pattern, gender))
    return f"preview_{name}.png"


# ---------------------------------------------------------------------------
# Export job
# ---------------------------------------------------------------------------

class Options:
    def __init__(self):
        self.manifest = default_manifest()
        self.pack = ""
        self.closet = ""
        self.anim_dir = ""
        self.out_dir = ""
        self.name = "Avatar"
        self.formats = {k: d for k, _, d, _ in FORMATS}
        self.formats["dae-xbox"] = False
        self.animations = False       # decode clips (CLI)
        self.pack_animations = False  # also the asset-pack clips (CLI)
        self.clip_files = False       # one .dae/.smd per clip (CLI)
        self.face_frames = False      # face flipbook frames (CLI)
        self.bake_size = 1024
        self.apply_scale = True
        self.carryable = True
        self.preview = True
        self.tool = find_avatarextract()


class ExportError(Exception):
    pass


def run_tool(args, log):
    tool = args[0]
    if not tool or not os.path.isfile(tool):
        raise ExportError("avatarextract.exe not found; build the SDK or put it next to this app")
    creation = subprocess.CREATE_NO_WINDOW if hasattr(subprocess, "CREATE_NO_WINDOW") else 0
    proc = subprocess.Popen(args, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                            text=True, encoding="utf-8", errors="replace", creationflags=creation)
    lines = []
    for line in proc.stdout:
        line = line.rstrip("\n")
        lines.append(line)
        log(line)
    proc.wait()
    return proc.returncode, lines


def bake(opts, bake_dir, log, quick=False, preview_dir=""):
    """Run avatarextract --avatar into bake_dir; returns the avatar.json path.

    quick=True is the inspect pass: 512px head bake plus the stock pose clips,
    each rendered into preview_dir. The export pass never decodes pose clips
    and renders only the T-pose preview."""
    manifest, pack, closet = expand(opts.manifest), expand(opts.pack), expand(opts.closet)
    if not manifest or not os.path.isfile(manifest):
        raise ExportError("pick the saved avatar (avatar_manifest.bin) first")
    if not pack or not os.path.isfile(pack):
        raise ExportError("AvatarAssetPack.toc not found, pick it")
    args = [opts.tool, "--avatar", manifest, bake_dir, "--toc", pack,
            "--bake-size", str(512 if quick else opts.bake_size)]
    if closet and os.path.isdir(closet):
        args += ["--closet", closet]
    if not opts.apply_scale:
        args.append("--no-scale")
    if not opts.carryable:
        args.append("--no-prop")
    if preview_dir:
        args += ["--preview-dir", preview_dir, "--preview-size", "384"]
    if quick and preview_dir:
        # the preview poses come from the stock asset-pack clips (always available)
        for f in pose_clip_filters(manifest_gender(manifest)):
            args += ["--pack-anim", f]
    if not quick:
        if opts.face_frames:
            args += ["--face-frames", "512"]
        if opts.animations and opts.anim_dir and os.path.isdir(expand(opts.anim_dir)):
            args += ["--anim-dir", expand(opts.anim_dir)]
        if opts.animations and opts.pack_animations:
            args.append("--pack-anims")
    rc, _ = run_tool(args, log)
    if rc != 0:
        raise ExportError(f"avatarextract --avatar failed (exit {rc}); see log")
    json_path = os.path.join(bake_dir, "avatar.json")
    if not os.path.isfile(json_path):
        raise ExportError("bake produced no avatar.json")
    return json_path


def inspect(opts, log):
    """Quick bake into a temp folder with every pose preview rendered.
    Returns (Avatar, {pose label: png path})."""
    d = os.path.join(tempfile.gettempdir(), "AvatarExport", "inspect")
    shutil.rmtree(d, ignore_errors=True)
    os.makedirs(d, exist_ok=True)
    json_path = bake(opts, d, log, quick=True, preview_dir=d if opts.preview else "")
    av = ac.Avatar(json_path)
    resolved = {c.get("guid") for c in av.components}
    av.missing_closet = [g for g in manifest_closet_guids(expand(opts.manifest)) if g not in resolved]
    gender = "Female" if str(av.info.get("body_type", "")).lower() == "female" else "Male"
    previews = {}
    for label, _, _ in POSES:
        p = os.path.join(d, pose_preview_file(label, gender))
        if os.path.isfile(p):
            previews[label] = p
    return av, previews


def export(opts, log, progress=None):
    out_dir = expand(opts.out_dir)
    if not out_dir:
        raise ExportError("pick an output folder")
    name = ac._safe_id(opts.name or "Avatar")
    root = os.path.join(out_dir, name)
    os.makedirs(root, exist_ok=True)
    written = []

    log("=== step 1/3: resolving + baking the avatar ===")
    json_path = bake(opts, os.path.join(root, "source"), log,
                     preview_dir=os.path.join(root, "source") if opts.preview else "")
    if progress:
        progress(40)

    log("")
    log("=== step 2/3: writing model files ===")
    av = ac.Avatar(json_path)
    log(f"  {len(av.meshes)} meshes, {len(av.materials)} materials, {len(av.animations)} clips, "
        f"height {av.height_m():.2f} m")
    xbox_rig = ac.build_source_rig(av, ac.FRAME_XBOX, 1.0)
    rigged = None
    if opts.formats.get("dae") or opts.formats.get("glb") or opts.clip_files:
        tpl_path = rig_template_path()
        if not tpl_path:
            raise ExportError("tu_rig_v3.json (Workshop skeleton template) not found")
        rigged = ac.build_tu_rig(av, ac.load_tu_template(tpl_path))
        unmapped = [av.skeleton[i]["name"] for i, b in enumerate(rigged.remap) if b < 0]
        if unmapped:
            log(f"  note: Xbox joints without a target bone: {', '.join(unmapped)}")

    if opts.formats.get("dae"):
        d = os.path.join(root, "dae")
        p = ac.write_dae(av, rigged, os.path.join(d, f"{name}.dae"), unit_meter=0.01, up_axis="Z_UP",
                         tu_mode=True)
        written.append(p)
        log(f"  .dae: {p}  (textures beside it as <Material>_Diffuse.png)")
        if opts.clip_files and av.animations:
            ad = os.path.join(d, "clips")
            for a in av.animations:
                p = ac.write_dae(av, rigged, os.path.join(ad, f"{name}_{ac._safe_id(a['name'])}.dae"),
                                 unit_meter=0.01, up_axis="Z_UP", tu_mode=True, animation=a)
                written.append(p)
            log(f"  {len(av.animations)} animated .dae clips: {ad}")
    if opts.formats.get("glb"):
        d = os.path.join(root, "glb")
        os.makedirs(d, exist_ok=True)
        # Workshop bone names/orientations, but in glTF's own frame (Y up, metres).
        glb_rig = rigged.reframed(ac.r3_t(ac.FRAME_TU), 0.01)
        p = ac.write_glb(av, glb_rig, os.path.join(d, f"{name}.glb"), include_animations=opts.animations)
        written.append(p)
        log(f"  .glb: {p}")
        if opts.animations:
            p = ac.write_glb(av, xbox_rig, os.path.join(d, f"{name}_xbox_skeleton.glb"),
                             include_animations=True)
            written.append(p)
            log(f"  .glb on the Xbox skeleton (with clips): {p}")
    if opts.formats.get("obj"):
        d = os.path.join(root, "obj")
        p = ac.write_obj(av, xbox_rig, os.path.join(d, f"{name}.obj"))
        written.append(p)
        log(f"  .obj: {p}")
    if opts.formats.get("dae-xbox"):
        d = os.path.join(root, "dae_xbox")
        p = ac.write_dae(av, xbox_rig, os.path.join(d, f"{name}.dae"), unit_meter=1.0, up_axis="Y_UP",
                         tu_mode=False)
        written.append(p)
        log(f"  .dae (Xbox skeleton): {p}")
        if opts.clip_files and av.animations:
            for a in av.animations:
                p = ac.write_dae(av, xbox_rig, os.path.join(d, "clips", f"{name}_{ac._safe_id(a['name'])}.dae"),
                                 unit_meter=1.0, up_axis="Y_UP", tu_mode=False, animation=a)
                written.append(p)
            log(f"  {len(av.animations)} animated clips: {os.path.join(d, 'clips')}")
    if opts.formats.get("smd"):
        d = os.path.join(root, "smd")
        src_rig = ac.build_source_rig(av, ac.FRAME_SOURCE, 100.0 / 2.54)  # Source units (inches)
        p = ac.write_smd_reference(av, src_rig, os.path.join(d, f"{name}_reference.smd"))
        written.append(p)
        if opts.animations:
            for a in av.animations:
                p = ac.write_smd_animation(av, src_rig, a, os.path.join(d, "anims", f"{ac._safe_id(a['name'])}.smd"))
                written.append(p)
        log(f"  .smd: {d}")
    if opts.face_frames:
        files = ac.copy_face_assets(av, root)
        log(f"  face frames: {len(files)} files under {os.path.join(root, 'face')}")
    if progress:
        progress(85)

    preview_png = ""
    if opts.preview:
        src = os.path.join(root, "source", "preview_T-Pose.png")
        if os.path.isfile(src):
            preview_png = os.path.join(root, "preview.png")
            shutil.copyfile(src, preview_png)
            written.append(preview_png)
            log("")
            log("=== step 3/3: preview ===")
            log(f"  {preview_png}")
    if progress:
        progress(100)
    return root, written, preview_png, av


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def run_cli(argv):
    import argparse
    ap_ = argparse.ArgumentParser(prog="AvatarExport", description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap_.add_argument("--cli", action="store_true", help="headless mode")
    ap_.add_argument("--manifest", help="saved avatar (default: the runtime save in userdata)")
    ap_.add_argument("--pack", help="AvatarAssetPack.toc")
    ap_.add_argument("--closet", help="closet folder")
    ap_.add_argument("--anim-dir", help="folder with *.AvatarAnimation")
    ap_.add_argument("--out", required=True, help="output folder (a <name> subfolder is created)")
    ap_.add_argument("--name", default="Avatar")
    ap_.add_argument("--formats", default="dae,glb,obj",
                     help="comma list of: dae (rigged), glb, obj, smd, dae-xbox (Xbox skeleton)")
    ap_.add_argument("--anims", action="store_true", help="decode the animation clips into .glb/.smd")
    ap_.add_argument("--pack-anims", action="store_true", help="...also the asset-pack clips")
    ap_.add_argument("--clip-files", action="store_true", help="one animated .dae per clip")
    ap_.add_argument("--face-frames", action="store_true", help="export the eye/brow/mouth frame stacks")
    ap_.add_argument("--no-scale", action="store_true", help="ignore the height/weight sliders")
    ap_.add_argument("--no-prop", action="store_true", help="skip the held item")
    ap_.add_argument("--no-preview", action="store_true")
    ap_.add_argument("--bake-size", type=int, default=1024, help="head texture size (default 1024)")
    ap_.add_argument("--tool", help="path to avatarextract.exe")
    a = ap_.parse_args(argv)
    o = Options()
    if a.manifest:
        o.manifest = a.manifest
    if a.pack:
        o.pack = a.pack
    o.closet = a.closet or ""
    o.anim_dir = a.anim_dir or ""
    if a.tool:
        o.tool = a.tool
    o.out_dir = a.out
    o.name = a.name
    o.formats = {k: False for k in list(o.formats)}
    for k in a.formats.split(","):
        k = k.strip().lower()
        if k == "tu":
            k = "dae"
        if k:
            o.formats[k] = True
    o.animations = a.anims or a.pack_anims or a.clip_files
    o.pack_animations = a.pack_anims
    o.clip_files = a.clip_files
    o.face_frames = a.face_frames
    o.apply_scale = not a.no_scale
    o.carryable = not a.no_prop
    o.preview = not a.no_preview
    o.bake_size = a.bake_size
    missing = manifest_closet_guids(expand(o.manifest))
    if missing and not (o.closet and os.path.isdir(expand(o.closet))):
        print(f"WARNING: the avatar wears {len(missing)} closet item(s) but no closet folder was found; "
              "they will fall back to defaults (pass --closet)")
    try:
        root, written, preview, _ = export(o, print)
    except ExportError as e:
        print(f"ERROR: {e}")
        return 1
    print(f"\nwrote {len(written)} files under {root}")
    return 0


# ---------------------------------------------------------------------------
# GUI
# ---------------------------------------------------------------------------

def run_gui():
    from PySide6.QtCore import QObject, QSize, Qt, QThread, QTimer, Signal
    from PySide6.QtGui import QIcon, QImage, QPixmap
    from PySide6.QtWidgets import (
        QApplication, QButtonGroup, QCheckBox, QFileDialog, QGridLayout, QGroupBox, QHBoxLayout,
        QLabel, QLineEdit, QListWidget, QListWidgetItem, QMessageBox, QProgressBar, QPushButton,
        QTextEdit, QVBoxLayout, QWidget,
    )

    class Worker(QObject):
        line = Signal(str)
        progress = Signal(int)
        done = Signal(object)
        failed = Signal(str)

        def __init__(self, opts, mode):
            super().__init__()
            self.opts = opts
            self.mode = mode

        def run(self):
            try:
                if self.mode == "inspect":
                    av, previews = inspect(self.opts, self.line.emit)
                    self.done.emit({"mode": "inspect", "previews": previews,
                                    "components": av.components, "info": av.info, "face": av.face,
                                    "missing_closet": av.missing_closet})
                else:
                    root, written, preview, av = export(self.opts, self.line.emit, self.progress.emit)
                    self.done.emit({"mode": "export", "root": root, "written": written, "preview": preview,
                                    "components": av.components, "info": av.info, "face": av.face})
            except ExportError as e:
                self.failed.emit(str(e))
            except Exception:  # noqa: BLE001, surface rather than kill the GUI
                self.failed.emit(traceback.format_exc())

    class Window(QWidget):
        def __init__(self):
            super().__init__()
            self.setWindowTitle(APP_NAME)
            self.resize(840, 780)
            self.setAcceptDrops(True)
            self.opts = Options()
            self.thread = None
            self.worker = None
            self.pose = "T-Pose"
            self.previews = {}        # pose label -> png (rendered by the inspect pass)

            root = QVBoxLayout(self)

            # -- inputs --
            box = QGroupBox("Avatar")
            g = QGridLayout(box)
            self.ed_manifest = self._path_row(g, 0, "Saved avatar:", self.opts.manifest,
                                              "avatar_manifest.bin", self.pick_manifest)
            self.ed_pack = self._path_row(g, 1, "Asset pack:", self.opts.pack, "AvatarAssetPack.toc",
                                          self.pick_pack)
            self.ed_closet = self._path_row(g, 2, "Closet:", self.opts.closet,
                                            "closet folder (marketplace items)", self.pick_closet)
            self.btn_refresh = QPushButton("Refresh")
            self.btn_refresh.setToolTip("Re-read the saved avatar and redraw the preview")
            self.btn_refresh.clicked.connect(self.inspect)
            g.addWidget(self.btn_refresh, 3, 2)
            root.addWidget(box)

            # -- preview + poses + wearing --
            mid = QHBoxLayout()
            left = QVBoxLayout()
            self.preview = QLabel()
            self.preview.setFixedSize(320, 320)
            self.preview.setAlignment(Qt.AlignCenter)
            self.preview.setStyleSheet("background:#2b4f2f; color:#cde;")
            self.preview.setText("reading the saved avatar...")
            left.addWidget(self.preview)
            pose_grid = QGridLayout()
            pose_grid.setSpacing(3)
            self.pose_group = QButtonGroup(self)
            self.pose_group.setExclusive(True)
            self.pose_buttons = {}
            for i, (label, _, _) in enumerate(POSES):
                b = QPushButton(label)
                b.setCheckable(True)
                b.setChecked(label == "T-Pose")
                b.setEnabled(False)
                b.setFixedWidth(60)
                b.clicked.connect(lambda _=False, lab=label: self.set_pose(lab))
                self.pose_group.addButton(b)
                self.pose_buttons[label] = b
                pose_grid.addWidget(b, i // 5, i % 5)
            left.addLayout(pose_grid)
            self.warn = QLabel("")
            self.warn.setWordWrap(True)
            self.warn.setFixedWidth(320)
            self.warn.setStyleSheet("color:#c0392b; font-weight:bold;")
            self.warn.hide()
            left.addWidget(self.warn)
            left.addStretch(1)
            mid.addLayout(left)
            sbox = QGroupBox("Wearing")
            sv = QVBoxLayout(sbox)
            self.items = QListWidget()
            self.items.setViewMode(QListWidget.IconMode)
            self.items.setIconSize(QSize(64, 64))
            self.items.setResizeMode(QListWidget.Adjust)
            self.items.setMovement(QListWidget.Static)
            self.items.setWrapping(True)
            self.items.setWordWrap(True)
            self.items.setSelectionMode(QListWidget.NoSelection)
            sv.addWidget(self.items, 1)
            self.summary = QLabel("")
            self.summary.setWordWrap(True)
            self.summary.setTextInteractionFlags(Qt.TextSelectableByMouse)
            sv.addWidget(self.summary)
            mid.addWidget(sbox, 1)
            root.addLayout(mid)

            # -- export --
            obox = QGroupBox("Export")
            og = QGridLayout(obox)
            og.addWidget(QLabel("Output folder:"), 0, 0)
            self.ed_out = QLineEdit("")
            self.ed_out.setPlaceholderText("a <Name> subfolder is created here, one folder per file type")
            og.addWidget(self.ed_out, 0, 1, 1, 3)
            b = QPushButton("Browse...")
            b.clicked.connect(self.pick_out)
            og.addWidget(b, 0, 4)
            og.addWidget(QLabel("Name:"), 1, 0)
            self.ed_name = QLineEdit(self.opts.name)
            og.addWidget(self.ed_name, 1, 1)
            self.fmt_checks = {}
            for i, (key, label, default, _) in enumerate(FORMATS):
                c = QCheckBox(label)
                c.setChecked(default)
                self.fmt_checks[key] = c
                og.addWidget(c, 2 + i // 2, (i % 2) * 2, 1, 2)
            root.addWidget(obox)

            run_row = QHBoxLayout()
            self.progress = QProgressBar()
            self.progress.setRange(0, 100)
            run_row.addWidget(self.progress, 1)
            self.btn_export = QPushButton("Export")
            self.btn_export.setDefault(True)
            self.btn_export.clicked.connect(self.export)
            run_row.addWidget(self.btn_export)
            root.addLayout(run_row)

            self.log_edit = QTextEdit()
            self.log_edit.setReadOnly(True)
            self.log_edit.setFontFamily("Consolas")
            root.addWidget(self.log_edit, 1)

            tool = find_avatarextract()
            if not tool:
                self.log("avatarextract.exe not found; build the SDK or put it next to this app")
            if not rig_template_path():
                self.log("tu_rig_v3.json not found, rigged .dae/.glb export unavailable")
            if self.opts.manifest and self.opts.pack and tool:
                QTimer.singleShot(150, self.inspect)
            else:
                self.preview.setText("pick the saved avatar and asset pack, then Refresh")

        def _path_row(self, grid, row, label, value, placeholder, slot):
            grid.addWidget(QLabel(label), row, 0)
            ed = QLineEdit(value)
            ed.setPlaceholderText(placeholder)
            grid.addWidget(ed, row, 1)
            b = QPushButton("Browse...")
            b.clicked.connect(slot)
            grid.addWidget(b, row, 2)
            return ed

        # -- drag & drop --
        def dragEnterEvent(self, event):
            if event.mimeData().hasUrls():
                event.acceptProposedAction()

        def dropEvent(self, event):
            for u in event.mimeData().urls():
                if u.isLocalFile():
                    p = u.toLocalFile()
                    if os.path.isdir(p):
                        if os.path.isfile(os.path.join(p, "closet_index.tsv")):
                            self.ed_closet.setText(p)
                        else:
                            self.ed_out.setText(p)
                    elif p.lower().endswith(".toc"):
                        self.ed_pack.setText(p)
                    else:
                        self.ed_manifest.setText(p)
                    break

        # -- pickers --
        def pick_manifest(self):
            p, _ = QFileDialog.getOpenFileName(self, "Saved avatar", expand(self.ed_manifest.text()),
                                               "Avatar manifest (*.bin *.Avatar);;All files (*)")
            if p:
                self.ed_manifest.setText(p)

        def pick_pack(self):
            p, _ = QFileDialog.getOpenFileName(self, "AvatarAssetPack.toc", expand(self.ed_pack.text()),
                                               "Asset pack (*.toc);;All files (*)")
            if p:
                self.ed_pack.setText(p)

        def pick_closet(self):
            p = QFileDialog.getExistingDirectory(self, "Closet folder", expand(self.ed_closet.text()))
            if p:
                self.ed_closet.setText(p)

        def pick_out(self):
            p = QFileDialog.getExistingDirectory(self, "Output folder", expand(self.ed_out.text()))
            if p:
                self.ed_out.setText(p)

        def log(self, msg):
            self.log_edit.append(msg)

        def collect(self):
            o = self.opts
            o.manifest = self.ed_manifest.text()
            o.pack = self.ed_pack.text()
            o.closet = self.ed_closet.text()
            o.out_dir = self.ed_out.text()
            o.name = self.ed_name.text().strip() or "Avatar"
            o.formats = {k: c.isChecked() for k, c in self.fmt_checks.items()}
            o.formats["dae-xbox"] = False
            o.animations = False
            o.face_frames = False
            o.clip_files = False
            o.tool = find_avatarextract()
            return o

        def _closet_warning(self, o):
            missing = manifest_closet_guids(expand(o.manifest))
            closet = expand(o.closet)
            if missing and not (closet and os.path.isdir(closet)):
                return (f"this avatar wears {len(missing)} closet item(s) and no Closet folder is set. "
                        "They will show as default clothing")
            if closet and not os.path.isfile(os.path.join(closet, "closet_index.tsv")):
                return "the Closet folder has no closet_index.tsv"
            return ""

        def _busy(self, busy):
            self.btn_export.setEnabled(not busy)
            self.btn_refresh.setEnabled(not busy)
            for label, b in self.pose_buttons.items():
                b.setEnabled(not busy and label in self.previews)

        def _start(self, mode):
            if self.thread is not None:
                return
            o = self.collect()
            if not expand(o.manifest) or not os.path.isfile(expand(o.manifest)):
                QMessageBox.warning(self, APP_NAME, "Pick the saved avatar (avatar_manifest.bin) first.")
                return
            if mode == "export" and not expand(o.out_dir):
                QMessageBox.warning(self, APP_NAME, "Pick an output folder first.")
                return
            self.log_edit.clear()
            warn = self._closet_warning(o)
            if warn:
                self.log(f"note: {warn}")
            self.progress.setValue(0)
            self._busy(True)
            if mode == "inspect":
                self.preview.setText("rendering...")
            self.thread = QThread()
            self.worker = Worker(o, mode)
            self.worker.moveToThread(self.thread)
            self.thread.started.connect(self.worker.run)
            self.worker.line.connect(self.log)
            self.worker.progress.connect(self.progress.setValue)
            self.worker.done.connect(self.on_done)
            self.worker.failed.connect(self.on_failed)
            self.worker.done.connect(self._finish)
            self.worker.failed.connect(self._finish)
            self.thread.start()

        def _finish(self, *_):
            if self.thread is not None:
                self.thread.quit()
                self.thread.wait()
                self.thread = None
                self.worker = None
            self._busy(False)

        def inspect(self):
            self.previews = {}
            self._start("inspect")

        def export(self):
            self._start("export")

        def set_pose(self, label):
            # previews are pre-rendered by the inspect pass: switching is instant
            self.pose = label
            self._show_preview(self.previews.get(label, ""))

        def on_failed(self, msg):
            self.summary.setText("Failed, see log")
            self.preview.setText("no preview")
            self.log(f"ERROR: {msg}")

        def _fill_items(self, info, components, face):
            self.items.clear()
            closet = expand(self.ed_closet.text())
            for c in components:
                label = c.get("name") or c.get("slot", "?")
                entry = QListWidgetItem(f"{c.get('slot', '')}\n{label}")
                entry.setToolTip(f"{c.get('guid')}\n{c.get('category_names')}\n{c.get('source')}")
                icon = os.path.join(closet, "icons", c.get("guid", "") + ".png") if closet else ""
                if c.get("source") == "closet" and os.path.isfile(icon):
                    img = QImage(icon)
                    if not img.isNull():
                        entry.setIcon(QIcon(QPixmap.fromImage(img)))
                self.items.addItem(entry)
            slots = (face or {}).get("slots", {})
            feats = ", ".join(f"{k}: {v.get('name', '?')}" for k, v in slots.items() if v.get("name"))
            colors = info.get("colors", ["FFFFFFFF"] * 9)
            self.summary.setText(
                f"<b>{info.get('body_type', '?')}</b> &nbsp; height {float(info.get('height_factor', 0)):+.2f} "
                f"&nbsp; weight {float(info.get('weight_factor', 0)):+.2f} &nbsp; "
                f"skin #{str(colors[0])[2:]} &nbsp; hair #{str(colors[1])[2:]}"
                + (f"<br>{feats}" if feats else ""))

        def _show_preview(self, png):
            if png and os.path.isfile(png):
                pm = QPixmap(png)
                if not pm.isNull():
                    self.preview.setPixmap(pm.scaled(320, 320, Qt.KeepAspectRatio, Qt.SmoothTransformation))
                    return
            self.preview.setText("no preview")

        def on_done(self, result):
            mode = result.get("mode")
            self._fill_items(result.get("info", {}), result.get("components", []), result.get("face"))
            if mode == "inspect":
                missing = result.get("missing_closet", [])
                if missing:
                    self.warn.setText(f"{len(missing)} worn item(s) are not in the Closet. Set the "
                                      "Closet folder and press Refresh (shown with default clothing)")
                    self.warn.show()
                else:
                    self.warn.hide()
                self.previews = result.get("previews", {})
                for label, b in self.pose_buttons.items():
                    b.setEnabled(label in self.previews)
                if self.pose not in self.previews:
                    self.pose = "T-Pose"
                    self.pose_buttons["T-Pose"].setChecked(True)
                self._show_preview(self.previews.get(self.pose, ""))
            else:
                # the export never changes the on-screen pose
                self._show_preview(self.previews.get(self.pose, result.get("preview", "")))
            if "root" in result:
                self.log("")
                self.log(f"done: {len(result['written'])} files under {result['root']}")

    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    win = Window()
    win.show()
    return app.exec()


def main():
    if "--cli" in sys.argv or (len(sys.argv) > 1 and sys.argv[1] in ("-h", "--help")):
        sys.exit(run_cli([a for a in sys.argv[1:]]))
    sys.exit(run_gui())


if __name__ == "__main__":
    main()
