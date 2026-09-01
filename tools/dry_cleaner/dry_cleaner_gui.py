"""Dry Cleaner: imports avatar item containers into the closet. Accepts a
single marketplace/award item (STFS file named by its product guid), a
title-shipped AvatarAwards container (every nested award inside), or a raw
YTGR/STRB item blob, named or nameless; a nameless blob is classified by its
asset-metadata block (STRB block 8) and gets a stable id synthesized from
the content. Outputs and usage: tools/README.md.

Everything is parsed defensively in pure python (mirrors the bounds logic of
the runtime consumers: strb.cpp WalkSTRBBlocks, compression.cpp
GetUncompressedSize, avatarextract DetectItemBodiesImpl). A container that
fails a gate the runtime itself enforces is refused, not written.
"""
import hashlib
import os
import re
import shutil
import struct
import subprocess
import sys
import tempfile
import time
import zlib

from PySide6.QtCore import QSize, Qt
from PySide6.QtGui import QIcon, QImage, QPainter, QPixmap
from PySide6.QtWidgets import (
    QApplication, QComboBox, QFileDialog, QGridLayout, QGroupBox,
    QHBoxLayout, QLabel, QLineEdit, QListWidget, QListWidgetItem,
    QMessageBox, QPushButton, QTextEdit, QVBoxLayout, QWidget,
)

# ---------------------------------------------------------------------------
# STFS container reading (defensive, read-only)
# ---------------------------------------------------------------------------

CATEGORY_BITS = [
    (1 << 0, "Head"), (1 << 1, "Body"), (1 << 2, "Hair"), (1 << 3, "Top"),
    (1 << 4, "Bottom"), (1 << 5, "Shoes"), (1 << 6, "Hat"), (1 << 7, "Gloves"),
    (1 << 8, "Glasses"), (1 << 9, "Wristwear"), (1 << 10, "Earrings"),
    (1 << 11, "Ring"), (1 << 12, "Prop"), (1 << 22, "Animation"),
]

BLOCK_NAMES = {1: "kAnimation", 2: "kTexture", 3: "kModel",
               4: "kShapeOverrides", 5: "kSkeleton",
               6: "kAssetMetadataUnversioned", 7: "kCustomColorTable",
               8: "kAssetMetadataVersioned"}


class PackageError(Exception):
    """Refusal: the container failed a gate the runtime itself enforces."""


class Result:
    def __init__(self):
        self.guid = ""            # lowercase dashed uuid
        self.categories = 0
        self.bodies = 3
        self.name = ""
        self.title_id = 0         # granting/publishing title (STFS header)
        self.title_name = ""      # game name (STFS header)
        self.description = ""     # container description = the award's reason
        self.is_award = False     # guid provenance nibble ((c >> 8) & 0xF) == 1
        self.meta = None          # STRB block 8: dict(bodies, flags, categories, subcategory)
        self.bin_bytes = b""
        self.icon_bytes = None    # validated PNG bytes or None
        self.log = []             # info lines
        self.warnings = []        # non-fatal oddities


def category_names(mask):
    names = [n for bit, n in CATEGORY_BITS if mask & bit]
    return "|".join(names) if names else "(none)"


def _u32be(d, o):
    return struct.unpack_from(">I", d, o)[0]


def _wstr_be(d, o, max_bytes):
    return d[o:o + max_bytes].decode("utf-16-be", "replace").split("\x00")[0]


def parse_stfs(data, res, expect_item=True):
    """Extract {name: bytes} from a LIVE/PIRS/CON STFS container."""
    if len(data) < 0xB000:
        raise PackageError("file too small to be an STFS container")
    magic = data[:4]
    if magic not in (b"LIVE", b"PIRS", b"CON "):
        raise PackageError(f"not an STFS container (magic {magic!r})")
    res.log.append(f"container: {magic.decode().strip()} STFS, {len(data)} bytes")

    header_size = _u32be(data, 0x340)
    content_type = _u32be(data, 0x344)
    if expect_item and content_type != 0x9000:
        raise PackageError(
            f"content type {content_type:#x} is not Avatar Item (0x9000)")
    res.name = _wstr_be(data, 0x411, 0x100).strip() or "(unnamed)"
    res.title_id = _u32be(data, 0x360)
    res.title_name = _wstr_be(data, 0x1691, 0x80).strip()
    res.description = " ".join(_wstr_be(data, 0xD11, 0x100).split())
    res.log.append(f"display name: {res.name!r}  title id: {res.title_id:08X}"
                   f"  title: {res.title_name!r}")

    # volume descriptor
    vd = 0x379
    ft_block_count = struct.unpack_from("<H", data, vd + 3)[0]
    ft_block_num = data[vd + 5] | (data[vd + 6] << 8) | (data[vd + 7] << 16)
    total_blocks = _u32be(data, vd + 0x1C)
    base = (header_size + 0xFFF) & ~0xFFF
    shift_holder = [0]

    def backing(block):
        adj = ((block + 0xAA) // 0xAA) << shift_holder[0]
        if block >= 0xAA:
            adj += ((block + 0x70E4) // 0x70E4) << shift_holder[0]
        return block + adj

    def read_block(block):
        off = base + backing(block) * 0x1000
        if off + 0x1000 > len(data):
            return None
        return data[off:off + 0x1000]

    def next_block(block):
        table_backing = backing((block // 0xAA) * 0xAA) - (1 << shift_holder[0])
        off = base + table_backing * 0x1000 + (block % 0xAA) * 0x18
        if off + 0x18 > len(data):
            return None
        e = data[off:off + 0x18]
        return (e[0x15] << 16) | (e[0x16] << 8) | e[0x17]

    # calibrate the backing shift: first file entry name must be ASCII
    first = None
    for sh in (0, 1):
        shift_holder[0] = sh
        b = read_block(ft_block_num)
        if b is None:
            continue
        n = b[0x28] & 0x3F
        if 0 < n <= 0x28 and all(0x20 <= c < 0x7F for c in b[:n]):
            first = b
            break
    if first is None:
        raise PackageError("cannot locate a sane file table (corrupt container)")

    ft = b""
    blk = ft_block_num
    for _ in range(max(1, min(ft_block_count, 16))):
        b = read_block(blk)
        if b is None:
            break
        ft += b
        nxt = next_block(blk)
        if nxt is None or nxt >= 0xFFFFFF:
            break
        blk = nxt

    files = {}
    for i in range(0, len(ft), 0x40):
        e = ft[i:i + 0x40]
        if len(e) < 0x40 or e[0] == 0:
            continue
        flags = e[0x28]
        if flags & 0x80:            # directory
            continue
        name = e[:flags & 0x3F].decode("ascii", "replace")
        alloc = e[0x29] | (e[0x2A] << 8) | (e[0x2B] << 16)
        start = e[0x2F] | (e[0x30] << 8) | (e[0x31] << 16)
        fsize = struct.unpack_from(">I", e, 0x34)[0]
        if fsize > 16 << 20 or alloc > total_blocks + 16:
            raise PackageError(f"file entry {name!r} claims absurd size {fsize}")
        out, blk, remaining = b"", start, fsize
        for _ in range(alloc + 2):
            if remaining <= 0:
                break
            b = read_block(blk)
            if b is None:
                raise PackageError(f"{name!r}: block chain leaves the file")
            take = min(0x1000, remaining)
            out += b[:take]
            remaining -= take
            nxt = next_block(blk)
            if nxt is None or nxt >= 0xFFFFFF:
                break
            blk = nxt
        if len(out) != fsize:
            raise PackageError(f"{name!r}: truncated ({len(out)}/{fsize} bytes)")
        files[name] = out
        res.log.append(f"  contains {name}: {fsize} bytes")
    if not files:
        raise PackageError("no files in the container")
    return files


# ---------------------------------------------------------------------------
# STRB validation (mirrors the runtime parsers)
# ---------------------------------------------------------------------------

def scan_signatures(blob, label, res):
    for pat in (b"MZ\x90", b"This program cannot", b"XEX2", b"PK\x03\x04",
                b"<script", b"powershell"):
        i = blob.find(pat)
        if i >= 0:
            raise PackageError(f"{label}: executable/script signature {pat!r} at {i}")


def _align(v, a):
    return (v + a - 1) // a * a


def validate_strb(blob, res):
    """Validate the STRB container; returns the bodies verdict (1/2/3)."""
    buf = blob
    if buf[:4] == b"YTGR":
        if len(buf) < 0x140:
            raise PackageError("YTGR wrapper truncated")
        buf = buf[0x140:]
        res.log.append("  YTGR signature wrapper skipped")
    if buf[:4] != b"STRB" or len(buf) < 30:
        raise PackageError(f"payload is not an STRB container (magic {buf[:4].hex()})")

    has_align = buf[4] != 0
    is_le = buf[5] != 0
    id_size, size_size = buf[22], buf[23]
    alignment = buf[26] if has_align else 1
    if alignment == 0 or id_size not in (1, 2, 4, 8) or size_size not in (1, 2, 4, 8):
        raise PackageError("malformed STRB header field widths")

    def rd(off, sz):
        return int.from_bytes(buf[off:off + sz], "little" if is_le else "big")

    bhs = _align(id_size + size_size + size_size, alignment)
    off = _align(30 if has_align else 26, alignment)
    blocks = []
    while off + bhs <= len(buf):
        bid = rd(off, id_size)
        data_size = rd(off + id_size, size_size)
        entry_size = rd(off + id_size + size_size, size_size)
        off += bhs
        payload = data_size * entry_size
        if bid not in BLOCK_NAMES:
            raise PackageError(f"unknown STRB block id {bid}")
        if off + payload > len(buf):
            raise PackageError(f"STRB block {BLOCK_NAMES[bid]} extends past container")
        blocks.append((bid, off, payload))
        off += _align(data_size, alignment)

    census = {}
    for bid, _, _ in blocks:
        census[bid] = census.get(bid, 0) + 1
    res.log.append("  blocks: " + ", ".join(
        f"{BLOCK_NAMES[k]}={v}" for k, v in sorted(census.items())))

    bodies = 3
    for bid, boff, blen in blocks:
        blk = buf[boff:boff + blen]
        if bid in (2, 3):  # kTexture/kModel: LZX chunk chain, LE headers
            if len(blk) < 12:
                raise PackageError(f"{BLOCK_NAMES[bid]} block under 12 bytes")
            o, total_unc, chunks = 0, 0, 0
            while o + 12 <= len(blk):
                cs, uo, us = struct.unpack_from("<III", blk, o)
                o += 12
                if cs > len(blk) - o:
                    raise PackageError(
                        f"{BLOCK_NAMES[bid]} chunk {chunks}: compressed data "
                        "extends past the block")
                if uo != total_unc:
                    raise PackageError(
                        f"{BLOCK_NAMES[bid]} chunk {chunks}: bad uncompressed offset")
                total_unc += us
                o += cs
                chunks += 1
            if total_unc > 64 << 20:
                raise PackageError(
                    f"{BLOCK_NAMES[bid]} claims {total_unc} uncompressed bytes")
            res.log.append(f"  {BLOCK_NAMES[bid]}: {chunks} LZX chunks, "
                           f"{total_unc} bytes uncompressed, chain in bounds")
        elif bid == 8 and len(blk) >= 15:
            # kAssetMetadataVersioned, the item's own classification:
            # version(u8)=2, bodies(u8), flags(u32 LE), categories(u32 LE),
            # subcategory(u32 LE), u8. Matches the product guid's first
            # dword on marketplace blobs and is the only category source
            # for game-shipped wearables.
            flags, cats, sub = struct.unpack_from("<III", blk, 2)
            res.meta = {"version": blk[0], "bodies": blk[1], "flags": flags,
                        "categories": cats, "subcategory": sub}
            res.log.append(f"  metadata: categories {cats:08X} = {category_names(cats)}, "
                           f"bodies {['?', 'male', 'female', 'both'][blk[1] & 3]}, "
                           f"subcategory {sub}")
        elif bid == 4:  # kShapeOverrides: raw, BitStream = LSB-first = LE
            if len(blk) < 24:
                continue
            count, _total = struct.unpack_from("<II", blk, 0)
            a, _b, c = struct.unpack_from("<IHH", blk, 8)
            if count > 8192:
                raise PackageError(
                    f"shape index_count {count} > 8192 (the runtime parser trusts it)")
            if a == 2 and c in (1, 2):
                bodies = c
    return bodies


def validate_png(d, label, res):
    """Full PNG structural validation; True = safe to carry verbatim."""
    if d[:8] != b"\x89PNG\r\n\x1a\n":
        res.warnings.append(f"{label}: not a PNG, dropped")
        return False
    o, idat, meta, saw_iend = 8, b"", None, False
    while o + 8 <= len(d):
        ln, ctype = struct.unpack_from(">I4s", d, o)
        if o + 12 + ln > len(d):
            res.warnings.append(f"{label}: truncated chunk, dropped")
            return False
        body = d[o + 8:o + 8 + ln]
        crc = struct.unpack_from(">I", d, o + 8 + ln)[0]
        if zlib.crc32(ctype + body) & 0xFFFFFFFF != crc:
            res.warnings.append(f"{label}: chunk CRC mismatch, dropped")
            return False
        if ctype == b"IHDR":
            w, h, depth, ctv, comp, filt, ilace = struct.unpack(">IIBBBBB", body)
            meta = (w, h, depth, ctv, ilace)
            if not (0 < w <= 2048 and 0 < h <= 2048) or comp or filt:
                res.warnings.append(f"{label}: bad IHDR ({w}x{h}), dropped")
                return False
        elif ctype == b"IDAT":
            idat += body
        elif not (ctype[0] & 0x20) and ctype not in (b"IHDR", b"PLTE", b"IEND"):
            res.warnings.append(f"{label}: unknown critical chunk {ctype!r}, dropped")
            return False
        o += 12 + ln
        if ctype == b"IEND":
            saw_iend = True
            break
    if not saw_iend or meta is None:
        res.warnings.append(f"{label}: no IEND/IHDR, dropped")
        return False
    if o != len(d):
        res.warnings.append(f"{label}: {len(d) - o} trailing bytes after IEND")
    try:
        raw = zlib.decompressobj().decompress(idat, 64 << 20)
    except zlib.error:
        res.warnings.append(f"{label}: IDAT inflate failed, dropped")
        return False
    w, h, depth, ctv, ilace = meta
    ch = {0: 1, 2: 3, 3: 1, 4: 2, 6: 4}.get(ctv, 0)
    if ilace == 0 and ch and len(raw) != h * (1 + (w * ch * depth + 7) // 8):
        res.warnings.append(f"{label}: raster size mismatch, dropped")
        return False
    res.log.append(f"  {label}: {w}x{h} PNG validated (CRCs, raster size)")
    return True


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------

def _guid_from_stem(stem):
    """Accept 32-hex or dashed-uuid names; return the dashed lowercase guid."""
    s = stem.lower()
    if len(s) == 32 and all(c in "0123456789abcdef" for c in s):
        return f"{s[0:8]}-{s[8:12]}-{s[12:16]}-{s[16:20]}-{s[20:32]}"
    if len(s) == 36 and s[8] == s[13] == s[18] == s[23] == "-":
        h = s.replace("-", "")
        if len(h) == 32 and all(c in "0123456789abcdef" for c in h):
            return s
    return None


def _finish_item(files, res):
    """Validate one avatar item's payloads into res (blob + optional icon)."""
    # the STRB payload: prefer asset_v2.bin, else first STRB/YTGR payload
    blob = None
    for name, content in files.items():
        if name.lower() == "asset_v2.bin":
            blob = content
            break
    if blob is None:
        for name, content in files.items():
            if content[:4] in (b"STRB", b"YTGR"):
                blob = content
                res.warnings.append(f"no asset_v2.bin; using {name!r}")
                break
    if blob is None:
        raise PackageError("container holds no STRB asset payload")

    scan_signatures(blob, "asset blob", res)
    res.bodies = validate_strb(blob, res)
    res.bin_bytes = blob
    if res.meta and res.meta["categories"] != res.categories:
        res.warnings.append(
            f"metadata says categories {res.meta['categories']:08X}, product id says "
            f"{res.categories:08X}; keeping the product id's value")

    for name, content in files.items():
        if name.lower().endswith(".png") and content is not blob:
            if validate_png(content, name, res):
                res.icon_bytes = content
            break

    for attr in ("name", "title_name", "description"):
        setattr(res, attr, getattr(res, attr).replace("\t", " ")
                .replace("\n", " ").replace("\r", " "))
    # Awards carry provenance nibble 1 in the id's c field (marketplace
    # purchases carry 2); the runtime classifies by the same nibble.
    res.is_award = ((int(res.guid[14:18], 16) >> 8) & 0xF) == 1
    res.log.append(f"verdict: clean. {['?', 'male', 'female', 'both'][res.bodies]}"
                   f" item, categories {res.categories:08X}"
                   f" = {category_names(res.categories)}"
                   + (f"; AWARD from {res.title_name or '?'} ({res.title_id:08X})"
                      if res.is_award else ""))


def _sidecar_name(path, guid):
    """Display name for a guid-named raw blob: the marketplace <guid>.xml's
    fullTitle, else the item folder's name, else the guid."""
    xml = os.path.join(os.path.dirname(path), guid + ".xml")
    for cand in (xml, os.path.splitext(path)[0] + ".xml"):
        if os.path.isfile(cand):
            try:
                text = open(cand, "rb").read().decode("utf-8", "replace")
            except OSError:
                continue
            m = re.search(r"<fullTitle>(.*?)</fullTitle>", text, re.S)
            if m and m.group(1).strip():
                return re.sub(r"&(amp|apos|quot|lt|gt);",
                              lambda k: {"amp": "&", "apos": "'", "quot": '"',
                                         "lt": "<", "gt": ">"}[k.group(1)],
                              m.group(1).strip())
    folder = os.path.basename(os.path.dirname(os.path.abspath(path)))
    return folder or guid


def _sidecar_icon(path, res):
    """Art beside a raw blob: <stem>.png, else, when the folder holds just
    this one .bin (multi-product folders pair art by download order, which a
    single file cannot prove), the marketplace thumbm/thumbsm.png."""
    folder = os.path.dirname(os.path.abspath(path))
    cands = [os.path.splitext(path)[0] + ".png"]
    try:
        bins = [f for f in os.listdir(folder) if f.lower().endswith(".bin")]
    except OSError:
        bins = []
    if len(bins) == 1:
        cands += [os.path.join(folder, "thumbm.png"), os.path.join(folder, "thumbsm.png")]
    for cand in cands:
        if os.path.isfile(cand):
            data = open(cand, "rb").read()
            if validate_png(data, os.path.basename(cand), res):
                return data
    return None


def process_raw_bin(path, data):
    """A bare YTGR/STRB item blob. Guid-named files keep their product id
    (categories from its first dword, cross-checked against the metadata
    block); nameless ones are classified by the metadata block and get a
    synthesized id: a = categories, b = content CRC16, c = bodies (provenance
    nibble 0, neither award (1) nor marketplace (2)), d = first 8 bytes of
    the content's SHA-1, so re-importing the same file yields the same id and
    dedupes."""
    res = Result()
    res.log.append(f"raw item blob: {data[:4].decode('latin1')!r}, {len(data)} bytes")
    stem = os.path.basename(path)
    if "." in stem:
        stem = stem.rsplit(".", 1)[0]
    guid = _guid_from_stem(stem)
    scan_signatures(data, "asset blob", res)
    res.bodies = validate_strb(data, res)
    res.bin_bytes = data
    if guid:
        res.guid = guid
        res.categories = int(guid.replace("-", "")[0:8], 16)
        res.title_id = int(guid.replace("-", "")[24:32], 16)
        res.name = _sidecar_name(path, guid)
        if res.meta and res.meta["categories"] != res.categories:
            res.warnings.append(
                f"metadata says categories {res.meta['categories']:08X}, product id says "
                f"{res.categories:08X}; keeping the product id's value")
    else:
        if not res.meta:
            raise PackageError(
                f"{os.path.basename(path)} has no product id and no asset-metadata "
                "block (STRB block 8), so its kind cannot be determined")
        res.categories = res.meta["categories"] & 0x1FFFFFF
        if res.meta["bodies"] & 3:
            res.bodies = res.meta["bodies"] & 3
        b = zlib.crc32(data) & 0xFFFF
        tail = hashlib.sha1(data).digest()[:8]
        if tail == bytes.fromhex("c1c8f109a19cb2e0"):  # the stock pack's tail
            tail = tail[:-1] + b"\x00"
        res.guid = (f"{res.categories:08x}-{b:04x}-{res.bodies:04x}-"
                    f"{tail[:2].hex()}-{tail[2:].hex()}")
        res.name = stem.replace("_", " ").strip() or "(unnamed)"
        res.log.append(f"no product id: classified by the metadata block, "
                       f"synthesized {res.guid} from the content")
    res.icon_bytes = _sidecar_icon(path, res)
    for attr in ("name", "title_name", "description"):
        setattr(res, attr, getattr(res, attr).replace("\t", " ")
                .replace("\n", " ").replace("\r", " "))
    res.is_award = ((int(res.guid[14:18], 16) >> 8) & 0xF) == 1
    res.log.append(f"verdict: clean. {['?', 'male', 'female', 'both'][res.bodies]}"
                   f" item, categories {res.categories:08X}"
                   f" = {category_names(res.categories)}"
                   + ("" if res.icon_bytes else "; no icon beside it"))
    return [res]


def process_package(path):
    """Returns a list of validated Results: one for a single avatar item
    container or raw blob, several for an avatar-award container (title-shipped
    AvatarAwards PIRS holding one nested <guid>.acp item container per award).
    """
    data = open(path, "rb").read()
    if len(data) > 64 << 20:
        raise PackageError("file over 64MB, not an avatar item container")
    if data[:4] in (b"YTGR", b"STRB"):
        return process_raw_bin(path, data)
    if len(data) < 0x348:
        raise PackageError("file too small to be an STFS container")
    stem = os.path.basename(path)
    if "." in stem:
        stem = stem.rsplit(".", 1)[0]

    if data[:4] in (b"LIVE", b"PIRS", b"CON ") and _u32be(data, 0x344) == 0x9000:
        # single avatar item: the closet guid comes from the FILENAME
        guid = _guid_from_stem(stem)
        if guid is None:
            raise PackageError(
                f"filename {stem!r} is not the item's product guid (32-hex "
                "or dashed). Item containers are named by their product id, "
                "which the closet needs")
        res = Result()
        res.guid = guid
        res.categories = int(guid.replace("-", "")[0:8], 16)
        files = parse_stfs(data, res)
        _finish_item(files, res)
        return [res]

    # container (e.g. a title's AvatarAwards, content type 0x2): import
    # every nested <guid>.acp avatar item container
    outer = Result()
    files = parse_stfs(data, outer, expect_item=False)
    results = []
    for name, content in files.items():
        child_stem = name.rsplit(".", 1)[0] if "." in name else name
        guid = _guid_from_stem(child_stem)
        if guid is None or content[:4] not in (b"LIVE", b"PIRS", b"CON "):
            continue
        res = Result()
        res.guid = guid
        res.categories = int(guid.replace("-", "")[0:8], 16)
        res.log.append(f"nested item: {name}")
        child_files = parse_stfs(content, res)
        _finish_item(child_files, res)
        if not res.title_name:
            # Award containers carry no title-name field of their own; the
            # game's name is the container's own display name.
            res.title_name = outer.name if outer.name != "(unnamed)" else ""
        results.append(res)
    if not results:
        raise PackageError(
            "not an avatar item container, and no nested avatar item containers "
            "found inside")
    return results


def index_line(res):
    return f"{res.guid}\t{res.categories:08X}\t{res.bodies}\t{res.name}"


def _filetime_now():
    return int((time.time() + 11644473600) * 10_000_000)


def award_line(res):
    return (f"{res.guid}\t{res.title_id:08X}\t{res.title_name}\t{res.description}"
            f"\t{_filetime_now()}")


def update_awards(results, out_dir):
    """Append a closet_awards.tsv row for every new award item (guid dedupe).
    Returns (message or None, written)."""
    awards = [r for r in results if r.is_award]
    if not awards:
        return None, []
    path = os.path.join(out_dir, "closet_awards.tsv")
    data = open(path, "rb").read() if os.path.exists(path) else b""
    listed = {l.split(b"\t")[0] for l in data.splitlines()}
    new = [r for r in awards if r.guid.encode() not in listed]
    if not new:
        return "award details already in closet_awards.tsv", []
    if data and not data.endswith(b"\n"):
        data += b"\r\n"
    seen = set()
    for r in new:
        if r.guid not in seen:
            seen.add(r.guid)
            data += award_line(r).encode("utf-8") + b"\r\n"
    with open(path, "wb") as f:
        f.write(data)
    return f"recorded {len(seen)} award detail line(s) in closet_awards.tsv", [path]


# ---------------------------------------------------------------------------
# Rendered previews for items that ship without icon art: the item is worn
# on a stock mannequin and rendered by avatarextract (the Avatar Export
# renderer). Preview only, never written to the closet.
# ---------------------------------------------------------------------------

HERE = os.path.dirname(os.path.abspath(__file__))


def _bundle_dir():
    return getattr(sys, "_MEIPASS", HERE)


def _exe_dir():
    if getattr(sys, "frozen", False):
        return os.path.dirname(os.path.abspath(sys.executable))
    return HERE


def find_avatarextract():
    for c in (os.path.join(_exe_dir(), "avatarextract.exe"),
              os.path.join(_bundle_dir(), "avatarextract.exe"),
              os.path.join(HERE, "avatarextract.exe"),
              os.path.normpath(os.path.join(HERE, "..", "..", "sdk", "out", "build", "win-amd64",
                                            "tools", "avatarextract", "RelWithDebInfo", "avatarextract.exe")),
              os.path.normpath(os.path.join(HERE, "..", "..", "sdk", "out", "build", "win-amd64",
                                            "tools", "avatarextract", "Release", "avatarextract.exe"))):
        if os.path.isfile(c):
            return c
    return ""


# The pack is never guessed from machine conventions: a copy placed next to
# the tool is the one explicit location. Without it the preview render is
# skipped (the import itself never needs the pack).
def find_asset_pack():
    for c in (os.path.join(_exe_dir(), "AvatarAssetPack.toc"),
              os.path.join(HERE, "AvatarAssetPack.toc")):
        if os.path.isfile(c):
            return c
    return ""


def mannequin_manifest(bodies):
    name = "mannequin_female.amd" if bodies == 2 else "mannequin_male.amd"
    for d in (_exe_dir(), _bundle_dir(), HERE):
        c = os.path.join(d, name)
        if os.path.isfile(c):
            return c
    return ""


def wear_on_manifest(base_path, res):
    """The 1000-byte X_AVATAR_METADATA of a blank white mannequin wearing the
    item. Component slots are 32 bytes at 0x160: asset id (16) + categories
    (u16 big-endian) + padding; the item takes the first one."""
    m = bytearray(open(base_path, "rb").read())
    if len(m) != 1000:
        raise PackageError("mannequin manifest is not 1000 bytes")
    # Blank mannequin, as the editor's own item tiles draw it: default
    # height/weight, no blend shapes, no face features, every avatar colour
    # white, no hair/clothes/props. Only the stock body and head stay.
    struct.pack_into(">ff", m, 4, 0.0, 0.0)
    m[0x0C:0x3C] = bytes(0x30)
    m[0x3C:0xFC] = bytes(0xC0)
    for i in range(9):
        struct.pack_into(">I", m, 0xFC + 4 * i, 0xFFFFFFFF)
    m[0x160:0x300] = bytes(0x1A0)
    m[0x300:0x380] = bytes(0x80)
    cats = res.categories & 0x1FFF
    free = None
    for i in range(13):
        off = 0x160 + 32 * i
        empty = m[off:off + 16] == bytes(16)
        if not empty and (struct.unpack_from(">H", m, off + 16)[0] & cats):
            m[off:off + 32] = bytes(32)
            empty = True
        if empty and free is None:
            free = off
    if free is None:
        raise PackageError("no free component slot on the mannequin")
    m[free:free + 16] = bytes.fromhex(res.guid.replace("-", ""))
    struct.pack_into(">H", m, free + 16, cats)
    return bytes(m)


def render_item_preview(res, log, size=192):
    """PNG bytes of the item worn on a mannequin in the Stand pose, or None
    (reason logged)."""
    tool, pack, base = find_avatarextract(), find_asset_pack(), mannequin_manifest(res.bodies)
    missing = [n for n, v in (("avatarextract.exe", tool), ("AvatarAssetPack.toc", pack),
                              ("mannequin manifest", base)) if not v]
    if missing:
        log("preview render unavailable, missing " + ", ".join(missing))
        return None
    work = tempfile.mkdtemp(prefix="drycleaner_")
    try:
        closet = os.path.join(work, "closet")
        os.makedirs(closet)
        with open(os.path.join(closet, res.guid + ".bin"), "wb") as f:
            f.write(res.bin_bytes)
        with open(os.path.join(closet, "closet_index.tsv"), "wb") as f:
            f.write((index_line(res) + "\r\n").encode("utf-8"))
        manifest = os.path.join(work, "wear.bin")
        with open(manifest, "wb") as f:
            f.write(wear_on_manifest(base, res))
        out_dir, prev_dir = os.path.join(work, "out"), os.path.join(work, "prev")
        # The stock "Generic Stand 0" clip at its midpoint: the relaxed stance
        # the Avatar Export GUI calls "Stand" (a T-pose reads as a rig test).
        args = [tool, "--avatar", manifest, out_dir, "--toc", pack, "--closet", closet,
                "--bake-size", "256", "--preview-dir", prev_dir, "--preview-size", str(size),
                "--no-scale", "--pack-anim", "Animation Generic Stand 0@0.5"]
        creation = subprocess.CREATE_NO_WINDOW if hasattr(subprocess, "CREATE_NO_WINDOW") else 0
        try:
            proc = subprocess.run(args, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                  text=True, encoding="utf-8", errors="replace",
                                  creationflags=creation, timeout=180)
        except subprocess.TimeoutExpired:
            log("preview render timed out")
            return None
        png = os.path.join(prev_dir, "preview_Animation_Generic_Stand_0.png")
        if not os.path.isfile(png) and os.path.isdir(prev_dir):
            stands = [f for f in os.listdir(prev_dir) if "Stand" in f and f.endswith(".png")]
            png = os.path.join(prev_dir, stands[0] if stands else "preview_T-Pose.png")
        if proc.returncode != 0 or not os.path.isfile(png):
            log(f"preview render failed (exit {proc.returncode})")
            for line in (proc.stdout or "").splitlines()[-6:]:
                log("  " + line)
            return None
        with open(png, "rb") as f:
            return f.read()
    except (OSError, PackageError) as e:
        log(f"preview render failed: {e}")
        return None
    finally:
        shutil.rmtree(work, ignore_errors=True)


def list_award_titles(out_dir):
    """[(title-id-hex, game name)] for every game with awards in the closet,
    from closet_awards.tsv (names) and closet_titles.tsv (fallback names)."""
    names = {}
    try:
        for line in open(os.path.join(out_dir, "closet_titles.tsv"), "rb").read().splitlines():
            f = line.decode("utf-8", "replace").lstrip("\ufeff").split("\t")
            if len(f) >= 2 and f[0]:
                names.setdefault(f[0].upper(), f[1])
    except OSError:
        pass
    titles = {}
    try:
        for line in open(os.path.join(out_dir, "closet_awards.tsv"), "rb").read().splitlines():
            f = line.decode("utf-8", "replace").lstrip("\ufeff").split("\t")
            if len(f) >= 2 and f[1]:
                tid = f[1].upper()
                titles.setdefault(tid, (f[2] if len(f) >= 3 and f[2] else "") or names.get(tid, ""))
    except OSError:
        pass
    return sorted(titles.items(), key=lambda kv: (kv[1].lower(), kv[0]))


ICON_EXTS = ("png", "jpg", "jpeg", "bmp", "gif")


def validate_icon_image(d, label, res):
    """Accept the common image formats the runtime's decoder (stb_image)
    reads. Returns the canonical extension or None (reason in res.warnings)."""
    if d[:8] == b"\x89PNG\r\n\x1a\n":
        return "png" if validate_png(d, label, res) else None
    if d[:3] == b"\xff\xd8\xff":
        if len(d) < 256 or b"\xff\xd9" not in d[-64:]:
            res.warnings.append(f"{label}: truncated JPEG, dropped")
            return None
        return "jpg"
    if d[:2] == b"BM" and len(d) >= 54:
        size = struct.unpack_from("<I", d, 2)[0]
        if size > len(d):
            res.warnings.append(f"{label}: BMP header size exceeds file, dropped")
            return None
        return "bmp"
    if d[:6] in (b"GIF87a", b"GIF89a") and len(d) >= 13:
        return "gif"
    res.warnings.append(f"{label}: not a png/jpg/bmp/gif image, dropped")
    return None


def require_game_icon(results, png_path):
    """The Extract & Install rule: a container that holds award items needs a
    valid game-icon PNG up front (nothing is written without it). Returns the
    PNG bytes (or None when the container has no awards); raises PackageError."""
    if not any(r.is_award for r in results):
        return None
    if not png_path:
        raise PackageError("this container holds avatar awards. Set the game's icon "
                           "first (Avatar Award Icons row). Nothing was written")
    if not os.path.isfile(png_path):
        raise PackageError(f"game icon not found: {png_path}")
    data = open(png_path, "rb").read()
    probe = Result()
    if not validate_icon_image(data, os.path.basename(png_path), probe):
        raise PackageError("game icon rejected: " + ("; ".join(probe.warnings) or "unreadable"))
    return data


def install_title_icon(out_dir, title_id, png_path):
    """Copy a validated image to <out_dir>/titles/<TITLEID>.<ext> (the Avatar
    Editor's XamReadTileEx game tile; the runtime resamples any size and
    re-encodes to PNG). Any other-format icon already stored for the title is
    removed so one file wins. Returns the written path; raises PackageError on
    a bad id or image."""
    tid = title_id.strip().upper()
    if len(tid) != 8 or any(c not in "0123456789ABCDEF" for c in tid):
        raise PackageError(f"title id {title_id!r} is not 8 hex digits")
    data = open(png_path, "rb").read()
    probe = Result()
    ext = validate_icon_image(data, os.path.basename(png_path), probe)
    if not ext:
        raise PackageError("; ".join(probe.warnings) or "unreadable image")
    icon_dir = os.path.join(out_dir, "titles")
    os.makedirs(icon_dir, exist_ok=True)
    for other in ICON_EXTS:
        if other != ext:
            try:
                os.remove(os.path.join(icon_dir, f"{tid}.{other}"))
            except OSError:
                pass
    dst = os.path.join(icon_dir, f"{tid}.{ext}")
    with open(dst, "wb") as f:
        f.write(data)
    return dst


def write_item_files(res, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    written = []
    bin_path = os.path.join(out_dir, res.guid + ".bin")
    with open(bin_path, "wb") as f:
        f.write(res.bin_bytes)
    written.append(bin_path)
    if res.icon_bytes:
        icon_dir = os.path.join(out_dir, "icons")
        os.makedirs(icon_dir, exist_ok=True)
        icon_path = os.path.join(icon_dir, res.guid + ".png")
        with open(icon_path, "wb") as f:
            f.write(res.icon_bytes)
        written.append(icon_path)
    return written


def update_index(results, out_dir):
    """Append every new item's line to closet_index.tsv (backup first), or
    create the file when there is none. Returns (action string, written)."""
    os.makedirs(out_dir, exist_ok=True)
    tsv_path = os.path.join(out_dir, "closet_index.tsv")
    if os.path.exists(tsv_path):
        data = open(tsv_path, "rb").read()
        indexed = {l.split(b"\t")[0] for l in data.splitlines()}
        new = [r for r in results if r.guid.encode() not in indexed]
        if not new:
            return "already in closet_index.tsv, index unchanged", []
        with open(tsv_path + ".bak", "wb") as f:
            f.write(data)
        if data and not data.endswith(b"\n"):
            data += b"\r\n"
        for r in new:
            data += index_line(r).encode("utf-8") + b"\r\n"
        with open(tsv_path, "wb") as f:
            f.write(data)
        return (f"appended {len(new)} line(s) to existing closet_index.tsv "
                "(backup: .bak)", [tsv_path])
    seen = set()
    data = b""
    for r in results:
        if r.guid not in seen:
            seen.add(r.guid)
            data += index_line(r).encode("utf-8") + b"\r\n"
    with open(tsv_path, "wb") as f:
        f.write(data)
    return f"created new closet_index.tsv ({len(seen)} line(s))", [tsv_path]


# ---------------------------------------------------------------------------
# GUI
# ---------------------------------------------------------------------------

def compose_item_icon(thumb_bytes, render_bytes, cell=128, gap=8):
    """One pixmap per item: the container thumbnail (when carried) beside the
    rendered Stand-pose preview, or whichever of the two exists."""
    images = []
    for data in (thumb_bytes, render_bytes):
        if data:
            img = QImage.fromData(data)
            if not img.isNull():
                images.append(img)
    if not images:
        return None
    width = cell * len(images) + gap * (len(images) - 1)
    canvas = QImage(width, cell, QImage.Format_ARGB32)
    canvas.fill(Qt.transparent)
    painter = QPainter(canvas)
    x = 0
    for img in images:
        scaled = img.scaled(cell, cell, Qt.KeepAspectRatio, Qt.SmoothTransformation)
        painter.drawImage(x + (cell - scaled.width()) // 2, (cell - scaled.height()) // 2, scaled)
        x += cell + gap
    painter.end()
    return QPixmap.fromImage(canvas)


class ImporterWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Dry Cleaner")
        self.setAcceptDrops(True)
        self.resize(640, 520)

        root = QVBoxLayout(self)

        io_box = QGroupBox("Container")
        grid = QGridLayout(io_box)
        grid.addWidget(QLabel("Input file:"), 0, 0)
        self.input_edit = QLineEdit()
        self.input_edit.setPlaceholderText("Avatar item container (.acp / guid file), AvatarAwards container, or raw .bin (YTGR/STRB)")
        grid.addWidget(self.input_edit, 0, 1)
        btn_in = QPushButton("Browse...")
        btn_in.clicked.connect(self.pick_input)
        grid.addWidget(btn_in, 0, 2)

        grid.addWidget(QLabel("Closet folder:"), 1, 0)
        self.output_edit = QLineEdit()
        self.output_edit.setPlaceholderText("path/to/closet")
        grid.addWidget(self.output_edit, 1, 1)
        btn_out = QPushButton("Browse...")
        btn_out.clicked.connect(self.pick_output)
        grid.addWidget(btn_out, 1, 2)
        root.addWidget(io_box)

        icon_box = QGroupBox("Avatar Award Icons")
        igrid = QGridLayout(icon_box)
        igrid.addWidget(QLabel("Path to Icon:"), 0, 0)
        self.icon_edit = QLineEdit()
        self.icon_edit.setPlaceholderText("png, jpg, bmp or gif (128x128 recommended)")
        igrid.addWidget(self.icon_edit, 0, 1)
        btn_icon = QPushButton("Browse...")
        btn_icon.clicked.connect(self.pick_icon)
        igrid.addWidget(btn_icon, 0, 2)
        note = QLabel("128x128 recommended.")
        note.setStyleSheet("color: #888;")
        igrid.addWidget(note, 1, 1)
        igrid.addWidget(QLabel("Installed game:"), 2, 0)
        self.title_combo = QComboBox()
        self.title_combo.setToolTip("Games that already have avatar awards in the closet "
                                    "folder. Pick one and Set icon to change its tile.")
        igrid.addWidget(self.title_combo, 2, 1)
        btn_replace = QPushButton("Set icon")
        btn_replace.setToolTip("Install the icon above for the selected game (no container needed).")
        btn_replace.clicked.connect(self.set_icon)
        igrid.addWidget(btn_replace, 2, 2)
        root.addWidget(icon_box)
        self.output_edit.textChanged.connect(self.refresh_titles)
        self.refresh_titles()

        run_row = QHBoxLayout()
        self.preview_btn = QPushButton("Preview")
        self.preview_btn.setToolTip("Parse the container and show its items. Writes nothing.")
        self.preview_btn.clicked.connect(self.preview)
        self.install_btn = QPushButton("Install")
        self.install_btn.setDefault(True)
        self.install_btn.clicked.connect(self.install)
        run_row.addStretch(1)
        run_row.addWidget(self.preview_btn)
        run_row.addWidget(self.install_btn)
        root.addLayout(run_row)

        info_box = QGroupBox("Items")
        info = QVBoxLayout(info_box)
        self.items_list = QListWidget()
        self.items_list.setViewMode(QListWidget.IconMode)
        self.items_list.setIconSize(QSize(264, 128))
        self.items_list.setResizeMode(QListWidget.Adjust)
        self.items_list.setMovement(QListWidget.Static)
        self.items_list.setWrapping(True)
        self.items_list.setWordWrap(True)
        self.items_list.setSelectionMode(QListWidget.NoSelection)
        self.items_list.setFixedHeight(200)
        info.addWidget(self.items_list)
        self.summary_label = QLabel("no container loaded")
        self.summary_label.setTextInteractionFlags(Qt.TextSelectableByMouse)
        self.summary_label.setWordWrap(True)
        info.addWidget(self.summary_label)
        root.addWidget(info_box)

        self.log_edit = QTextEdit()
        self.log_edit.setReadOnly(True)
        self.log_edit.setFontFamily("Consolas")
        root.addWidget(self.log_edit, 1)

    # -- drag & drop ---------------------------------------------------------
    def dragEnterEvent(self, event):
        if event.mimeData().hasUrls():
            event.acceptProposedAction()

    def dropEvent(self, event):
        paths = [u.toLocalFile() for u in event.mimeData().urls()
                 if u.isLocalFile()]
        if paths:
            self.input_edit.setText(paths[0])

    # -- slots ---------------------------------------------------------------
    def pick_input(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Select an avatar item container, AvatarAwards container or raw .bin", self.input_edit.text() or "",
            "Avatar items (*.bin *.acp *.*);;All files (*)")
        if path:
            self.input_edit.setText(path)

    def pick_output(self):
        path = QFileDialog.getExistingDirectory(
            self, "Select closet folder", self.output_edit.text() or "")
        if path:
            self.output_edit.setText(path)

    def refresh_titles(self):
        """Fill the Installed game list from the closet folder's award sidecars."""
        out_dir = self.output_edit.text().strip().strip('"')
        self.title_combo.clear()
        if not out_dir or not os.path.isdir(out_dir):
            self.title_combo.addItem("(set the closet folder first)", None)
            return
        titles = list_award_titles(out_dir)
        if not titles:
            self.title_combo.addItem("(no avatar award games in this closet)", None)
            return
        icon_dir = os.path.join(out_dir, "titles")
        for tid, name in titles:
            have = any(os.path.isfile(os.path.join(icon_dir, f"{tid}.{e}")) for e in ICON_EXTS)
            self.title_combo.addItem(f"{name or tid} ({tid})" + ("" if have else "  [no icon]"), tid)

    def set_icon(self):
        """Install the Path-to-Icon image for the selected installed game."""
        out_dir = self.output_edit.text().strip().strip('"')
        tid = self.title_combo.currentData()
        png = self.icon_edit.text().strip().strip('"')
        if not out_dir or not tid:
            QMessageBox.warning(self, "Dry Cleaner",
                                "Pick the closet folder and an installed game first.")
            return
        if not png:
            QMessageBox.warning(self, "Dry Cleaner", "Set the Path to Icon first.")
            return
        try:
            dst = install_title_icon(out_dir, tid, png)
        except (PackageError, OSError) as e:
            self.log(f"REFUSED: {e}")
            QMessageBox.warning(self, "Dry Cleaner", str(e))
            return
        self.log(f"wrote {dst}")
        self.log(f"game icon set for {self.title_combo.currentText()}. "
                 "Restart the Avatar Editor to see it.")
        self.refresh_titles()

    def pick_icon(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Select game icon", self.icon_edit.text() or "",
            "Images (*.png *.jpg *.jpeg *.bmp *.gif);;All files (*)")
        if path:
            self.icon_edit.setText(path)

    def log(self, msg):
        self.log_edit.append(msg)

    def _analyze(self, verb):
        """Parse the input, fill the log and the Items preview. Returns the
        validated results, or None after a refusal/error. Writes nothing."""
        in_path = self.input_edit.text().strip().strip('"')
        if not in_path or not os.path.isfile(in_path):
            QMessageBox.warning(self, "Dry Cleaner", "Pick an input file first.")
            return None
        self.log_edit.clear()
        self.items_list.clear()
        try:
            results = process_package(in_path)
        except PackageError as e:
            self.summary_label.setText("REFUSED, see log")
            self.log(f"REFUSED: {e}")
            self.log("nothing was written.")
            return None
        except Exception as e:  # noqa: BLE001 -- surface, never crash the GUI
            self.summary_label.setText("ERROR, see log")
            self.log(f"ERROR: {e!r}")
            return None
        for res in results:
            for line in res.log:
                self.log(line)
            for warn in res.warnings:
                self.log(f"note: {warn}")
        self._show_items(results, verb)
        return results

    def _show_items(self, results, verb):
        bodies_names = {1: "male", 2: "female", 3: "both"}
        for res in results:
            label = f"{res.name} ({bodies_names[res.bodies]})"
            entry = QListWidgetItem(label)
            entry.setToolTip(
                f"{res.guid}\ncategories: {res.categories:08X} "
                f"({category_names(res.categories)})\n"
                f"blob: {len(res.bin_bytes)} bytes"
                + (f"\navatar award from title {res.title_id:08X}" if res.is_award else ""))
            self.log(f"rendering preview for {res.name}")
            QApplication.processEvents()
            render = render_item_preview(res, self.log)
            pix = compose_item_icon(res.icon_bytes, render)
            if pix is not None:
                entry.setIcon(QIcon(pix))
            if res.icon_bytes and render:
                entry.setToolTip(entry.toolTip() + "\nleft: container thumbnail, right: rendered preview")
            elif render:
                entry.setToolTip(entry.toolTip() + "\nrendered preview (no icon in the container)")
            self.items_list.addItem(entry)
            QApplication.processEvents()
        if len(results) == 1:
            res = results[0]
            self.summary_label.setText(
                f"<b>{res.name}</b> &nbsp; {res.guid} &nbsp; "
                f"bodies: {bodies_names[res.bodies]} &nbsp; "
                f"categories: {res.categories:08X} "
                f"({category_names(res.categories)})"
                + ("" if res.icon_bytes else " &nbsp; (no icon carried, preview rendered)")
                + f" &nbsp; ({verb})")
        else:
            self.summary_label.setText(
                f"<b>{len(results)} items {verb}</b>. Hover an item for "
                "its guid and categories.")

    def preview(self):
        results = self._analyze("previewed")
        if results is None:
            return
        awards = [r for r in results if r.is_award]
        if awards:
            titles = ", ".join(sorted({f"{r.title_id:08X}" for r in awards}))
            png = self.icon_edit.text().strip().strip('"')
            self.log(f"{len(awards)} avatar award(s) from title {titles}: Install will "
                     "require the game icon" + ("" if png else " (none set yet)"))
        self.log("preview only, nothing was written.")

    def install(self):
        out_dir = self.output_edit.text().strip().strip('"')
        if not out_dir:
            QMessageBox.warning(self, "Dry Cleaner",
                                "Pick a closet folder first.")
            return
        results = self._analyze("installed")
        if results is None:
            return
        png = self.icon_edit.text().strip().strip('"')
        try:
            icon_bytes = require_game_icon(results, png)
        except PackageError as e:
            self.summary_label.setText("REFUSED, game icon required")
            self.log(f"REFUSED: {e}")
            QMessageBox.warning(self, "Dry Cleaner", str(e))
            return
        try:
            written = []
            for res in results:
                written += write_item_files(res, out_dir)
            index_action, idx_written = update_index(results, out_dir)
            written += idx_written
            award_action, aw_written = update_awards(results, out_dir)
            written += aw_written
            if icon_bytes is not None:
                for tid in sorted({f"{r.title_id:08X}" for r in results if r.is_award}):
                    written.append(install_title_icon(out_dir, tid, png))
        except OSError as e:
            self.log(f"ERROR writing outputs: {e}")
            return
        self.log("")
        for p in written:
            self.log(f"wrote {p}")
        self.log(f"index: {index_action}")
        if award_action:
            self.log(f"awards: {award_action}")
        if icon_bytes is not None:
            self.log("game icon installed. Restart the Avatar Editor to see it.")
        elif png:
            self.log("note: icon ignored, this container holds no avatar awards")


def run_cli(in_path, out_dir=None, icon_path=None):
    """Headless: py dry_cleaner_gui.py <container|item.bin> [outdir] [--icon <file>]."""
    out_dir = out_dir or os.path.join(os.path.dirname(os.path.abspath(in_path)),
                                      "closet_staging")
    try:
        results = process_package(in_path)
        for res in results:
            for line in res.log:
                print(line)
            for warn in res.warnings:
                print(f"note: {warn}")
        icon_bytes = require_game_icon(results, icon_path)
    except (PackageError, OSError) as e:
        print(f"REFUSED: {e}")
        return 1
    written = []
    for res in results:
        written += write_item_files(res, out_dir)
    index_action, idx_written = update_index(results, out_dir)
    award_action, aw_written = update_awards(results, out_dir)
    if icon_bytes is not None:
        for tid in sorted({f"{r.title_id:08X}" for r in results if r.is_award}):
            written.append(install_title_icon(out_dir, tid, icon_path))
    for p in written + idx_written + aw_written:
        print(f"wrote {p}")
    print(f"index: {index_action}")
    if award_action:
        print(f"awards: {award_action}")
    if icon_path and icon_bytes is None:
        print("note: icon ignored, this container holds no avatar awards")
    return 0


def run_cli_game_icon(out_dir, title_id, png_path):
    """py dry_cleaner_gui.py --game-icon <closet> <TITLEID> <png>"""
    try:
        dst = install_title_icon(out_dir, title_id, png_path)
    except (PackageError, OSError) as e:
        print(f"REFUSED: {e}")
        return 1
    print(f"wrote {dst}")
    return 0


def main():
    if len(sys.argv) > 1 and sys.argv[1] == "--game-icon":
        if len(sys.argv) != 5:
            print("usage: py dry_cleaner_gui.py --game-icon <closet> <TITLEID> <png|jpg|bmp|gif>")
            sys.exit(2)
        sys.exit(run_cli_game_icon(sys.argv[2], sys.argv[3], sys.argv[4]))
    if len(sys.argv) > 1:
        args = sys.argv[1:]
        icon = None
        if "--icon" in args:
            i = args.index("--icon")
            icon = args[i + 1] if i + 1 < len(args) else None
            del args[i:i + 2]
        sys.exit(run_cli(args[0], args[1] if len(args) > 1 else None, icon))
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    win = ImporterWindow()
    win.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
