"""Extract the Tower Unite Workshop Rig 3.0 bone template from the reference
character shipped in WorkshopRigv3_Release.zip (Davier/Davier.dae) into
tu_rig_v3.json, the template the Avatar Export uses to re-skin an Xbox avatar
onto the TU skeleton.

The template keeps, per bone: name, parent, the local 4x4 matrix of the rig's
rest pose (rotation is what matters; TU's animations are authored against
these bone orientations, and the Workshop guide allows only the bone lengths
to change), and the Blender extras (tip, roll, collection) so a re-exported
DAE round-trips into Blender with sane bone display.

    py make_tu_rig.py <path-to-Davier.dae> [tu_rig_v3.json]
"""
import json
import sys
import xml.etree.ElementTree as ET

NS = {"c": "http://www.collada.org/2005/11/COLLADASchema"}


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        return 1
    src = sys.argv[1]
    dst = sys.argv[2] if len(sys.argv) > 2 else "tu_rig_v3.json"
    root = ET.parse(src).getroot()
    scene = root.find("c:library_visual_scenes/c:visual_scene", NS)
    bones = []

    def walk(node, parent):
        name = node.get("name") or node.get("id")
        typ = node.get("type")
        mat = node.find("c:matrix", NS)
        m = [float(x) for x in mat.text.split()] if mat is not None else None
        extra = {}
        tech = node.find("c:extra/c:technique[@profile='blender']", NS)
        if tech is not None:
            for child in tech:
                tag = child.tag.split("}")[1]
                txt = (child.text or "").strip()
                if tag in ("roll", "tip_x", "tip_y", "tip_z"):
                    extra[tag] = float(txt)
                elif tag == "collections":
                    extra["collection"] = txt.split()[0] if txt else ""
        if typ == "JOINT":
            bones.append({
                "name": name,
                "parent": parent,
                "local_matrix": m,  # row-major 4x4 (COLLADA order)
                "tip": [extra.get("tip_x", 0.0), extra.get("tip_y", 0.0), extra.get("tip_z", 0.0)],
                "roll": extra.get("roll", 0.0),
                "collection": extra.get("collection", ""),
            })
            parent = name
        for child in node.findall("c:node", NS):
            walk(child, parent)

    for n in scene.findall("c:node", NS):
        walk(n, None)

    unit = root.find("c:asset/c:unit", NS)
    up = root.find("c:asset/c:up_axis", NS)
    out = {
        "source": "Tower Unite Workshop Rig 3.0 (WorkshopRigv3_Release/Davier/Davier.dae)",
        "unit_meter": float(unit.get("meter")) if unit is not None else 0.01,
        "up_axis": up.text if up is not None else "Z_UP",
        "forward": "-Y",
        "right": "-X",
        "bones": bones,
    }
    with open(dst, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=1)
    print(f"wrote {dst}: {len(bones)} bones")
    return 0


if __name__ == "__main__":
    sys.exit(main())
