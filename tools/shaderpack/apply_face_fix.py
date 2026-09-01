# Face-feature channel fix for pack shader FB79EB740C314B1F (the avatar
# head PS): the recompiled swizzle reads the wrong color channels for the
# feature tint. Applied after pack generation, then recompiled.
#
#   py apply_face_fix.py <pack_dir> <XenosRecomp.exe>
import io
import subprocess
import sys

pack = sys.argv[1]
xr = sys.argv[2]
hlsl = pack + "/FB79EB740C314B1F.ps.hlsl"

OLD_A = "r12.xyz = r3.xxx * xe_fc218.xyz + r3.zzz;"
NEW_A = "r12.xyz = r3.xxx * xe_fc218.xyz + r3.yyy;"
OLD_B = "r12.xyz = r3.yyy * xe_fc215.xyz + r12.xyz;"
NEW_B = "r12.xyz = r3.zzz * xe_fc215.xyz + r12.xyz;"

t = io.open(hlsl, encoding="utf-8").read()
if NEW_A in t and NEW_B in t:
    print("face fix already applied")
    sys.exit(0)
if t.count(OLD_A) != 1 or t.count(OLD_B) != 1:
    sys.exit("face fix anchors not found in " + hlsl)
t = t.replace(OLD_A, NEW_A).replace(OLD_B, NEW_B)
io.open(hlsl, "w", encoding="utf-8", newline="").write(t)

for out, args in ((".dxil", ["ps_6_0"]), (".spirv", ["ps_6_0", "spirv"])):
    r = subprocess.run([xr, "--compile", hlsl,
                        pack + "/FB79EB740C314B1F.ps" + out] + args)
    if r.returncode != 0:
        sys.exit("recompile failed for " + out)
print("face fix applied and recompiled")
