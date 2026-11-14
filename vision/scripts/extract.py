# Extract the VISNAV_PURE regions of pchie_lk_visnav.cpp into pure.inc for visnav_pure_test.cpp.
# Usage, from vision/: python scripts/extract.py pchie_lk_visnav.cpp pure.inc
import re, sys

s = open(sys.argv[1], encoding="utf-8").read()
out = []
for m in re.finditer(r"// VISNAV_PURE_BEGIN (\w+)\n(.*?)// VISNAV_PURE_END \1\n", s, re.S):
    out.append("// ---- region %s ----\n" % m.group(1) + m.group(2))
open(sys.argv[2], "w", encoding="utf-8").write("\n".join(out))
print("regions", len(out))
