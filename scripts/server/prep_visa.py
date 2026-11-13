import csv, os
from collections import Counter
from PIL import Image
import numpy as np

SRC = "/data/disk1/shared/aiaa3201/project2_anomaly/VisA"
DST = "/tmp/yzhang176/data/VisA_pytorch/1cls"
CLASSES = ["candle", "capsules", "cashew", "chewinggum", "fryum", "pipe_fryum"]

os.makedirs(DST, exist_ok=True)
rows = list(csv.DictReader(open(os.path.join(SRC, "split_csv/1cls.csv"), encoding="utf-8")))
stat = Counter(); missing_mask = 0; dup = 0; seen = set()

def link(src, dst):
    global dup
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    if os.path.lexists(dst):
        dup += 1
        return False
    os.symlink(src, dst)
    return True

for r in rows:
    c = r["object"].strip()
    if c not in CLASSES:
        continue
    split, label = r["split"].strip(), r["label"].strip()
    img, mask = r["image"].strip(), (r["mask"] or "").strip()
    src = os.path.join(SRC, img)
    name = os.path.basename(img)
    if split == "train":
        link(src, os.path.join(DST, c, "train/good", name)); stat[(c, "train/good")] += 1
    elif label == "normal":
        link(src, os.path.join(DST, c, "test/good", name)); stat[(c, "test/good")] += 1
    else:
        link(src, os.path.join(DST, c, "test/bad", name)); stat[(c, "test/bad")] += 1
        gdst = os.path.join(DST, c, "ground_truth/bad", os.path.splitext(name)[0] + ".png")
        if mask and os.path.exists(os.path.join(SRC, mask)):
            link(os.path.join(SRC, mask), gdst)
        else:
            missing_mask += 1
            os.makedirs(os.path.dirname(gdst), exist_ok=True)
            if not os.path.lexists(gdst):
                Image.fromarray(np.zeros((256, 256), dtype=np.uint8)).save(gdst)

print("== per class ==")
for c in CLASSES:
    print(f"{c:12s} train={stat[(c,'train/good')]:4d} test_good={stat[(c,'test/good')]:4d} test_bad={stat[(c,'test/bad')]:4d}")
print("total bad:", sum(stat[(c, 'test/bad')] for c in CLASSES))
print("bad images with empty mask column -> zero mask created:", missing_mask)
print("duplicate name collisions:", dup)
