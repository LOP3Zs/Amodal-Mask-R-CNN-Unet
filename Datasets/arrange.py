import json, os, shutil

anno_file = "Dataset/test/_annotations.coco.json"
depth_src = "Dataset/Depth/chunk_000"
depth_dst = "Dataset/test/depths"
os.makedirs(depth_dst, exist_ok=True)

with open(anno_file, "r", encoding="utf-8") as f:
    data = json.load(f)

for img in data["images"]:
    fname = img["file_name"]
    prefix = fname[:6]
    depth_file = f"{prefix}_depth.png"
    src = os.path.join(depth_src, depth_file)
    dst = os.path.join(depth_dst, depth_file)
    # print(f"Copying {src} to {dst}")
    shutil.copy2(src, dst)
