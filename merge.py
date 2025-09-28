#!/usr/bin/env python3
import argparse
import glob
import json
import os
import re

def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def is_coco_dict(d):
    return isinstance(d, dict) and "images" in d and "annotations" in d and "categories" in d

# ---- Category normalization ----
# Fixed IDs:
# 0: warehouse (any WHousse/warehouse variants)
# 1: buffer
# 2: pallet
# 3: shelf
# 4: transporter
FIXED_CATEGORIES = [
    {"id": 0, "name": "warehouse", "supercategory": "env"},
    {"id": 1, "name": "buffer", "supercategory": "object"},
    {"id": 2, "name": "pallet", "supercategory": "object"},
    {"id": 3, "name": "shelf", "supercategory": "object"},
    {"id": 4, "name": "transporter", "supercategory": "object"},
]

ALIASES = {
    "warehouse": {"warehouse", "whouse", "whousse", "wh", "whs", "ware house", "ware-house"},
    "buffer": {"buffer", "bufffer", "buf", "buffer_zone", "bufferzone"},
    "pallet": {"pallet", "palette", "palet"},
    "shelf": {"shelf", "shelve", "shelves", "rack"},
    "transporter": {"transporter", "forklift", "agv", "robot", "amr", "carrier"}
}

ALIAS_TO_ID = {}
for fixed in FIXED_CATEGORIES:
    name = fixed["name"]
    for alias in ALIASES.get(name, {name}):
        ALIAS_TO_ID[alias] = fixed["id"]

def normalize_cat_name(name: str) -> str:
    if not isinstance(name, str):
        return ""
    s = name.strip().lower().replace("-", " ").replace("_", " ")
    s = re.sub(r"\s+", " ", s)
    return s

def map_to_fixed_id(name: str, on_unknown: str = "to_warehouse"):
    norm = normalize_cat_name(name)
    if norm in ALIAS_TO_ID:
        return ALIAS_TO_ID[norm]
    # heuristic keywords
    if any(k in norm for k in ["ware", "wh"]):
        return 0
    if "buffer" in norm:
        return 1
    if "pallet" in norm or "palet" in norm or "palette" in norm:
        return 2
    if "shelf" in norm or "rack" in norm:
        return 3
    if any(k in norm for k in ["transport", "forklift", "agv", "amr", "robot", "carrier"]):
        return 4
    return 0 if on_unknown == "to_warehouse" else None

def merge_coco(files, out_file, dedupe_images=True, verbose=True, on_unknown="to_warehouse"):
    merged = {
        "images": [],
        "annotations": [],
        "categories": FIXED_CATEGORIES.copy(),
        "info": {"description": "Merged COCO dataset (fixed categories 0..4)", "version": "1.0"}
    }
    file_name_to_new_img_id = {} if dedupe_images else None

    stats = {"files": 0, "images_added": 0, "images_skipped": 0, "annotations_added": 0, "annotations_skipped": 0}
    next_img_id = 1
    next_ann_id = 1

    for path in files:
        if os.path.isdir(path):
            subfiles = glob.glob(os.path.join(path, "**", "*.json"), recursive=True)
            if verbose:
                print(f"[INFO] Expanding directory {path}: {len(subfiles)} json files")
        else:
            subfiles = [path]

        for f in subfiles:
            try:
                data = load_json(f)
            except Exception as e:
                if verbose:
                    print(f"[WARN] Skip {f}: cannot read JSON ({e})")
                continue
            if not is_coco_dict(data):
                if verbose:
                    print(f"[WARN] Skip {f}: not COCO-style")
                continue

            stats["files"] += 1

            # Map old category id -> fixed id
            oldcat_to_fixed = {}
            for cat in data.get("categories", []):
                fixed_id = map_to_fixed_id(cat.get("name", ""), on_unknown=on_unknown)
                if fixed_id is None:
                    continue
                oldcat_to_fixed[cat["id"]] = fixed_id

            # Map images
            oldimg_to_newimg = {}
            id2name = {img["id"]: img.get("file_name") for img in data.get("images", [])}

            for img in data.get("images", []):
                fn = img.get("file_name")
                if dedupe_images and fn in file_name_to_new_img_id:
                    new_id = file_name_to_new_img_id[fn]
                    oldimg_to_newimg[img["id"]] = new_id
                    stats["images_skipped"] += 1
                    continue

                new_id = next_img_id
                next_img_id += 1
                new_img = dict(img)
                new_img["id"] = new_id
                merged["images"].append(new_img)
                oldimg_to_newimg[img["id"]] = new_id
                stats["images_added"] += 1
                if dedupe_images:
                    file_name_to_new_img_id[fn] = new_id

            # Map annotations
            for ann in data.get("annotations", []):
                old_img_id = ann.get("image_id")
                if old_img_id not in oldimg_to_newimg:
                    fn = id2name.get(old_img_id, None)
                    if dedupe_images and fn and fn in file_name_to_new_img_id:
                        new_img_id = file_name_to_new_img_id[fn]
                    else:
                        stats["annotations_skipped"] += 1
                        continue
                else:
                    new_img_id = oldimg_to_newimg[old_img_id]

                old_cat = ann.get("category_id")
                fixed_id = oldcat_to_fixed.get(old_cat, 0 if on_unknown == "to_warehouse" else None)
                if fixed_id is None:
                    stats["annotations_skipped"] += 1
                    continue

                new_ann = dict(ann)
                new_ann["id"] = next_ann_id
                next_ann_id += 1
                new_ann["image_id"] = new_img_id
                new_ann["category_id"] = fixed_id
                merged["annotations"].append(new_ann)
                stats["annotations_added"] += 1

    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(merged, f, ensure_ascii=False, indent=2)

    if verbose:
        print("[DONE] Wrote:", out_file)
        print(json.dumps(stats, indent=2))

def main():
    parser = argparse.ArgumentParser(description="Merge COCO JSON files and remap categories to fixed IDs {0..4}.")
    parser.add_argument("inputs", nargs="+", help="Input JSON files or directories")
    parser.add_argument("--out", default="merged_fixed_categories.json", help="Output JSON path")
    parser.add_argument("--no-dedupe-images", action="store_true", help="Disable deduplication by file_name")
    parser.add_argument("--unknown-policy", choices=["to_warehouse","skip"], default="to_warehouse",
                        help="If a category is unknown, map to warehouse (0) or skip its annotations")
    parser.add_argument("--quiet", action="store_true", help="Suppress logs")
    args = parser.parse_args()

    merge_coco(
        files=args.inputs,
        out_file=args.out,
        dedupe_images=not args.no_dedupe_images,
        verbose=not args.quiet,
        on_unknown=args.unknown_policy
    )

if __name__ == "__main__":
    main()
