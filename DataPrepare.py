import cv2
import numpy as np
import PIL.Image as Image
from typing import List, Dict, Any
from pycocotools import mask as maskUtils
import os
import json

def segm_to_mask(segmentation: List[List[float]], height: int, width: int) -> np.ndarray:
    """Chuyển đổi segmentation polygon thành binary mask"""
    if isinstance(segmentation, list):
        rles = maskUtils.frPyObjects(segmentation, height, width)
        rle = maskUtils.merge(rles)
    else:
        rle = segmentation
    return maskUtils.decode(rle)

def mask_to_polygon(mask):
    """Chuyển đổi binary mask thành polygon"""
    mask = (mask > 0).astype(np.uint8)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    polygons = []
    for contour in contours:
        if len(contour) >= 3:
            contour = contour.reshape(-1, 2)
            polygon = contour.flatten().tolist()
            polygons.append(polygon)
    if len(polygons) == 0:
        return []
    return max(polygons, key=len)  # Trả về contour lớn nhất

def mask_to_bbox(mask):
    """Tính bbox từ mask theo format [x_min, y_min, x_max, y_max]"""
    coords = np.where(mask > 0)
    if len(coords[0]) == 0:
        return [0, 0, 0, 0]
    
    y_min, y_max = coords[0].min(), coords[0].max()
    x_min, x_max = coords[1].min(), coords[1].max()
    
    return [int(x_min), int(y_min), int(x_max), int(y_max)]

def create_class_object_matrix(image_id: str, coco_json: str) -> np.ndarray:
    """Tạo ma trận 3D với class IDs từ COCO annotations"""
    with open(coco_json, 'r') as f:
        data = json.load(f)
    
    # Tìm image info
    target_image_info = None
    for img in data['images']:
        img_basename = os.path.splitext(img['file_name'])[0]
        if img_basename == image_id:
            target_image_info = img
            break
    
    if not target_image_info:
        print(f"❌ Không tìm thấy ảnh: {image_id}")
        return None, None
    
    img_width = target_image_info['width']
    img_height = target_image_info['height']
    
    # Lấy tất cả annotations của ảnh này
    image_annotations = [ann for ann in data['annotations'] 
                        if ann['image_id'] == target_image_info['id']]
    
    if not image_annotations:
        print(f"❌ Không có annotations cho ảnh: {image_id}")
        return None, None
    
    num_objects = len(image_annotations)
    print(f"✅ Ảnh {image_id}: {img_width}x{img_height}, {num_objects} objects")
    
    # Tạo ma trận 3D: (height, width, num_objects)
    object_matrix = np.zeros((img_height, img_width, num_objects), dtype=np.uint8)
    
    # Xử lý từng annotation
    for idx, ann in enumerate(image_annotations):
        segmentation = ann['segmentation']
        mask = segm_to_mask(segmentation, img_height, img_width)
        class_id = ann['category_id']
        object_matrix[:, :, idx] = np.where(mask, class_id, 0).astype(np.uint8)
    
    return object_matrix, image_annotations

def getDepth(instance_mask, filedepth):
    """Tính depth trung bình của object"""
    coords = (instance_mask != 0).astype(np.uint8)
    if not os.path.exists(filedepth):
        return 0
    depth = np.array(Image.open(filedepth))
    depth = np.resize(depth, (512, 512))
    vals = depth[coords == 1]
    mean_depth = np.mean(vals) if vals.size > 0 else 0
    return mean_depth

def arrange_occlusion(matrices, filedepth):
    """Sắp xếp objects theo depth"""
    H, W, N = matrices.shape
    depths = []
    class_ids = []

    for i in range(N):
        instance_mask = matrices[:, :, i]
        uniq_vals = np.unique(instance_mask)
        uniq_vals = uniq_vals[uniq_vals != 0]
        
        if len(uniq_vals) > 0:
            cid = int(uniq_vals[0])
        else:
            cid = -1

        if cid == 1:  # buffer luôn dưới cùng
            d = float("inf")
        else:
            d = getDepth(instance_mask, filedepth)

        depths.append((i, d))
        class_ids.append(cid)

    # Sắp xếp theo depth tăng dần (gần nhất trước)
    depths_sorted = sorted(depths, key=lambda x: x[1])

    # Tạo mảng mới đã sắp xếp
    sorted_matrices = np.zeros_like(matrices)
    sorted_class_ids = []

    for new_idx, (old_idx, _) in enumerate(depths_sorted):
        sorted_matrices[:, :, new_idx] = matrices[:, :, old_idx]
        sorted_class_ids.append(class_ids[old_idx])

    return sorted_matrices

def getVisibleObjects(matrix):
    """Tạo visible masks bằng cách loại bỏ phần bị che"""
    H, W, N = matrix.shape
    visible_objects = np.zeros((H, W, N), dtype=np.uint8)

    for i in range(N):
        instance_mask = matrix[:, :, i]
        class_ids = np.unique(instance_mask)
        class_ids = class_ids[class_ids != 0]

        if len(class_ids) == 0:
            continue
        cid = int(class_ids[0])

        # Chuẩn hóa mask hiện tại
        current_mask = (instance_mask > 0).astype(np.uint8)

        # Loại bỏ phần bị che bởi các mask trước đó (gần hơn)
        for j in range(i):
            prev_mask = (matrix[:, :, j] > 0).astype(np.uint8)
            current_mask = current_mask * (1 - prev_mask)

        # Gán class_id cho visible part
        visible_objects[:, :, i] = current_mask * cid

    return visible_objects

def find_occluders(visible_mask, amodal_mask, sorted_matrices, current_idx):
    """Tìm các objects che khuất object hiện tại"""
    occluders = []
    
    # Phần bị che = amodal - visible
    occluded_part = np.logical_and(amodal_mask > 0, visible_mask == 0)
    
    if not np.any(occluded_part):
        return occluders
    
    # Kiểm tra các objects trước (gần hơn) có che không
    for j in range(current_idx):
        occluder_mask = (sorted_matrices[:, :, j] > 0).astype(np.uint8)
        
        # Nếu có overlap với phần bị che
        if np.any(np.logical_and(occluded_part, occluder_mask)):
            class_ids = np.unique(sorted_matrices[:, :, j])
            class_ids = class_ids[class_ids != 0]
            if len(class_ids) > 0:
                occluders.append(int(class_ids[0]))
    
    return occluders

def create_cocoa_format(coco_json: str, output_json: str, depth_dir: str):
    """
    Tạo file JSON theo format COCOA tương thích với AIstron preprocessing
    """
    with open(coco_json, 'r') as f:
        data = json.load(f)
    
    categories = data['categories']
    images = []
    annotations = []
    annotation_id = 1
    
    for img in data['images']:
        img_id = img['id']
        img_filename = img['file_name']
        height = img['height']
        width = img['width']
        
        # Thêm depth file info (nếu cần)
        depth_path = img_filename[:6]
        depth_file = f"{depth_path}_depth.png"
        
        # Format images theo COCO standard
        new_image = {
            "id": img_id,
            "file_name": img_filename,
            "depth_file": depth_file,
            "width": width,
            "height": height
        }
        images.append(new_image)
        
        # Xử lý annotations cho ảnh này
        file_name = os.path.splitext(img_filename)[0]
        amodal_matrix, original_annotations = create_class_object_matrix(file_name, coco_json)
        
        if amodal_matrix is None:
            continue
            
        # Sắp xếp theo depth
        depth_file_path = os.path.join(depth_dir, depth_file)
        sorted_matrix = arrange_occlusion(amodal_matrix, depth_file_path)
        
        # Tạo visible masks
        visible_matrix = getVisibleObjects(sorted_matrix)
        
        # Tạo annotations cho từng object
        H, W, N = amodal_matrix.shape
        
        for i in range(N):
            amodal_mask = sorted_matrix[:, :, i]
            visible_mask = visible_matrix[:, :, i]
            
            # Lấy class ID
            class_ids = np.unique(amodal_mask)
            class_ids = class_ids[class_ids != 0]
            
            if len(class_ids) == 0:
                continue
                
            class_id = int(class_ids[0])
            
            # Tạo polygons cho segmentation (amodal)
            amodal_polygon = mask_to_polygon(amodal_mask)
            if len(amodal_polygon) == 0:
                continue
            
            # Tạo visible mask polygon
            visible_polygon = mask_to_polygon(visible_mask) if np.any(visible_mask) else []
            
            # Tính bbox từ amodal mask (theo COCO format)
            amodal_bbox = mask_to_bbox(amodal_mask)
            
            # Tìm occluders
            occluders = find_occluders(visible_mask, amodal_mask, sorted_matrix, i)
            
            # Format annotation để tương thích với AIstron preprocessing
            annotation = {
                "id": annotation_id,
                "image_id": img_id,
                "category_id": class_id,
                "amodal_mask": [amodal_polygon],  
                "bbox": amodal_bbox,              
                "visible_mask": [visible_polygon] if len(visible_polygon) > 0 else [], 
                "iscrowd": 0,
                "area": int(np.sum(amodal_mask > 0)),
                "visible_area": int(np.sum(visible_mask > 0)),
                "occluders": occluders  # Thông tin bổ sung
            }
            
            annotations.append(annotation)
            annotation_id += 1
    
    # In thông tin classes
    print("🏷️  Classes được sử dụng:")
    used_classes = set(ann['category_id'] for ann in annotations)
    for cat in categories:
        if cat['id'] in used_classes:
            print(f"   {cat['id']}: {cat['name']}")
    
    # Tạo dữ liệu theo COCO format chuẩn (sẽ được convert bởi AIstron script)
    cocoa_data = {
        "images": images,
        "annotations": annotations,
        "categories": categories,
        "info": {
            "description": "COCOA format dataset for amodal segmentation",
            "version": "1.0",
            "year": 2024
        }
    }
    
    # Ghi file
    with open(output_json, 'w') as f:
        json.dump(cocoa_data, f, indent=2)
    
    print(f"✅ Đã tạo file COCOA format: {output_json}")
    print(f"📊 Tổng số ảnh: {len(images)}")
    print(f"📊 Tổng số annotations: {len(annotations)}")
    print(f"💡 File này sẽ được convert bởi script AIstron preprocessing")

if __name__ == "__main__":
    coco_json = "Datasets/train/_annotations.coco.json"
    output_json = "Datasets/train/cocoa_format_annotations.json"
    depth_dir = "Datasets/train/depth"
    
    create_cocoa_format(coco_json, output_json, depth_dir)