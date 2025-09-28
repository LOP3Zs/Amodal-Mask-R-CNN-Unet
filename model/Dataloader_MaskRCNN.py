import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import cv2
import json
import os

def polygon_to_mask(polygon, height, width):
    polygon = np.array(polygon).reshape((-1, 2))
    mask = np.zeros((height, width), dtype=np.uint8)
    cv2.fillPoly(mask, [polygon.astype(np.int32)], 1)
    return mask

class COCOADataset(Dataset):
    def __init__(self, json_path, img_dir):
        self.img_dir = img_dir
        
        with open(json_path, 'r') as f:
            self.cocoa_data = json.load(f)
        
        self.images = self.cocoa_data['images']
        self.annotations = self.cocoa_data['annotations']
        
        # Gom annotation theo image_id
        self.img_id_to_anns = {}
        for ann in self.annotations:
            if 'coarse_visible_mask' in ann and ann['coarse_visible_mask']:
                img_id = ann['image_id']
                if img_id not in self.img_id_to_anns:
                    self.img_id_to_anns[img_id] = []
                self.img_id_to_anns[img_id].append(ann)

        # Danh sách ảnh hợp lệ (có ít nhất 1 annotation)
        self.valid_images = [img for img in self.images if img['id'] in self.img_id_to_anns]
        print(f"Found {len(self.valid_images)} images with valid annotations")
    
    def __len__(self):
        return len(self.valid_images)
    
    def __getitem__(self, idx):
        img_info = self.valid_images[idx]
        img_id = img_info['id']
        height, width = img_info['height'], img_info['width']
        
        # Load RGB image
        img_path = os.path.join(self.img_dir, img_info['file_name'])
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Chuyển sang tensor chuẩn Mask R-CNN (CxHxW, float32, [0,1])
        image = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0
        
        # Gom annotations của ảnh này
        anns = self.img_id_to_anns[img_id]
        masks, boxes, labels = [], [], []
        
        for ann in anns:
            # Mask từ polygon
            visible_mask = polygon_to_mask(ann['visible_mask'], height, width)
            masks.append(visible_mask)
            
            # BBox [x,y,w,h] -> [x1,y1,x2,y2]
            x,y,w,h = ann['bbox']
            boxes.append([x, y, x+w, y+h])
            
            # Category id (chỉnh về 1..K, background=0 là mặc định của model)
            labels.append(ann['category_id'])
        
        if len(boxes) == 0:
            boxes  = torch.zeros((0,4), dtype=torch.float32)
            labels = torch.zeros((0,), dtype=torch.int64)
            masks  = torch.zeros((0, height, width), dtype=torch.uint8)
        else:
            boxes  = torch.as_tensor(boxes, dtype=torch.float32)
            labels = torch.as_tensor(labels, dtype=torch.int64)
            masks  = torch.as_tensor(np.stack(masks, axis=0), dtype=torch.uint8)
        
        target = {
            "boxes": boxes,
            "labels": labels,
            "masks": masks,
            "image_id": torch.tensor([img_id]),
            "iscrowd": torch.zeros((len(labels),), dtype=torch.int64)
        }
        
        return image, target

def collate_fn(batch):
    return tuple(zip(*batch))

def create_dataloader(json_path, img_dir, batch_size=2, shuffle=True, num_workers=2):
    dataset = COCOADataset(json_path, img_dir)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collate_fn
    )
    return dataloader

# Example usage
if __name__ == "__main__":
    dataloader = create_dataloader(
        json_path="train/cocoa_format_annotations.json",
        img_dir="train/images",
        batch_size=2
    )
    
    for images, targets in dataloader:
        print("Batch size:", len(images))
        print("Image tensor:", images[0].shape)
        print("Target keys:", targets[0].keys())
        print("Boxes:", targets[0]['boxes'])
        print("Labels:", targets[0]['labels'])
        print("Masks shape:", targets[0]['masks'].shape)
        break
