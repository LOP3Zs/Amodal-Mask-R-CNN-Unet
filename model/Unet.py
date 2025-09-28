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
    def __init__(self, json_path, img_dir, depth_dir):
        self.img_dir = img_dir
        self.depth_dir = depth_dir
        
        with open(json_path, 'r') as f:
            self.cocoa_data = json.load(f)
        
        self.images = self.cocoa_data['images']
        self.annotations = self.cocoa_data['annotations']
        
        # Filter annotations that have coarse_visible_mask
        self.valid_annotations = []
        for ann in self.annotations:
            if 'coarse_visible_mask' in ann and ann['coarse_visible_mask']:
                self.valid_annotations.append(ann)
        
        print(f"Found {len(self.valid_annotations)} valid annotations")
    
    def __len__(self):
        return len(self.valid_annotations)
    
    def __getitem__(self, idx):
        ann = self.valid_annotations[idx]
        img_id = ann['image_id']
        
        # Find image info
        img_info = next(img for img in self.images if img['id'] == img_id)
        height, width = img_info['height'], img_info['width']
        
        # Load RGB image
        img_path = os.path.join(self.img_dir, img_info['file_name'])
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Load depth
        depth_filename = img_info.get('depth_file', img_info['file_name'].replace('.jpg', '_depth.png'))
        depth_path = os.path.join(self.depth_dir, depth_filename)
        if os.path.exists(depth_path):
            depth = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)
            if depth.ndim == 3:
                depth = cv2.cvtColor(depth, cv2.COLOR_BGR2GRAY)
            # Resize depth to match image size
            if depth.shape != (height, width):
                depth = cv2.resize(depth, (width, height), interpolation=cv2.INTER_NEAREST)
        else:
            depth = np.zeros((height, width), dtype=np.uint16)
        
        # Process masks
        coarse_visible_mask = polygon_to_mask(ann['coarse_visible_mask'], height, width)
        amodal_mask = polygon_to_mask(ann['amodal_mask'], height, width)
        
        # Get category (0-indexed)
        category_id = ann['category_id'] - 1
        
        # Convert to tensors
        image = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0
        depth = torch.from_numpy(depth).unsqueeze(0).float() / 65535.0
        coarse_visible = torch.from_numpy(coarse_visible_mask).unsqueeze(0).float()
        gt_amodal = torch.from_numpy(amodal_mask).unsqueeze(0).float()
        
        return {
            'rgb': image,
            'depth': depth,
            'coarse_visible': coarse_visible,
            'gt_amodal': gt_amodal,
            'class_label': torch.tensor(category_id, dtype=torch.long)
        }

def create_dataloader(json_path, img_dir, depth_dir, batch_size=4, shuffle=True, num_workers=2):
    dataset = COCOADataset(json_path, img_dir, depth_dir)
    
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers
    )
    
    return dataloader

# Usage
if __name__ == "__main__":
    dataloader = create_dataloader(
        json_path="train/cocoa_format_annotations.json",
        img_dir="train/images",
        depth_dir="train/depths",
        batch_size=4
    )
    
    # Test
    for batch in dataloader:
        print("RGB shape:", batch['rgb'].shape)
        print("Depth shape:", batch['depth'].shape)
        print("Coarse visible shape:", batch['coarse_visible'].shape)
        print("GT amodal shape:", batch['gt_amodal'].shape)
        print("Class labels:", batch['class_label'])
        break