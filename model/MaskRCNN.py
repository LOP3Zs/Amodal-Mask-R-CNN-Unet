#!/usr/bin/env python3
import os, torch, math
from torch.utils.data import DataLoader
from torchvision.models.detection import maskrcnn_resnet50_fpn, MaskRCNN_ResNet50_FPN_Weights
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn   import MaskRCNNPredictor

# ==== import dataset, collate_fn bạn đã viết ====
# from your_dataset_file import COCOADataset, collate_fn   # nếu tách file
# Ở đây giả sử bạn đã có COCOADataset(json_path, img_dir) và collate_fn

def make_model(num_classes=5, freeze_backbone_epochs=0):
    """num_classes = 1 + K; với K=4 => num_classes=5."""
    weights = MaskRCNN_ResNet50_FPN_Weights.DEFAULT  # pretrained COCO
    model = maskrcnn_resnet50_fpn(weights=weights)

    # thay box head
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    # thay mask head
    in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    hidden = 256
    model.roi_heads.mask_predictor = MaskRCNNPredictor(in_features_mask, hidden, num_classes)

    # tùy chọn freeze backbone giai đoạn đầu
    if freeze_backbone_epochs > 0:
        for p in model.backbone.parameters():
            p.requires_grad = False
        model._frozen_backbone_epochs = freeze_backbone_epochs
    else:
        model._frozen_backbone_epochs = 0

    return model

def split_dataset(full_ds, val_ratio=0.2, seed=42):
    n = len(full_ds)
    n_val = int(math.ceil(n * val_ratio))
    g = torch.Generator().manual_seed(seed)
    indices = torch.randperm(n, generator=g).tolist()
    val_idx = set(indices[:n_val])
    train_subset = torch.utils.data.Subset(full_ds, [i for i in range(n) if i not in val_idx])
    val_subset   = torch.utils.data.Subset(full_ds, [i for i in indices[:n_val]])
    return train_subset, val_subset

def train_one_epoch(model, loader, optimizer, scaler, device, max_grad_norm=5.0):
    model.train()
    running = 0.0
    for images, targets in loader:
        images  = [img.to(device) for img in images]
        targets = [{k: v.to(device) for k,v in t.items()} for t in targets]

        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type='cuda', dtype=torch.float16, enabled=(device.type=='cuda')):
            loss_dict = model(images, targets)    # dict các loss
            loss = sum(loss_dict.values())

        scaler.scale(loss).backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
        scaler.step(optimizer)
        scaler.update()
        running += loss.item()
    return running / max(1, len(loader))

@torch.inference_mode()
def evaluate_loss(model, loader, device):
    model.eval()
    total = 0.0
    for images, targets in loader:
        images  = [img.to(device) for img in images]
        targets = [{k: v.to(device) for k,v in t.items()} for t in targets]
        with torch.autocast(device_type='cuda', dtype=torch.float16, enabled=(device.type=='cuda')):
            loss = sum(model(images, targets).values()).item()
        total += loss
    return total / max(1, len(loader))

def main():
    # ==== đường dẫn dữ liệu ====
    json_path = "train/cocoa_format_annotations.json"
    img_dir   = "train/images"

    # ==== tham số ====
    K = 4
    num_classes = 1 + K            # 1 background + 4 lớp
    batch_size  = 2
    epochs      = 12
    freeze_bb_for = 2              # freeze backbone 2 epoch đầu cho ổn định (0 nếu không muốn)
    base_lr     = 0.005
    weight_decay = 5e-4
    momentum     = 0.9
    milestones   = [8, 11]         # hạ LR tại các mốc

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    from Dataloader_MaskRCNN import COCOADataset, collate_fn
    # ==== dataset & dataloader ====
    full_ds = COCOADataset(json_path, img_dir)
    train_ds, val_ds = split_dataset(full_ds, val_ratio=0.2, seed=42)

    from torch.utils.data import DataLoader
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=4, collate_fn=collate_fn)
    val_loader   = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                              num_workers=4, collate_fn=collate_fn)

    # ==== model, optim, scheduler ====
    model = make_model(num_classes=num_classes, freeze_backbone_epochs=freeze_bb_for).to(device)

    def params_that_require_grad(m):
        return [p for p in m.parameters() if p.requires_grad]

    optimizer = torch.optim.SGD(params_that_require_grad(model),
                                lr=base_lr, momentum=momentum, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=milestones, gamma=0.1)
    scaler = torch.amp.GradScaler(device.type, enabled=(device.type=='cuda'))

    best_val = float('inf')
    os.makedirs("checkpoints", exist_ok=True)

    for epoch in range(1, epochs+1):
        # Unfreeze backbone sau khi đủ số epoch (nếu có)
        if hasattr(model, "_frozen_backbone_epochs") and model._frozen_backbone_epochs > 0 and epoch == model._frozen_backbone_epochs + 1:
            for p in model.backbone.parameters():
                p.requires_grad = True
            optimizer = torch.optim.SGD(params_that_require_grad(model),
                                        lr=base_lr, momentum=momentum, weight_decay=weight_decay)
            # có thể reset scheduler nếu muốn, ở đây giữ nguyên

        train_loss = train_one_epoch(model, train_loader, optimizer, scaler, device)
        val_loss   = evaluate_loss(model, val_loader, device)
        scheduler.step()

        print(f"Epoch {epoch:02d}/{epochs} | train_loss: {train_loss:.4f} | val_loss: {val_loss:.4f} | lr: {scheduler.get_last_lr()[0]:.6f}")

        # Lưu best theo val_loss
        if val_loss < best_val:
            best_val = val_loss
            torch.save({"epoch": epoch,
                        "model": model.state_dict(),
                        "optimizer": optimizer.state_dict(),
                        "scheduler": scheduler.state_dict(),
                        "val_loss": val_loss},
                       f"checkpoints/maskrcnn_best.pth")
        # Lưu checkpoint mỗi epoch
        torch.save({"epoch": epoch,
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict(),
                    "val_loss": val_loss},
                   f"checkpoints/maskrcnn_e{epoch:02d}.pth")

    print("Done. Best val_loss:", best_val)

    # ==== ví dụ suy luận sau khi train ====
    model.eval()
    # Lấy 1 mẫu từ val để test
    img, _ = val_ds[0]
    with torch.inference_mode():
        out = model([img.to(device)])[0]
    print("Pred keys:", out.keys())  # boxes, labels, scores, masks

if __name__ == "__main__":
    main()
