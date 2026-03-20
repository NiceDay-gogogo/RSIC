import argparse
import os
import json
import sys
import math

import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from sklearn.metrics import f1_score, average_precision_score
from PIL import Image
from torchvision import transforms
import open_clip

# 直接导入，避免 models/__init__.py 的导入问题
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from models.RemoteCLIPAttributeExtractor import RemoteCLIPAttributeExtractor


class AttributeRSICDImageDataset(Dataset):
    """RSICD 属性预测数据集，从原始图像读取并端到端训练（配合 RemoteCLIP）。

    每个样本：
      - image: [3, H, W] 经过 RemoteCLIP 对应 preprocess
      - labels: [num_attributes]
    """

    def __init__(self, json_path, images_root, split="train", transform=None):
        super().__init__()
        self.json_path = json_path
        self.images_root = images_root
        self.split = split
        self.transform = transform

        with open(self.json_path, "r", encoding="utf-8") as f:
            info = json.load(f)
        images = info["images"]

        self.images = [img for img in images if img.get("split", "train") == split]
        if len(self.images) == 0:
            raise ValueError(f"No images found for split '{split}' in {json_path}")

        first_labels = self.images[0]["attribute_labels"]
        self.num_attributes = len(first_labels)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_info = self.images[idx]
        filename = img_info.get("filename")
        if filename is None:
            raise KeyError("Image entry must contain 'filename'")

        img_path = os.path.join(self.images_root, filename)
        image = Image.open(img_path).convert("RGB")

        if self.transform is not None:
            image = self.transform(image)

        labels = np.array(img_info["attribute_labels"], dtype="float32")
        labels = torch.from_numpy(labels)
        return image, labels


def compute_class_weights_from_json(json_path, split="train"):
    """根据 JSON 统计正负样本数，生成 BCEWithLogitsLoss 的 pos_weight。"""
    with open(json_path, "r", encoding="utf-8") as f:
        info = json.load(f)
    images = [img for img in info["images"] if img.get("split", "train") == split]
    if len(images) == 0:
        raise ValueError(f"No images found for split '{split}' in {json_path}")

    all_labels = [np.array(img["attribute_labels"], dtype="float32") for img in images]
    label_mat = np.stack(all_labels, axis=0)  # [N, A]
    pos_counts = label_mat.sum(axis=0)        # [A]
    neg_counts = label_mat.shape[0] - pos_counts
    pos_weight = neg_counts / (pos_counts + 1e-6)
    pos_weight = np.clip(pos_weight, 1.0, 20.0)
    return torch.from_numpy(pos_weight.astype("float32"))


def train_one_epoch(model, dataloader, optimizer, device):
    model.train()
    total_loss = 0.0
    total_samples = 0

    for images, labels in dataloader:
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        loss = model.compute_loss(images, labels)
        loss.backward()
        optimizer.step()

        bs = labels.size(0)
        total_loss += loss.item() * bs
        total_samples += bs

    return total_loss / max(total_samples, 1)


@torch.no_grad()
def evaluate(model, dataloader, device, threshold=0.5):
    model.eval()

    total_loss = 0.0
    total_samples = 0
    all_labels = []
    all_probs = []

    bce = nn.BCEWithLogitsLoss(reduction="none")

    for images, labels in dataloader:
        images = images.to(device)
        labels = labels.to(device)

        logits = model.forward_logits(images)
        loss_mat = bce(logits, labels)
        loss = loss_mat.mean()

        probs = torch.sigmoid(logits)

        bs = labels.size(0)
        total_loss += loss.item() * bs
        total_samples += bs

        all_labels.append(labels.cpu())
        all_probs.append(probs.cpu())

    if total_samples == 0:
        return 0.0, 0.0, 0.0, 0.0

    avg_loss = total_loss / total_samples

    labels_np = torch.cat(all_labels, dim=0).numpy()  # [N, A]
    probs_np = torch.cat(all_probs, dim=0).numpy()    # [N, A]
    preds_np = (probs_np >= threshold).astype("int32")

    f1_micro = f1_score(labels_np, preds_np, average="micro", zero_division=0)
    f1_macro = f1_score(labels_np, preds_np, average="macro", zero_division=0)

    pos_per_class = labels_np.sum(axis=0)
    valid_mask = pos_per_class > 0
    if valid_mask.sum() == 0:
        map_macro = 0.0
    else:
        map_macro = average_precision_score(
            labels_np[:, valid_mask],
            probs_np[:, valid_mask],
            average="macro",
        )

    return avg_loss, f1_micro, f1_macro, map_macro


def compute_sampling_weights(dataset: AttributeRSICDImageDataset, class_weights: torch.Tensor):
    class_weights = class_weights.cpu().numpy()
    sample_weights = []
    for img in dataset.images:
        labels = np.array(img["attribute_labels"], dtype="float32")
        positives = labels > 0.5
        if positives.any():
            weight = float(class_weights[positives].mean())
        else:
            weight = 1.0
        sample_weights.append(weight)
    sample_weights = np.array(sample_weights, dtype="float32")
    sample_weights /= max(sample_weights.mean(), 1e-6)
    return torch.from_numpy(sample_weights)


def build_optimizer(model, head_lr, backbone_lr, weight_decay):
    head_params = list(model.attr_head.parameters())
    backbone_params = [p for p in model.clip.parameters()]
    param_groups = [
        {"params": head_params, "lr": head_lr},
        {"params": backbone_params, "lr": backbone_lr},
    ]
    optimizer = torch.optim.AdamW(param_groups, betas=(0.9, 0.999), weight_decay=weight_decay)
    return optimizer


def main():
    parser = argparse.ArgumentParser(description="End-to-end train attribute extractor with RemoteCLIP backbone on RSICD")
    parser.add_argument("--input_json", type=str, default="data/rsicd_with_attributes_top25.json",
                        help="JSON with images and attribute_labels")
    parser.add_argument("--images_root", type=str, default="data/RSICD_images",
                        help="directory containing raw RSICD images")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--num_epochs", type=int, default=40)
    parser.add_argument("--head_lr", type=float, default=1e-4)
    parser.add_argument("--backbone_lr", type=float, default=5e-5)
    parser.add_argument("--weight_decay", type=float, default=1e-2)
    parser.add_argument("--lr_warmup_epochs", type=int, default=4,
                        help="epochs for linear LR warmup")
    parser.add_argument("--min_lr", type=float, default=1e-6,
                        help="minimum LR ratio for cosine schedule")
    parser.add_argument("--freeze_backbone_epochs", type=int, default=0,
                        help="freeze RemoteCLIP backbone for first N epochs")
    parser.add_argument("--d_model", type=int, default=512)
    parser.add_argument("--nhead", type=int, default=8)
    parser.add_argument("--num_layers", type=int, default=6)
    parser.add_argument("--dim_feedforward", type=int, default=2048)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--threshold", type=float, default=0.5,
                        help="threshold for turning probabilities into binary labels")
    parser.add_argument("--loss_type", type=str, default="bce", choices=["bce", "focal"],
                        help="loss function type: 'bce' or 'focal'")
    parser.add_argument("--focal_gamma", type=float, default=2.0,
                        help="gamma parameter for focal loss (only used when loss_type=focal)")
    parser.add_argument("--clip_model_name", type=str, default="ViT-B-32",
                        help="RemoteCLIP / OpenCLIP model name, e.g., 'RN50', 'ViT-B-32', 'ViT-L-14'")
    parser.add_argument("--remoteclip_ckpt", type=str, default="RemoteCLIP/RemoteCLIP-ViT-B-32.pt",
                        help="path to RemoteCLIP checkpoint .pt file")
    parser.add_argument("--save_path", type=str, default="save/attribute_extractor_remoteclip_vitb32.pth")
    parser.add_argument("--log_path", type=str, default=None,
                        help="path to save training log text file")
    parser.add_argument("--disable_weighted_sampler", action="store_true",
                        help="disable weighted sampler and fall back to random shuffling")

    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    save_dir = os.path.dirname(args.save_path)
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)

    if args.log_path is None:
        if save_dir:
            args.log_path = os.path.join(save_dir, "train_attribute_extractor_remoteclip_e2e.log")
        else:
            args.log_path = "train_attribute_extractor_remoteclip_e2e.log"

    log_dir = os.path.dirname(args.log_path)
    if log_dir:
        os.makedirs(log_dir, exist_ok=True)

    log_f = open(args.log_path, "a", encoding="utf-8")

    def log(msg: str):
        print(msg)
        print(msg, file=log_f)
        log_f.flush()

    # 使用 open_clip 提供的 preprocess 作为图像变换
    _, _, preprocess = open_clip.create_model_and_transforms(args.clip_model_name)
    crop_size = 224
    if hasattr(preprocess, "transforms"):
        for t in preprocess.transforms:
            if hasattr(t, "size"):
                size = t.size
                if isinstance(size, (tuple, list)):
                    crop_size = size[0]
                else:
                    crop_size = size
                break
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(crop_size, scale=(0.8, 1.0)),
        transforms.RandomHorizontalFlip(),
        preprocess,
    ])
    val_transform = preprocess

    # datasets & dataloaders
    train_dataset = AttributeRSICDImageDataset(args.input_json, args.images_root, split="train", transform=train_transform)
    try:
        val_dataset = AttributeRSICDImageDataset(args.input_json, args.images_root, split="val", transform=val_transform)
    except ValueError:
        val_dataset = AttributeRSICDImageDataset(args.input_json, args.images_root, split="test", transform=val_transform)

    num_attributes = train_dataset.num_attributes
    log(f"Num train images: {len(train_dataset)}")
    log(f"Num val images: {len(val_dataset)}")
    log(f"Num attributes: {num_attributes}")

    class_weights = compute_class_weights_from_json(args.input_json, split="train")
    sample_weights = None
    if not args.disable_weighted_sampler:
        sample_weights = compute_sampling_weights(train_dataset, class_weights)

    model = RemoteCLIPAttributeExtractor(
        num_attributes=num_attributes,
        clip_model_name=args.clip_model_name,
        remoteclip_ckpt_path=args.remoteclip_ckpt,
        d_model=args.d_model,
        nhead=args.nhead,
        num_layers=args.num_layers,
        dim_feedforward=args.dim_feedforward,
        dropout=args.dropout,
        class_weights=class_weights,
        loss_type=args.loss_type,
        focal_gamma=args.focal_gamma,
    ).to(device)

    if args.freeze_backbone_epochs > 0:
        model.set_backbone_trainable(False)

    optimizer = build_optimizer(model, args.head_lr, args.backbone_lr, args.weight_decay)

    warmup_epochs = max(args.lr_warmup_epochs, 0)

    def lr_lambda(epoch):
        if warmup_epochs > 0 and epoch < warmup_epochs:
            return (epoch + 1) / float(warmup_epochs)
        progress_epochs = max(args.num_epochs - warmup_epochs, 1)
        progress = (epoch - warmup_epochs) / progress_epochs
        progress = min(max(progress, 0.0), 1.0)
        cosine = 0.5 * (1 + math.cos(math.pi * progress))
        min_lr_ratio = args.min_lr / max(args.head_lr, 1e-8)
        return min_lr_ratio + (1 - min_lr_ratio) * cosine

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)

    train_sampler = None
    shuffle = True
    if sample_weights is not None:
        train_sampler = WeightedRandomSampler(
            weights=sample_weights.double(),
            num_samples=len(train_dataset),
            replacement=True,
        )
        shuffle = False

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=shuffle,
        sampler=train_sampler,
        num_workers=args.num_workers,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    best_map = -1.0

    for epoch in range(1, args.num_epochs + 1):
        if args.freeze_backbone_epochs > 0 and epoch == args.freeze_backbone_epochs + 1:
            model.set_backbone_trainable(True)

        train_loss = train_one_epoch(model, train_loader, optimizer, device)
        val_loss, f1_micro, f1_macro, map_macro = evaluate(model, val_loader, device, threshold=args.threshold)

        log(
            f"Epoch {epoch:03d}: "
            f"train_loss={train_loss:.4f}, "
            f"val_loss={val_loss:.4f}, "
            f"F1_micro={f1_micro:.4f}, "
            f"F1_macro={f1_macro:.4f}, "
            f"mAP_macro={map_macro:.4f}"
        )

        if map_macro > best_map:
            best_map = map_macro
            torch.save({
                "model_state": model.state_dict(),
                "num_attributes": num_attributes,
                "args": vars(args),
                "best_map": best_map,
            }, args.save_path)
            log(f"  Saved best model to {args.save_path} (mAP_macro={best_map:.4f})")

        scheduler.step()


if __name__ == "__main__":
    main()
