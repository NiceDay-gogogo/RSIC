import argparse
import os
import json

import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import f1_score, average_precision_score
from PIL import Image
from torchvision import transforms

from models.ResNetAttributeExtractor import ResNetAttributeExtractor


class AttributeRSICDImageDataset(Dataset):
    """RSICD 属性预测数据集，从原始图像读取并端到端训练。

    每个样本：
      - image: [3, H, W] 经过标准 ImageNet 归一化
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


def main():
    parser = argparse.ArgumentParser(description="End-to-end train attribute extractor with ResNet backbone on RSICD")
    parser.add_argument("--input_json", type=str, default="data/rsicd_with_attributes_top25.json",
                        help="JSON with images and attribute_labels")
    parser.add_argument("--images_root", type=str, default="data/RSICD_images",
                        help="directory containing raw RSICD images")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--num_epochs", type=int, default=20)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--lr_step", type=int, default=5,
                        help="decay learning rate every lr_step epochs")
    parser.add_argument("--lr_gamma", type=float, default=0.5,
                        help="multiplicative factor of learning rate decay")
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
    parser.add_argument("--backbone", type=str, default="resnet101",
                        help="backbone model name defined in misc.resnet, e.g., resnet101")
    parser.add_argument("--imagenet_weights_dir", type=str, default="data/imagenet_weights",
                        help="directory containing pretrained backbone weights, e.g., resnet101.pth")
    parser.add_argument("--save_path", type=str, default="save/attribute_extractor_e2e_resnet101.pth")
    parser.add_argument("--log_path", type=str, default=None,
                        help="path to save training log text file")

    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    save_dir = os.path.dirname(args.save_path)
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)

    if args.log_path is None:
        if save_dir:
            args.log_path = os.path.join(save_dir, "train_attribute_extractor_e2e.log")
        else:
            args.log_path = "train_attribute_extractor_e2e.log"

    log_dir = os.path.dirname(args.log_path)
    if log_dir:
        os.makedirs(log_dir, exist_ok=True)

    log_f = open(args.log_path, "a", encoding="utf-8")

    def log(msg: str):
        print(msg)
        print(msg, file=log_f)
        log_f.flush()

    # transforms
    train_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])
    val_transform = train_transform

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

    model = ResNetAttributeExtractor(
        num_attributes=num_attributes,
        d_model=args.d_model,
        nhead=args.nhead,
        num_layers=args.num_layers,
        dim_feedforward=args.dim_feedforward,
        dropout=args.dropout,
        class_weights=class_weights,
        loss_type=args.loss_type,
        focal_gamma=args.focal_gamma,
        backbone=args.backbone,
        imagenet_weights_dir=args.imagenet_weights_dir,
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.lr_step, gamma=args.lr_gamma)

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
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
