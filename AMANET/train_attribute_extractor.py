import argparse
import os
import json
import math

import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from sklearn.metrics import f1_score, average_precision_score

from dataloader import HybridLoader
from models.AttributeFeatureExtractor import AdvancedAttributeFeatureExtractor


class AttributeRSICDDataset(Dataset):
    """RSICD 属性预测数据集，直接读取 rsicd_with_attributes.json 和预提取的 att_feats。

    每个样本：
      - att_feats: [num_regions, feat_dim]
      - labels:   [num_attributes]
    """

    def __init__(self, json_path, att_dir, split="train"):
        super().__init__()
        self.json_path = json_path
        self.att_dir = att_dir
        self.split = split

        with open(self.json_path, "r") as f:
            info = json.load(f)
        images = info["images"]

        self.images = [img for img in images if img.get("split", "train") == split]
        if len(self.images) == 0:
            raise ValueError(f"No images found for split '{split}' in {json_path}")

        # 预提取区域特征加载器（.npz，键为 imgid 或 id）
        self.att_loader = HybridLoader(self.att_dir, ".npz")

        first_labels = self.images[0]["attribute_labels"]
        self.num_attributes = len(first_labels)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_info = self.images[idx]
        # 优先使用 imgid，如果没有则使用 id
        imgid = img_info.get("imgid", img_info.get("id"))
        if imgid is None:
            raise KeyError("Image entry must contain 'imgid' or 'id'")

        # att_feats: [K, feat_dim]
        att_feats = self.att_loader.get(str(imgid))
        att_feats = att_feats.reshape(-1, att_feats.shape[-1]).astype("float32")

        labels = np.array(img_info["attribute_labels"], dtype="float32")

        att_feats = torch.from_numpy(att_feats)
        labels = torch.from_numpy(labels)
        return att_feats, labels


def attribute_collate_fn(batch):
    """将变长的区域特征 padding 成统一长度，并生成 mask。

    输入 batch: List[(att_feats, labels)]
      - att_feats: [K_i, D]
      - labels:   [A]

    输出：
      - feats: [B, max_K, D]
      - masks: [B, max_K]，有效位置为 1
      - labels: [B, A]
    """
    feats_list, labels_list = zip(*batch)
    batch_size = len(feats_list)
    feat_dim = feats_list[0].shape[1]
    max_len = max(f.shape[0] for f in feats_list)

    feats = torch.zeros(batch_size, max_len, feat_dim, dtype=torch.float32)
    masks = torch.zeros(batch_size, max_len, dtype=torch.float32)

    for i, f in enumerate(feats_list):
        L = f.shape[0]
        feats[i, :L] = f
        masks[i, :L] = 1.0

    labels = torch.stack(labels_list, dim=0)
    return feats, masks, labels


def compute_class_weights(train_dataset: AttributeRSICDDataset):
    """根据训练集统计正负样本数，生成 BCEWithLogitsLoss 的 pos_weight。"""
    all_labels = [np.array(img["attribute_labels"], dtype="float32") for img in train_dataset.images]
    label_mat = np.stack(all_labels, axis=0)  # [N, A]
    pos_counts = label_mat.sum(axis=0)        # [A]
    neg_counts = label_mat.shape[0] - pos_counts
    pos_weight = neg_counts / (pos_counts + 1e-6)
    # 裁剪 pos_weight 上限为 20，避免极端长尾类权重过大
    pos_weight = np.clip(pos_weight, 1.0, 20.0)
    return torch.from_numpy(pos_weight.astype("float32"))


def compute_sampling_weights(train_dataset: AttributeRSICDDataset, class_weights: torch.Tensor):
    """根据每张图片所含稀有属性分配采样权重。"""
    class_weights = class_weights.cpu().numpy()
    sample_weights = []
    for img in train_dataset.images:
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


def train_one_epoch(model, dataloader, optimizer, device):
    model.train()
    total_loss = 0.0
    total_samples = 0

    for feats, masks, labels in dataloader:
        feats = feats.to(device)
        masks = masks.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        loss = model.compute_loss(feats, labels, img_masks=masks)
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

    for feats, masks, labels in dataloader:
        feats = feats.to(device)
        masks = masks.to(device)
        labels = labels.to(device)

        logits, _ = model.forward_logits(feats, masks)
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

    # 多标签指标（在所有类别上计算 F1）
    f1_micro = f1_score(labels_np, preds_np, average="micro", zero_division=0)
    f1_macro = f1_score(labels_np, preds_np, average="macro", zero_division=0)

    # 仅在当前 split 中有正样本的属性上计算 mAP_macro，避免全 0 列导致 sklearn warning
    pos_per_class = labels_np.sum(axis=0)  # [A]
    valid_mask = pos_per_class > 0
    if valid_mask.sum() == 0:
        map_macro = 0.0
    else:
        map_macro = average_precision_score(labels_np[:, valid_mask],
                                            probs_np[:, valid_mask],
                                            average="macro")

    return avg_loss, f1_micro, f1_macro, map_macro


def main():
    parser = argparse.ArgumentParser(description="Train attribute feature extractor on RSICD")
    parser.add_argument("--input_json", type=str, default="data/rsicd_with_attributes.json",
                        help="JSON with images and attribute_labels")
    parser.add_argument("--input_att_dir", type=str, default="data/rsicdtalk_att",
                        help="directory containing pre-extracted att feats (.npz)")
    parser.add_argument("--feat_dim", type=int, default=2048,
                        help="dimension of region features (att_feat_size)")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--num_epochs", type=int, default=40)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--min_lr", type=float, default=1e-6,
                        help="minimum learning rate for cosine schedule")
    parser.add_argument("--lr_warmup_epochs", type=int, default=4,
                        help="epochs used for linear warmup before cosine decay")
    parser.add_argument("--weight_decay", type=float, default=1e-2)
    parser.add_argument("--d_model", type=int, default=512)
    parser.add_argument("--nhead", type=int, default=8)
    parser.add_argument("--num_layers", type=int, default=6)
    parser.add_argument("--dim_feedforward", type=int, default=2048)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--threshold", type=float, default=0.5,
                        help="threshold for turning probabilities into binary labels")
    parser.add_argument("--loss_type", type=str, default='focal', choices=['bce', 'focal'],
                        help="loss function type: 'bce' or 'focal'")
    parser.add_argument("--focal_gamma", type=float, default=2.0,
                        help="gamma parameter for focal loss (only used when loss_type=focal)")
    parser.add_argument("--save_path", type=str, default="save/attribute_extractor.pth")
    parser.add_argument("--log_path", type=str, default=None,
                        help="path to save training log text file")
    parser.add_argument("--disable_weighted_sampler", action="store_true",
                        help="disable weighted sampler and fall back to random shuffling")

    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Ensure save and log directories exist
    save_dir = os.path.dirname(args.save_path)
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)

    if args.log_path is None:
        if save_dir:
            args.log_path = os.path.join(save_dir, "train_attribute_extractor.log")
        else:
            args.log_path = "train_attribute_extractor.log"

    log_dir = os.path.dirname(args.log_path)
    if log_dir:
        os.makedirs(log_dir, exist_ok=True)

    log_f = open(args.log_path, "a", encoding="utf-8")

    def log(msg: str):
        print(msg)
        print(msg, file=log_f)
        log_f.flush()

    # 数据集与 DataLoader
    train_dataset = AttributeRSICDDataset(args.input_json, args.input_att_dir, split="train")
    # 如果没有 val，则退化为 test
    try:
        val_dataset = AttributeRSICDDataset(args.input_json, args.input_att_dir, split="val")
    except ValueError:
        val_dataset = AttributeRSICDDataset(args.input_json, args.input_att_dir, split="test")

    num_attributes = train_dataset.num_attributes
    log(f"Num train images: {len(train_dataset)}")
    log(f"Num val images: {len(val_dataset)}")
    log(f"Num attributes: {num_attributes}")

    class_weights = compute_class_weights(train_dataset)
    sample_weights = None
    if not args.disable_weighted_sampler:
        sample_weights = compute_sampling_weights(train_dataset, class_weights)

    model = AdvancedAttributeFeatureExtractor(
        feat_dim=args.feat_dim,
        num_attributes=num_attributes,
        d_model=args.d_model,
        nhead=args.nhead,
        num_layers=args.num_layers,
        dim_feedforward=args.dim_feedforward,
        dropout=args.dropout,
        class_weights=class_weights,
        loss_type=args.loss_type,
        focal_gamma=args.focal_gamma,
    ).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        betas=(0.9, 0.999),
        weight_decay=args.weight_decay,
    )

    warmup_epochs = max(args.lr_warmup_epochs, 0)

    def lr_lambda(epoch):
        if warmup_epochs > 0 and epoch < warmup_epochs:
            return (epoch + 1) / float(warmup_epochs)
        progress_epochs = max(args.num_epochs - warmup_epochs, 1)
        progress = (epoch - warmup_epochs) / progress_epochs
        progress = min(max(progress, 0.0), 1.0)
        cosine = 0.5 * (1 + math.cos(math.pi * progress))
        min_lr_ratio = args.min_lr / max(args.lr, 1e-8)
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
        collate_fn=attribute_collate_fn,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=attribute_collate_fn,
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
