import argparse

import torch
from torch.utils.data import DataLoader
from torchvision import transforms

from train_attribute_extractor_e2e import AttributeRSICDImageDataset, evaluate
from models.ResNetAttributeExtractor import ResNetAttributeExtractor


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate end-to-end ResNet-based attribute extractor on RSICD",
    )
    parser.add_argument(
        "--input_json",
        type=str,
        default="data/rsicd_with_attributes_top25.json",
        help="JSON with images and attribute_labels",
    )
    parser.add_argument(
        "--images_root",
        type=str,
        default="data/RSICD_images",
        help="directory containing raw RSICD images",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="val",
        choices=["train", "val", "test"],
        help="which split to evaluate on",
    )
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="threshold for turning probabilities into binary labels",
    )
    parser.add_argument(
        "--thresholds",
        type=str,
        default=None,
        help="comma-separated list of thresholds to scan, e.g. '0.1,0.2,0.3'",
    )
    parser.add_argument(
        "--ckpt",
        type=str,
        default="save/attribute_extractor_top25_e2e_resnet101.pth",
        help="path to trained checkpoint",
    )

    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 使用与 train_attribute_extractor_e2e.py 相同的图像预处理
    transform = transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ]
    )

    dataset = AttributeRSICDImageDataset(
        args.input_json, args.images_root, split=args.split, transform=transform
    )

    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    print(f"Eval split: {args.split}")
    print(f"Num images: {len(dataset)}")
    print(f"Num attributes: {dataset.num_attributes}")

    # Load checkpoint
    ckpt = torch.load(args.ckpt, map_location=device)
    num_attributes = ckpt.get("num_attributes", dataset.num_attributes)
    ckpt_args = ckpt.get("args", {})

    d_model = ckpt_args.get("d_model", 512)
    nhead = ckpt_args.get("nhead", 8)
    num_layers = ckpt_args.get("num_layers", 6)
    dim_feedforward = ckpt_args.get("dim_feedforward", 2048)
    dropout = ckpt_args.get("dropout", 0.1)
    loss_type = ckpt_args.get("loss_type", "bce")
    focal_gamma = ckpt_args.get("focal_gamma", 2.0)
    backbone = ckpt_args.get("backbone", "resnet101")
    imagenet_weights_dir = ckpt_args.get("imagenet_weights_dir", "data/imagenet_weights")

    # Build model with the same architecture as training
    model = ResNetAttributeExtractor(
        num_attributes=num_attributes,
        d_model=d_model,
        nhead=nhead,
        num_layers=num_layers,
        dim_feedforward=dim_feedforward,
        dropout=dropout,
        class_weights=None,
        loss_type=loss_type,
        focal_gamma=focal_gamma,
        backbone=backbone,
        imagenet_weights_dir=imagenet_weights_dir,
    ).to(device)

    model.load_state_dict(ckpt["model_state"], strict=False)

    if args.thresholds is not None:
        thr_list = [float(t) for t in args.thresholds.split(",") if t.strip()]

        print("\nThreshold scan:")
        print(f"  thresholds: {thr_list}")

        best_thr = None
        best_f1_macro = -1.0
        best_metrics = None  # (loss, f1_micro, map_macro)

        for th in thr_list:
            val_loss, f1_micro, f1_macro, map_macro = evaluate(
                model, dataloader, device, threshold=th
            )
            print(
                f"  thr={th:.3f} | loss={val_loss:.4f}, "
                f"F1_micro={f1_micro:.4f}, F1_macro={f1_macro:.4f}, mAP_macro={map_macro:.4f}"
            )

            if f1_macro > best_f1_macro:
                best_f1_macro = f1_macro
                best_thr = th
                best_metrics = (val_loss, f1_micro, map_macro)

        if best_thr is not None and best_metrics is not None:
            best_loss, best_f1_micro, best_map_macro = best_metrics
            print("\nBest threshold by F1_macro:")
            print(f"  thr       = {best_thr:.3f}")
            print(f"  loss      = {best_loss:.4f}")
            print(f"  F1_micro  = {best_f1_micro:.4f}")
            print(f"  F1_macro  = {best_f1_macro:.4f}")
            print(f"  mAP_macro = {best_map_macro:.4f}")
    else:
        val_loss, f1_micro, f1_macro, map_macro = evaluate(
            model, dataloader, device, threshold=args.threshold
        )

        print("\nEvaluation results:")
        print(f"  loss      = {val_loss:.4f}")
        print(f"  F1_micro  = {f1_micro:.4f}")
        print(f"  F1_macro  = {f1_macro:.4f}")
        print(f"  mAP_macro = {map_macro:.4f}")


if __name__ == "__main__":
    main()
