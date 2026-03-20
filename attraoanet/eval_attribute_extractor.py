import argparse

import torch
from torch.utils.data import DataLoader

from train_attribute_extractor import AttributeRSICDDataset, attribute_collate_fn, evaluate
from models.AttributeFeatureExtractor import AttributeFeatureExtractor


def main():
    parser = argparse.ArgumentParser(description="Evaluate trained attribute feature extractor on RSICD")
    parser.add_argument("--input_json", type=str, default="data/rsicd_with_attributes.json",
                        help="JSON with images and attribute_labels")
    parser.add_argument("--input_att_dir", type=str, default="data/rsicdtalk_att",
                        help="directory containing pre-extracted att feats (.npz)")
    parser.add_argument("--split", type=str, default="val", choices=["train", "val", "test"],
                        help="which split to evaluate on")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--threshold", type=float, default=0.5,
                        help="threshold for turning probabilities into binary labels")
    parser.add_argument("--thresholds", type=str, default=None,
                        help="comma-separated list of thresholds to scan, e.g. '0.1,0.2,0.3'")
    parser.add_argument("--ckpt", type=str, default="save/attribute_extractor.pth",
                        help="path to trained checkpoint")

    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset = AttributeRSICDDataset(args.input_json, args.input_att_dir, split=args.split)
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=attribute_collate_fn,
        pin_memory=True,
    )

    print(f"Eval split: {args.split}")
    print(f"Num images: {len(dataset)}")
    print(f"Num attributes: {dataset.num_attributes}")

    # Load checkpoint
    ckpt = torch.load(args.ckpt, map_location=device)
    num_attributes = ckpt.get("num_attributes", dataset.num_attributes)
    ckpt_args = ckpt.get("args", {})

    feat_dim = ckpt_args.get("feat_dim", 2048)
    d_model = ckpt_args.get("d_model", 512)
    nhead = ckpt_args.get("nhead", 8)
    num_layers = ckpt_args.get("num_layers", 6)
    dim_feedforward = ckpt_args.get("dim_feedforward", 2048)
    dropout = ckpt_args.get("dropout", 0.1)

    # Build model with the same architecture as training
    model = AttributeFeatureExtractor(
        feat_dim=feat_dim,
        num_attributes=num_attributes,
        d_model=d_model,
        nhead=nhead,
        num_layers=num_layers,
        dim_feedforward=dim_feedforward,
        dropout=dropout,
        class_weights=None,
    ).to(device)

    # Allow extra keys like 'class_weights' that may be present in the checkpoint
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
