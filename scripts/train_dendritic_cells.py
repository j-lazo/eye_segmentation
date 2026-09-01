"""Train a residual U-Net on the project's SAM dendritic-cell pseudo-masks.

The repository does not contain the original manual pixel masks.  This script
therefore treats one of the saved, box-prompted SAM result sets as weak labels
and keeps subjects disjoint between training, validation, and testing.
"""

from __future__ import annotations

import argparse
import json
import random
import re
from datetime import date
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split

try:
    from .models.Unet import UNetRes
    from .train_unet import fit
    from .utils.data_loaders import AUGMENTATIONS, make_dataloader
except ImportError:
    from models.Unet import UNetRes
    from train_unet import fit
    from utils.data_loaders import AUGMENTATIONS, make_dataloader


def subject_id(name: str) -> str:
    match = re.search(r"p_case_0*(\d+)", name)
    if not match:
        raise ValueError(f"Cannot extract subject from {name}")
    return match.group(1)


def find_image(dataset: Path, mask: Path) -> Path:
    subject = subject_id(mask.name)
    candidates = list((dataset / subject / "images").glob(mask.stem + ".*"))
    candidates = [p for p in candidates if not p.name.startswith(".")]
    if len(candidates) != 1:
        raise FileNotFoundError(f"Expected one source image for {mask.name}; found {candidates}")
    return candidates[0]


def build_pairs(dataset: Path, pseudo_masks: Path) -> list[dict[str, str]]:
    pairs = []
    for mask in sorted(pseudo_masks.glob("*")):
        if mask.is_file() and not mask.name.startswith("."):
            pairs.append({"path_img": str(find_image(dataset, mask)), "path_mask": str(mask)})
    if not pairs:
        raise ValueError(f"No pseudo-masks found in {pseudo_masks}")
    return pairs


def split_by_subject(pairs, seed: int):
    subjects = sorted({subject_id(Path(p["path_img"]).name) for p in pairs})
    train_subjects, holdout = train_test_split(subjects, test_size=0.30, random_state=seed)
    val_subjects, test_subjects = train_test_split(holdout, test_size=2 / 3, random_state=seed)
    def select(ids):
        return [p for p in pairs if subject_id(Path(p["path_img"]).name) in ids]
    return select(set(train_subjects)), select(set(val_subjects)), select(set(test_subjects)), {
        "train": sorted(train_subjects), "validation": sorted(val_subjects), "test": sorted(test_subjects)
    }


@torch.no_grad()
def evaluate(model, loader, device, output: Path):
    pred_dir = output / "predictions"
    pred_dir.mkdir()
    rows, examples = [], []
    totals = np.zeros(4, dtype=np.float64)  # tp, fp, fn, tn
    model.eval()
    for images, masks, paths in loader:
        probabilities = torch.sigmoid(model(images.to(device))).cpu().numpy()[:, 0]
        truth = masks.numpy()[:, 0] >= 0.5
        for image, probability, target, source in zip(images.numpy(), probabilities, truth, paths):
            prediction = probability >= 0.5
            tp = np.logical_and(prediction, target).sum()
            fp = np.logical_and(prediction, ~target).sum()
            fn = np.logical_and(~prediction, target).sum()
            tn = np.logical_and(~prediction, ~target).sum()
            totals += (tp, fp, fn, tn)
            rows.append({
                "file": source,
                "subject": subject_id(Path(source).name),
                "dice": (2 * tp + 1e-7) / (2 * tp + fp + fn + 1e-7),
                "precision": (tp + 1e-7) / (tp + fp + 1e-7),
                "recall": (tp + 1e-7) / (tp + fn + 1e-7),
            })
            stem = Path(source).stem
            cv2.imwrite(str(pred_dir / f"{stem}_probability.png"), np.uint8(probability * 255))
            cv2.imwrite(str(pred_dir / f"{stem}_binary.png"), np.uint8(prediction) * 255)
            if len(examples) < 6:
                examples.append((image.transpose(1, 2, 0)[..., ::-1], target, probability, prediction))
    frame = pd.DataFrame(rows)
    frame.to_csv(output / "test_per_image.csv", index=False)
    tp, fp, fn, tn = totals
    summary = {
        "test_images": len(frame),
        "mean_image_dice": float(frame.dice.mean()),
        "global_dice": float((2 * tp + 1e-7) / (2 * tp + fp + fn + 1e-7)),
        "precision": float((tp + 1e-7) / (tp + fp + 1e-7)),
        "recall": float((tp + 1e-7) / (tp + fn + 1e-7)),
        "specificity": float((tn + 1e-7) / (tn + fp + 1e-7)),
    }
    with (output / "test_summary.json").open("w") as f:
        json.dump(summary, f, indent=2)
    if examples:
        fig, axes = plt.subplots(len(examples), 4, figsize=(12, 3 * len(examples)))
        axes = np.atleast_2d(axes)
        for row, (image, target, probability, prediction) in zip(axes, examples):
            for ax, value, title in zip(row, (image, target, probability, prediction),
                                        ("Image", "Pseudo-label", "Probability", "Binary prediction")):
                ax.imshow(value, cmap=None if value.ndim == 3 else "gray", vmin=0, vmax=1)
                ax.set_title(title); ax.axis("off")
        fig.tight_layout(); fig.savefig(output / "test_examples.png", dpi=160); plt.close(fig)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=Path("dataset/nerves_and_dendritic_cells"))
    parser.add_argument("--pseudo-masks", type=Path, default=Path(
        "results/SAM_models_results/SAM_boxes_1_ch_2_26_02_2025_15_37/predictions"))
    parser.add_argument("--output", type=Path, default=Path(f"results/dendritic_cells_augmented_{date.today()}") )
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--image-size", type=int, default=384)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    if not torch.cuda.is_available():
        raise RuntimeError("This experiment requires CUDA")
    args.output.mkdir(parents=True, exist_ok=False)
    pairs = build_pairs(args.dataset, args.pseudo_masks)
    train_pairs, val_pairs, test_pairs, splits = split_by_subject(pairs, args.seed)
    loader_args = dict(img_size=args.image_size, num_workers=args.workers)
    train_loader = make_dataloader(train_pairs, batch_size=args.batch_size, training_mode=True,
                                   augmentation_functions=AUGMENTATIONS, drop_remainder=False, **loader_args)
    val_loader = make_dataloader(val_pairs, batch_size=1, training_mode=False,
                                 augmentation_functions=("none",), drop_remainder=False, **loader_args)
    test_loader = make_dataloader(test_pairs, batch_size=1, analyze_dataset=True,
                                  drop_remainder=False, **loader_args)
    config = vars(args).copy()
    config.update(device="cuda", subjects=splits, images={"train": len(train_pairs),
                  "validation": len(val_pairs), "test": len(test_pairs)},
                  augmentations=list(AUGMENTATIONS), label_type="SAM box-prompt pseudo-mask")
    with (args.output / "experiment.json").open("w") as f:
        json.dump(config, f, indent=2, default=str)
    model = UNetRes(num_filters=(32, 64, 128, 256, 512)).cuda()
    fit(model, train_loader, val_loader, torch.device("cuda"), args.output,
        args.learning_rate, args.epochs)
    summary = evaluate(model, test_loader, torch.device("cuda"), args.output)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
