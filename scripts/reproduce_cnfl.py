"""Subject-stratified 5-fold reproduction of the paper's CNFL segmentation model."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import KFold, train_test_split
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm

try:
    from .models.cnfl_resunet import CNFLResUNet, two_class_dice_loss
except ImportError:
    from models.cnfl_resunet import CNFLResUNet, two_class_dice_loss


def discover_pairs(root: Path) -> dict[str, list[tuple[Path, Path]]]:
    """Use the paper dataset's density-pre field to select annotated CNFL images."""
    result: dict[str, list[tuple[Path, Path]]] = {}
    for patient in sorted((p for p in root.iterdir() if p.is_dir()), key=lambda p: int(p.name)):
        csv_files = [p for p in patient.glob("*.csv") if not p.name.startswith("._")]
        if len(csv_files) != 1:
            continue
        frame = pd.read_csv(csv_files[0])
        names = frame.loc[frame["density-pre"].notna(), "image name"].astype(str)
        pairs = []
        for name in names:
            image = patient / "images" / f"{name}_.jpg"
            mask = patient / "masks" / f"mask_{name}_.jpg"
            if image.is_file() and mask.is_file():
                pairs.append((image, mask))
        if pairs:
            result[patient.name] = pairs
    return result


class CNFLDataset(Dataset):
    def __init__(self, pairs: list[tuple[Path, Path]]) -> None:
        self.pairs = pairs

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, index: int):
        image_path, mask_path = self.pairs[index]
        image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        if image is None or mask is None:
            raise FileNotFoundError(f"Unreadable pair: {image_path}, {mask_path}")
        image_tensor = torch.from_numpy(image.astype(np.float32) / 255.0)[None]
        mask_tensor = torch.from_numpy((mask > 127).astype(np.int64))[None]
        return image_tensor, mask_tensor, str(image_path)


def loader(pairs, batch_size, shuffle, workers):
    return DataLoader(CNFLDataset(pairs), batch_size=batch_size, shuffle=shuffle,
                      num_workers=workers, pin_memory=True, persistent_workers=workers > 0)


def flatten(patient_pairs, patients):
    return [pair for patient in patients for pair in patient_pairs[patient]]


def train_epoch(model, data_loader, optimizer, scaler, device, epoch):
    model.train()
    total = 0.0
    progress = tqdm(data_loader, desc=f"epoch {epoch} train", dynamic_ncols=True)
    for images, masks, _ in progress:
        images, masks = images.to(device, non_blocking=True), masks.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=device.type == "cuda"):
            loss = two_class_dice_loss(model(images), masks)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        total += loss.item()
        progress.set_postfix(loss=f"{total / (progress.n + 1):.4f}")
    return total / len(data_loader)


@torch.no_grad()
def evaluate(model, data_loader, device, collect_predictions=False):
    model.eval()
    losses, image_dice, foreground_dice = [], [], []
    tp = fp = tn = fn = 0
    predictions = []
    for images, masks, paths in tqdm(data_loader, desc="evaluate", leave=False, dynamic_ncols=True):
        images, masks = images.to(device, non_blocking=True), masks.to(device, non_blocking=True)
        logits = model(images)
        losses.append(two_class_dice_loss(logits, masks).item())
        pred = logits.argmax(dim=1)
        truth = masks[:, 0]
        tp += int(((pred == 1) & (truth == 1)).sum())
        fp += int(((pred == 1) & (truth == 0)).sum())
        tn += int(((pred == 0) & (truth == 0)).sum())
        fn += int(((pred == 0) & (truth == 1)).sum())
        for index in range(len(pred)):
            class_scores = []
            for value in (0, 1):
                p, t = pred[index] == value, truth[index] == value
                score = (2.0 * (p & t).sum().float() + 1.0) / (p.sum() + t.sum() + 1.0)
                class_scores.append(score.item())
            image_dice.append(sum(class_scores) / 2.0)
            foreground_dice.append(class_scores[1])
            if collect_predictions:
                predictions.append((paths[index], pred[index].byte().cpu().numpy() * 255))
    eps = 1e-12
    return {
        "loss": float(np.mean(losses)),
        "dice": float(np.mean(image_dice)),
        "foreground_dice": float(np.mean(foreground_dice)),
        "recall": tp / (tp + fn + eps),
        "precision": tp / (tp + fp + eps),
        "specificity": tn / (tn + fp + eps),
    }, predictions


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=Path("dataset/nerves_and_dendritic_cells_old"))
    parser.add_argument("--output", type=Path, default=Path("results/cnfl_paper_reproduction"))
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--max_epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--learning_rate", type=float, default=1e-3)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=10)
    parser.add_argument("--early_stopping_patience", type=int, default=10)
    parser.add_argument("--lr_patience", type=int, default=5)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--save_predictions", action="store_true")
    parser.add_argument("--filters", type=int, nargs=4, default=(64, 128, 256, 512))
    parser.add_argument("--bridge_channels", type=int, default=1024)
    parser.add_argument("--only_fold", type=int, choices=range(1, 6), help="Run only one fold (useful for staged reproduction)")
    return parser.parse_args()


def main():
    args = parse_args()
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    device = torch.device(args.device)
    patient_pairs = discover_pairs(args.dataset)
    patients = np.array(sorted(patient_pairs, key=int))
    if len(patients) < args.folds:
        raise ValueError("Fewer annotated subjects than folds")
    args.output.mkdir(parents=True, exist_ok=args.only_fold is not None)
    configuration = vars(args).copy()
    configuration.update(annotated_subjects=len(patients), image_mask_pairs=sum(map(len, patient_pairs.values())),
                         torch_version=str(torch.__version__), gpu=torch.cuda.get_device_name(0) if device.type == "cuda" else None)
    configuration = {key: str(value) if isinstance(value, Path) else value for key, value in configuration.items()}
    (args.output / "configuration.json").write_text(json.dumps(configuration, indent=2))

    fold_rows = []
    splitter = KFold(n_splits=args.folds, shuffle=True, random_state=args.seed)
    for fold, (train_val_indices, test_indices) in enumerate(splitter.split(patients), start=1):
        if args.only_fold is not None and fold != args.only_fold:
            continue
        train_val_subjects, test_subjects = patients[train_val_indices], patients[test_indices]
        train_subjects, val_subjects = train_test_split(
            train_val_subjects, test_size=0.125, random_state=args.seed + fold
        )
        fold_dir = args.output / f"fold_{fold}"
        fold_dir.mkdir(exist_ok=False)
        split = {"train": train_subjects.tolist(), "validation": val_subjects.tolist(), "test": test_subjects.tolist()}
        (fold_dir / "subjects.json").write_text(json.dumps(split, indent=2))
        train_loader = loader(flatten(patient_pairs, train_subjects), args.batch_size, True, args.workers)
        val_loader = loader(flatten(patient_pairs, val_subjects), args.batch_size, False, args.workers)
        test_loader = loader(flatten(patient_pairs, test_subjects), args.batch_size, False, args.workers)

        model = CNFLResUNet(args.filters, args.bridge_channels).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
        scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=0.1,
                                      patience=args.lr_patience, min_lr=1e-7)
        scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
        best_loss, bad_epochs, history = float("inf"), 0, []
        for epoch in range(1, args.max_epochs + 1):
            train_loss = train_epoch(model, train_loader, optimizer, scaler, device, epoch)
            validation, _ = evaluate(model, val_loader, device)
            row = {"epoch": epoch, "train_loss": train_loss,
                   **{f"val_{key}": value for key, value in validation.items()},
                   "learning_rate": optimizer.param_groups[0]["lr"]}
            history.append(row)
            pd.DataFrame(history).to_csv(fold_dir / "history.csv", index=False)
            if validation["loss"] < best_loss:
                best_loss, bad_epochs = validation["loss"], 0
                torch.save({"model_state": model.state_dict(), "epoch": epoch,
                            "validation_loss": best_loss}, fold_dir / "best_model.pth")
            else:
                bad_epochs += 1
            scheduler.step(validation["loss"])
            print(f"fold {fold}/{args.folds} epoch {epoch}: train={train_loss:.4f} "
                  f"val={validation['loss']:.4f} dice={validation['dice']:.4f}")
            if bad_epochs >= args.early_stopping_patience:
                break

        checkpoint = torch.load(fold_dir / "best_model.pth", map_location=device, weights_only=True)
        model.load_state_dict(checkpoint["model_state"])
        test_metrics, predictions = evaluate(model, test_loader, device, args.save_predictions)
        test_metrics.update(fold=fold, best_epoch=checkpoint["epoch"],
                            train_subjects=len(train_subjects), val_subjects=len(val_subjects),
                            test_subjects=len(test_subjects), test_images=len(test_loader.dataset))
        fold_rows.append(test_metrics)
        (fold_dir / "metrics.json").write_text(json.dumps(test_metrics, indent=2))
        if predictions:
            prediction_dir = fold_dir / "predictions"; prediction_dir.mkdir()
            for source, prediction in predictions:
                cv2.imwrite(str(prediction_dir / Path(source).name), prediction)
        del model, optimizer, scaler
        torch.cuda.empty_cache()

    folds = pd.DataFrame(fold_rows).sort_values("fold")
    fold_metrics_path = args.output / "fold_metrics.csv"
    if fold_metrics_path.exists():
        previous = pd.read_csv(fold_metrics_path)
        folds = pd.concat((previous[previous["fold"] != folds.iloc[0]["fold"]], folds), ignore_index=True).sort_values("fold")
    folds.to_csv(fold_metrics_path, index=False)
    metric_names = ["dice", "foreground_dice", "recall", "precision", "specificity"]
    summary = {name: {"mean": float(folds[name].mean()), "sample_sd": float(folds[name].std(ddof=1))}
               for name in metric_names}
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
