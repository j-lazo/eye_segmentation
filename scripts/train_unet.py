"""Train and evaluate the PyTorch Residual U-Net port."""

from __future__ import annotations

import argparse
import random
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
import yaml
from sklearn.model_selection import train_test_split
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm.auto import tqdm

try:
    from torch.utils.tensorboard import SummaryWriter
except ImportError:  # TensorBoard logging is useful, but not required to train.
    SummaryWriter = None

try:  # Supports both direct execution and ``python -m scripts_tf.train_unet``.
    from .models.Unet import UNetRes
    from .utils.data_loaders import AUGMENTATIONS, build_list_dict, make_dataloader
    from .utils.loss_functions import dice_coef_loss_from_logits
    from .utils.metric_functions import bin_metrics_from_logits, dice_coef
    from .utils.trainer_auxiliars import EarlyStopping, save_checkpoint
except ImportError:
    from models.Unet import UNetRes
    from utils.data_loaders import AUGMENTATIONS, build_list_dict, make_dataloader
    from utils.loss_functions import dice_coef_loss_from_logits
    from utils.metric_functions import bin_metrics_from_logits, dice_coef
    from utils.trainer_auxiliars import EarlyStopping, save_checkpoint


def _run_epoch(model, loader, device, optimizer=None, description=""):
    training = optimizer is not None
    model.train(training)
    totals = {key: 0.0 for key in ("loss", "acc", "precision", "recall", "dice_coef")}
    if not len(loader):
        raise ValueError("The data loader has zero batches; reduce batch_size or disable drop_remainder")
    context = torch.enable_grad() if training else torch.no_grad()
    with context:
        for images, masks in tqdm(loader, desc=description, dynamic_ncols=True):
            images = images.to(device, non_blocking=True)
            masks = masks.to(device, non_blocking=True)
            if training:
                optimizer.zero_grad(set_to_none=True)
            logits = model(images)
            loss = dice_coef_loss_from_logits(logits, masks)
            if training:
                loss.backward()
                optimizer.step()
            acc, precision, recall, dice = bin_metrics_from_logits(logits, masks)
            for key, value in zip(totals, (loss.item(), acc, precision, recall, dice)):
                totals[key] += value
    return {key: value / len(loader) for key, value in totals.items()}


def fit(model, train_loader, val_loader, device, results_directory, lr, epochs):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=0.1, patience=10)
    early_stopper = EarlyStopping(patience=15, restore_best_weights=True)
    checkpoint_path = results_directory / "best_model.pth"
    writer = SummaryWriter(str(results_directory / "tensorboard")) if SummaryWriter else None
    rows = []
    best_val_loss = float("inf")

    for epoch in range(1, epochs + 1):
        train = _run_epoch(model, train_loader, device, optimizer, f"Epoch {epoch}/{epochs} [train]")
        val = _run_epoch(model, val_loader, device, description=f"Epoch {epoch}/{epochs} [val]")
        row = {**train, **{f"val_{key}": value for key, value in val.items()},
               "learning_rate": optimizer.param_groups[0]["lr"]}
        rows.append(row)
        pd.DataFrame(rows).to_csv(results_directory / "training_history.csv", index=False)
        if writer:
            for key, value in row.items():
                writer.add_scalar(key, value, epoch)
        if val["loss"] < best_val_loss:
            best_val_loss = val["loss"]
            save_checkpoint(checkpoint_path, model, optimizer, epoch, best_val_loss)
        scheduler.step(val["loss"])
        print(f"Epoch {epoch}: loss={train['loss']:.4f}, val_loss={val['loss']:.4f}, "
              f"val_dice={val['dice_coef']:.4f}, lr={row['learning_rate']:.2e}")
        if early_stopper.step(val["loss"], model):
            break

    early_stopper.restore(model)
    if writer:
        writer.close()
    return pd.DataFrame(rows), checkpoint_path


@torch.no_grad()
def evaluate(model, loader, device, results_directory, image_size):
    predictions_dir = results_directory / "predictions"
    predictions_dir.mkdir(exist_ok=True)
    rows = []
    model.eval()
    for images, masks, paths in tqdm(loader, desc="Testing", dynamic_ncols=True):
        probabilities = torch.sigmoid(model(images.to(device))).cpu()
        for probability, mask, source_path in zip(probabilities, masks, paths):
            score = dice_coef(probability[None], mask[None]).item()
            output_path = predictions_dir / Path(source_path).name
            cv2.imwrite(str(output_path), (probability[0].numpy() * 255).astype(np.uint8))
            rows.append({"file name": source_path, "DSC": score})
    frame = pd.DataFrame(rows)
    frame.to_csv(results_directory / f"predictions_test_ds_{image_size}x{image_size}.csv", index=False)
    print(f"Mean DSC test dataset: {frame['DSC'].mean():.6f}")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--path_dataset", type=Path, default=Path.cwd() / "dataset")
    parser.add_argument("--project_folder", type=Path, default=Path.cwd())
    parser.add_argument("--name_model", default="Res-UNet")
    parser.add_argument("--run_name", help="Exact output directory name under project_folder/results")
    parser.add_argument("--learning_rate", type=float, default=1e-3)
    parser.add_argument("--max_epochs", type=int, default=4)
    parser.add_argument("--image_size", type=int, default=64)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--num_filters", type=int, nargs="+", default=[32, 64, 128, 256, 512])
    parser.add_argument("--augmentation_functions", nargs="+", default=["all"])
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=10)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def main():
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available; pass --device cpu")
    device = torch.device(args.device)
    cases = sorted(path.name for path in args.path_dataset.iterdir()
                   if path.is_dir() and (path / "images").is_dir() and (path / "masks").is_dir())
    if len(cases) < 4:
        raise ValueError(f"Expected at least four patient folders containing images/ and masks/ in {args.path_dataset}")
    train_cases, val_test_cases = train_test_split(cases, test_size=0.30, random_state=args.seed)
    val_cases, test_cases = train_test_split(val_test_cases, test_size=0.40, random_state=args.seed)
    train_pairs, val_pairs = build_list_dict(args.path_dataset, train_cases), build_list_dict(args.path_dataset, val_cases)
    test_pairs = build_list_dict(args.path_dataset, test_cases)
    augmentations = AUGMENTATIONS if args.augmentation_functions == ["all"] else args.augmentation_functions
    loader_args = dict(batch_size=args.batch_size, img_size=args.image_size, num_workers=args.num_workers)
    train_loader = make_dataloader(train_pairs, training_mode=True, augmentation_functions=augmentations, **loader_args)
    val_loader = make_dataloader(val_pairs, training_mode=False, augmentation_functions=("none",), **loader_args)
    test_loader = make_dataloader(test_pairs, batch_size=1, img_size=args.image_size,
                                  analyze_dataset=True, drop_remainder=False, num_workers=args.num_workers)

    model = UNetRes(num_filters=args.num_filters).to(device)
    run_id = args.run_name or f"{args.name_model}_lr_{args.learning_rate}_bs_{args.batch_size}_{datetime.now():%d_%m_%Y_%H_%M}"
    results_directory = args.project_folder / "results" / run_id
    results_directory.mkdir(parents=True, exist_ok=False)
    parameters = vars(args).copy()
    parameters.update(train_cases=train_cases, val_cases=val_cases, test_cases=test_cases,
                      pytorch_version=str(torch.__version__), device=str(device))
    parameters = {key: str(value) if isinstance(value, Path) else value for key, value in parameters.items()}
    with (results_directory / "parameters_training.yaml").open("w") as stream:
        yaml.safe_dump(parameters, stream, sort_keys=False)
    fit(model, train_loader, val_loader, device, results_directory, args.learning_rate, args.max_epochs)
    evaluate(model, test_loader, device, results_directory, args.image_size)
    print(f"Experiment finished: {results_directory}")


if __name__ == "__main__":
    main()
