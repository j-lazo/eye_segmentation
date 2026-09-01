"""PyTorch dataset and image/mask preprocessing utilities."""

from __future__ import annotations

import os
import random
from pathlib import Path
from typing import Sequence

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

AUGMENTATIONS = (
    "none", "rotate_90", "rotate_180", "rotate_270",
    "flip_vertical", "flip_horizontal", "add_salt_and_pepper_noise",
    "add_gaussian_noise", "adjust_brightness",
)


def _visible_files(directory: Path) -> list[Path]:
    return sorted(path for path in directory.iterdir() if path.is_file() and not path.name.startswith("."))


def build_list_dict(path_dataset: str | os.PathLike, patient_cases: Sequence[str] | None = None):
    """Pair images and masks in patient folders, preserving the TF data contract."""
    root = Path(path_dataset)
    cases = list(patient_cases) if patient_cases is not None else [p.name for p in root.iterdir() if p.is_dir()]
    pairs: list[dict[str, str]] = []
    for case in sorted(cases):
        images_dir, masks_dir = root / case / "images", root / case / "masks"
        if not images_dir.is_dir() or not masks_dir.is_dir():
            continue
        masks = _visible_files(masks_dir)
        for image in _visible_files(images_dir):
            matches = [mask for mask in masks if image.stem in mask.stem]
            if len(matches) > 1:
                raise ValueError(f"Multiple masks match {image}: {matches}")
            if matches:
                pairs.append({"path_img": str(image), "path_mask": str(matches[0])})
    return pairs


def read_img(path_img: str, img_size: tuple[int, int] = (256, 256)) -> np.ndarray:
    image = cv2.imread(path_img, cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(f"Could not read image: {path_img}")
    # The TensorFlow implementation trains on OpenCV's BGR channel order.
    return cv2.resize(image, img_size, interpolation=cv2.INTER_AREA).astype(np.float32) / 255.0


def read_mask(path_mask: str, img_size: tuple[int, int] = (256, 256)) -> np.ndarray:
    mask = cv2.imread(path_mask, cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise FileNotFoundError(f"Could not read mask: {path_mask}")
    mask = cv2.resize(mask, img_size, interpolation=cv2.INTER_AREA)
    _, mask = cv2.threshold(mask, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    return (mask.astype(np.float32) / 255.0)[..., None]


def _augment(image: np.ndarray, mask: np.ndarray, choice: str) -> tuple[np.ndarray, np.ndarray]:
    if choice == "rotate_90":
        image, mask = np.rot90(image, 1), np.rot90(mask, 1)
    elif choice == "rotate_180":
        image, mask = np.rot90(image, 2), np.rot90(mask, 2)
    elif choice == "rotate_270":
        image, mask = np.rot90(image, 3), np.rot90(mask, 3)
    elif choice == "flip_vertical":
        image, mask = np.flip(image, 1), np.flip(mask, 1)
    elif choice == "flip_horizontal":
        image, mask = np.flip(image, 0), np.flip(mask, 0)
    elif choice == "add_salt_and_pepper_noise":
        count = int(image.shape[0] * image.shape[1] * 0.05)
        rows = np.random.randint(0, image.shape[0], count)
        cols = np.random.randint(0, image.shape[1], count)
        image = image.copy()
        image[rows, cols] = np.random.randint(0, 2, (count, 1))
    elif choice == "add_gaussian_noise":
        image = np.clip(image + np.random.normal(0, 0.5 / 255.0, image.shape), 0, 1)
    elif choice == "adjust_brightness":
        gamma = random.choice((0.85, 0.9, 1.1, 1.2, 1.3, 1.4, 1.5))
        image = np.clip(image ** (1.0 / gamma), 0, 1)
    elif choice != "none":
        raise ValueError(f"Unknown augmentation: {choice}")
    return np.ascontiguousarray(image, dtype=np.float32), np.ascontiguousarray(mask, dtype=np.float32)


class SegmentationDataset(Dataset):
    def __init__(self, annotations_list, img_size=256, training_mode=False,
                 analyze_dataset=False, augmentation_functions: Sequence[str] = ("none",)):
        self.annotations = list(annotations_list)
        self.img_size = int(img_size)
        self.training_mode = bool(training_mode)
        self.analyze_dataset = bool(analyze_dataset)
        self.augmentation_functions = tuple(value.lower() for value in augmentation_functions)
        unknown = set(self.augmentation_functions) - set(AUGMENTATIONS)
        if unknown:
            raise ValueError(f"Unknown augmentations: {sorted(unknown)}")

    def __len__(self) -> int:
        return len(self.annotations)

    def __getitem__(self, index: int):
        item = self.annotations[index]
        size = (self.img_size, self.img_size)
        image, mask = read_img(item["path_img"], size), read_mask(item["path_mask"], size)
        if self.training_mode:
            image, mask = _augment(image, mask, random.choice(self.augmentation_functions))
        image_tensor = torch.from_numpy(image).permute(2, 0, 1)
        mask_tensor = torch.from_numpy(mask).permute(2, 0, 1)
        if self.analyze_dataset:
            return image_tensor, mask_tensor, item["path_img"]
        return image_tensor, mask_tensor


def make_dataloader(annotations_list, batch_size=8, img_size=256, training_mode=False,
                    analyze_dataset=False, augmentation_functions=("none",), num_workers=4,
                    pin_memory=None, drop_remainder=True):
    dataset = SegmentationDataset(annotations_list, img_size, training_mode,
                                  analyze_dataset, augmentation_functions)
    if pin_memory is None:
        pin_memory = torch.cuda.is_available()
    return DataLoader(dataset, batch_size=batch_size, shuffle=training_mode,
                      drop_last=drop_remainder, num_workers=num_workers,
                      pin_memory=pin_memory, persistent_workers=num_workers > 0)
