# eye_segmentation

## PyTorch Residual U-Net

The PyTorch port of the original TensorFlow/Keras binary segmentation pipeline
is in `scripts`. 

The dataset root must contain one directory per patient, with an `images/` and
`masks/` directory inside each patient directory. Train from the repository
root with:

```bash
python scripts_tf/train_unet.py \
  --path_dataset /path/to/patient_dataset \
  --image_size 256 \
  --batch_size 8 \
  --num_filters 32 64 128 256 512 \
  --max_epochs 100
```

The run writes its parameters, CSV history, TensorBoard events, best `.pth`
checkpoint, test Dice scores, and predicted masks under `results/`.
