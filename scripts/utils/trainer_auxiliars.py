import torch
import copy 
from tqdm.auto import tqdm

from .metric_functions import bin_metrics_from_logits
from .loss_functions import dice_coef_loss

# Early stopping 

class EarlyStopping:
    def __init__(self, patience=15, restore_best_weights=True):
        self.patience = patience
        self.restore_best_weights = restore_best_weights
        self.best_loss = float("inf")
        self.best_state = None
        self.bad_epochs = 0

    def step(self, val_loss, model):
        improved = val_loss < self.best_loss - 1e-12
        if improved:
            self.best_loss = val_loss
            self.bad_epochs = 0
            if self.restore_best_weights:
                self.best_state = copy.deepcopy(model.state_dict())
        else:
            self.bad_epochs += 1

        stop = self.bad_epochs >= self.patience
        return stop

    def restore(self, model):
        if self.restore_best_weights and self.best_state is not None:
            model.load_state_dict(self.best_state)

# Save Checkpoints 
def save_checkpoint(path, model, optimizer, epoch, best_val_loss):
    torch.save(
        {
            "epoch": epoch,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "best_val_loss": best_val_loss,
        },
        path,
    )


# train 1 epoch 

def train_one_epoch(model, loader, optimizer, device, epoch, epochs):
    model.train()
    running_loss = 0.0
    running_acc = running_prec = running_rec = running_dice = 0.0
    n_batches = 0

    running = {
        "loss": 0.0,
        "acc": 0.0,
        "precision": 0.0,
        "recall": 0.0,
        "dice_coef": 0.0,
    }

    pbar = tqdm(loader, desc=f"Epoch {epoch}/{epochs} [train]", dynamic_ncols=True)
    for imgs, masks in pbar:
        imgs = imgs.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        logits = model(imgs)
        loss = dice_coef_loss(torch.sigmoid(logits), masks)
        loss.backward()
        optimizer.step()
        acc, prec, rec, dice = bin_metrics_from_logits(logits, masks)

        # accumulate
        running["loss"] += loss.item()
        running["acc"] += acc
        running["precision"] += prec
        running["recall"] += rec
        running["dice_coef"] += dice

        n = pbar.n + 1  # number of processed batches

        # update progress bar (averages so far)
        pbar.set_postfix(
            loss=f"{running['loss']/n:.4f}",
            acc=f"{running['acc']/n:.3f}",
            prec=f"{running['precision']/n:.3f}",
            rec=f"{running['recall']/n:.3f}",
            dsc=f"{running['dice_coef']/n:.3f}",
            lr=f"{optimizer.param_groups[0]['lr']:.2e}",
        )

    # epoch averages
    for k in running:
        running[k] /= len(loader)

    return running
