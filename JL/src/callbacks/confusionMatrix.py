import os
from typing import List

import torch
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from pytorch_lightning import Callback, Trainer, LightningModule

class ConfusionMatrixCallback(Callback):
    def __init__(
        self,
        num_classes: int,
        class_names: List[str] | None = None,
        save_dir: str = "confusion_matrices",
        every_n_epoch: int = 1,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.class_names     = class_names if class_names is not None else [str(i) for i in range(num_classes)]
        self.save_dir        = save_dir
        self.every_n_epoch   = every_n_epoch
        os.makedirs(save_dir, exist_ok=True)

        self.preds   = []
        self.targets = []

    @torch.no_grad()
    def on_validation_batch_end(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
        outputs,
        batch,
        batch_idx,
        dataloader_idx: int = 0,
    ):
        logits  = outputs["logits"]        # (B, C)
        targets = outputs["targets"]       # (B,) hard-label

        preds = torch.argmax(logits, dim=1)
        self.preds.append(preds.cpu())
        self.targets.append(targets.cpu())

    def on_validation_epoch_end(self, trainer: Trainer, pl_module: LightningModule):
        epoch = trainer.current_epoch
        if (epoch + 1) % self.every_n_epoch != 0:
            self._reset(); return

        preds   = torch.cat(self.preds).numpy()
        targets = torch.cat(self.targets).numpy()

        cm = confusion_matrix(targets, preds, labels=range(self.num_classes))
        cm_norm = cm.astype("float") / (cm.sum(axis=1, keepdims=True) + 1e-8)

        fig, ax = plt.subplots(figsize=(14, 12))
        disp = ConfusionMatrixDisplay(confusion_matrix=cm_norm, display_labels=self.class_names)

        disp.plot(include_values=True, cmap="Blues", ax=ax, colorbar=True)

        ax.set_xticklabels(self.class_names, rotation=45, ha="right", fontsize=8)
        ax.set_yticklabels(self.class_names, rotation=0, fontsize=8)

        ax.set_title(f"Confusion Matrix @ epoch {epoch}")
        plt.tight_layout()

        fn = os.path.join(self.save_dir, f"cm_epoch_{epoch:03d}.png")
        fig.savefig(fn, dpi=150)
        plt.close(fig)

        self._reset()

    def _reset(self):
        self.preds.clear()
        self.targets.clear()