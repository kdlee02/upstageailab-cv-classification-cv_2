import os
from typing import List

import pandas as pd
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from pytorch_lightning import Callback

class ErrorAnalysisCallback(Callback):
    def __init__(self, 
                 num_classes: int, 
                 class_names: List[str] | None = None,
                 save_dir="error_analysis", top_k=10):
        super().__init__()
        self.num_classes = num_classes
        self.class_names = class_names if class_names is not None else [str(i) for i in range(num_classes)]
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)
        self.top_k = top_k
        self.reset_buffer()

    def reset_buffer(self):
        self.all_preds = []
        self.all_probs = []
        self.all_targets = []
        self.all_fnames = []
    
    def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0):
        x, y = batch
        logits = pl_module(x)
        probs = F.softmax(logits, dim=1)
        preds = torch.argmax(probs, dim=1)

        if hasattr(batch, 'filenames'):
            fnames = batch.filenames
        elif hasattr(pl_module.trainer.datamodule.val_ds, "samples"):
            fnames = [pl_module.trainer.datamodule.val_ds.samples[idx][0] for idx in range(batch_idx * len(x), (batch_idx + 1) * len(x))]
        else:
            fnames = [f"sample_{batch_idx}_{i}.png" for i in range(len(x))]

        self.all_preds.extend(preds.cpu().numpy())
        self.all_probs.extend(probs.max(dim=1).values.cpu().numpy())
        self.all_targets.extend(y.cpu().numpy())
        self.all_fnames.extend(fnames)
    

    def on_validation_epoch_end(self, trainer, pl_module):
        df = pd.DataFrame({
            "filename": self.all_fnames,
            "id": list(range(len(self.all_fnames))),  # Add ID for each sample
            "target": self.all_targets,
            "pred": self.all_preds,
            "confidence": self.all_probs
        })

        error_df = df[df["target"] != df["pred"]]
        error_df = error_df.sort_values("confidence", ascending=False)

        save_df = error_df.head(self.top_k).copy()

        save_df["target_name"] = save_df["target"].apply(lambda x: self.class_names[int(x)])
        save_df["pred_name"]   = save_df["pred"].apply(lambda x: self.class_names[int(x)])

        save_df = save_df[['filename', 'id', 'target_name', 'pred_name', 'confidence']]

        save_path = os.path.join(self.save_dir, f"val_errors_epoch{trainer.current_epoch:03d}.csv")
        save_df.to_csv(save_path, index=False)

        print(f"[ErrorAnalysis] Saved {len(save_df)} misclassified samples → {save_path}")

        N = len(save_df)
        n_cols = 5
        n_rows = int(np.ceil(N / n_cols))
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 3, n_rows * 3))

        for ax, (_, row) in zip(axes.flatten(), save_df.iterrows()):
            try:
                image_root = os.path.join(trainer.datamodule.data_path, trainer.datamodule.full_data_name)

                img_path = os.path.join(image_root, row["filename"])
                img = Image.open(img_path).convert("RGB")
                ax.imshow(img)
                ax.set_title(f'ID: {os.path.basename(row["filename"])}\nActual: {row["target_name"]}\nPredicted: {row["pred_name"]}\nConfidence: {row["confidence"]:.2f}',
                             fontsize=8)
                ax.axis("off")
            except Exception as e:
                ax.axis("off")
                print(f"Error loading image {row['filename']}: {e}")

        # 빈칸 제거
        for ax in axes.flatten()[N:]:
            ax.axis("off")

        plt.tight_layout()
        img_save_path = os.path.join(self.save_dir, f"val_errors_epoch{trainer.current_epoch:03d}.png")
        plt.savefig(img_save_path, dpi=150)
        plt.close()
        print(f"[ErrorAnalysis] Saved error thumbnails → {img_save_path}")

        self.reset_buffer()