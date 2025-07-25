import os
import warnings
from typing import List

import torch
import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA
import umap.umap_ as umap
from sklearn.exceptions import ConvergenceWarning
from pytorch_lightning import LightningModule, Trainer, Callback
from matplotlib import cm

# Filter out the specific warning
warnings.filterwarnings('ignore', message='Graph is not fully connected, spectral embedding may not work as expected.')

class UMAPCallback(Callback):
    def __init__(self, 
                 num_classes: int, 
                 class_names: List[str] | None = None,
                 save_dir="error_umap", every_n_epoch: int = 5, sample_limit: int = 1500):
        super().__init__()
        self.num_classes = num_classes
        self.class_names = class_names if class_names is not None else [str(i) for i in range(num_classes)]
        self.every_n_epoch = every_n_epoch
        self.sample_limit = sample_limit
        self.save_dir = save_dir
    
    @torch.no_grad()
    def on_validation_epoch_end(self, trainer: Trainer, pl_module: LightningModule):
        epoch = trainer.current_epoch
        if (epoch + 1) % self.every_n_epoch != 0:
            return

        pl_module.eval()

        feats, labels = [], []
        collected = 0
        val_loader = trainer.datamodule.val_dataloader()

        for x, y in val_loader:
            x = x.to(pl_module.device, non_blocking=True)

            f = pl_module.model.backbone.forward_features(x)
            if f.ndim == 4:
                f = torch.nn.functional.adaptive_avg_pool2d(f, 1).squeeze(-1).squeeze(-1)

            feats.append(f.cpu())
            labels.append(y.cpu())

            collected += f.size(0)
            if collected >= self.sample_limit:
                break

        feats  = torch.cat(feats)[: self.sample_limit]
        labels = torch.cat(labels)[: self.sample_limit]

        # one-hot 또는 soft-label → argmax
        if labels.ndim == 2:
            labels = torch.argmax(labels, dim=1)

        feats  = feats.numpy()
        labels = labels.numpy()

        # ── UMAP 계산 ─────────────────────
        n_samples, n_feats = feats.shape
        n_pca = min(50, n_samples, n_feats)
        if n_pca < 2:
            print("[UMAP] Too few samples for PCA")
            return
        
        feats_50 = PCA(n_components=n_pca, random_state=42).fit_transform(feats)
        
        # UMAP 파라미터 설정
        reducer = umap.UMAP(
            n_components=2,
            n_neighbors=15,
            min_dist=0.1,
            metric='euclidean',
            random_state=42
        )
        emb = reducer.fit_transform(feats_50)

        # ── 시각화 ─────────────────────────
        fig, ax = plt.subplots(figsize=(8, 8))
        cmap = cm.get_cmap("tab20", self.num_classes)

        for c in range(self.num_classes):
            idx = labels == c
            if idx.any():
                ax.scatter(
                    emb[idx, 0], emb[idx, 1],
                    s=10, alpha=0.8, color=cmap(c),
                    label=self.class_names[c]
                )

        ax.set_title(f"UMAP (val) @ epoch {epoch}")
        ax.axis("off")
        ax.legend(fontsize=8, loc="best", frameon=False, markerscale=2)

        # ── 저장 ───────────────────────────
        os.makedirs(self.save_dir, exist_ok=True)
        fig.savefig(os.path.join(self.save_dir,
                                 f"umap_epoch_{epoch:03d}.png"),
                    dpi=150, bbox_inches="tight")
        plt.close(fig)