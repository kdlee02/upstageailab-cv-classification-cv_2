import os
from pathlib import Path
from typing import Tuple, Dict

import albumentations as A
import hydra
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torchmetrics
from albumentations.pytorch import ToTensorV2
from PIL import Image
from omegaconf import DictConfig
import pytorch_lightning as pl
from pytorch_lightning import LightningDataModule, LightningModule, Trainer, seed_everything
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
import torch.nn.functional as F  
from torch.optim import Adam, AdamW
from torch.utils.data import DataLoader, Dataset
from torch.distributions.beta import Beta
from hydra.utils import instantiate
from sklearn.model_selection import train_test_split
from transformers import get_cosine_schedule_with_warmup
import wandb
from timm.data.mixup import Mixup

from src.models.swinTransformer import SwinTransformer
from src.models.efficientnet import EfficientNet
from src.models.convnextArcFace import ConvNeXtArcFace
from src.losses.focalloss import FocalLoss
from src.callbacks.averaging import EMA

class HardNegativeMiningTrainerModule(LightningModule):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg

        if "convnext" in cfg.model.model_name:
            self.model = ConvNeXtArcFace(cfg)
        elif "swin" in cfg.model.model_name:
            self.model = SwinTransformer(cfg)
        elif "efficientnet" in cfg.model.model_name:
            self.model = EfficientNet(cfg)
        
        self.criterion = FocalLoss(**cfg.loss.loss)
        
        #if cfg.model.arcFace:
            #self.criterion = nn.CrossEntropyLoss()

        n_classes = cfg.model.num_classes

        self.train_acc = torchmetrics.Accuracy(task="multiclass", num_classes=n_classes)
        self.val_acc = torchmetrics.Accuracy(task="multiclass", num_classes=n_classes)
        self.train_f1 = torchmetrics.F1Score(task="multiclass", num_classes=n_classes, average="macro")
        self.val_f1 = torchmetrics.F1Score(task="multiclass", num_classes=n_classes, average="macro")

        self.ema = None

        if cfg.trainer.use_mixup == True:
            self.mixup = Mixup(
                mixup_alpha=0.4,
                cutmix_alpha=0.0,
                prob=1.0,
                switch_prob= 0.0,
                mode="batch",
                label_smoothing=0.0,
                num_classes=n_classes
            )
        else:
            self.mixup = None
        
        self.register_buffer("_cls_loss_sum", torch.zeros(n_classes))
        self.register_buffer("_cls_sample_cnt", torch.zeros(n_classes))

    def forward(self, x):
        return self.model(x)
    
    def on_train_epoch_start(self):
        self._cls_loss_sum.zero_()
        self._cls_sample_cnt.zero_()

    def on_train_start(self):
        if self.cfg.trainer.use_ema == True:
            self.ema = EMA(self.model, decay=0.995)
            if hasattr(self.ema, "ema_model"):
                self.ema.ema_model.to(self.device)
                self.ema.ema_model.eval()

    def on_predict_start(self):
        if self.cfg.trainer.use_ema == True and self.ema is not None:
            self.ema.ema_model.to(self.device)

    def _shared_step(self, batch, stage: str):
        x, y = batch
        
        # Convert labels to long if they're not already
        if y.dtype != torch.long:
            y = y.long()
        
        # Handle mixup for training
        if stage == "train" and self.mixup is not None and self.current_epoch % 2 == 0:
            x, y_mix = self.mixup(x, y)
            # For mixup, we need to use the mixed targets for loss computation
            if self.cfg.model.arcFace:
                # For ArcFace, we need to use the original labels for the forward pass
                logits = self.model.forward(x, y)
                loss = self.criterion(logits, y_mix)  # But compute loss with mixed targets
            else:
                logits = self.model.forward(x)
                loss = self.criterion(logits, y_mix)
        else:
            # No mixup or not in training mode
            if self.cfg.model.arcFace:
                logits = self.model.forward(x, y)  # Use original labels for ArcFace
            else:
                logits = self.model.forward(x)
            loss = self.criterion(logits, y)
            
        # Ensure we have the right number of dimensions for metrics
        if y.dim() > 1:
            y_hard = y.argmax(1)
        else:
            y_hard = y

        # Hard negative mining (only when mixup is off)
        if stage == "train" and hasattr(self.cfg.trainer, 'hnm') and self.cfg.trainer.hnm.use_hnm:
            if self.current_epoch % 2 == 1:  # Only on odd epochs
                with torch.no_grad():
                    # Use y_hard which already handles one-hot if needed
                    l_each = F.cross_entropy(logits, y_hard, reduction="none")  # B
                    for cls in range(self.cfg.model.num_classes):
                        m = y_hard == cls
                        if m.any():
                            self._cls_loss_sum[cls] += l_each[m].sum()
                            self._cls_sample_cnt[cls] += m.sum().item()  # Ensure scalar value

        # Get the metrics for the current stage
        acc_metric = getattr(self, f"{stage}_acc")
        f1_metric = getattr(self, f"{stage}_f1")
        
        # Ensure we're not passing one-hot encoded labels to metrics
        if y_hard.dim() > 1 and y_hard.size(1) > 1:
            y_hard = y_hard.argmax(1)
        
        # Only update metrics if we have valid predictions and targets
        if logits is not None and y_hard is not None:
            acc_metric.update(logits, y_hard)
            f1_metric.update(logits, y_hard)
        
        # Only log the loss here, metrics will be logged in _log_epoch_metrics
        self.log(f"{stage}_loss", loss, prog_bar=(stage == "val"), on_step=False, on_epoch=True)
        return {"logits": logits, "targets": y, "loss": loss}

    def training_step(self, batch, batch_idx):
        return self._shared_step(batch, "train")

    def per_class_loss(self) -> Dict[int, float]:
        eps = 1e-6
        return {i: (self._cls_loss_sum[i] / (self._cls_sample_cnt[i] + eps)).item()
                for i in range(len(self._cls_loss_sum))}
    
    def on_train_batch_end(self, outputs, batch, batch_idx):
        
        if self.cfg.trainer.use_ema == True and self.ema is not None:
            if self.current_epoch >= self.cfg.trainer.ema_update_epochs:
                self.ema.update(self.model)

    def validation_step(self, batch, batch_idx):
        if self.cfg.trainer.use_ema == True and self.ema is not None:
            backup_state = {k: v.detach().clone() for k, v in self.model.state_dict().items()}
            self.model.load_state_dict(self.ema.state_dict())
            out = self._shared_step(batch, "val")
            self.model.load_state_dict(backup_state)
            return out
        
        return self._shared_step(batch, "val")

    def predict_step(self, batch, batch_idx):
        x, _ = batch
        if self.cfg.trainer.use_ema == True and self.ema is not None:
            logits = self.ema.ema_model(x)
        else:
            logits = self(x)
        return F.softmax(logits, dim=1)

    def on_train_epoch_end(self):
        if self.current_epoch == self.cfg.model.freeze_epochs:
            print(f"Epoch {self.current_epoch+1}: Start Feature Extractor unfreeze and full-model fine-tuning")
            self.model.unfreeze()

            # optimizer = self.trainer.optimizers[0]
            # for pg in optimizer.param_groups:
            #     old_lr = pg["lr"]
            #     pg["lr"] = old_lr * 0.1
            #     print(f"LR {old_lr:.6f} → {pg['lr']:.6f}")

            # scheduler = self.lr_schedulers()
            # if hasattr(scheduler, "base_lrs"):
            #     scheduler.base_lrs = [pg["lr"] for pg in optimizer.param_groups]

        # Debug logging for better understanding
        if wandb.run is not None:
            wandb.log({
                "debug/epoch": self.current_epoch,
                "debug/mixup_epoch": self.current_epoch % 2 == 0 if self.mixup is not None else False,
                "debug/train_samples": self.train_acc.total if hasattr(self.train_acc, 'total') else 0,
            })

        self._log_epoch_metrics("train")

    def on_validation_epoch_end(self):
        self._log_epoch_metrics("val")

    def _log_epoch_metrics(self, stage: str):
        acc_metric = getattr(self, f"{stage}_acc")
        f1_metric = getattr(self, f"{stage}_f1")
        
        # Only compute metrics if they've been updated
        if acc_metric.update_called and f1_metric.update_called:
            acc = acc_metric.compute()
            f1 = f1_metric.compute()
            
            self.log(f"{stage}_acc", acc, prog_bar=True)
            self.log(f"{stage}_f1", f1, prog_bar=True)
            
            if wandb.run is not None:  # Only log to wandb if it's initialized
                # Create the log dict
                log_dict = {
                    f"Accuracy/{stage}": acc,
                    f"F1score/{stage}": f1,
                    "epoch": self.current_epoch,
                }
                
                # Only log learning rate during training
                if stage == "train":
                    log_dict["LR"] = self.trainer.optimizers[0].param_groups[0]["lr"]
                    # Log whether mixup was used this epoch
                    if self.mixup is not None:
                        log_dict["mixup_used"] = self.current_epoch % 2 == 0
                
                wandb.log(log_dict)
        
        # Always reset the metrics
        acc_metric.reset()
        f1_metric.reset()

    def on_save_checkpoint(self, checkpoint):
        if self.cfg.trainer.use_ema == True and self.ema is not None:
            checkpoint['ema_state'] = self.ema.state_dict()
    
    def on_load_checkpoint(self, checkpoint):
        if checkpoint.get("ema_state") and self.cfg.trainer.use_ema == True:
            decay = float(getattr(self.cfg.trainer, "ema_decay", 0.995))
            self.ema = EMA(self.model, decay=decay)
            self.ema.ema_model.load_state_dict(checkpoint["ema_state"])
            self.ema.ema_model.eval()

    def configure_optimizers(self):
        optimizer_name = str(self.cfg.optimizer._target_)
        print(f"=========== {optimizer_name} ==============")
        if "AdamW" in optimizer_name:
            print("======== AdamW =========")
            opt = AdamW(
                self.parameters(),
                lr=self.cfg.optimizer.lr,
                weight_decay=self.cfg.optimizer.weight_decay,
            )

            scheduler = get_cosine_schedule_with_warmup(
                opt,
                num_warmup_steps=self.cfg.scheduler.warmup_steps,
                num_training_steps=self.cfg.scheduler.total_steps
            )
            return {
                "optimizer":   opt,
                "lr_scheduler": {
                    "scheduler":  scheduler,
                    "interval":   "step",   # ← 매 step마다 step()
                    "frequency":  1,
                    "name":       "cosine_warmup",
                },
            }
        else:
            opt = Adam(
                self.parameters(),
                lr=self.cfg.optimizer.lr,
                weight_decay=self.cfg.optimizer.weight_decay,
            )

        return opt