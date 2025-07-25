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


class ClassBalancedLoss(nn.Module):
    
    def __init__(self, samples_per_class, num_classes, beta=0.9999, gamma=2.0, loss_type="focal"):
        super(ClassBalancedLoss, self).__init__()
        self.samples_per_class = samples_per_class
        self.num_classes = num_classes
        self.beta = beta
        self.gamma = gamma
        self.loss_type = loss_type
        
        effective_num = 1.0 - np.power(beta, samples_per_class)
        weights = (1.0 - beta) / np.array(effective_num)
        weights = weights / np.sum(weights) * num_classes
        
        self.register_buffer('weights', torch.tensor(weights, dtype=torch.float32))
    
    def forward(self, logits, labels):
        if labels.dim() > 1 and labels.size(1) > 1:
            labels = labels.argmax(1)
            
        cb_loss = F.cross_entropy(logits, labels, weight=self.weights, reduction='none')
        
        if self.loss_type == "focal":
            # Apply focal loss weighting
            pt = torch.exp(-cb_loss)
            focal_weight = (1 - pt) ** self.gamma
            cb_loss = focal_weight * cb_loss
        
        return cb_loss.mean()


class OHEMCBLossTrainerModule(LightningModule):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg

        if "convnext" in cfg.model.model_name:
            self.model = ConvNeXtArcFace(cfg)
        elif "swin" in cfg.model.model_name:
            self.model = SwinTransformer(cfg)
        elif "efficientnet" in cfg.model.model_name:
            self.model = EfficientNet(cfg)
        
        if hasattr(cfg.loss, 'use_cb_loss') and cfg.loss.use_cb_loss:
            samples_per_class = cfg.loss.samples_per_class
            self.cb_criterion = ClassBalancedLoss(
                samples_per_class=samples_per_class,
                num_classes=cfg.model.num_classes,
                beta=cfg.loss.get('cb_beta', 0.9999),
                gamma=cfg.loss.get('cb_gamma', 2.0),
                loss_type=cfg.loss.get('cb_loss_type', 'focal')
            )
        else:
            self.cb_criterion = None
            
        self.criterion = FocalLoss(**cfg.loss.loss)

        n_classes = cfg.model.num_classes

        self.train_acc = torchmetrics.Accuracy(task="multiclass", num_classes=n_classes)
        self.val_acc = torchmetrics.Accuracy(task="multiclass", num_classes=n_classes)
        self.train_f1 = torchmetrics.F1Score(task="multiclass", num_classes=n_classes, average="macro")
        self.val_f1 = torchmetrics.F1Score(task="multiclass", num_classes=n_classes, average="macro")

        self.ema = None

        # OHEM configuration
        self.ohem_enabled = hasattr(cfg.trainer, 'ohem') and cfg.trainer.ohem.use_ohem
        if self.ohem_enabled:
            self.ohem_ratio = cfg.trainer.ohem.get('ratio', 0.7)  # Keep top 70% hardest examples
            self.ohem_min_samples = cfg.trainer.ohem.get('min_samples', 16)  # Minimum samples to keep
            
        # Mixup configuration
        if cfg.trainer.use_mixup == True:
            self.mixup = Mixup(
                mixup_alpha=0.4,
                cutmix_alpha=0.0,
                prob=1.0,
                switch_prob=0.0,
                mode="batch",
                label_smoothing=0.0,
                num_classes=n_classes
            )
        else:
            self.mixup = None

        # Statistics tracking
        self.register_buffer("_cls_loss_sum", torch.zeros(n_classes))
        self.register_buffer("_cls_sample_cnt", torch.zeros(n_classes))
        self.register_buffer("_ohem_selected_count", torch.zeros(1))
        self.register_buffer("_total_samples_count", torch.zeros(1))

    def forward(self, x):
        return self.model(x)
    
    def on_train_epoch_start(self):
        self._cls_loss_sum.zero_()
        self._cls_sample_cnt.zero_()
        self._ohem_selected_count.zero_()
        self._total_samples_count.zero_()

    def on_train_start(self):
        if self.cfg.trainer.use_ema == True:
            self.ema = EMA(self.model, decay=0.995)
            if hasattr(self.ema, "ema_model"):
                self.ema.ema_model.to(self.device)
                self.ema.ema_model.eval()

    def on_predict_start(self):
        if self.cfg.trainer.use_ema == True and self.ema is not None:
            self.ema.ema_model.to(self.device)

    def apply_ohem(self, logits, labels, loss_per_sample):
        """
        Apply Online Hard Example Mining
        Args:
            logits: model predictions [N, C]
            labels: ground truth labels [N]
            loss_per_sample: loss for each sample [N]
        Returns:
            selected_loss: loss after OHEM selection
            selected_indices: indices of selected hard examples
        """
        batch_size = logits.size(0)
        
        # Calculate number of samples to keep
        num_keep = max(int(batch_size * self.ohem_ratio), self.ohem_min_samples)
        num_keep = min(num_keep, batch_size)
        
        # Select top-k hardest examples
        _, hard_indices = torch.topk(loss_per_sample, num_keep, largest=True)
        
        # Update statistics
        self._ohem_selected_count += num_keep
        self._total_samples_count += batch_size
        
        return loss_per_sample[hard_indices].mean(), hard_indices

    def _shared_step(self, batch, stage: str):
        x, y = batch
        
        # Convert labels to long if they're not already
        if y.dtype != torch.long:
            y = y.long()
        
        # Handle mixup for training (disable OHEM during mixup)
        use_mixup = (stage == "train" and self.mixup is not None and 
                    self.current_epoch % 2 == 0)
        
        if use_mixup:
            x, y_mix = self.mixup(x, y)
            if self.cfg.model.arcFace:
                logits = self.model.forward(x, y)
                # Use CB loss if available, otherwise use standard criterion
                if self.cb_criterion is not None:
                    loss = self.cb_criterion(logits, y_mix)
                else:
                    loss = self.criterion(logits, y_mix)
            else:
                logits = self.model.forward(x)
                if self.cb_criterion is not None:
                    loss = self.cb_criterion(logits, y_mix)
                else:
                    loss = self.criterion(logits, y_mix)
        else:
            # No mixup - can apply OHEM
            if self.cfg.model.arcFace:
                logits = self.model.forward(x, y)
            else:
                logits = self.model.forward(x)
            
            # Calculate loss per sample for OHEM
            if stage == "train" and self.ohem_enabled:
                # Get per-sample losses
                if self.cb_criterion is not None:
                    # For CB loss, we need to compute per-sample losses
                    y_for_loss = y.argmax(1) if y.dim() > 1 and y.size(1) > 1 else y
                    loss_per_sample = F.cross_entropy(logits, y_for_loss, 
                                                    weight=self.cb_criterion.weights, 
                                                    reduction='none')
                    if self.cb_criterion.loss_type == "focal":
                        pt = torch.exp(-loss_per_sample)
                        focal_weight = (1 - pt) ** self.cb_criterion.gamma
                        loss_per_sample = focal_weight * loss_per_sample
                else:
                    y_for_loss = y.argmax(1) if y.dim() > 1 and y.size(1) > 1 else y
                    loss_per_sample = F.cross_entropy(logits, y_for_loss, reduction='none')
                
                # Apply OHEM
                loss, selected_indices = self.apply_ohem(logits, y, loss_per_sample)
            else:
                # Standard loss computation
                if self.cb_criterion is not None:
                    loss = self.cb_criterion(logits, y)
                else:
                    loss = self.criterion(logits, y)
            
        # Ensure we have the right number of dimensions for metrics
        if y.dim() > 1:
            y_hard = y.argmax(1)
        else:
            y_hard = y

        # Statistics tracking for hard negative mining
        if stage == "train":
            with torch.no_grad():
                y_for_stats = y_hard if y_hard.dim() == 1 else y_hard.argmax(1)
                l_each = F.cross_entropy(logits, y_for_stats, reduction="none")
                
                for cls in range(self.cfg.model.num_classes):
                    m = y_for_stats == cls
                    if m.any():
                        self._cls_loss_sum[cls] += l_each[m].sum()
                        self._cls_sample_cnt[cls] += m.sum().item()

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
        
        # Log the loss
        self.log(f"{stage}_loss", loss, prog_bar=(stage == "val"), on_step=False, on_epoch=True)
        
        # Log OHEM statistics
        if stage == "train" and self.ohem_enabled:
            ohem_ratio = (self._ohem_selected_count / (self._total_samples_count + 1e-8)).item()
            self.log("train_ohem_ratio", ohem_ratio, on_step=False, on_epoch=True)
        
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

        # Enhanced debug logging
        if wandb.run is not None:
            log_dict = {
                "debug/epoch": self.current_epoch,
                "debug/mixup_epoch": self.current_epoch % 2 == 0 if self.mixup is not None else False,
                "debug/train_samples": self.train_acc.total if hasattr(self.train_acc, 'total') else 0,
            }
            
            # Add OHEM statistics
            if self.ohem_enabled:
                ohem_ratio = (self._ohem_selected_count / (self._total_samples_count + 1e-8)).item()
                log_dict.update({
                    "debug/ohem_selection_ratio": ohem_ratio,
                    "debug/ohem_selected_samples": self._ohem_selected_count.item(),
                    "debug/total_samples": self._total_samples_count.item(),
                })
            
            # Add per-class loss statistics
            per_class_losses = self.per_class_loss()
            for cls, loss_val in per_class_losses.items():
                log_dict[f"debug/class_{cls}_loss"] = loss_val
            
            wandb.log(log_dict)

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
            
            if wandb.run is not None:
                log_dict = {
                    f"Accuracy/{stage}": acc,
                    f"F1score/{stage}": f1,
                    "epoch": self.current_epoch,
                }
                
                if stage == "train":
                    log_dict["LR"] = self.trainer.optimizers[0].param_groups[0]["lr"]
                    if self.mixup is not None:
                        log_dict["mixup_used"] = self.current_epoch % 2 == 0
                    
                    # Add OHEM and CB loss indicators
                    log_dict["ohem_enabled"] = self.ohem_enabled
                    log_dict["cb_loss_enabled"] = self.cb_criterion is not None
                
                wandb.log(log_dict)
        
        # Always reset the metrics
        acc_metric.reset()
        f1_metric.reset()

    def on_save_checkpoint(self, checkpoint):
        if self.cfg.trainer.use_ema == True and self.ema is not None:
            checkpoint['ema_state'] = self.ema.state_dict()
        
        # Save OHEM and CB loss configurations
        checkpoint['ohem_enabled'] = self.ohem_enabled
        checkpoint['cb_loss_enabled'] = self.cb_criterion is not None
    
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
                "optimizer": opt,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "interval": "step",
                    "frequency": 1,
                    "name": "cosine_warmup",
                },
            }
        else:
            opt = Adam(
                self.parameters(),
                lr=self.cfg.optimizer.lr,
                weight_decay=self.cfg.optimizer.weight_decay,
            )

        return opt