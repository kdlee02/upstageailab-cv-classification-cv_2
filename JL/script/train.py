import os
import sys

sys.path.append(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)

from dotenv import load_dotenv
load_dotenv()

from omegaconf import DictConfig, OmegaConf
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor, EarlyStopping
import hydra
import wandb
import torch
import numpy as np

from src.datasets.dataset import DatasetModule
from src.trainer.HNMTrainer import HardNegativeMiningTrainerModule
from src.trainer.ohemcb import OHEMCBLossTrainerModule
from src.callbacks.hardNegativeMining import HNMCallback
from src.callbacks.evaluation import ErrorAnalysisCallback
from src.callbacks.umap import UMAPCallback
from src.callbacks.confusionMatrix import ConfusionMatrixCallback
from src.utils.utils import auto_increment_run_suffix, get_latest_run, project_path, make_error_run_dir

@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig):
    api_key = os.environ["WANDB_API_KEY"]
    wandb.login(key=api_key)

    project_name = ""
    try:
        run_name = get_latest_run(project_name, cfg.experiment_name)
    except Exception as e:
        print(f"[W&B WARNING] Failed to get previous runs: {e}")
        run_name = f"{cfg.experiment_name.replace('_', '-')}-000"
    next_run_name = auto_increment_run_suffix(run_name)
    wandb.init(
        project=project_name,
        id=next_run_name,
        notes="content-based classification model",
        tags=["content-based", "classification"],
        config={
            "experiment_name": cfg.experiment_name,
            "model_name": cfg.model.model_name,
            "freeze_epochs": cfg.model.freeze_epochs,
            "batch_size": cfg.data.batch_size
        }
    )

    seed_everything(cfg.seed if "seed" in cfg else 42, workers=True)

    data_module = DatasetModule(cfg)
    data_module.setup('train')

    samples = data_module.train_dataloader().dataset.samples
    labels = [label for _, label in samples]
    cls_counts = np.bincount(labels)
    total_cnt  = cls_counts.sum()
    alpha_np   = total_cnt / (len(cls_counts) * cls_counts)
    alpha_np   = alpha_np / alpha_np.sum()

    # ---------- 3. cfg 수정 (primitive 타입만!) ----------
    OmegaConf.set_struct(cfg, False)   # 구조 잠금 해제
    cfg.loss.loss.alpha = alpha_np.tolist()   # ← 리스트(float) OK
    
    # Add CB Loss samples_per_class if using CB Loss
    if hasattr(cfg.loss, 'use_cb_loss') and cfg.loss.use_cb_loss:
        cfg.loss.samples_per_class = cls_counts.tolist()  # Automatically set from dataset
    
    cfg.scheduler.total_steps = len(data_module.train_dataloader()) * cfg.trainer.max_epochs
    cfg.scheduler.warmup_steps = len(data_module.train_dataloader()) * 3
    
    # Model selection logic - now with OHEM+CB Loss support
    use_hnm = cfg.trainer.hnm.use_hnm if hasattr(cfg.trainer, 'hnm') else False
    use_ohem = cfg.trainer.ohem.use_ohem if hasattr(cfg.trainer, 'ohem') else False
    use_cb_loss = cfg.loss.use_cb_loss if hasattr(cfg.loss, 'use_cb_loss') else False
    
    print(f"Training Configuration:")
    print(f"  - HNM (Hard Negative Mining): {use_hnm}")
    print(f"  - OHEM (Online Hard Example Mining): {use_ohem}")
    print(f"  - CB Loss (Class-balanced Loss): {use_cb_loss}")
    
    if use_ohem or use_cb_loss:
        print("Using OHEM+CB Loss Trainer Module")
        model = OHEMCBLossTrainerModule(cfg)
    elif use_hnm:
        print("Using Hard Negative Mining Trainer Module")
        model = HardNegativeMiningTrainerModule(cfg)
    else:
        print("Using Standard Trainer Module")
        model = TrainerModule(cfg)

    ckpt_cb = ModelCheckpoint(
        monitor="val_f1",
        mode="max",
        save_top_k = 1,
        filename="best-{epoch:02d}-{val_f1:.3f}",
    )

    lr_monitor = LearningRateMonitor(logging_interval="step")

    early_stop_cb = EarlyStopping(monitor=cfg.callback.monitor, mode=cfg.callback.mode, patience=cfg.callback.patience, min_delta=0.0005, verbose=True)
    
    error_root_dir = make_error_run_dir()

    error_cb = ErrorAnalysisCallback(num_classes=cfg.model.num_classes, 
                                     class_names=data_module.meta_df["class_name"].unique(), 
                                     save_dir=os.path.join(project_path(), error_root_dir, cfg.trainer.error.analysis.save_dir), 
                                     top_k=cfg.trainer.error.analysis.top_k)

    umap_cb = UMAPCallback(num_classes=cfg.model.num_classes, 
                           class_names=data_module.meta_df["class_name"].unique(), 
                           save_dir=os.path.join(project_path(), error_root_dir, cfg.trainer.error.umap.save_dir), 
                           every_n_epoch=cfg.trainer.error.umap.every_n_epoch)

    cm_cb = ConfusionMatrixCallback(num_classes=cfg.model.num_classes, 
                                    class_names=data_module.meta_df["class_name"].unique(), 
                                    save_dir=os.path.join(project_path(), error_root_dir, cfg.trainer.error.confusion_matrix.save_dir))

    # Callback setup - HNM callback works with any trainer module
    callbacks = [ckpt_cb, lr_monitor, early_stop_cb, error_cb, umap_cb, cm_cb]
    
    if use_hnm:
        hnm_cb = HNMCallback(data_module.train_df, train_idx=data_module.train_idx, cfg=cfg)
        callbacks.append(hnm_cb)
    
    trainer = Trainer(
        max_epochs=cfg.trainer.max_epochs,
        accelerator="auto",
        devices="auto",
        precision="bf16-mixed" if cfg.get("bf16", False) else 32,
        callbacks=callbacks,
        log_every_n_steps=1,
        reload_dataloaders_every_n_epochs=1,
    )

    trainer.fit(model, datamodule=data_module)

    wandb.finish()

if __name__ == "__main__":
    main()