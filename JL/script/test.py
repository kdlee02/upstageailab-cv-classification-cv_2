import os
import sys
from typing import List, Optional

sys.path.append(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)

from dotenv import load_dotenv
load_dotenv()

import hydra
import pandas as pd
import torch
from omegaconf import DictConfig
from pytorch_lightning import Trainer, seed_everything

from src.datasets.dataset import DatasetModule
from src.trainer.trainer import TrainerModule
from src.trainer.HNMTrainer import HardNegativeMiningTrainerModule

# Import TTA module
from scripts import tta


def _find_latest_ckpt() -> str | None:
    from pathlib import Path
    ckpts = list(Path(".").rglob("best-*.ckpt"))
    return str(sorted(ckpts, key=lambda p: p.stat().st_mtime)[-1]) if ckpts else None


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    seed_everything(cfg.get("seed", 42), workers=True)

    ckpt_path: str | None = cfg.get("test", {}).get("ckpt_path", None)  # hydra override 가능
    ckpt_path = ckpt_path or _find_latest_ckpt()
    if ckpt_path is None:
        raise FileNotFoundError("No checkpoint (.ckpt) found. Train the model first or specify test.ckpt_path=<path>.")

    print(f"Using checkpoint: {ckpt_path}")

    dm = DatasetModule(cfg)
    if cfg.trainer.hnm.use_hnm == True:
        model = src.trainer.HNMTrainer.HardNegativeMiningTrainerModule.load_from_checkpoint(
            ckpt_path, 
            cfg=cfg,
            strict=False  # This will ignore unexpected keys
        )
    else:
        model = src.trainer.trainer.TrainerModule.load_from_checkpoint(
            ckpt_path, 
            cfg=cfg,
            strict=False  # This will ignore unexpected keys
        )
    # Setup datamodule for testing
    dm.setup(stage="test")
    
    # Check if TTA is enabled in config
    use_tta = cfg.test.get("tta", {}).get("enabled", False)
    
    if use_tta:
        # Get TTA configuration
        tta_config = cfg.test.tta
        n_augmentations = tta_config.get("n_augmentations", 5)
        use_softmax = tta_config.get("use_softmax", True)
        
        print(f"Running TTA with {n_augmentations} augmentations")
        
        # Get test dataloader
        test_loader = dm.predict_dataloader()
        
        # Get predictions with TTA
        preds = tta.apply_tta(
            model=model,
            dataloader=test_loader,
            config={
                'n_augmentations': n_augmentations,
                'use_softmax': use_softmax
            }
        )
        
        # Convert to class predictions
        preds = torch.argmax(preds, dim=1).cpu().numpy()
        output_file = "pred_tta.csv"
        print("TTA predictions saved to pred_tta.csv")
    else:
        # Standard prediction without TTA
        print("Running standard prediction (no TTA)")
        trainer = Trainer(accelerator="auto", 
                         devices="auto", 
                         precision="bf16-mixed" if cfg.get("bf16", False) else 32)
        preds = trainer.predict(model, datamodule=dm)
        preds = torch.cat(preds, dim=0)
        preds = torch.argmax(preds, dim=1).cpu().numpy()
        output_file = "pred.csv"
        print("Standard predictions saved to pred.csv")
    
    # Save predictions
    submission = pd.read_csv(os.path.join(cfg.data.data_path, "sample_submission.csv"))
    submission["target"] = preds
    submission.to_csv(output_file, index=False)


if __name__ == "__main__":
    main()
