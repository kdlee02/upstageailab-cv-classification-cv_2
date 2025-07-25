import os
import sys
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from PIL import Image
import pandas as pd
from tqdm import tqdm
import hydra
from omegaconf import DictConfig
import cv2
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.datasets.dataset import DatasetModule
from src.trainer.trainer import TrainerModule
from src.trainer.HNMTrainer import HardNegativeMiningTrainerModule

class ModelWrapper(torch.nn.Module):
    """Wrapper class to make the model compatible with Grad-CAM"""
    def __init__(self, model):
        super().__init__()
        self.model = model
        
        # Get the target layer for Grad-CAM
        if hasattr(model.model, 'backbone'):
            backbone = model.model.backbone
            # Try to find the last convolutional layer
            if hasattr(backbone, 'stages'):  # For ConvNeXt
                self.target_layer = backbone.stages[-1].blocks[-1]
            elif hasattr(backbone, 'layers'):  # For ResNet
                self.target_layer = backbone.layers[-1][-1].conv3
            elif hasattr(backbone, 'blocks'):  # For ViT
                self.target_layer = backbone.blocks[-1].norm1
            else:
                # Fallback to the last layer of the backbone
                self.target_layer = list(backbone.children())[-1]
        else:
            self.target_layer = list(model.model.children())[-1]

    def forward(self, x):
        return self.model(x)

def get_misclassified_samples(model, dataloader, device):
    """Get misclassified samples from the validation set"""
    model.eval()
    misclassified = []
    
    with torch.no_grad():
        for batch_idx, (images, targets) in enumerate(tqdm(dataloader, desc="Finding misclassified samples")):
            images, targets = images.to(device), targets.to(device)
            outputs = model(images)
            _, preds = torch.max(outputs, 1)
            
            # Get indices of misclassified samples
            misclassified_idx = (preds != targets).nonzero(as_tuple=True)[0]
            
            for idx in misclassified_idx:
                misclassified.append({
                    'image': images[idx].cpu(),
                    'pred': preds[idx].item(),
                    'target': targets[idx].item(),
                    'batch_idx': batch_idx,
                    'sample_idx': idx.item()
                })
    
    return misclassified

def save_gradcam(misclassified_samples, model_wrapper, dataloader, output_dir, num_samples=5):
    """Generate and save Grad-CAM visualizations for misclassified samples"""
    # Create output directories for each class
    os.makedirs(output_dir, exist_ok=True)
    
    # Initialize Grad-CAM
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    cam = GradCAM(model=model_wrapper, target_layers=[model_wrapper.target_layer])
    cam.device = device
    
    # Process each misclassified sample
    for i, sample in enumerate(tqdm(misclassified_samples[:num_samples], desc="Generating Grad-CAM visualizations")):
        input_tensor = sample['image'].unsqueeze(0)
        target_class = sample['target']
        pred_class = sample['pred']
        
        # Generate Grad-CAM
        targets = [ClassifierOutputTarget(target_class)]
        grayscale_cam = cam(input_tensor=input_tensor, targets=targets)
        grayscale_cam = grayscale_cam[0, :]
        
        # Convert input tensor to image
        img = input_tensor.squeeze(0).permute(1, 2, 0).cpu().numpy()
        img = (img - img.min()) / (img.max() - img.min())  # Normalize to [0, 1]
        
        # Apply colormap to the CAM
        cam_image = show_cam_on_image(img, grayscale_cam, use_rgb=True)
        
        # Create output directory for target class
        class_dir = os.path.join(output_dir, f"class_{target_class}")
        os.makedirs(class_dir, exist_ok=True)
        
        # Save the image with Grad-CAM
        output_path = os.path.join(class_dir, f"misclassified_{i}_pred_{pred_class}_target_{target_class}.png")
        plt.figure(figsize=(10, 5))
        
        # Plot original image
        plt.subplot(1, 2, 1)
        plt.imshow(img)
        plt.title(f"Original (True: {target_class})")
        plt.axis('off')
        
        # Plot Grad-CAM
        plt.subplot(1, 2, 2)
        plt.imshow(cam_image)
        plt.title(f"Grad-CAM (Pred: {pred_class})")
        plt.axis('off')
        
        plt.tight_layout()
        plt.savefig(output_path, bbox_inches='tight', pad_inches=0.1)
        plt.close()

def find_latest_ckpt():
    """Find the latest checkpoint in the current directory"""
    from pathlib import Path
    ckpts = list(Path(".").rglob("best-*.ckpt"))
    return str(sorted(ckpts, key=lambda p: p.stat().st_mtime)[-1]) if ckpts else None

@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load model
    ckpt_path = cfg.get("test", {}).get("ckpt_path", None) or find_latest_ckpt()
    if ckpt_path is None:
        raise FileNotFoundError("No checkpoint (.ckpt) found. Train the model first or specify test.ckpt_path=<path>.")
    
    print(f"Using checkpoint: {ckpt_path}")
    
    # Initialize model
    if cfg.trainer.hnm.use_hnm:
        model = HardNegativeMiningTrainerModule.load_from_checkpoint(
            ckpt_path, 
            cfg=cfg,
            strict=False
        )
    else:
        model = TrainerModule.load_from_checkpoint(
            ckpt_path, 
            cfg=cfg,
            strict=False
        )
    
    model = model.to(device)
    model_wrapper = ModelWrapper(model)
    
    # Setup dataloader
    dm = DatasetModule(cfg)
    dm.setup(stage="train")  # Setup both train and validation datasets
    val_loader = dm.val_dataloader()
    
    # Get misclassified samples
    print("Finding misclassified samples...")
    misclassified = get_misclassified_samples(model, val_loader, device)
    
    if not misclassified:
        print("No misclassified samples found!")
        return
    
    print(f"Found {len(misclassified)} misclassified samples")
    
    # Create output directory
    output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "gradcam_results")
    
    # Generate and save Grad-CAM visualizations
    save_gradcam(misclassified, model_wrapper, val_loader, output_dir, num_samples=min(20, len(misclassified)))
    print(f"Grad-CAM visualizations saved to: {output_dir}")

if __name__ == "__main__":
    main()
