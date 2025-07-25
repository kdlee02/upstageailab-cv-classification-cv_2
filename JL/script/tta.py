from typing import List, Optional, Union, Tuple, Dict, Any

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm


def predict_with_tta(
    model: torch.nn.Module,
    dataloader: DataLoader,
    n_augmentations: int = 5,
    use_softmax: bool = True,
    device: Optional[torch.device] = None
) -> torch.Tensor:
    """
    Perform Test-Time Augmentation (TTA) on the given model and dataloader.
    
    Args:
        model: The model to use for predictions
        dataloader: DataLoader containing test data
        n_augmentations: Number of augmentations per sample
        use_softmax: Whether to apply softmax to logits
        device: Device to run the model on. If None, uses the same device as model parameters.
        
    Returns:
        torch.Tensor: Averaged predictions
    """
    model.eval()
    if device is None:
        device = next(model.parameters()).device
    
    all_preds = []
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="TTA"):
            # Handle different batch formats
            if isinstance(batch, (list, tuple)):
                x = batch[0]
            else:
                x = batch
            
            if not isinstance(x, list):
                x = [x]
                
            batch_preds = []
            for _ in range(n_augmentations):
                # Forward pass with different augmentations
                outputs = model(*[xi.to(device) for xi in x])
                if use_softmax:
                    outputs = F.softmax(outputs, dim=1)
                batch_preds.append(outputs.cpu())
                
            # Average predictions across augmentations
            avg_preds = torch.stack(batch_preds).mean(dim=0)
            all_preds.append(avg_preds)
            
    return torch.cat(all_preds, dim=0)


def apply_tta(
    model: torch.nn.Module,
    dataloader: DataLoader,
    config: Optional[Dict[str, Any]] = None
) -> torch.Tensor:
    """
    Convenience function to apply TTA with configuration dictionary.
    
    Args:
        model: The model to use for predictions
        dataloader: DataLoader containing test data
        config: Configuration dictionary with TTA parameters
            - n_augmentations: Number of augmentations per sample (default: 5)
            - use_softmax: Whether to apply softmax to logits (default: True)
            - device: Device to run the model on (default: model's device)
            
    Returns:
        torch.Tensor: Averaged predictions
    """
    if config is None:
        config = {}
        
    return predict_with_tta(
        model=model,
        dataloader=dataloader,
        n_augmentations=config.get("n_augmentations", 5),
        use_softmax=config.get("use_softmax", True),
        device=config.get("device", None)
    )
