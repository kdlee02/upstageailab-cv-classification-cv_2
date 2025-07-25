import copy

import torch

class SWA:
    def __init__(self, model):
        self.swa_model = copy.deepcopy(model)
        self.swa_model.eval()
        self.n_averaged = 0
        self._device = next(model.parameters()).device
        for param in self.swa_model.parameters():
            param.requires_grad = False
    
    def update(self, model):
        # Ensure both models are on the same device
        model = model.to(self._device)
        self.swa_model = self.swa_model.to(self._device)
        
        with torch.no_grad():
            self.n_averaged += 1
            # Use numerically stable update formula: swa_param = swa_param + (param - swa_param) / n_averaged
            for swa_param, param in zip(self.swa_model.parameters(), model.parameters()):
                # Move parameters to the same device if needed
                param = param.to(self._device)
                # Calculate the update
                update = (param - swa_param) / self.n_averaged
                # Check for NaN/Inf values before applying update
                if not torch.isfinite(update).all():
                    print(f"Warning: NaN/Inf detected in SWA update. Skipping this update.")
                    continue
                # Apply the update
                swa_param.add_(update)
            
            # Update buffers (like running mean/variance in BatchNorm)
            for swa_buf, buf in zip(self.swa_model.buffers(), model.buffers()):
                buf = buf.to(self._device)
                if not torch.isfinite(buf).all():
                    print("Warning: NaN/Inf detected in buffer. Skipping buffer update.")
                    continue
                swa_buf.copy_(buf)

    def state_dict(self):
        return {
            'model_state_dict': self.swa_model.state_dict(),
            'n_averaged': self.n_averaged
        }

class EMA:
    def __init__(self, model, decay=0.999):
        self.ema_model = copy.deepcopy(model)
        self.ema_model.eval()
        self.decay = decay
        for param in self.ema_model.parameters():
            param.requires_grad = False
    
    def update(self, model):
        with torch.no_grad():
            for ema_param, param in zip(self.ema_model.parameters(), model.parameters()):
                ema_param.copy_(self.decay * ema_param + (1. - self.decay) * param)
            for ema_buf, buf in zip(self.ema_model.buffers(), model.buffers()):
                ema_buf.copy_(buf)

    def state_dict(self):
        return self.ema_model.state_dict()
        