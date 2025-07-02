import hydra
from omegaconf import OmegaConf

@hydra.main(config_path="configs", config_name="config", version_base=None)
def check_config(cfg):
    print("=== 설정 확인 ===")
    print(f"training.max_epochs: {cfg.training.max_epochs}")
    print(f"loss.label_smoothing: {cfg.loss.label_smoothing}")
    print(f"optimizer.betas: {cfg.optimizer.betas if hasattr(cfg.optimizer, 'betas') else 'None'}")

if __name__ == '__main__':
    check_config() 