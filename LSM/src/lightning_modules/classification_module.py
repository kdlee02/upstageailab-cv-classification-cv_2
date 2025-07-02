import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from torchmetrics import Accuracy, F1Score
import torchvision.transforms as transforms

from src.models.resnet50 import ResNet50Model
from src.models.efficientnet import EfficientNetModel
from src.models.vit import VisionTransformerModel
from src.optimizers.adam import AdamOptimizer
from src.optimizers.adamw import AdamWOptimizer
from src.optimizers.sgd import SGDOptimizer
from src.schedulers.cosine import CosineScheduler
from src.schedulers.step import StepScheduler
from src.schedulers.exponential import ExponentialScheduler
from src.schedulers.plateau import PlateauScheduler
from src.schedulers.warmup_cosine import WarmupCosineScheduler
from src.schedulers.warmup_linear import WarmupLinearScheduler
from src.schedulers.cosine_warm_restart import CosineWarmRestartScheduler
from src.losses.cross_entropy import CrossEntropyLoss
from src.losses.focal import FocalLoss


class ClassificationModule(pl.LightningModule):
    """문서 분류를 위한 PyTorch Lightning 모듈"""
    
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.save_hyperparameters()
        
        # 모델 초기화
        self.model = self._create_model()
        
        # 손실 함수 초기화
        self.criterion = self._create_loss()
        
        # 메트릭 초기화
        self.train_acc = Accuracy(task='multiclass', num_classes=config['num_classes'])
        self.val_acc = Accuracy(task='multiclass', num_classes=config['num_classes'])
        self.train_f1 = F1Score(task='multiclass', num_classes=config['num_classes'], average='macro')
        self.val_f1 = F1Score(task='multiclass', num_classes=config['num_classes'], average='macro')
    
    def _create_model(self):
        """설정에 따라 모델 생성"""
        model_config = self.config['model']
        model_name = model_config['name']
        
        if model_name == 'resnet50':
            return ResNet50Model(
                pretrained=model_config.get('pretrained', True),
                num_classes=self.config['num_classes'],
                dropout_rate=model_config.get('dropout_rate', 0.5)
            )
        elif model_name == 'efficientnet':
            return EfficientNetModel(
                model_name=model_config.get('model_name', 'efficientnet-b0'),
                num_classes=self.config['num_classes'],
                dropout_rate=model_config.get('dropout_rate', 0.3)
            )
        elif model_name == 'vit':
            return VisionTransformerModel(
                model_name=model_config.get('model_name', 'vit_base_patch16_224'),
                num_classes=self.config['num_classes'],
                dropout_rate=model_config.get('dropout_rate', 0.1)
            )
        else:
            raise ValueError(f"지원하지 않는 모델: {model_name}")
    
    def _create_loss(self):
        """설정에 따라 손실 함수 생성"""
        loss_config = self.config['loss']
        loss_name = loss_config['name']
        
        if loss_name == 'cross_entropy':
            return CrossEntropyLoss(
                label_smoothing=loss_config.get('label_smoothing', 0.0)
            )
        elif loss_name == 'focal':
            return FocalLoss(
                alpha=loss_config.get('alpha', 1.0),
                gamma=loss_config.get('gamma', 2.0)
            )
        else:
            raise ValueError(f"지원하지 않는 손실 함수: {loss_name}")
    
    def forward(self, x):
        return self.model(x)
    
    def training_step(self, batch, batch_idx):
        images, targets = batch
        outputs = self(images)
        loss = self.criterion(outputs, targets)
        
        # 메트릭 계산
        self.train_acc(outputs, targets)
        self.train_f1(outputs, targets)
        
        # 로깅
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log('train_acc', self.train_acc, on_step=False, on_epoch=True, prog_bar=True)
        self.log('train_f1', self.train_f1, on_step=False, on_epoch=True, prog_bar=True)
        
        return loss
    
    def validation_step(self, batch, batch_idx):
        images, targets = batch
        outputs = self(images)
        loss = self.criterion(outputs, targets)
        
        # 메트릭 계산
        self.val_acc(outputs, targets)
        self.val_f1(outputs, targets)
        
        # 로깅
        self.log('val_loss', loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log('val_acc', self.val_acc, on_step=False, on_epoch=True, prog_bar=True)
        self.log('val_f1', self.val_f1, on_step=False, on_epoch=True, prog_bar=True)
        
        return loss
    
    def configure_optimizers(self):
        """설정에 따라 옵티마이저와 스캐줄러 생성"""
        optimizer_config = self.config['optimizer']
        optimizer_name = optimizer_config['name']
        
        if optimizer_name == 'adam':
            optimizer = AdamOptimizer(
                lr=optimizer_config.get('lr', 0.001),
                weight_decay=optimizer_config.get('weight_decay', 0.0001)
            )
        elif optimizer_name == 'adamw':
            optimizer = AdamWOptimizer(
                lr=optimizer_config.get('lr', 0.001),
                weight_decay=optimizer_config.get('weight_decay', 0.01)
            )
        elif optimizer_name == 'sgd':
            optimizer = SGDOptimizer(
                lr=optimizer_config.get('lr', 0.01),
                momentum=optimizer_config.get('momentum', 0.9),
                weight_decay=optimizer_config.get('weight_decay', 0.0001)
            )
        else:
            raise ValueError(f"지원하지 않는 옵티마이저: {optimizer_name}")
        
        optimizer = optimizer(self.parameters())
        
        # 스캐줄러 설정
        scheduler_config = self.config.get('scheduler', {})
        scheduler_name = scheduler_config.get('name', 'cosine')
        
        if scheduler_name == 'cosine':
            params = scheduler_config.get('params', {})
            scheduler = CosineScheduler(
                T_max=params.get('T_max', scheduler_config.get('T_max', 100)),
                eta_min=params.get('eta_min', scheduler_config.get('eta_min', 0.0))
            )
        elif scheduler_name == 'step':
            params = scheduler_config.get('params', {})
            scheduler = StepScheduler(
                step_size=params.get('step_size', scheduler_config.get('step_size', 30)),
                gamma=params.get('gamma', scheduler_config.get('gamma', 0.1))
            )
        elif scheduler_name == 'exponential':
            params = scheduler_config.get('params', {})
            scheduler = ExponentialScheduler(
                gamma=params.get('gamma', scheduler_config.get('gamma', 0.95))
            )
        elif scheduler_name == 'plateau':
            params = scheduler_config.get('params', {})
            scheduler = PlateauScheduler(
                mode=params.get('mode', scheduler_config.get('mode', 'min')),
                factor=params.get('factor', scheduler_config.get('factor', 0.1)),
                patience=params.get('patience', scheduler_config.get('patience', 10)),
                verbose=params.get('verbose', scheduler_config.get('verbose', False)),
                threshold=params.get('threshold', scheduler_config.get('threshold', 1e-4)),
                threshold_mode=params.get('threshold_mode', scheduler_config.get('threshold_mode', 'rel')),
                cooldown=params.get('cooldown', scheduler_config.get('cooldown', 0)),
                min_lr=params.get('min_lr', scheduler_config.get('min_lr', 0)),
                eps=params.get('eps', scheduler_config.get('eps', 1e-8))
            )
        elif scheduler_name == 'warmup_cosine':
            params = scheduler_config.get('params', {})
            scheduler = WarmupCosineScheduler(
                warmup_steps=params.get('warmup_steps', scheduler_config.get('warmup_steps', 1000)),
                max_steps=params.get('max_steps', scheduler_config.get('max_steps', 10000)),
                min_lr=params.get('min_lr', scheduler_config.get('min_lr', 0.0)),
                warmup_start_lr=params.get('warmup_start_lr', scheduler_config.get('warmup_start_lr', 0.0))
            )
        elif scheduler_name == 'warmup_linear':
            params = scheduler_config.get('params', {})
            scheduler = WarmupLinearScheduler(
                warmup_steps=params.get('warmup_steps', scheduler_config.get('warmup_steps', 1000)),
                max_steps=params.get('max_steps', scheduler_config.get('max_steps', 10000)),
                min_lr=params.get('min_lr', scheduler_config.get('min_lr', 0.0)),
                warmup_start_lr=params.get('warmup_start_lr', scheduler_config.get('warmup_start_lr', 0.0))
            )
        elif scheduler_name == 'cosine_warm_restart':
            # params 섹션에서 파라미터를 가져오거나 직접 설정에서 가져옴
            params = scheduler_config.get('params', {})
            scheduler = CosineWarmRestartScheduler(
                T_0=params.get('T_0', scheduler_config.get('T_0', 10)),
                T_mult=params.get('T_mult', scheduler_config.get('T_mult', 2)),
                eta_min=params.get('eta_min', scheduler_config.get('eta_min', 0.0))
            )
        else:
            raise ValueError(f"지원하지 않는 스캐줄러: {scheduler_name}")
        
        scheduler = scheduler(optimizer)
        
        # 스캐줄러 설정에 따라 반환
        if scheduler_name == 'plateau':
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "monitor": "val_loss",
                    "interval": "epoch",
                    "frequency": 1
                }
            }
        else:
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "interval": "step",
                    "frequency": 1
                }
            }