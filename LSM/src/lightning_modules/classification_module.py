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
        """설정에 따라 옵티마이저 생성"""
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
        
        return optimizer(self.parameters())