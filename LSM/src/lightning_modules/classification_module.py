import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from torchmetrics import Accuracy, F1Score, ConfusionMatrix
import torchvision.transforms as transforms
import numpy as np
from collections import defaultdict

from src.models.resnet50 import ResNet50Model
from src.models.efficientnet import EfficientNetModel
from src.models.vit import VisionTransformerModel
from src.models.convnext import create_convnext_model
from src.optimizers.adam import AdamOptimizer
from src.optimizers.adamw import AdamWOptimizer
from src.optimizers.sgd import SGDOptimizer
from src.optimizers.rmsprop import RMSpropOptimizer
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
        
        # 클래스별 메트릭 초기화
        self.train_class_acc = Accuracy(task='multiclass', num_classes=config['num_classes'], average='none')
        self.val_class_acc = Accuracy(task='multiclass', num_classes=config['num_classes'], average='none')
        self.train_class_f1 = F1Score(task='multiclass', num_classes=config['num_classes'], average='none')
        self.val_class_f1 = F1Score(task='multiclass', num_classes=config['num_classes'], average='none')
        
        # 혼동 행렬 초기화
        self.train_confusion = ConfusionMatrix(task='multiclass', num_classes=config['num_classes'])
        self.val_confusion = ConfusionMatrix(task='multiclass', num_classes=config['num_classes'])
        
        # 클래스별 손실을 저장할 딕셔너리
        self.class_losses = defaultdict(list)
        
        # 클래스명 설정 (기본값)
        self.class_names = [f"class_{i}" for i in range(config['num_classes'])]
    
    def set_class_names(self, class_names):
        """클래스명 설정"""
        self.class_names = class_names
    
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
        elif model_name == 'convnext':
            return create_convnext_model(
                model_name=model_config.get('model_name', 'convnext_tiny'),
                num_classes=self.config['num_classes'],
                dropout_rate=model_config.get('dropout_rate', 0.1),
                use_attention=model_config.get('use_attention', False),
                use_focal_loss=model_config.get('use_focal_loss', False),
                pretrained=model_config.get('pretrained', True),
                alpha=model_config.get('alpha', 1.0),
                gamma=model_config.get('gamma', 2.0),
                attention_dim=model_config.get('attention_dim', 256)
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
    
    def _calculate_class_losses(self, outputs, targets):
        """클래스별 손실 계산"""
        class_losses = {}
        for class_idx in range(self.config['num_classes']):
            class_mask = (targets == class_idx)
            if class_mask.sum() > 0:
                class_outputs = outputs[class_mask]
                class_targets = targets[class_mask]
                class_loss = self.criterion(class_outputs, class_targets)
                class_losses[class_idx] = class_loss.item()
        return class_losses
    
    def training_step(self, batch, batch_idx):
        images, targets = batch
        # 라벨 스무딩이 적용된 경우 클래스 인덱스로 변환
        if targets.dim() > 1:
            targets = targets.argmax(dim=1)
        targets = targets.long()
        outputs = self(images)
        loss = self.criterion(outputs, targets)
        
        # 메트릭 계산
        self.train_acc(outputs, targets)
        self.train_f1(outputs, targets)
        self.train_class_acc(outputs, targets)
        self.train_class_f1(outputs, targets)
        self.train_confusion(outputs, targets)
        
        # 클래스별 손실 계산
        class_losses = self._calculate_class_losses(outputs, targets)
        for class_idx, class_loss in class_losses.items():
            self.class_losses[f'train_class_{class_idx}_loss'].append(class_loss)
        
        # 로깅
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log('train_acc', self.train_acc, on_step=False, on_epoch=True, prog_bar=True)
        self.log('train_f1', self.train_f1, on_step=False, on_epoch=True, prog_bar=True)
        
        # 클래스별 메트릭 로깅
        for i in range(self.config['num_classes']):
            class_name = self.class_names[i] if i < len(self.class_names) else f'class_{i}'
            self.log(f'train_{class_name}_acc', self.train_class_acc.compute()[i], on_step=False, on_epoch=True)
            self.log(f'train_{class_name}_f1', self.train_class_f1.compute()[i], on_step=False, on_epoch=True)
        
        return loss
    
    def validation_step(self, batch, batch_idx):
        images, targets = batch
        # 라벨 스무딩이 적용된 경우 클래스 인덱스로 변환
        if targets.dim() > 1:
            targets = targets.argmax(dim=1)
        targets = targets.long()
        outputs = self(images)
        loss = self.criterion(outputs, targets)
        
        # 메트릭 계산
        self.val_acc(outputs, targets)
        self.val_f1(outputs, targets)
        self.val_class_acc(outputs, targets)
        self.val_class_f1(outputs, targets)
        self.val_confusion(outputs, targets)
        
        # 클래스별 손실 계산
        class_losses = self._calculate_class_losses(outputs, targets)
        for class_idx, class_loss in class_losses.items():
            self.class_losses[f'val_class_{class_idx}_loss'].append(class_loss)
        
        # 로깅
        self.log('val_loss', loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log('val_acc', self.val_acc, on_step=False, on_epoch=True, prog_bar=True)
        self.log('val_f1', self.val_f1, on_step=False, on_epoch=True, prog_bar=True)
        
        # 클래스별 메트릭 로깅
        for i in range(self.config['num_classes']):
            class_name = self.class_names[i] if i < len(self.class_names) else f'class_{i}'
            self.log(f'val_{class_name}_acc', self.val_class_acc.compute()[i], on_step=False, on_epoch=True)
            self.log(f'val_{class_name}_f1', self.val_class_f1.compute()[i], on_step=False, on_epoch=True)
        
        return loss
    
    def on_validation_epoch_end(self):
        """검증 에포크 종료 시 클래스별 평균 손실 계산"""
        # 클래스별 평균 손실 계산 및 로깅
        for loss_key, losses in self.class_losses.items():
            if losses:  # 손실이 있는 경우에만
                avg_loss = np.mean(losses)
                self.log(f'avg_{loss_key}', avg_loss, on_epoch=True)
                # 리스트 초기화
                self.class_losses[loss_key] = []
    
    def get_class_performance_summary(self, stage='val'):
        """클래스별 성능 요약 반환"""
        if stage == 'train':
            class_acc = self.train_class_acc.compute()
            class_f1 = self.train_class_f1.compute()
            confusion = self.train_confusion.compute()
        else:
            class_acc = self.val_class_acc.compute()
            class_f1 = self.val_class_f1.compute()
            confusion = self.val_confusion.compute()
        
        summary = {}
        for i in range(self.config['num_classes']):
            class_name = self.class_names[i] if i < len(self.class_names) else f'class_{i}'
            summary[class_name] = {
                'accuracy': class_acc[i].item(),
                'f1_score': class_f1[i].item(),
                'confusion_matrix': confusion[i].tolist()
            }
        
        return summary
    
    def predict_with_class_metrics(self, dataloader, device='cuda'):
        """추론 시 클래스별 메트릭 계산"""
        self.eval()
        all_predictions = []
        all_targets = []
        all_outputs = []
        
        with torch.no_grad():
            for batch in dataloader:
                if len(batch) == 2:
                    images, targets = batch
                else:
                    images = batch
                    targets = None
                
                images = images.to(device)
                outputs = self(images)
                
                if targets is not None:
                    all_targets.append(targets.cpu())
                all_outputs.append(outputs.cpu())
                all_predictions.append(outputs.argmax(dim=1).cpu())
        
        all_predictions = torch.cat(all_predictions)
        all_outputs = torch.cat(all_outputs)
        
        if all_targets:
            all_targets = torch.cat(all_targets)
            
            # 클래스별 메트릭 계산
            class_acc = Accuracy(task='multiclass', num_classes=self.config['num_classes'], average='none')
            class_f1 = F1Score(task='multiclass', num_classes=self.config['num_classes'], average='none')
            confusion = ConfusionMatrix(task='multiclass', num_classes=self.config['num_classes'])
            
            class_accuracies = class_acc(all_outputs, all_targets)
            class_f1_scores = class_f1(all_outputs, all_targets)
            confusion_matrix = confusion(all_outputs, all_targets)
            
            # 클래스별 손실 계산
            class_losses = {}
            for class_idx in range(self.config['num_classes']):
                class_mask = (all_targets == class_idx)
                if class_mask.sum() > 0:
                    class_outputs = all_outputs[class_mask]
                    class_targets = all_targets[class_mask]
                    class_loss = self.criterion(class_outputs, class_targets)
                    class_losses[class_idx] = class_loss.item()
            
            # 결과 요약
            results = {
                'predictions': all_predictions.numpy(),
                'class_metrics': {}
            }
            
            for i in range(self.config['num_classes']):
                class_name = self.class_names[i] if i < len(self.class_names) else f'class_{i}'
                results['class_metrics'][class_name] = {
                    'accuracy': class_accuracies[i].item(),
                    'f1_score': class_f1_scores[i].item(),
                    'loss': class_losses.get(i, 0.0),
                    'sample_count': (all_targets == i).sum().item()
                }
            
            results['confusion_matrix'] = confusion_matrix.numpy()
            results['overall_accuracy'] = (all_predictions == all_targets).float().mean().item()
            
            return results
        else:
            return {
                'predictions': all_predictions.numpy(),
                'outputs': all_outputs.numpy()
            }
    
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
        elif optimizer_name == 'rmsprop':
            optimizer = RMSpropOptimizer(
                lr=optimizer_config.get('lr', 0.001),
                weight_decay=optimizer_config.get('weight_decay', 0.0001),
                alpha=optimizer_config.get('alpha', 0.99),
                eps=optimizer_config.get('eps', 1e-08),
                momentum=optimizer_config.get('momentum', 0)
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