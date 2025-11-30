"""
Módulos de PyTorch Lightning para el Proyecto II
Incluye: LossFunctions, CNNClassifierLightning, AutoencoderLightning
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from torchmetrics import StructuralSimilarityIndexMeasure
from torchmetrics.classification import Accuracy
from torchvision.models import resnet18


class LossFunctions:
    """Funciones de pérdida para el entrenamiento"""

    @staticmethod
    def l1_loss(pred, target):
        return F.l1_loss(pred, target)

    @staticmethod
    def l2_loss(pred, target):
        return F.mse_loss(pred, target)

    @staticmethod
    def ssim_loss(pred, target):
        ssim = StructuralSimilarityIndexMeasure(data_range=2.0)
        ssim_val = ssim(pred, target)
        return 1 - ssim_val

    @staticmethod
    def ssim_l1_loss(pred, target, alpha=0.5):
        ssim = LossFunctions.ssim_loss(pred, target)
        l1 = LossFunctions.l1_loss(pred, target)
        return alpha * ssim + (1 - alpha) * l1


class CNNClassifierLightning(pl.LightningModule):
    """Módulo Lightning para entrenar CNN clasificadores (Modelo A y B)"""

    def __init__(self, model, num_classes=10, learning_rate=0.001, weight_decay=1e-5,
                 scheduler_config=None, model_type="scratch", teacher_model=None,
                 distillation_config=None):
        super().__init__()
        self.model = model
        self.num_classes = num_classes
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.model_type = model_type
        self.scheduler_config = scheduler_config or {"name": "step", "step_size": 15, "gamma": 0.5}

        # Configuración de destilación (solo para Modelo B)
        self.distillation_config = distillation_config or {}
        self.teacher_model = teacher_model
        if model_type == "distilled" and teacher_model is None:
            # Cargar ResNet-18 pre-entrenado como teacher
            try:
                from torchvision.models import ResNet18_Weights
                self.teacher_model = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
            except:
                self.teacher_model = resnet18(pretrained=True)
            self.teacher_model.fc = nn.Linear(self.teacher_model.fc.in_features, num_classes)
            self.teacher_model.eval()
            for param in self.teacher_model.parameters():
                param.requires_grad = False

        # Métricas
        self.train_acc = Accuracy(task="multiclass", num_classes=num_classes)
        self.val_acc = Accuracy(task="multiclass", num_classes=num_classes)
        self.test_acc = Accuracy(task="multiclass", num_classes=num_classes)

        # Criterio de pérdida
        self.criterion = nn.CrossEntropyLoss()

        # Guardar hiperparámetros
        self.save_hyperparameters(ignore=['model', 'teacher_model'])

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        images, labels = batch
        logits, embeddings = self(images)

        # Pérdida de clasificación
        loss = self.criterion(logits, labels)

        # Pérdida de destilación (solo para Modelo B)
        if self.model_type == "distilled" and self.teacher_model is not None:
            with torch.no_grad():
                teacher_logits = self.teacher_model(images)

            temperature = self.distillation_config.get("temperature", 4.0)
            alpha = self.distillation_config.get("alpha", 0.7)

            # Softmax con temperatura
            student_soft = F.log_softmax(logits / temperature, dim=1)
            teacher_soft = F.softmax(teacher_logits / temperature, dim=1)

            # Pérdida de destilación (KL divergence)
            distillation_loss = F.kl_div(student_soft, teacher_soft, reduction='batchmean') * (temperature ** 2)

            # Combinar pérdidas
            loss = alpha * distillation_loss + (1 - alpha) * loss

        # Logging
        self.log('train/loss', loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log('train/acc', self.train_acc(logits, labels), on_step=True, on_epoch=True, prog_bar=True)
        self.log('train/learning_rate', self.optimizers().param_groups[0]['lr'], on_step=True)

        return loss

    def validation_step(self, batch, batch_idx):
        images, labels = batch
        logits, embeddings = self(images)
        loss = self.criterion(logits, labels)

        # Logging
        self.log('val/loss', loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log('val/acc', self.val_acc(logits, labels), on_step=False, on_epoch=True, prog_bar=True)

        return loss

    def test_step(self, batch, batch_idx):
        images, labels = batch
        logits, embeddings = self(images)
        loss = self.criterion(logits, labels)

        # Logging
        self.log('test/loss', loss, on_step=False, on_epoch=True)
        self.log('test/acc', self.test_acc(logits, labels), on_step=False, on_epoch=True)

        return {'logits': logits, 'labels': labels, 'embeddings': embeddings}

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)

        scheduler_name = self.scheduler_config.get("name", "step")
        if scheduler_name == "step":
            scheduler = torch.optim.lr_scheduler.StepLR(
                optimizer,
                step_size=self.scheduler_config.get("step_size", 15),
                gamma=self.scheduler_config.get("gamma", 0.5)
            )
        elif scheduler_name == "cosine":
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=self.scheduler_config.get("T_max", 50)
            )
        elif scheduler_name == "plateau":
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                mode='min',
                factor=self.scheduler_config.get("factor", 0.5),
                patience=self.scheduler_config.get("patience", 5)
            )
        else:
            scheduler = None

        if scheduler is None:
            return optimizer
        else:
            # Para ReduceLROnPlateau, necesitamos incluir el monitor
            if scheduler_name == "plateau":
                return {
                    "optimizer": optimizer,
                    "lr_scheduler": {
                        "scheduler": scheduler,
                        "interval": "epoch",
                        "monitor": "val/loss"
                    }
                }
            else:
                return {
                    "optimizer": optimizer,
                    "lr_scheduler": {
                        "scheduler": scheduler,
                        "interval": "epoch"
                    }
                }


class AutoencoderLightning(pl.LightningModule):
    """Módulo Lightning para entrenar autoencoders (Modelo C)"""

    def __init__(self, model, learning_rate=0.001, loss_function="L2", scheduler_config=None):
        super().__init__()
        self.model = model
        self.learning_rate = learning_rate
        self.loss_function = loss_function
        self.scheduler_config = scheduler_config or {"name": "step", "step_size": 15, "gamma": 0.5}

        # Inicializar función de pérdida
        if loss_function == "L1":
            self.criterion = LossFunctions.l1_loss
        elif loss_function == "L2":
            self.criterion = LossFunctions.l2_loss
        elif loss_function == "SSIM":
            self.criterion = LossFunctions.ssim_loss
        elif loss_function == "SSIM_L1":
            self.criterion = LossFunctions.ssim_l1_loss
        else:
            raise ValueError(f"Función de pérdida no reconocida: {loss_function}")

        # Métricas
        self.ssim_metric = StructuralSimilarityIndexMeasure(data_range=2.0)

        # Guardar hiperparámetros
        self.save_hyperparameters(ignore=['model'])

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x = batch[0] if isinstance(batch, tuple) else batch
        x_recon = self(x)
        loss = self.criterion(x_recon, x)

        # Logging
        self.log('train/loss', loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log('train/learning_rate', self.optimizers().param_groups[0]['lr'], on_step=True)

        return loss

    def validation_step(self, batch, batch_idx):
        x = batch[0] if isinstance(batch, tuple) else batch
        x_recon = self(x)
        loss = self.criterion(x_recon, x)

        # Calcular SSIM
        ssim_val = self.ssim_metric(x_recon, x)

        # Logging
        self.log('val/loss', loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log('val/ssim', ssim_val, on_step=False, on_epoch=True, prog_bar=True)

        return loss

    def test_step(self, batch, batch_idx):
        x = batch[0] if isinstance(batch, tuple) else batch
        x_recon = self(x)
        loss = self.criterion(x_recon, x)

        # Extraer embeddings
        embeddings = self.model.get_embedding(x)

        # Logging
        self.log('test/loss', loss, on_step=False, on_epoch=True)

        return {'reconstructions': x_recon, 'originals': x, 'embeddings': embeddings}

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)

        scheduler_name = self.scheduler_config.get("name", "step")
        if scheduler_name == "step":
            scheduler = torch.optim.lr_scheduler.StepLR(
                optimizer,
                step_size=self.scheduler_config.get("step_size", 15),
                gamma=self.scheduler_config.get("gamma", 0.5)
            )
        elif scheduler_name == "cosine":
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=self.scheduler_config.get("T_max", 50)
            )
        elif scheduler_name == "plateau":
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                mode='min',
                factor=self.scheduler_config.get("factor", 0.5),
                patience=self.scheduler_config.get("patience", 5)
            )
        else:
            scheduler = None

        if scheduler is None:
            return optimizer
        else:
            if scheduler_name == "plateau":
                return {
                    "optimizer": optimizer,
                    "lr_scheduler": {
                        "scheduler": scheduler,
                        "interval": "epoch",
                        "monitor": "val/loss"
                    }
                }
            else:
                return {
                    "optimizer": optimizer,
                    "lr_scheduler": {
                        "scheduler": scheduler,
                        "interval": "epoch"
                    }
                }

