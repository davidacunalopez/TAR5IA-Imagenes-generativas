"""
Módulo de Pytorch Lightning para el entrenamiento de autoencoders
"""
import torch
import torch.nn as nn
import pytorch_lightning as pl
from torchmetrics import StructuralSimilarityIndexMeasure
import torch.nn.functional as F


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
        ssim = StructuralSimilarityIndexMeasure(data_range=1.0)
        ssim_val = ssim(pred, target)
        return 1 - ssim_val  # SSIM es una métrica de similitud, convertimos a pérdida
    
    @staticmethod
    def ssim_l1_loss(pred, target, alpha=0.5):
        ssim = LossFunctions.ssim_loss(pred, target)
        l1 = LossFunctions.l1_loss(pred, target)
        return alpha * ssim + (1 - alpha) * l1


class AutoencoderLightning(pl.LightningModule):
    """Módulo de Lightning para entrenar autoencoders"""
    
    def __init__(self, model, learning_rate=0.001, loss_function="L2", scheduler_config=None):
        super().__init__()
        self.model = model
        self.learning_rate = learning_rate
        self.loss_function = loss_function
        self.scheduler_config = scheduler_config or {"step_size": 10, "gamma": 0.5}
        
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
        self.ssim_metric = StructuralSimilarityIndexMeasure(data_range=1.0)
        
        # Guardar pérdidas de entrenamiento
        self.train_losses = []
        
        # Guardar hiperparámetros
        self.save_hyperparameters(ignore=['model'])
    
    def forward(self, x):
        return self.model(x)
    
    def training_step(self, batch, batch_idx):
        x = batch
        x_recon = self(x)
        loss = self.criterion(x_recon, x)
        
        # Logging
        self.log('train/loss', loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log('train/learning_rate', self.optimizers().param_groups[0]['lr'], on_step=True)
        
        return loss
    
    def on_train_epoch_end(self):
        # Guardar pérdida promedio de la época
        epoch_loss = self.trainer.callback_metrics.get('train/loss_epoch', None)
        if epoch_loss is not None:
            self.train_losses.append(epoch_loss.item())
    
    def validation_step(self, batch, batch_idx):
        x = batch
        x_recon = self(x)
        loss = self.criterion(x_recon, x)
        
        # Calcular SSIM
        ssim_val = self.ssim_metric(x_recon, x)
        
        # Logging
        self.log('val/loss', loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log('val/ssim', ssim_val, on_step=False, on_epoch=True, prog_bar=True)
        
        return loss
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size=self.scheduler_config.get("step_size", 10),
            gamma=self.scheduler_config.get("gamma", 0.5)
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "epoch"
            }
        }

