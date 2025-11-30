"""
Utilidades de entrenamiento con soporte mejorado para Hydra
"""

import os
import torch
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint, LearningRateMonitor
from omegaconf import DictConfig, OmegaConf

from .models import CNNClassifier, UNetAutoencoder, transfer_resnet18_weights
from .lightning_modules import CNNClassifierLightning, AutoencoderLightning


def train_model_with_hydra(cfg: DictConfig, model_type: str, experiment_name: str, 
                           data_module, categories, drive_base_path=None):
    """
    Entrena un modelo usando configuraciones de Hydra.
    
    Esta función mejora train_model() para usar configuraciones de Hydra en lugar de diccionarios hardcodeados.
    
    Args:
        cfg: Configuración de Hydra (DictConfig)
        model_type: "cnn_scratch", "cnn_distilled", o "unet"
        experiment_name: Nombre del experimento para WandB
        data_module: Instancia de MVTecDataModule
        categories: Lista de categorías
        drive_base_path: Ruta base para guardar checkpoints (opcional)
    """
    print(f"\n{'='*80}")
    print(f"ENTRENANDO: {experiment_name}")
    print(f"{'='*80}\n")

    # Obtener configuraciones de Hydra
    model_cfg = cfg.model
    trainer_cfg = cfg.trainer
    logger_cfg = cfg.logger

    # Extraer parámetros del modelo
    model_config = OmegaConf.to_container(model_cfg, resolve=True)
    # Remover _target_ si existe (no es un parámetro del modelo)
    model_config.pop('_target_', None)
    model_config.pop('architecture', None)
    
    # Extraer parámetros del trainer para Lightning module
    scheduler_config = {
        "name": trainer_cfg.scheduler.get("name", "step"),
        "step_size": trainer_cfg.scheduler.get("step_size", 15),
        "gamma": trainer_cfg.scheduler.get("gamma", 0.5),
    }
    if trainer_cfg.scheduler.get("name") == "cosine":
        scheduler_config["T_max"] = trainer_cfg.scheduler.get("T_max", 50)
    elif trainer_cfg.scheduler.get("name") == "plateau":
        scheduler_config["factor"] = trainer_cfg.scheduler.get("factor", 0.5)
        scheduler_config["patience"] = trainer_cfg.scheduler.get("patience", 5)

    lightning_params = {
        'learning_rate': trainer_cfg.get('learning_rate', 0.001),
        'weight_decay': trainer_cfg.get('weight_decay', 1e-5),
        'scheduler_config': scheduler_config
    }

    # Para modelos con destilación
    if model_type == "cnn_distilled" and hasattr(trainer_cfg, 'distillation_config'):
        lightning_params['distillation_config'] = OmegaConf.to_container(
            trainer_cfg.distillation_config, resolve=True
        )

    # Crear modelo base
    if model_type == "cnn_scratch":
        base_model = CNNClassifier(
            num_classes=len(categories),
            model_type="scratch",
            **model_config
        )
        lightning_model = CNNClassifierLightning(
            model=base_model,
            num_classes=len(categories),
            model_type="scratch",
            **lightning_params
        )
    elif model_type == "cnn_distilled":
        base_model = CNNClassifier(
            num_classes=len(categories),
            model_type="distilled",
            **model_config
        )
        
        # TRANSFERIR PESOS DE RESNET-18 A LAS PRIMERAS 3 CONVOLUCIONES
        base_model = transfer_resnet18_weights(base_model)
        
        lightning_model = CNNClassifierLightning(
            model=base_model,
            num_classes=len(categories),
            model_type="distilled",
            **lightning_params
        )
    elif model_type == "unet":
        base_model = UNetAutoencoder(**model_config)
        # Para AutoencoderLightning, también necesitamos loss_function
        autoencoder_params = {
            'learning_rate': trainer_cfg.get('learning_rate', 0.001),
            'loss_function': trainer_cfg.get('loss_function', 'L2'),
            'scheduler_config': scheduler_config
        }
        lightning_model = AutoencoderLightning(
            model=base_model,
            **autoencoder_params
        )
    else:
        raise ValueError(f"Tipo de modelo no reconocido: {model_type}")

    # Configurar logger usando Hydra
    wandb_logger = WandbLogger(
        project=logger_cfg.get("project", "proyecto-ii-anomaly-detection"),
        name=experiment_name,
        config=OmegaConf.to_container(cfg, resolve=True),
        reinit=True
    )

    # Callbacks usando configuración de Hydra
    early_stopping = EarlyStopping(
        monitor=trainer_cfg.early_stopping.get("monitor", "val/loss"),
        mode=trainer_cfg.early_stopping.get("mode", "min"),
        patience=trainer_cfg.early_stopping.get("patience", 10),
        min_delta=trainer_cfg.early_stopping.get("min_delta", 0.001)
    )

    checkpoint_dir = os.path.join(drive_base_path, 'checkpoints', experiment_name) if drive_base_path else None
    checkpoint_callback = ModelCheckpoint(
        monitor=trainer_cfg.checkpoint.get("monitor", "val/loss"),
        mode=trainer_cfg.checkpoint.get("mode", "min"),
        save_top_k=trainer_cfg.checkpoint.get("save_top_k", 3),
        save_last=trainer_cfg.checkpoint.get("save_last", True),
        dirpath=checkpoint_dir,
        filename=f'{experiment_name}-{{epoch:02d}}-{{val/loss:.4f}}'
    )

    lr_monitor = LearningRateMonitor(logging_interval='step')

    # Crear Trainer usando configuración de Hydra
    trainer = pl.Trainer(
        max_epochs=trainer_cfg.get("max_epochs", 50),
        accelerator='auto',
        devices=1,
        logger=wandb_logger,
        callbacks=[early_stopping, checkpoint_callback, lr_monitor],
        log_every_n_steps=10,
        enable_progress_bar=True,
        gradient_clip_val=trainer_cfg.get("gradient_clip_val", 1.0),
        accumulate_grad_batches=trainer_cfg.get("accumulate_grad_batches", 1)
    )

    # Validar que el data_module está configurado
    try:
        if not hasattr(data_module, 'train_paths') or len(data_module.train_paths) == 0:
            raise ValueError("❌ ERROR: El data_module no está configurado correctamente. Ejecuta data_module.setup() primero.")
    except AttributeError:
        raise ValueError("❌ ERROR: El data_module no está configurado. Ejecuta data_module.setup() primero.")

    # Entrenar
    trainer.fit(lightning_model, data_module)

    # Evaluar
    trainer.test(lightning_model, data_module)

    return lightning_model

