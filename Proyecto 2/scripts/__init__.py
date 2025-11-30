"""
Scripts auxiliares para el Proyecto II
"""

from .models import BasicBlock, CNNClassifier, UNetAutoencoder, transfer_resnet18_weights
from .lightning_modules import LossFunctions, CNNClassifierLightning, AutoencoderLightning
from .data_module import AnomalyDataset, load_dataset_paths, MVTecDataModule
from .evaluation import (
    calculate_mahalanobis_distance,
    extract_embeddings,
    estimate_normal_distribution,
    evaluate_anomaly_detection,
    quantize_model,
    compare_model_sizes,
    dbscan_analysis,
    visualize_dbscan_results
)
from .train_utils import train_model_with_hydra

__all__ = [
    'BasicBlock',
    'CNNClassifier',
    'UNetAutoencoder',
    'transfer_resnet18_weights',
    'LossFunctions',
    'CNNClassifierLightning',
    'AutoencoderLightning',
    'AnomalyDataset',
    'load_dataset_paths',
    'MVTecDataModule',
    'calculate_mahalanobis_distance',
    'extract_embeddings',
    'estimate_normal_distribution',
    'evaluate_anomaly_detection',
    'quantize_model',
    'compare_model_sizes',
    'dbscan_analysis',
    'visualize_dbscan_results',
    'train_model_with_hydra',
]

