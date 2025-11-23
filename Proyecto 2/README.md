# Proyecto II - Detección de Anomalías con Destilación de Modelos

## Estructura del Proyecto

```
Proyecto 2/
├── conf/
│   ├── config.yaml                    # Configuración principal
│   ├── model/
│   │   ├── cnn_classifier_scratch.yaml # Modelo A
│   │   ├── cnn_classifier_distilled.yaml # Modelo B
│   │   └── unet_autoencoder.yaml      # Modelo C
│   ├── trainer/
│   │   └── default.yaml               # Configuración de entrenamiento
│   └── logger/
│       └── wandb.yaml                 # Configuración de WandB
├── models.py                          # Arquitecturas de modelos
├── lightning_modules.py               # Módulos de Pytorch Lightning
├── data_module.py                     # DataModule para carga de datos
├── evaluation.py                      # Funciones de evaluación
├── Proyecto_II.ipynb                  # Notebook principal (contenido del PDF)
└── Proyecto_II_Implementation.ipynb   # Notebook de implementación (a crear)
```

## Componentes Implementados

### 1. Modelos
- **CNNClassifier** (Modelo A y B): Basado en ResNet-18 para primeras 3 convoluciones
- **UNetAutoencoder** (Modelo C): Autoencoder con skip connections

### 2. Módulos Lightning
- **CNNClassifierLightning**: Para entrenar clasificadores (con soporte para destilación)
- **AutoencoderLightning**: Para entrenar autoencoders

### 3. DataModule
- **MVTecDataModule**: Carga datos de MVTec AD con 10 clases, solo datos sin defectos para entrenamiento

### 4. Evaluación
- **evaluate_anomaly_detection**: Evaluación con distancia de Mahalanobis, euclidiana o reconstruction loss
- **quantize_model**: Cuantización de modelos
- **dbscan_analysis**: Análisis DBSCAN con PCA y t-SNE

## Uso

1. Instalar dependencias (ver notebook)
2. Configurar rutas del dataset en `conf/config.yaml`
3. Ejecutar el notebook `Proyecto_II_Implementation.ipynb`

## Configuración

Los archivos de configuración están en `conf/`. Se pueden modificar hiperparámetros usando Hydra overrides.

## Notas

- Todos los modelos se entrenan solo con datos sin defectos
- Se requiere EarlyStopping durante el entrenamiento
- Cada modelo debe entrenarse con al menos 3 configuraciones diferentes de hiperparámetros
- Los mejores 3 modelos se cuantizan para comparación


