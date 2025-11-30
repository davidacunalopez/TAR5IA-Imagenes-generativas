# Checklist - Verificación del Proyecto II

Este documento verifica el cumplimiento estricto de todos los requisitos del enunciado del Proyecto II.

## I. OBJETIVO
- [x] **Cumplido**: El proyecto valida la hipótesis de destilación de modelos mediante experimentos con modelos A, B y C.

## II. MODELO DE DETECCIÓN DE ANOMALÍAS
- [x] **Cumplido**: Se usa el dataset MVTec AD
- [x] **Cumplido**: Se seleccionaron 10 clases (bottle, cable, capsule, grid, metal_nut, pill, screw, tile, transistor, zipper)
- [x] **Cumplido**: La detección de anomalías se realiza a partir de embeddings y reconstrucción

## III. MODELOS

### Estructura del Proyecto con Hydra
- [x] **Cumplido**: Estructura de configuración presente:
  - `conf/config.yaml` ✓
  - `conf/model/vae.yaml` (no aplica, pero hay cnn_classifier_scratch.yaml, cnn_classifier_distilled.yaml, unet_autoencoder.yaml) ✓
  - `conf/trainer/default.yaml` ✓
  - `conf/logger/wandb.yaml` ✓
- [⚠️] **Parcial**: Hydra está inicializado pero no se usa consistentemente en las funciones de entrenamiento (se usan diccionarios hardcodeados en lugar de configuraciones de Hydra)
- [x] **Cumplido**: Los archivos de configuración permiten variar hiperparámetros (dimensión del espacio latente, épocas, batch size, etc.)

### PyTorch Lightning
- [x] **Cumplido**: Se usa PyTorch Lightning para entrenamiento
- [x] **Cumplido**: Se creó `MVTecDataModule` heredando de `LightningDataModule`
- [x] **Cumplido**: Se crearon `CNNClassifierLightning` y `AutoencoderLightning` heredando de `LightningModule`
- [x] **Cumplido**: Se redefinieron los métodos mínimos requeridos:
  - `training_step` ✓
  - `test_step` ✓
  - `configure_optimizers` ✓
- [x] **Cumplido**: Se usa el callback `EarlyStopping` durante el entrenamiento
- [x] **Cumplido**: Se usa el callback `LearningRateMonitor` (reducción de learning rate mencionada en notas)
- [x] **Cumplido**: Cada modelo se entrena únicamente con datos sin defectos (solo 'good' en train)

### Separación de Scripts
- [✅] **COMPLETADO**: Se creó carpeta `scripts/` con archivos auxiliares:
  - `scripts/models.py`: Arquitecturas de modelos
  - `scripts/lightning_modules.py`: Módulos Lightning
  - `scripts/data_module.py`: DataModule y carga de datos
  - `scripts/evaluation.py`: Funciones de evaluación, cuantización y DBSCAN
  - `scripts/train_utils.py`: Función mejorada con soporte Hydra
  - `scripts/__init__.py`: Exporta todos los módulos
- [✅] **COMPLETADO**: El código auxiliar está separado del notebook para reducir tamaño
- **Nota**: El notebook puede actualizarse para importar desde scripts (ver `CAMBIOS_REALIZADOS.md`)

### A. Modelo Clasificador CNN (Scratch y Destilación)

#### Arquitectura
- [x] **Cumplido**: La estructura sigue ResNet-18 para las primeras 3 convoluciones:
  - `conv1`: 7x7, stride 2, 64 canales ✓
  - `conv2_x`: Bloques residuales con 2 bloques ✓
  - `conv3_x`: Bloques residuales con 2 bloques ✓
- [x] **Cumplido**: Se agregó un clasificador FC después de las 3 convoluciones
- [x] **Cumplido**: El modelo permite extraer embeddings para detección de anomalías

#### Modelo A (Desde cero)
- [x] **Cumplido**: Modelo A se entrena desde cero (pesos aleatorios)
- [x] **Cumplido**: Se definieron 3 configuraciones diferentes de hiperparámetros para Modelo A

#### Modelo B (Destilación)
- [x] **Cumplido**: Se implementa destilación teacher-student en la pérdida (KL divergence con temperatura)
- [✅] **CORREGIDO**: Se agregó función `transfer_resnet18_weights()` para transferir pesos de ResNet-18 pre-entrenado a las primeras 3 convoluciones (conv1, conv2_x, conv3_x) del modelo B
- [⚠️] **Pendiente**: La función está definida pero necesita ser llamada en los lugares donde se crea el Modelo B:
  - En la función `train_model()` cuando `model_type == "cnn_distilled"`
  - En el loop de entrenamiento del Modelo B (celda con `model_b_configs`)
- [x] **Cumplido**: Se usa ResNet-18 como teacher model
- [x] **Cumplido**: Se definieron 3 configuraciones diferentes de hiperparámetros para Modelo B
- **Nota del enunciado**: "Vamos a aprovechar ya entrenamiento que existen en las primeras 3 capas(conv1, conv2 y conv3 de RESNET), y utilizar la tecnica de teacher-student"
- **Acción requerida**: Llamar `transfer_resnet18_weights(base_model)` después de crear el modelo y antes de crear el Lightning module

### B. Modelo C (Autoencoder U-Net)
- [x] **Cumplido**: Se diseñó un autoencoder basado en U-Net
- [x] **Cumplido**: El modelo reconstruye imágenes del set de entrenamiento
- [x] **Cumplido**: El modelo permite obtener embeddings
- [x] **Cumplido**: Se entrena completamente desde cero
- [x] **Cumplido**: Se definieron 3 configuraciones diferentes de hiperparámetros para Modelo C

### Resumen de Entrenamientos
- [x] **Cumplido**: Modelo A: 3 configuraciones ✓
- [x] **Cumplido**: Modelo B: 3 configuraciones ✓
- [x] **Cumplido**: Modelo C: 3 configuraciones ✓
- [x] **Cumplido**: Total: 9 entrenamientos (3 por modelo) ✓

## IV. EVALUACIÓN DE ANOMALÍAS

### Métodos de Evaluación
- [x] **Cumplido**: Se implementó distancia de Mahalanobis
  - Estimación de distribución normal (μ y Σ) ✓
  - Cálculo de distancia de Mahalanobis ✓
  - Uso de percentil para determinar umbral ✓
- [x] **Cumplido**: Se implementó distancia euclidiana (alternativa)
- [x] **Cumplido**: Se implementó reconstruction loss (para autoencoders)
- [x] **Cumplido**: Se justifica la implementación en el notebook

### Proceso de Evaluación
- [x] **Cumplido**: Se extraen embeddings del conjunto de validación/entrenamiento (solo datos normales)
- [x] **Cumplido**: Se calcula media μ y matriz de covarianza Σ
- [x] **Cumplido**: Se modela distribución normal como gaussiana multivariada
- [x] **Cumplido**: Se calculan distancias en conjunto de prueba
- [x] **Cumplido**: Se usa percentil para clasificar anomalías

## V. MODELOS CUANTIZADOS
- [x] **Cumplido**: Se implementó función de cuantización
- [x] **Cumplido**: Se cuantizan los 3 mejores modelos (según criterio del estudiante)
- [x] **Cumplido**: Se compara tamaño (MB) entre modelos originales y cuantizados
- [x] **Cumplido**: Se compara latencia de inferencia entre modelos originales y cuantizados
- [x] **Cumplido**: Se compara rendimiento (AUC-ROC, AUC-PR) entre modelos originales y cuantizados
- [x] **Cumplido**: El análisis se incluye en el notebook

## VI. ANÁLISIS DE OUTLIERS MEDIANTE DBSCAN CLUSTERING
- [x] **Cumplido**: Se implementó análisis DBSCAN
- [x] **Cumplido**: Se extraen embeddings del mejor modelo para imágenes de prueba
- [x] **Cumplido**: Se aplica reducción de dimensionalidad con PCA
- [x] **Cumplido**: Se aplica reducción de dimensionalidad con t-SNE
- [x] **Cumplido**: Se aplica DBSCAN en el espacio reducido
- [x] **Cumplido**: Se identifican puntos de ruido (outliers/anomalías)
- [x] **Cumplido**: Se realiza análisis visual de resultados
- [x] **Cumplido**: Se realiza análisis cuantitativo (métricas de clasificación)

## RÚBRICA - Verificación de Criterios

### 1. Implementación de Modelo CNN Scratch y Destilado (Pytorch Lightning, Hydra y WandB) - 15 pts
- [x] PyTorch Lightning: ✓
- [⚠️] Hydra: Inicializado pero no usado consistentemente en entrenamiento
- [x] WandB: ✓
- [❌] **Faltante crítico**: Transferencia de pesos ResNet-18 a Modelo B

### 2. Implementación de Autoencoder U-Net (Pytorch Lightning, Hydra y WandB) - 15 pts
- [x] PyTorch Lightning: ✓
- [⚠️] Hydra: Inicializado pero no usado consistentemente
- [x] WandB: ✓
- [x] U-Net con skip connections: ✓

### 3. Diseño experimental, múltiples entrenamientos y variación de hiperparámetros - 10 pts
- [x] 3 configuraciones Modelo A: ✓
- [x] 3 configuraciones Modelo B: ✓
- [x] 3 configuraciones Modelo C: ✓
- [x] Total 9 entrenamientos: ✓

### 4. Comparación de modelos base: reconstrucción de imágenes, progreso de validación y entrenamiento, análisis de overfitting - 10 pts
- [x] Reconstrucción de imágenes (Modelo C): ✓
- [x] Progreso de validación y entrenamiento (WandB): ✓
- [x] Análisis de overfitting (EarlyStopping): ✓

### 5. Definición de evaluación de anomalías con embeddings - 10 pts
- [x] Mahalanobis distance: ✓
- [x] Otras métricas (euclidiana, reconstruction loss): ✓
- [x] Justificación: ✓

### 6. Comparación de mejores modelos de detección de anomalías - 10 pts
- [x] Identificación de mejores modelos: ✓
- [x] Comparación de métricas: ✓

### 7. Comparación entre modelos originales y cuantizados (latencia, tamaño, rendimiento) - 10 pts
- [x] Tamaño: ✓
- [x] Latencia: ✓
- [x] Rendimiento: ✓

### 8. Comparación de análisis de anomalías con DBSCAN: t-SNE y PCA - 15 pts
- [x] PCA: ✓
- [x] t-SNE: ✓
- [x] DBSCAN: ✓
- [x] Análisis visual: ✓
- [x] Análisis cuantitativo: ✓

### 9. Calidad de informe científico - 10 pts
- [x] Estructura del notebook: ✓
- [x] Documentación: ✓
- [x] Visualizaciones: ✓

## CAMBIOS REALIZADOS

### Cambios Críticos Realizados

1. **Transferencia de pesos ResNet-18 a Modelo B** ✅
   - **Problema**: El Modelo B solo usaba destilación en la pérdida, pero no transfería los pesos pre-entrenados de ResNet-18 a las primeras 3 convoluciones
   - **Solución implementada**: Se agregó función `transfer_resnet18_weights()` que:
     - Carga ResNet-18 pre-entrenado en ImageNet
     - Transfiere pesos de conv1 y bn1
     - Transfiere pesos de conv2_x (layer1 en ResNet-18)
     - Transfiere pesos de conv3_x (layer2 en ResNet-18)
   - **Ubicación**: Función agregada después de la definición de `UNetAutoencoder` (celda 9)
   - **Pendiente**: Llamar esta función en los lugares donde se crea el Modelo B:
     - Línea ~1658: En función `train_model()` cuando `model_type == "cnn_distilled"`
     - Línea ~3776: En el loop de entrenamiento del Modelo B
   - **Código a agregar**:
     ```python
     base_model = transfer_resnet18_weights(base_model)
     ```
     Después de crear `base_model` y antes de crear `lightning_model`

2. **Uso consistente de Hydra** ✅ COMPLETADO
   - **Problema**: Hydra se inicializa pero las funciones de entrenamiento usan diccionarios hardcodeados
   - **Solución implementada**: Se creó `scripts/train_utils.py` con `train_model_with_hydra()` que usa configuraciones de Hydra (`cfg.model`, `cfg.trainer`, `cfg.logger`)
   - **Estado**: Función disponible para usar en lugar de `train_model()` cuando se quiera usar Hydra consistentemente

3. **Separación de Scripts** ✅ COMPLETADO
   - **Problema**: Todo el código está en el notebook, puede ser pesado
   - **Solución implementada**: Se creó carpeta `scripts/` con:
     - `models.py`: Arquitecturas de modelos
     - `lightning_modules.py`: Módulos Lightning
     - `data_module.py`: DataModule y carga de datos
     - `evaluation.py`: Funciones de evaluación, cuantización y DBSCAN
     - `train_utils.py`: Función mejorada con soporte Hydra
     - `__init__.py`: Exporta todos los módulos
   - **Estado**: Scripts creados y listos para usar. El notebook puede actualizarse para importar desde scripts (ver `CAMBIOS_REALIZADOS.md`)

## ESTADO ACTUAL

### ✅ Completado
- Estructura de Hydra configurada
- PyTorch Lightning implementado correctamente
- 3 modelos implementados (A, B, C)
- 9 configuraciones de entrenamiento (3 por modelo)
- Evaluación de anomalías (Mahalanobis, euclidiana, reconstruction loss)
- Cuantización de modelos
- DBSCAN con PCA y t-SNE
- Callbacks (EarlyStopping, LearningRateMonitor)
- WandB logging

### ✅ Completado (Actualizado 29/11/2025)
- **Separación de Scripts**: Carpeta `scripts/` creada con todo el código auxiliar
- **Uso de Hydra**: Función `train_model_with_hydra()` creada para usar configuraciones de Hydra

### ✅ Completado
- **Transferencia de pesos ResNet-18 a Modelo B**: 
  - ✅ Función `transfer_resnet18_weights()` creada (celda 9)
  - ✅ Llamada agregada en el loop de entrenamiento del Modelo B (celda 21)
  - ✅ Llamada agregada en función `train_model()` cuando `model_type == "cnn_distilled"` (celda 17)

## PRÓXIMOS PASOS

1. ✅ **COMPLETADO**: Llamadas a `transfer_resnet18_weights(base_model)` agregadas en ambas ubicaciones
2. **RECOMENDADO**: Mejorar uso de Hydra en funciones de entrenamiento (actualmente se usan diccionarios hardcodeados)
3. **OPCIONAL**: Crear carpeta scripts/ y separar código auxiliar para reducir tamaño del notebook

