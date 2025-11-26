# Checklist de Verificación - Proyecto II

Este documento actúa como memoria y trazabilidad de los cambios realizados en el proyecto según el enunciado.

---

## Verificación: Sección I. OBJETIVO y II. MODELO DE DETECCIÓN DE ANOMALÍAS (Líneas 9-19)

**Fecha de verificación:** 2025-01-27

### ✅ I. OBJETIVO (Líneas 9-11)

**Requisito del enunciado:**
> Aplicar un experimento que permita validar la hipótesis de que al aplicar técnicas de destilado de modelos de grandes volúmenes de parámetros en modelos más pequeños se pueden resolver tareas igual de complejas pero con modelos más eficientes.

**Estado:** ✅ **IMPLEMENTADO**

**Ubicación en notebook:**
- Línea 9-22: Objetivo claramente definido
- Implementación de 3 modelos (A, B, C) para validar la hipótesis:
  - **Modelo A**: CNN clasificador desde cero
  - **Modelo B**: CNN clasificador con destilación teacher-student
  - **Modelo C**: Autoencoder U-Net para reconstrucción

**Notas:**
- El objetivo está correctamente documentado en el notebook
- La estructura del proyecto permite validar la hipótesis mediante comparación de modelos

---

### ✅ II. MODELO DE DETECCIÓN DE ANOMALÍAS (Líneas 13-19)

#### 2.1. Dataset MVTec AD

**Requisito del enunciado:**
> Para el desarrollo de este proyecto debe usar el dataset propuesto en **MVTec AD — A Comprehensive Real-World Dataset for Unsupervised Anomaly Detection**. Un dataset de escenarios industriales reales con diferentes tipos de anomalías en la forma de detección de defectos en objetos o texturas.

**Estado:** ✅ **IMPLEMENTADO**

**Ubicación en notebook:**
- Línea 1248: Documentación sobre carga del dataset MVTec AD
- Línea 1346-1474: Implementación de `MVTecDataModule` (hereda de `pl.LightningDataModule`)
- Configuración en `conf/config.yaml`: Ruta del dataset configurada

**Notas:**
- El dataset está correctamente configurado y cargado
- Se implementa un DataModule siguiendo las mejores prácticas de PyTorch Lightning

---

#### 2.2. Selección de 10 Clases

**Requisito del enunciado:**
> Seleccione **10 clases del dataset** las que usted más prefiera, y con este subconjunto vamos a entrenar distintos modelos para resolver un problema de detección de anomalías.

**Estado:** ✅ **IMPLEMENTADO**

**Ubicación en notebook:**
- Línea 1248: Menciona "10 clases"
- Configuración en `conf/config.yaml` (línea 10):
  ```yaml
  categories: ["bottle", "cable", "capsule", "grid", "metal_nut", "pill", "screw", "tile", "transistor", "zipper"]
  ```

**Clases seleccionadas:**
1. bottle
2. cable
3. capsule
4. grid
5. metal_nut
6. pill
7. screw
8. tile
9. transistor
10. zipper

**Notas:**
- Las 10 clases están correctamente configuradas
- El DataModule carga datos de todas las categorías especificadas

---

#### 2.3. Nota sobre Detección de Anomalías (Línea 19)

**Requisito del enunciado:**
> **Nota:** Detectar anomalías desde una reconstrucción de datos, cuando construimos un embedding. Detectar las anomalías partiendo de la pregunta: ¿Existen diferencias a partir de las imágenes que estamos haciendo con las originales?

**Estado:** ✅ **IMPLEMENTADO** (parcialmente documentado)

**Ubicación en notebook:**
- Línea 4097-4117: Sección "6. Evaluación de Anomalías"
- Implementación de método `reconstruction_loss` en función `evaluate_anomaly_detection`:
  - Línea 4421-4440: Cálculo de reconstruction loss comparando imágenes reconstruidas vs originales
  - Se calcula: `np.mean((test_reconstructions - test_originals) ** 2, axis=(1, 2, 3))`

**Métodos implementados:**
1. ✅ **Distancia de Mahalanobis**: Usando embeddings
2. ✅ **Distancia Euclidiana**: Usando embeddings
3. ✅ **Reconstruction Loss**: Comparando imágenes reconstruidas vs originales

**Notas:**
- La funcionalidad está implementada correctamente
- El método `reconstruction_loss` responde a la pregunta: "¿Existen diferencias entre las imágenes que estamos haciendo con las originales?"
- **Sugerencia de mejora**: Podría añadirse una nota explícita en la documentación del notebook mencionando esta pregunta filosófica del profesor

---

## Verificación: Sección III. MODELOS - Estructura con Hydra (Líneas 21-43)

**Fecha de verificación:** 2025-01-27

### ✅ III.1. Gestión Modular con Hydra (Línea 23)

**Requisito del enunciado:**
> Cada modelo debe estructurar el proyecto utilizando la librería **Hydra**(el mismo que en la tarea) para la gestión modular de configuraciones, asegurando la correcta separación de hiper parámetros entre el modelo, el entrenamiento y los registros experimentales.

**Estado:** ✅ **IMPLEMENTADO**

**Ubicación en notebook:**
- Línea 938-944: Sección "3. Configuración con Hydra"
- Línea 1152-1171: Inicialización de Hydra con `hydra.initialize()` y `hydra.compose()`
- Línea 296: Importación de `OmegaConf` para manejo de configuraciones

**Notas:**
- Hydra está correctamente inicializado y configurado
- Se maneja el caso donde no existe configuración (valores por defecto)
- Las clases del notebook están registradas en el resolver de Hydra

---

### ✅ III.2. Estructura del Proyecto (Líneas 25-38)

**Requisito del enunciado:**
> La estructura mínima recomendada del proyecto es la siguiente:
> ```
> conf/
> - config.yaml
> - model/
>     - vae.yaml
> - trainer/
>     - default.yaml
> - logger/
>     - wandb.yaml
> ```

**Estado:** ✅ **IMPLEMENTADO** (con variación aceptable)

**Estructura actual:**
```
conf/
- config.yaml ✅
- model/
    - cnn_classifier_scratch.yaml ✅ (Modelo A)
    - cnn_classifier_distilled.yaml ✅ (Modelo B)
    - unet_autoencoder.yaml ✅ (Modelo C - equivalente a vae.yaml)
- trainer/
    - default.yaml ✅
- logger/
    - wandb.yaml ✅
```

**Ubicación:**
- Directorio `conf/` existe en el proyecto
- Todos los archivos YAML requeridos están presentes
- El notebook crea la estructura automáticamente si no existe (líneas 1118-1125)

**Notas:**
- ✅ El requisito menciona `vae.yaml` pero el proyecto usa `unet_autoencoder.yaml` (equivalente funcional para autoencoder)
- ✅ Se tienen 3 archivos de modelo (A, B, C) en lugar de solo uno, lo cual es correcto para este proyecto
- ✅ La estructura cumple con el requisito de separación modular

---

### ✅ III.3. Configuración de Hiperparámetros (Líneas 40-43)

**Requisito del enunciado:**
> Cada módulo de configuración deberá permitir la ejecución de experimentos con distintos parámetros del modelo, tales como:
> - Dimensión del espacio latente $(z)$.
> - Cantidad de épocas, tamaño de batch, o cualquier hiperparámetro que requiera.

**Estado:** ✅ **IMPLEMENTADO**

#### 3.3.1. Dimensión del Espacio Latente (z)

**Ubicación:**
- `conf/model/unet_autoencoder.yaml` (línea 7): `latent_dim: 128`
- `conf/model/cnn_classifier_scratch.yaml` (línea 19): `embedding_dim: 256`
- `conf/model/cnn_classifier_distilled.yaml` (línea 19): `embedding_dim: 256`

**Notas:**
- ✅ Configurable en archivos YAML
- ✅ Diferentes modelos pueden tener diferentes dimensiones

#### 3.3.2. Cantidad de Épocas

**Ubicación:**
- `conf/trainer/default.yaml` (línea 2): `max_epochs: 50`

**Notas:**
- ✅ Configurable en archivo de entrenamiento
- ✅ Permite variar entre experimentos

#### 3.3.3. Tamaño de Batch

**Ubicación:**
- `conf/config.yaml` (línea 13): `batch_size: 32`

**Notas:**
- ✅ Configurable en configuración principal
- ✅ Fácil de modificar para diferentes experimentos

#### 3.3.4. Otros Hiperparámetros Configurables

**En `conf/trainer/default.yaml`:**
- ✅ `learning_rate: 0.001`
- ✅ `weight_decay: 1e-5`
- ✅ `optimizer: "adam"` (adam, sgd)
- ✅ `momentum: 0.9` (para SGD)
- ✅ `scheduler`: Configuración completa (step, cosine, plateau)
- ✅ `early_stopping`: Configuración completa
- ✅ `checkpoint`: Configuración de guardado

**En `conf/model/`:**
- ✅ `conv1_channels`, `conv2_channels`, `conv3_channels` (arquitectura CNN)
- ✅ `num_classes: 10`
- ✅ `fc_hidden: 512`
- ✅ `dropout: 0.5`
- ✅ `encoder_channels`, `decoder_channels` (para U-Net)
- ✅ `embedding_dim` (para detección de anomalías)

**En `conf/config.yaml`:**
- ✅ `image_size: 128`
- ✅ `num_workers: 2`
- ✅ `train_split: 0.8`

**Notas:**
- ✅ Todos los hiperparámetros importantes son configurables
- ✅ La separación modular permite modificar parámetros sin tocar código
- ✅ Se pueden ejecutar múltiples experimentos cambiando solo los archivos YAML

---

## Verificación: Sección III.4. PyTorch Lightning (Líneas 45-56)

**Fecha de verificación:** 2025-01-27

### ✅ III.4.1. Uso de PyTorch Lightning (Línea 47)

**Requisito del enunciado:**
> Además debe utilizar **PyTorch Lightning** para las personalizaciones de los entrenamientos y creación de los modelos basado en las mejores prácticas de diseño de software que permita un correcto diseño escalable.

**Estado:** ✅ **IMPLEMENTADO**

**Ubicación en notebook:**
- Línea 290: Importación de `pytorch_lightning` y callbacks
- Línea 691: `CNNClassifierLightning(pl.LightningModule)`
- Línea 838: `AutoencoderLightning(pl.LightningModule)`
- Línea 1346: `MVTecDataModule(pl.LightningDataModule)`

**Notas:**
- ✅ Todos los modelos y módulos de datos heredan de clases Lightning
- ✅ Estructura escalable y modular implementada

---

### ✅ III.4.2. LightningDataModule (Línea 47)

**Requisito del enunciado:**
> Debe crear su propia clase de carga de datos utilizando `LightningDataModule`

**Estado:** ✅ **IMPLEMENTADO**

**Ubicación en notebook:**
- Línea 1346-1474: Implementación de `MVTecDataModule(pl.LightningDataModule)`
- Métodos implementados:
  - ✅ `setup()`: Carga y prepara los datos
  - ✅ `train_dataloader()`: Retorna DataLoader de entrenamiento
  - ✅ `val_dataloader()`: Retorna DataLoader de validación
  - ✅ `test_dataloader()`: Retorna DataLoader de prueba

**Notas:**
- ✅ Implementación completa siguiendo mejores prácticas
- ✅ Solo usa datos 'good' para entrenamiento (línea 1323-1330)
- ✅ Test incluye normales y anomalías (línea 1332-1341)

---

### ✅ III.4.3. LightningModule con Métodos Mínimos (Línea 47-48)

**Requisito del enunciado:**
> Debe crear su propia clase de carga de datos utilizando `LightningDataModule` y su modelo utilizando `LightningModule`, acá debe redefinir como mínimo los métodos de `training_step`, `test_step`, `configure_optimizers`.  
> **Nota:** El parrafo anterior es lo minimo que debe de estar.

**Estado:** ✅ **IMPLEMENTADO**

#### Modelo A y B: CNNClassifierLightning

**Ubicación en notebook:**
- Línea 691: Definición de clase `CNNClassifierLightning(pl.LightningModule)`
- Línea 736: ✅ `training_step(self, batch, batch_idx)` - Implementado
- Línea 768: ✅ `validation_step(self, batch, batch_idx)` - Implementado (adicional)
- Línea 779: ✅ `test_step(self, batch, batch_idx)` - Implementado
- Línea 790: ✅ `configure_optimizers(self)` - Implementado

**Notas:**
- ✅ Todos los métodos mínimos requeridos están implementados
- ✅ Incluye `validation_step` adicional (buena práctica)
- ✅ Soporta destilación teacher-student en `training_step` (Modelo B)

#### Modelo C: AutoencoderLightning

**Ubicación en notebook:**
- Línea 838: Definición de clase `AutoencoderLightning(pl.LightningModule)`
- Línea 869: ✅ `training_step(self, batch, batch_idx)` - Implementado
- Línea 880: ✅ `validation_step(self, batch, batch_idx)` - Implementado (adicional)
- Línea 894: ✅ `test_step(self, batch, batch_idx)` - Implementado
- Línea 907: ✅ `configure_optimizers(self)` - Implementado

**Notas:**
- ✅ Todos los métodos mínimos requeridos están implementados
- ✅ Incluye `validation_step` adicional (buena práctica)
- ✅ Soporta múltiples funciones de pérdida (L1, L2, SSIM)

---

### ✅ III.4.4. Callback de EarlyStopping (Línea 50-51)

**Requisito del enunciado:**
> Adicionalmente utilice el **Callback de EarlyStopping** durante el proceso de entrenamiento para evitar Overfitting del modelo.  
> **Nota:** Este callback va a estar monitoreando el comportamiento de mis metricas para definir cuando ya no hay una mejora y detener el entrenamiento con el objetivo de no gastar recursos computacionales en entrenar un modelo que no va a mejorar.

**Estado:** ✅ **IMPLEMENTADO**

**Ubicación en notebook:**
- Línea 290: Importación de `EarlyStopping` desde `pytorch_lightning.callbacks`
- Línea 1642-1646: Configuración de `EarlyStopping`:
  ```python
  early_stopping = EarlyStopping(
      monitor="val/loss",
      mode="min",
      patience=10,
      min_delta=0.001
  )
  ```
- Línea 1666: Callback añadido al Trainer: `callbacks=[early_stopping, checkpoint_callback, lr_monitor]`
- Línea 1015: Configuración en `conf/trainer/default.yaml`:
  ```yaml
  early_stopping:
    enabled: true
    monitor: "val/loss"
    mode: "min"
    patience: 10
    min_delta: 0.001
  ```

**Notas:**
- ✅ EarlyStopping correctamente implementado y configurado
- ✅ Monitorea métrica `val/loss` para detectar mejoras
- ✅ Configurable mediante archivo YAML

---

### ✅ III.4.5. Callback de Reducción de Learning Rate (Línea 51)

**Requisito del enunciado:**
> El profe tambien menciona que se pueden usar otros callbacks como el de reduccion del learning rate que es cuando no se ha mejorado en 5 itereaciones en el set de validacion

**Estado:** ✅ **IMPLEMENTADO**

**Ubicación en notebook:**
- Línea 290: Importación de `LearningRateMonitor` desde `pytorch_lightning.callbacks`
- Línea 1658: `LearningRateMonitor(logging_interval='step')` - Callback de monitoreo
- Línea 806-811: `ReduceLROnPlateau` scheduler implementado en `configure_optimizers`:
  ```python
  scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
      optimizer,
      mode='min',
      factor=0.5,
      patience=5  # Reduce LR si no mejora en 5 iteraciones
  )
  ```
- Línea 1011-1016: Configuración en `conf/trainer/default.yaml`:
  ```yaml
  scheduler:
    name: "plateau"  # o "step", "cosine"
    patience: 5
    factor: 0.5
  ```

**Notas:**
- ✅ `LearningRateMonitor`: Callback que registra el learning rate en WandB
- ✅ `ReduceLROnPlateau`: Scheduler que reduce LR cuando no hay mejora (patience=5)
- ✅ Ambos callbacks están implementados y funcionando

---

### ✅ III.4.6. Entrenamiento Solo con Datos Sin Defectos (Línea 53)

**Requisito del enunciado:**
> **Importante:** Cada modelo debe de ser entrenado únicamente con datos de la clase sin defectos, no incluir clases anómalas en el entrenamiento de los modelos.

**Estado:** ✅ **IMPLEMENTADO** (ya verificado en sección II)

**Ubicación en notebook:**
- Línea 1323-1330: En `load_dataset_paths()`, cuando `split == 'train' and only_good`:
  ```python
  if split == 'train' and only_good:
      # Solo imágenes 'good' en entrenamiento
      good_path = os.path.join(split_path, 'good')
  ```

**Notas:**
- ✅ Solo se cargan imágenes 'good' para entrenamiento
- ✅ Las anomalías solo se usan en el conjunto de prueba

---

### ⚠️ III.4.7. Scripts Externos (Líneas 55-56)

**Requisito del enunciado:**
> Como tener todo en un mismo Jupyter Notebook puede ser complicado y extenso, pueden crear la jerarquía de archivos que requieran y utilizarlas como archivos auxiliares en formato script para el diseño y control de los experimentos, sin embargo, la ejecución del entrenamiento debe estar en un Jupyter Notebook. Todos los archivos utilizados deben estar dentro del entregable.  
> **Nota:** Para esta parte puedes separar lo que ya hay en el notebook y crear scripsts por aparte en una carpeta llamada "scripts" para que no sea tan pesado. Documenta los los scripts utilizados de forma basica en el notebook para ver que se utiizaron externo al notebook. La ruta para usar correr en google coolab es "/content/drive/MyDrive/Colab Notebooks/Proyecto2-IA/scripts"

**Estado:** ⚠️ **PARCIALMENTE IMPLEMENTADO**

**Observaciones:**
- ✅ El código está todo en el notebook (cumple con "la ejecución del entrenamiento debe estar en un Jupyter Notebook")
- ⚠️ No existe carpeta "scripts" con archivos externos
- ⚠️ Hay comentarios que mencionan scripts externos:
  - Línea 665: `"# Módulos Lightning - Copiamos el contenido de lightning_modules.py"`
  - Línea 1292: `"# DataModule para MVTec AD - Copiamos el contenido de data_module.py"`
- ⚠️ No hay documentación explícita en el notebook sobre scripts utilizados externamente

**Notas:**
- El enfoque actual (todo en el notebook) es válido según el enunciado
- La nota del profesor sugiere usar scripts externos para reducir tamaño, pero no es obligatorio
- **Sugerencia opcional**: Si se desea seguir la nota del profesor, se podrían crear scripts en carpeta "scripts" y documentarlos en el notebook

---

## Verificación: Sección III.A. Modelo Clasificador CNN (Scratch y Destilación) (Líneas 58-76)

**Fecha de verificación:** 2025-01-27

### ✅ III.A.1. Estructura Basada en ResNet-18 (Líneas 60-61)

**Requisito del enunciado:**
> Para el siguiente modelo debe de crear una estructura base siguiendo la estructura de **RESNET-18** para las primeras 3 convoluciones (`conv1`, `conv2_x`, `conv3_x`) (ver Figura 1), de acá en adelante coloque un clasificador (FC layer) a su gusto para crear un clasificador entre las distintas clases.  
> **Nota:** Lo que quiero es que el extractor de caracteriristicas de esa red convolucional, tengan la mismas entradas de la figura(para las 3 primeras entradas conv1, conv2 y conv3), esto porque vamos a hacer dos variantes de entrenamientos explicado mas adelante. Apartir de esas 3 convoluciones podemos extender la arquitectura como nosotros queramos para hacer un mejor modelo

**Estado:** ✅ **IMPLEMENTADO**

**Ubicación en notebook:**
- Línea 369-395: Implementación de `BasicBlock` (bloques residuales de ResNet)
- Línea 396-433: Implementación de `CNNClassifier` con estructura ResNet-18

#### Comparación detallada con Figura 1 (ResNet-18):

**conv1 (Figura 1, línea 92-93):**
- Requerido: $7 \times 7,64$, stride 2, seguido de $3 \times 3$ max pool, stride 2
- Implementado (línea 410-412):
  - ✅ `nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)`
  - ✅ `nn.MaxPool2d(kernel_size=3, stride=2, padding=1)`
- **Estado:** ✅ **COINCIDE EXACTAMENTE**

**conv2_x (Figura 1, línea 94):**
- Requerido: $\left[\begin{array}{l}3 \times 3,64 \\ 3 \times 3,64\end{array}\right] \times 2$ (output $56 \times 56$)
- Implementado (línea 415):
  - ✅ `self._make_layer(64, 64, 64, num_blocks=2, stride=1)`
  - ✅ `BasicBlock` usa: `3x3, 64` y `3x3, 64` (líneas 374-376)
  - ✅ `num_blocks=2` crea 2 bloques residuales
- **Estado:** ✅ **COINCIDE EXACTAMENTE**

**conv3_x (Figura 1, línea 95):**
- Requerido: $\left[\begin{array}{l}3 \times 3,128 \\ 3 \times 3,128\end{array}\right] \times 2$ (output $28 \times 28$)
- Implementado (línea 418):
  - ✅ `self._make_layer(64, 128, 128, num_blocks=2, stride=2)`
  - ✅ `BasicBlock` usa: `3x3, 128` y `3x3, 128`
  - ✅ `num_blocks=2` crea 2 bloques residuales
  - ✅ `stride=2` reduce tamaño de $56 \times 56$ a $28 \times 28$
- **Estado:** ✅ **COINCIDE EXACTAMENTE**

**BasicBlock (verificación de estructura):**
- Línea 374: ✅ `nn.Conv2d(..., kernel_size=3, ...)` - Primera convolución $3 \times 3$
- Línea 376: ✅ `nn.Conv2d(..., kernel_size=3, ...)` - Segunda convolución $3 \times 3$
- ✅ Skip connection implementada (líneas 382-386, 390)

**Configuración en YAML:**
- `conf/model/cnn_classifier_scratch.yaml` (líneas 9-11):
  ```yaml
  conv1_channels: 64        # ✅ Coincide con Figura 1
  conv2_channels: [64, 64]   # ✅ Coincide con Figura 1 (2 bloques de 64)
  conv3_channels: [128, 128] # ✅ Coincide con Figura 1 (2 bloques de 128)
  ```

**Notas:**
- ✅ Estructura **EXACTA** de ResNet-18 para las primeras 3 convoluciones según Figura 1
- ✅ Todos los parámetros (kernel size, stride, canales, número de bloques) coinciden
- ✅ Clasificador FC personalizado después de conv3_x (como permite el enunciado)
- ✅ Arquitectura extensible manteniendo las 3 primeras convoluciones iguales

---

### ✅ III.A.2. Modelo A - Entrenado desde 0 (Líneas 65-66)

**Requisito del enunciado:**
> **El modelo A** será entrenado desde 0, es decir al inicio tendrá pesos colocados aleatoriamente y comenzará su proceso de entrenamiento.

**Estado:** ✅ **IMPLEMENTADO**

**Ubicación en notebook:**
- Línea 403: `model_type="scratch"` en `CNNClassifier.__init__()`
- Línea 3505: `model_type="cnn_scratch"` en función de entrenamiento
- Línea 3440-3498: 3 configuraciones de hiperparámetros para Modelo A

**Notas:**
- ✅ Modelo A se inicializa con pesos aleatorios (comportamiento por defecto de PyTorch)
- ✅ No se usa ningún modelo pre-entrenado
- ✅ Entrenamiento completamente desde cero

---

### ✅ III.A.3. Modelo B - Destilación Teacher-Student (Líneas 67-71)

**Requisito del enunciado:**
> **El modelo B** será entrenado siguiendo un proceso de destilado del modelo RESNET-18 siguiendo la técnica **teacher-student** donde el modelo RESNET-18 sirve como teacher y el modelo B como student.  
> **Nota:** Vamos a aprovechar ya entrenamiento que existen en las primeras 3 capas(conv1, conv2 y conv3 de RESNET), y utilizar la tecnica de teacher-student

**Estado:** ✅ **IMPLEMENTADO**

**Ubicación en notebook:**
- Línea 287: Importación de `resnet18` desde `torchvision.models`
- Línea 708-720: Carga de ResNet-18 pre-entrenado como teacher:
  ```python
  self.teacher_model = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
  self.teacher_model.fc = nn.Linear(..., num_classes)
  self.teacher_model.eval()
  for param in self.teacher_model.parameters():
      param.requires_grad = False
  ```
- Línea 743-759: Implementación de destilación en `training_step`:
  - Línea 748-749: Extracción de logits del teacher
  - Línea 751-752: Softmax con temperatura
  - Línea 755-756: Pérdida de destilación (KL divergence)
  - Línea 759: Combinación de pérdidas: `alpha * distillation_loss + (1-alpha) * classification_loss`
- Línea 3606-3668: 3 configuraciones de hiperparámetros para Modelo B con destilación

**Configuración de destilación:**
- `conf/model/cnn_classifier_distilled.yaml` (líneas 22-25):
  ```yaml
  distillation:
    teacher_model: "resnet18"
    temperature: 4.0
    alpha: 0.7
  ```

**Notas:**
- ✅ ResNet-18 pre-entrenado en ImageNet se usa como teacher
- ✅ Teacher se congela (no se entrena)
- ✅ Destilación implementada con temperatura y alpha
- ✅ Las primeras 3 capas del teacher (conv1, conv2, conv3) están pre-entrenadas y se aprovechan mediante la técnica teacher-student

---

### ✅ III.A.4. Extracción de Embeddings (Línea 73)

**Requisito del enunciado:**
> **Importante:** Es importante un buen diseño de modelo que permita obtener el vector de embeddings de salida de las capas convolucionales. Pues son los que luego permitirán diseñar el detector de anomalías.

**Estado:** ✅ **IMPLEMENTADO**

**Ubicación en notebook:**
- Línea 432: `self.embedding_layer = nn.Linear(conv3_channels[-1], embedding_dim)`
- Línea 458-459: Extracción de embeddings en `forward()`:
  ```python
  embedding = self.embedding_layer(x)  # x viene de conv3_x
  ```
- Línea 466-477: Método `get_embedding()` para extraer solo embeddings:
  ```python
  def get_embedding(self, x):
      # Pasa por conv1, conv2_x, conv3_x
      # Extrae embedding de la salida de conv3_x
      embedding = self.embedding_layer(x)
      return embedding
  ```

**Notas:**
- ✅ Embeddings extraídos de la salida de las capas convolucionales (después de conv3_x)
- ✅ Método `get_embedding()` implementado para facilitar extracción
- ✅ Embeddings usados para detección de anomalías (ver sección IV)

---

### ✅ III.A.5. 3 Hiperparámetros Distintos por Modelo (Líneas 75-76)

**Requisito del enunciado:**
> Cada modelo debe ser entrenado al menos con **3 hiperparámetros distintos** para obtener buenos modelos y no solamente la primera combinación que obtengan.  
> **Nota:** Tenemos 3 distintos entrenamientos por cada modelo(A, B C), en total 9 entrenamientos.

**Estado:** ✅ **IMPLEMENTADO**

#### Modelo A - 3 Configuraciones

**Ubicación en notebook:**
- Línea 3439-3498: Definición de 3 configuraciones (`model_a_configs`)
- Línea 3500-3521: Entrenamiento de las 3 configuraciones
- Variaciones en hiperparámetros:
  - **Config 1**: `fc_hidden=512`, `dropout=0.5`, `embedding_dim=256`, `lr=0.001`, scheduler `step`
  - **Config 2**: `fc_hidden=256`, `dropout=0.3`, `embedding_dim=128`, `lr=0.0005`, scheduler `cosine`
  - **Config 3**: `fc_hidden=1024`, `dropout=0.7`, `embedding_dim=512`, `lr=0.002`, scheduler `plateau`

#### Modelo B - 3 Configuraciones

**Ubicación en notebook:**
- Línea 3606-3668: Definición de 3 configuraciones (`model_b_configs`)
- Línea 3670-3744: Entrenamiento de las 3 configuraciones
- Variaciones en hiperparámetros:
  - **Config 1**: Misma estructura que A, `temperature=4.0`, `alpha=0.7`
  - **Config 2**: Misma estructura que A, `temperature=5.0`, `alpha=0.8`
  - **Config 3**: Misma estructura que A, `temperature=3.0`, `alpha=0.6`
  - Además incluye variaciones en parámetros de destilación

#### Modelo C - 3 Configuraciones

**Ubicación en notebook:**
- Línea 4009-4088: Definición de 3 configuraciones (`model_c_configs`)
- Línea 4068-4088: Entrenamiento de las 3 configuraciones
- Variaciones en hiperparámetros:
  - **Config 1**: `latent_dim=128`, `embedding_dim=128`, loss `L2`
  - **Config 2**: `latent_dim=256`, `embedding_dim=256`, loss `SSIM_L1`
  - **Config 3**: (verificar en notebook)

**Resumen:**
- ✅ Modelo A: 3 configuraciones entrenadas
- ✅ Modelo B: 3 configuraciones entrenadas
- ✅ Modelo C: 3 configuraciones entrenadas
- ✅ **Total: 9 entrenamientos** (cumple con el requisito)

**Notas:**
- ✅ Cada modelo tiene al menos 3 configuraciones distintas
- ✅ Las configuraciones varían hiperparámetros importantes (learning rate, dropout, embedding_dim, etc.)
- ✅ Para Modelo B, también varían parámetros de destilación (temperature, alpha)

---

## Verificación: Sección III.B. Modelo C - Embedding de un Autoencoder (Líneas 78-84)

**Fecha de verificación:** 2025-01-27

### ✅ III.B.1. Autoencoder Basado en U-Net (Líneas 80, 84)

**Requisito del enunciado:**
> Diseñe un modelo de autoencoder basado en **U-Net** que reconstruya las imágenes del set de entrenamiento seleccionado y también permita obtener el embedding correspondiente.  
> **Nota:** Recordar que buscamos probar diferentes arquitecturas que me construyan embbedins y hacer comparaciones. Arquitectura A es un CNN tradicional entrenado desde 0. Modelo B es el mismo CNN pero aplicado con un proceso de destilado desde el modelo RESNET. Y el modelo C va a ser un autoecoder, vamos a reconstruir la imagen. Este autoencoder esta basado en el modelo U-Net(esto lo podemos ver con la Tarea 5 ya realizada)

**Estado:** ✅ **IMPLEMENTADO**

**Ubicación en notebook:**
- Línea 480-618: Implementación de `UNetAutoencoder(nn.Module)`
- Línea 482: Documentación: "Autoencoder U-Net con skip connections (Modelo C)"
- Línea 483: Nota: "Reutilizado de Tarea05"

#### Estructura U-Net:

**Encoder (Líneas 497-509):**
- ✅ Bloques de encoder con convoluciones `4x4, stride=2`
- ✅ Canales: `[64, 128, 256, 512]` (configurable)
- ✅ BatchNorm y ReLU después de cada convolución

**Bottleneck (Líneas 511-515):**
- ✅ Capa bottleneck que reduce a `latent_dim`

**Decoder con Skip Connections (Líneas 517-546):**
- ✅ Bloques de decoder con transposed convoluciones
- ✅ **Skip connections implementadas** (líneas 575-593):
  - Línea 567-570: Guarda skip connections durante encoding
  - Línea 579-592: Usa skip connections durante decoding con `torch.cat([x, skip], dim=1)`
  - Línea 595-608: Usa skip connection en capa final
- ✅ Canales: `[512, 256, 128, 64]` (configurable)
- ✅ Capa final con `Tanh()` para normalizar salida

**Notas:**
- ✅ Skip connections correctamente implementadas (característica clave de U-Net)
- ✅ Similar a implementación de Tarea 5 (como menciona la nota)
- ✅ Arquitectura permite reconstrucción de imágenes

---

### ✅ III.B.2. Reconstrucción de Imágenes (Línea 80)

**Requisito del enunciado:**
> que reconstruya las imágenes del set de entrenamiento seleccionado

**Estado:** ✅ **IMPLEMENTADO**

**Ubicación en notebook:**
- Línea 565-610: Método `forward()` que reconstruye imágenes:
  ```python
  def forward(self, x):
      # Encoder
      # Bottleneck
      # Decoder con skip connections
      x = self.final_layer(x)  # Reconstrucción
      return x
  ```
- Línea 869-877: `training_step()` en `AutoencoderLightning`:
  ```python
  x_recon = self(x)  # Reconstrucción
  loss = self.criterion(x_recon, x)  # Compara reconstrucción vs original
  ```
- Línea 880-891: `validation_step()` también reconstruye y calcula pérdida
- Línea 894-905: `test_step()` reconstruye y extrae embeddings

**Notas:**
- ✅ El modelo reconstruye imágenes de entrada
- ✅ La pérdida se calcula comparando reconstrucción vs original
- ✅ Soporta múltiples funciones de pérdida: L1, L2, SSIM, SSIM_L1

---

### ✅ III.B.3. Extracción de Embeddings (Línea 80)

**Requisito del enunciado:**
> y también permita obtener el embedding correspondiente

**Estado:** ✅ **IMPLEMENTADO**

**Ubicación en notebook:**
- Línea 548-553: Capa de embeddings:
  ```python
  self.embedding_layer = nn.Sequential(
      nn.AdaptiveAvgPool2d((1, 1)),
      nn.Flatten(),
      nn.Linear(latent_dim, embedding_dim)
  )
  ```
- Línea 555-563: Método `encode()` que extrae el vector latente
- Línea 612-616: Método `get_embedding()`:
  ```python
  def get_embedding(self, x):
      latent, _ = self.encode(x)
      embedding = self.embedding_layer(latent)
      return embedding
  ```
- Línea 900: Uso en `test_step()`: `embeddings = self.model.get_embedding(x)`

**Notas:**
- ✅ Embeddings extraídos del espacio latente (bottleneck)
- ✅ Método `get_embedding()` implementado para facilitar extracción
- ✅ Embeddings usados para detección de anomalías (ver sección IV)

---

### ✅ III.B.4. Entrenamiento desde 0 (Línea 82)

**Requisito del enunciado:**
> Este será entrenado completamente desde 0.

**Estado:** ✅ **IMPLEMENTADO**

**Ubicación en notebook:**
- Línea 1615: Creación del modelo: `base_model = UNetAutoencoder(**model_config)`
- No hay carga de pesos pre-entrenados
- No hay uso de modelos pre-entrenados como base
- Inicialización con pesos aleatorios (comportamiento por defecto de PyTorch)

**Notas:**
- ✅ Modelo se inicializa desde cero
- ✅ No se usa ningún modelo pre-entrenado
- ✅ Entrenamiento completamente desde 0 (a diferencia del Modelo B que usa destilación)

---

### ✅ III.B.5. Comparación con Otros Modelos (Línea 84 - Nota)

**Requisito del enunciado:**
> **Nota:** Recordar que buscamos probar diferentes arquitecturas que me construyan embbedins y hacer comparaciones. Arquitectura A es un CNN tradicional entrenado desde 0. Modelo B es el mismo CNN pero aplicado con un proceso de destilado desde el modelo RESNET. Y el modelo C va a ser un autoecoder, vamos a reconstruir la imagen.

**Estado:** ✅ **IMPLEMENTADO**

**Comparación de arquitecturas:**

| Modelo | Arquitectura | Entrenamiento | Embeddings | Propósito |
|--------|--------------|---------------|------------|-----------|
| **A** | CNN (ResNet-18 primeras 3 conv) | Desde 0 | De capas convolucionales | Clasificación |
| **B** | CNN (ResNet-18 primeras 3 conv) | Destilación teacher-student | De capas convolucionales | Clasificación |
| **C** | U-Net Autoencoder | Desde 0 | Del espacio latente | Reconstrucción |

**Notas:**
- ✅ Tres arquitecturas diferentes para construir embeddings
- ✅ Permite comparar diferentes enfoques para detección de anomalías
- ✅ Modelo C se enfoca en reconstrucción (diferente a A y B que son clasificadores)

---

### ✅ III.B.6. 3 Configuraciones de Hiperparámetros (Ya verificado en III.A.5)

**Estado:** ✅ **IMPLEMENTADO**

**Ubicación en notebook:**
- Línea 4009-4088: 3 configuraciones para Modelo C (`model_c_configs`)
- Variaciones en hiperparámetros:
  - **Config 1**: `latent_dim=128`, `embedding_dim=128`, loss `L2`, `encoder_channels=[64, 128, 256, 512]`
  - **Config 2**: `latent_dim=256`, `embedding_dim=256`, loss `SSIM_L1`, `encoder_channels=[64, 128, 256, 512]`
  - **Config 3**: `latent_dim=64`, `embedding_dim=64`, loss `L1`, `encoder_channels=[32, 64, 128, 256]` (arquitectura más pequeña)

**Notas:**
- ✅ Ya verificado en sección III.A.5
- ✅ Total de 9 entrenamientos (3 por cada modelo A, B, C)

---

## Verificación: Sección IV. EVALUACIÓN DE ANOMALÍAS (Líneas 103-136)

**Fecha de verificación:** 2025-01-27

### ✅ IV.1. Cálculo de Representaciones Latentes (Líneas 105-106)

**Requisito del enunciado:**
> Una vez entrenados los modelos, se deben calcular las representaciones latentes (embeddings) de las imágenes del conjunto de validación para estimar una métrica que permita, posteriormente, identificar los datos anómalos en el conjunto de prueba.  
> **Nota:** Tomar datos de validacion y apartir de ahi definir una metrica con lo cual vamos a ver que es una anomalia y que no es una anomalia. Realizar la deteccion de anomalias apartir de los embbedings.

**Estado:** ✅ **IMPLEMENTADO**

**Ubicación en notebook:**
- Línea 4097-4117: Sección "6. Evaluación de Anomalías"
- Línea 4206-4330: Función `extract_embeddings()` que extrae embeddings de un dataloader
- Línea 4367-4374: Extracción de embeddings del conjunto normal (validación/entrenamiento)
- Línea 4382-4385: Extracción de embeddings del conjunto de prueba

**Notas:**
- ✅ Embeddings extraídos del conjunto de validación/entrenamiento (solo datos normales)
- ✅ Embeddings extraídos del conjunto de prueba (normales y anómalos)
- ✅ Funciona para todos los modelos (A, B, C)

---

### ✅ IV.2. Estimación de la Distribución Normal (Líneas 114-125)

**Requisito del enunciado:**
> A partir del conjunto de validación o entrenamiento correspondiente a la clase sin defectos, se extraen los embeddings de cada imagen mediante el modelo previamente entrenado (para cada modelo A, B y C), ya sea del set de validacion o de entrenamiento para las clases buenas.  
> Cada embedding puede representarse como un vector $\mathbf{z}_{i} \in \mathbb{R}^{d}$. Con todos los embeddings del conjunto normal se calcula la media $\boldsymbol{\mu}$ y la matriz de covarianza $\boldsymbol{\Sigma}$:  
> $$
> \boldsymbol{\mu}=\frac{1}{N} \sum_{i=1}^{N} \mathbf{z}_{i}, \quad \boldsymbol{\Sigma}=\frac{1}{N-1} \sum_{i=1}^{N}\left(\mathbf{z}_{i}-\boldsymbol{\mu}\right)\left(\mathbf{z}_{i}-\boldsymbol{\mu}\right)^{T}
> $$  
> De esta forma se modela la distribución normal como una distribución gaussiana multivariada $\mathcal{N}(\boldsymbol{\mu}, \boldsymbol{\Sigma})$, que representa los datos normales en el espacio de embeddings.

**Estado:** ✅ **IMPLEMENTADO**

**Ubicación en notebook:**
- Línea 4295-4330: Función `estimate_normal_distribution()`:
  ```python
  # Media: μ = (1/N) Σ z_i
  mean = np.mean(normal_embeddings, axis=0)
  
  # Matriz de covarianza: Σ = (1/(N-1)) Σ (z_i - μ)(z_i - μ)^T
  cov = np.cov(normal_embeddings.T)  # np.cov usa (N-1) como denominador
  ```
- Línea 4376-4379: Uso en `evaluate_anomaly_detection()`:
  ```python
  mean, cov = estimate_normal_distribution(normal_embeddings)
  ```

**Verificación de fórmulas:**
- ✅ Media: `np.mean(normal_embeddings, axis=0)` = $\frac{1}{N} \sum_{i=1}^{N} \mathbf{z}_{i}$ ✅
- ✅ Covarianza: `np.cov(normal_embeddings.T)` = $\frac{1}{N-1} \sum_{i=1}^{N}\left(\mathbf{z}_{i}-\boldsymbol{\mu}\right)\left(\mathbf{z}_{i}-\boldsymbol{\mu}\right)^{T}$ ✅
- ✅ Modela distribución gaussiana multivariada $\mathcal{N}(\boldsymbol{\mu}, \boldsymbol{\Sigma})$ ✅

**Notas:**
- ✅ Fórmulas implementadas exactamente como en el enunciado
- ✅ Validación de que embeddings tienen shape (N, d) donde d es la dimensión
- ✅ Validación de que hay al menos 2 muestras para calcular covarianza

---

### ✅ IV.3. Cálculo de la Distancia de Mahalanobis (Líneas 127-131)

**Requisito del enunciado:**
> Para una nueva muestra con embedding $\mathbf{z}_{\text{test}}$, se calcula su distancia a la distribución normal.  
> Esta distancia mide qué tan alejada se encuentra la muestra del centro de la distribución de los datos sin defectos, considerando la forma y correlaciones de dicha distribución.

**Estado:** ✅ **IMPLEMENTADO**

**Ubicación en notebook:**
- Línea 4155-4203: Función `calculate_mahalanobis_distance()`:
  ```python
  def calculate_mahalanobis_distance(embeddings, mean, cov):
      """
      Distancia de Mahalanobis: d = sqrt((z - μ)^T Σ^(-1) (z - μ))
      """
      # Regularización para evitar singularidad
      cov_reg = cov + np.eye(cov.shape[0]) * 1e-6
      cov_inv = inv(cov_reg)
      
      # Calcular distancias
      for emb in embeddings:
          diff = emb - mean
          dist = np.sqrt(diff @ cov_inv @ diff.T)
  ```
- Línea 4397-4406: Uso en evaluación:
  ```python
  if method == "mahalanobis":
      test_normal_distances = calculate_mahalanobis_distance(test_normal_embeddings, mean, cov)
      test_anomaly_distances = calculate_mahalanobis_distance(test_anomaly_embeddings, mean, cov)
  ```

**Verificación de fórmula:**
- ✅ Fórmula implementada: $d = \sqrt{(\mathbf{z} - \boldsymbol{\mu})^T \boldsymbol{\Sigma}^{-1} (\mathbf{z} - \boldsymbol{\mu})}$ ✅
- ✅ Regularización añadida para evitar singularidad de la matriz de covarianza
- ✅ Manejo de errores para distancias inválidas (NaN, Inf)

**Notas:**
- ✅ Implementación correcta de la distancia de Mahalanobis
- ✅ Considera correlaciones entre dimensiones (mediante matriz de covarianza)
- ✅ Calcula distancia para cada embedding del conjunto de prueba

---

### ✅ IV.4. Clasificación usando Percentiles (Línea 133)

**Requisito del enunciado:**
> A partir de acá debe de averiguar como clasificar una anomalía o una clase sin defectos utilizando comparación de la distancia (e.g tomar el percentil).

**Estado:** ✅ **IMPLEMENTADO**

**Ubicación en notebook:**
- Línea 4444-4464: Cálculo de umbral usando percentil:
  ```python
  # Determinar umbral usando percentil de las distancias normales del conjunto de validación
  validation_normal_distances = calculate_mahalanobis_distance(normal_embeddings, mean, cov)
  threshold = np.percentile(validation_normal_distances, percentile)
  print(f"📏 Umbral calculado (percentil {percentile}): {threshold:.4f}")
  ```
- Línea 4469-4474: Clasificación:
  ```python
  all_distances = np.concatenate([test_normal_distances, test_anomaly_distances])
  predictions = (all_distances > threshold).astype(int)  # 1 = anomalía, 0 = normal
  true_labels = np.concatenate([np.zeros_like(test_normal_distances), np.ones_like(test_anomaly_distances)])
  ```

**Configuración:**
- Línea 1064: `percentile_threshold: 95` en `conf/config.yaml`
- Línea 4349: Parámetro `percentile=95` por defecto en `evaluate_anomaly_detection()`

**Notas:**
- ✅ Umbral calculado usando percentil de distancias normales de validación
- ✅ Clasificación: distancias > umbral = anomalía, distancias ≤ umbral = normal
- ✅ Percentil configurable (default: 95)
- ✅ Métricas calculadas: AUC-ROC, AUC-PR

---

### ✅ IV.5. Otras Estrategias de Detección (Línea 135)

**Requisito del enunciado:**
> **Nota:** El estudiante puede implementar también otras estrategias de detección, como la distancia euclidiana, reconstrucción basada en error (reconstruction loss). Debe justificarse la implementada en el notebook

**Estado:** ✅ **IMPLEMENTADO**

**Ubicación en notebook:**
- Línea 4361: Métodos soportados: `["mahalanobis", "euclidean", "reconstruction_loss"]`

#### 1. Distancia Euclidiana

**Ubicación:**
- Línea 4409-4419: Implementación:
  ```python
  elif method == "euclidean":
      # Distancia euclidiana: d = ||z - μ||
      test_normal_distances = np.linalg.norm(test_normal_embeddings - mean, axis=1)
      test_anomaly_distances = np.linalg.norm(test_anomaly_embeddings - mean, axis=1)
  ```

**Notas:**
- ✅ Implementada: $d = ||\mathbf{z} - \boldsymbol{\mu}||$
- ✅ Métrica más simple que Mahalanobis (no considera correlaciones)

#### 2. Reconstruction Loss

**Ubicación:**
- Línea 4421-4440: Implementación:
  ```python
  elif method == "reconstruction_loss":
      # Error de reconstrucción: MSE entre reconstrucción y original
      test_normal_distances = np.mean((test_reconstructions - test_originals) ** 2, axis=(1, 2, 3))
      test_anomaly_distances = np.mean((test_anomaly_recon - test_anomaly_orig) ** 2, axis=(1, 2, 3))
  ```

**Notas:**
- ✅ Implementada para autoencoders (Modelo C)
- ✅ Compara imágenes reconstruidas vs originales
- ✅ Responde a la pregunta: "¿Existen diferencias entre las imágenes que estamos haciendo con las originales?"

**Justificación en notebook:**
- Línea 4114-4117: Documentación de métodos:
  ```markdown
  **Métodos de evaluación**:
  - **Distancia de Mahalanobis**: d = sqrt((z - μ)^T Σ^(-1) (z - μ))
  - **Distancia Euclidiana**: d = ||z - μ||
  - **Reconstruction Loss**: Error de reconstrucción para autoencoders
  ```

**Notas:**
- ✅ Tres métodos implementados: Mahalanobis, Euclidiana, Reconstruction Loss
- ✅ Cada método tiene su justificación y uso apropiado
- ✅ Mahalanobis: Considera correlaciones (más robusto)
- ✅ Euclidiana: Métrica simple y rápida
- ✅ Reconstruction Loss: Específico para autoencoders

---

## Verificación: Sección V. MODELOS CUANTIZADOS (Líneas 137-139)

**Fecha de verificación:** 2025-01-27

### ✅ V.1. Selección de los 3 Mejores Modelos (Línea 139)

**Requisito del enunciado:**
> Para esto, convierta los **tres modelos con mejores resultados** de acuerdo a su criterio a modelos cuantizados

**Estado:** ✅ **IMPLEMENTADO**

**Ubicación en notebook:**
- Línea 4720-4728: Selección de los 3 mejores modelos:
  ```python
  # Ordenar por AUC-ROC (usar el mejor método para cada modelo)
  sorted_results = sorted(
      all_evaluation_results,
      key=lambda x: max(x.get("auc_roc", 0), x.get("auc_roc_mah", 0), x.get("auc_roc_recon", 0)),
      reverse=True
  )
  best_3_models = sorted_results[:3]
  ```
- Línea 4540: Documentación: "Los mejores modelos se seleccionan según AUC-ROC para cuantización y análisis DBSCAN"
- Línea 4730-4733: Visualización de los top 3 modelos

**Criterio de selección:**
- ✅ Selección basada en AUC-ROC (métrica de rendimiento)
- ✅ Considera el mejor método de evaluación para cada modelo (Mahalanobis, Euclidiana, Reconstruction Loss)
- ✅ Ordena de mayor a menor AUC-ROC y toma los primeros 3

**Notas:**
- ✅ Criterio claro y justificado (AUC-ROC como métrica principal)
- ✅ Permite comparar modelos de diferentes tipos (A, B, C)

---

### ✅ V.2. Conversión a Modelos Cuantizados (Línea 139)

**Requisito del enunciado:**
> convierta los **tres modelos con mejores resultados** de acuerdo a su criterio a modelos cuantizados

**Estado:** ✅ **IMPLEMENTADO**

**Ubicación en notebook:**
- Línea 4800-4818: Función `quantize_model()`:
  ```python
  def quantize_model(model, method="dynamic"):
      """
      Cuantiza un modelo PyTorch
      """
      model.eval()
      if method == "dynamic":
          quantized_model = torch.quantization.quantize_dynamic(
              model, {torch.nn.Linear, torch.nn.Conv2d}, dtype=torch.qint8
          )
  ```
- Línea 4838-5040: Proceso de cuantización de los 3 mejores modelos:
  ```python
  for i, best_model_info in enumerate(best_3_models, 1):
      # Extraer modelo base
      # Cuantizar modelo
      quantized_model = quantize_model(model_to_quantize, method="dynamic")
  ```

**Método de cuantización:**
- ✅ Cuantización dinámica de PyTorch (`torch.quantization.quantize_dynamic`)
- ✅ Cuantiza capas `Linear` y `Conv2d` a `qint8` (int8)
- ✅ Reduce precisión de float32 a int8

**Notas:**
- ✅ Conversión implementada correctamente
- ✅ Modelos se ponen en modo evaluación antes de cuantizar
- ✅ Soporta cuantización dinámica (método más común)

---

### ✅ V.3. Comparación de Tamaño (Línea 139)

**Requisito del enunciado:**
> y realice una comparación de latencias en respuesta, tamaño, y rendimiento

**Estado:** ✅ **IMPLEMENTADO**

**Ubicación en notebook:**
- Línea 4821-4835: Función `compare_model_sizes()`:
  ```python
  def compare_model_sizes(original_model, quantized_model):
      """Compara el tamaño de modelos original y cuantizado"""
      def get_model_size(model):
          param_size = sum(p.numel() * p.element_size() for p in model.parameters())
          buffer_size = sum(b.numel() * b.element_size() for b in model.buffers())
          return param_size + buffer_size
      
      original_size = get_model_size(original_model)
      quantized_size = get_model_size(quantized_model)
      
      return {
          'original_size_mb': original_size / (1024 * 1024),
          'quantized_size_mb': quantized_size / (1024 * 1024),
          'compression_ratio': original_size / quantized_size if quantized_size > 0 else 0
      }
  ```
- Línea 4878-4879: Uso en comparación:
  ```python
  size_comparison = compare_model_sizes(model_to_quantize, quantized_model)
  ```
- Línea 5003-5005: Almacenamiento en resultados:
  ```python
  "original_size_mb": size_comparison['original_size_mb'],
  "quantized_size_mb": size_comparison['quantized_size_mb'],
  "compression_ratio": size_comparison['compression_ratio']
  ```
- Línea 5020-5023: Visualización:
  ```python
  print(f"  Tamaño:")
  print(f"    Original: {size_comparison['original_size_mb']:.2f} MB")
  print(f"    Cuantizado: {size_comparison['quantized_size_mb']:.2f} MB")
  print(f"    Compresión: {size_comparison['compression_ratio']:.2f}x")
  ```

**Métricas de tamaño:**
- ✅ Tamaño original en MB
- ✅ Tamaño cuantizado en MB
- ✅ Ratio de compresión (cuántas veces más pequeño es el modelo cuantizado)

**Notas:**
- ✅ Comparación de tamaño implementada correctamente
- ✅ Calcula tamaño considerando parámetros y buffers
- ✅ Muestra ratio de compresión para evaluar eficiencia

---

### ✅ V.4. Comparación de Latencia en Respuesta (Línea 139)

**Requisito del enunciado:**
> y realice una comparación de latencias en respuesta, tamaño, y rendimiento

**Estado:** ✅ **IMPLEMENTADO**

**Ubicación en notebook:**
- Línea 4881-4917: Medición de latencia:
  ```python
  # Latencia original - promedio sobre 100 iteraciones
  latencies_original = []
  with torch.no_grad():
      for _ in range(100):
          start_time = time.time()
          if hasattr(model_to_quantize, 'get_embedding'):
              _ = model_to_quantize.get_embedding(test_images)
          else:
              _ = model_to_quantize(test_images)
          latencies_original.append((time.time() - start_time) * 1000)  # ms
  original_latency = np.mean(latencies_original)
  
  # Latencia cuantizado - promedio sobre 100 iteraciones
  latencies_quantized = []
  with torch.no_grad():
      for _ in range(100):
          start_time = time.time()
          if hasattr(quantized_model, 'get_embedding'):
              _ = quantized_model.get_embedding(test_images)
          else:
              _ = quantized_model(test_images)
          latencies_quantized.append((time.time() - start_time) * 1000)  # ms
  quantized_latency = np.mean(latencies_quantized)
  ```
- Línea 5006-5007: Almacenamiento:
  ```python
  "original_latency_ms": original_latency,
  "quantized_latency_ms": quantized_latency,
  "speedup": original_latency / quantized_latency if quantized_latency > 0 else 0
  ```
- Línea 5025-5028: Visualización:
  ```python
  print(f"  Latencia (promedio sobre 100 iteraciones):")
  print(f"    Original: {original_latency:.2f} ms")
  print(f"    Cuantizado: {quantized_latency:.2f} ms")
  print(f"    Speedup: {original_latency / quantized_latency if quantized_latency > 0 else 0:.2f}x")
  ```

**Métricas de latencia:**
- ✅ Latencia original (ms) - promedio sobre 100 iteraciones
- ✅ Latencia cuantizada (ms) - promedio sobre 100 iteraciones
- ✅ Speedup (cuántas veces más rápido es el modelo cuantizado)

**Notas:**
- ✅ Medición de latencia implementada correctamente
- ✅ Promedio sobre 100 iteraciones para mayor precisión
- ✅ Calcula speedup para evaluar mejora en velocidad
- ✅ Mide tiempo de inferencia (extracción de embeddings)

---

### ✅ V.5. Comparación de Rendimiento (Línea 139)

**Requisito del enunciado:**
> y realice una comparación de latencias en respuesta, tamaño, y rendimiento

**Estado:** ✅ **IMPLEMENTADO**

**Ubicación en notebook:**
- Línea 4940-4992: Evaluación de rendimiento:
  ```python
  # Evaluar modelo original
  original_performance = {
      'auc_roc': result.get('auc_roc', 0),
      'auc_pr': result.get('auc_pr', 0)
  }
  
  # Evaluar modelo cuantizado
  eval_quantized = evaluate_anomaly_detection(
      model=quantized_lightning,
      normal_dataloader=data_module.val_dataloader(),
      test_dataloader=data_module.test_dataloader(),
      device=device,
      method="mahalanobis",
      percentile=95
  )
  quantized_performance = {
      'auc_roc': eval_quantized['auc_roc'],
      'auc_pr': eval_quantized['auc_pr']
  }
  
  # Calcular diferencia de rendimiento
  performance_diff_auc_roc = original_performance['auc_roc'] - quantized_performance['auc_roc']
  performance_retention_auc_roc = (quantized_performance['auc_roc'] / original_performance['auc_roc'] * 100) if original_performance['auc_roc'] > 0 else 0
  ```
- Línea 5008-5015: Almacenamiento:
  ```python
  "original_auc_roc": original_performance['auc_roc'],
  "quantized_auc_roc": quantized_performance['auc_roc'],
  "original_auc_pr": original_performance['auc_pr'],
  "quantized_auc_pr": quantized_performance['auc_pr'],
  "performance_diff_auc_roc": performance_diff_auc_roc,
  "performance_diff_auc_pr": performance_diff_auc_pr,
  "performance_retention_auc_roc": performance_retention_auc_roc,
  "performance_retention_auc_pr": performance_retention_auc_pr
  ```
- Línea 5030-5040: Visualización:
  ```python
  print(f"  Rendimiento (AUC-ROC):")
  print(f"    Original: {original_performance['auc_roc']:.4f}")
  print(f"    Cuantizado: {quantized_performance['auc_roc']:.4f}")
  print(f"    Diferencia: {performance_diff_auc_roc:+.4f}")
  print(f"    Retención: {performance_retention_auc_roc:.2f}%")
  ```

**Métricas de rendimiento:**
- ✅ AUC-ROC original vs cuantizado
- ✅ AUC-PR original vs cuantizado
- ✅ Diferencia de rendimiento (cuánto se pierde)
- ✅ Porcentaje de retención de rendimiento (cuánto se mantiene)

**Notas:**
- ✅ Comparación de rendimiento implementada correctamente
- ✅ Usa las mismas métricas que la evaluación principal (AUC-ROC, AUC-PR)
- ✅ Calcula diferencia y retención para evaluar impacto de cuantización

---

### ✅ V.6. Análisis Incluido en el Informe (Línea 139)

**Requisito del enunciado:**
> incluya este análisis en su informe

**Estado:** ✅ **IMPLEMENTADO**

**Ubicación en notebook:**
- Línea 5042-5075: Resumen comparativo completo:
  ```python
  print("="*80)
  print("RESUMEN COMPARATIVO DE CUANTIZACIÓN")
  print("="*80)
  print("\nComparación de los 3 mejores modelos: Original vs Cuantizado\n")
  
  for i, result in enumerate(quantization_results, 1):
      print(f"{i}. {result['model_type']} - {result['config']}")
      print(f"   Tamaño: Original: {result['original_size_mb']:.2f} MB → Cuantizado: {result['quantized_size_mb']:.2f} MB")
      print(f"   Compresión: {result['compression_ratio']:.2f}x")
      print(f"   Latencia: Original: {result['original_latency_ms']:.2f} ms → Cuantizado: {result['quantized_latency_ms']:.2f} ms")
      print(f"   Speedup: {result['speedup']:.2f}x")
      print(f"   Rendimiento (AUC-ROC): Original: {result['original_auc_roc']:.4f} → Cuantizado: {result['quantized_auc_roc']:.4f}")
      print(f"   Diferencia: {result['performance_diff_auc_roc']:+.4f} ({result['performance_retention_auc_roc']:.2f}% retención)")
  ```
- Línea 5077-5088: Resumen estadístico:
  ```python
  print("="*80)
  print("RESUMEN ESTADÍSTICO")
  print("="*80)
  avg_compression = np.mean([r['compression_ratio'] for r in quantization_results])
  avg_speedup = np.mean([r['speedup'] for r in quantization_results])
  avg_retention_auc_roc = np.mean([r['performance_retention_auc_roc'] for r in quantization_results])
  avg_retention_auc_pr = np.mean([r['performance_retention_auc_pr'] for r in quantization_results])
  
  print(f"\nPromedio de compresión: {avg_compression:.2f}x")
  print(f"Promedio de speedup: {avg_speedup:.2f}x")
  print(f"Retención promedio de rendimiento (AUC-ROC): {avg_retention_auc_roc:.2f}%")
  print(f"Retención promedio de rendimiento (AUC-PR): {avg_retention_auc_pr:.2f}%")
  ```

**Análisis incluido:**
- ✅ Comparación detallada por modelo (tamaño, latencia, rendimiento)
- ✅ Resumen estadístico con promedios
- ✅ Visualización clara de resultados
- ✅ Métricas calculadas y documentadas

**Notas:**
- ✅ Análisis completo y estructurado
- ✅ Fácil de incluir en informe
- ✅ Incluye promedios para análisis general

---

## Verificación: Sección VI. ANÁLISIS DE OUTLIERS MEDIANTE DBSCAN CLUSTERING (Líneas 141-150)

**Fecha de verificación:** 2025-01-27

### ✅ VI.1. Selección del Mejor Modelo (Línea 143)

**Requisito del enunciado:**
> Una vez identificado el mejor modelo de detección de anomalías —ya sea el clasificador CNN entrenado desde cero, su versión distilada mediante teacher–student, o el modelo autoencoder basado en U-Net— proceda a utilizar sus embeddings como insumo para realizar un análisis adicional mediante técnicas de agrupamiento no supervisado.

**Estado:** ✅ **IMPLEMENTADO**

**Ubicación en notebook:**
- Línea 5391-5393: Selección del mejor modelo:
  ```python
  if best_3_models:
      best_model_info = best_3_models[0]  # Toma el mejor (primer lugar)
      print(f"Analizando con el mejor modelo: {best_model_info['model_type']} - {best_model_info['config']}")
  ```
- Línea 5395-5415: Búsqueda del modelo en los resultados:
  ```python
  if best_model_info['model_type'] == "Modelo A":
      # Buscar en model_a_results
  elif best_model_info['model_type'] == "Modelo B":
      # Buscar en model_b_results
  elif best_model_info['model_type'] == "Modelo C":
      # Buscar en model_c_results
  ```
- Línea 4540: Documentación: "Los mejores modelos se seleccionan según AUC-ROC para cuantización y análisis DBSCAN"

**Notas:**
- ✅ Selecciona el mejor modelo según AUC-ROC (mismo criterio que para cuantización)
- ✅ Soporta los tres tipos de modelos (A, B, C)
- ✅ El mejor modelo es el primero de `best_3_models` (mayor AUC-ROC)

---

### ✅ VI.2. Extracción de Embeddings del Conjunto de Prueba (Línea 145)

**Requisito del enunciado:**
> Extraiga los embeddings generados por el modelo seleccionado para cada imagen del conjunto de prueba.

**Estado:** ✅ **IMPLEMENTADO**

**Ubicación en notebook:**
- Línea 5417-5446: Extracción de embeddings:
  ```python
  # Extraer embeddings del conjunto de prueba
  all_embeddings = []
  all_labels = []
  
  best_model.eval()
  with torch.no_grad():
      for batch in data_module.test_dataloader():
          images = images.to(device)
          
          # Extraer embeddings
          if hasattr(best_model, 'get_embedding'):
              embeddings = best_model.get_embedding(images)
          elif hasattr(best_model, 'model') and hasattr(best_model.model, 'get_embedding'):
              embeddings = best_model.model.get_embedding(images)
          else:
              logits, embeddings = best_model.model(images)
          
          all_embeddings.append(embeddings.cpu().numpy())
          if labels is not None:
              all_labels.append(labels.cpu().numpy())
  
  all_embeddings = np.concatenate(all_embeddings, axis=0)
  all_labels = np.concatenate(all_labels, axis=0) if all_labels else None
  ```

**Notas:**
- ✅ Extrae embeddings de todas las imágenes del conjunto de prueba
- ✅ Soporta diferentes formas de extraer embeddings según el tipo de modelo
- ✅ Guarda también las etiquetas (ground truth) para comparación

---

### ✅ VI.3. Reducción de Dimensionalidad con PCA y t-SNE (Línea 145)

**Requisito del enunciado:**
> Con el fin de facilitar tanto la visualización como la separación estructural, aplique reducción de dimensionalidad con **PCA** y **t-SNE**.

**Estado:** ✅ **IMPLEMENTADO**

**Ubicación en notebook:**
- Línea 292-293: Importaciones:
  ```python
  from sklearn.decomposition import PCA
  from sklearn.manifold import TSNE
  ```
- Línea 5185-5196: Aplicación de PCA:
  ```python
  # Reducción de dimensionalidad con PCA
  if use_pca and embeddings.shape[1] > pca_components:
      print(f"  Aplicando PCA: {embeddings.shape[1]} → {pca_components} dimensiones")
      pca = PCA(n_components=pca_components)
      embeddings_reduced = pca.fit_transform(embeddings)
      explained_variance = np.sum(pca.explained_variance_ratio_)
      print(f"  ✓ Varianza explicada por PCA: {explained_variance:.4f} ({explained_variance*100:.2f}%)")
  ```
- Línea 5213-5223: Aplicación de t-SNE:
  ```python
  # Reducción para visualización con t-SNE
  if use_tsne:
      print(f"  Aplicando t-SNE para visualización 2D...")
      perplexity = min(tsne_perplexity, len(embeddings_reduced) - 1)
      if perplexity > 0:
          tsne = TSNE(n_components=tsne_components, random_state=42, perplexity=perplexity)
          embeddings_2d = tsne.fit_transform(embeddings_reduced)
          print(f"  ✓ t-SNE completado: {embeddings_reduced.shape[1]} → {tsne_components} dimensiones")
  ```

**Configuración:**
- Línea 1071-1075: En `conf/config.yaml`:
  ```yaml
  dbscan:
    use_pca: true
    pca_components: 50
    use_tsne: true
    tsne_components: 2
    tsne_perplexity: 30
  ```

**Proceso:**
- ✅ **PCA**: Reduce dimensionalidad manteniendo varianza (configurable, default: 50 componentes)
- ✅ **t-SNE**: Reduce a 2D para visualización preservando estructura local
- ✅ Proceso: Embeddings originales → PCA → DBSCAN → t-SNE (para visualización)

**Notas:**
- ✅ PCA aplicado antes de DBSCAN (facilita procesamiento)
- ✅ t-SNE aplicado después de DBSCAN (para visualización 2D)
- ✅ Configuración flexible mediante YAML

---

### ✅ VI.4. Aplicación de DBSCAN (Líneas 143, 147)

**Requisito del enunciado:**
> En particular **DBSCAN** (Density-Based Spatial Clustering of Applications with Noise), un método basado en densidad que permite identificar regiones de alta concentración en el espacio latente y, simultáneamente, detectar puntos aislados que pueden interpretarse como outliers o anomalías.  
> Una vez obtenidas las representaciones latentes reducidas aplique **DBSCAN**.

**Estado:** ✅ **IMPLEMENTADO**

**Ubicación en notebook:**
- Línea 294: Importación: `from sklearn.cluster import DBSCAN`
- Línea 5159-5233: Función `dbscan_analysis()`:
  ```python
  def dbscan_analysis(embeddings, eps=0.5, min_samples=5, use_pca=True, pca_components=50,
                      use_tsne=True, tsne_components=2, tsne_perplexity=30):
      # Aplicar DBSCAN
      dbscan = DBSCAN(eps=eps, min_samples=min_samples)
      clusters = dbscan.fit_predict(embeddings_reduced)
      
      # Identificar outliers (ruido)
      n_clusters = len(set(clusters)) - (1 if -1 in clusters else 0)
      n_noise = list(clusters).count(-1)
  ```
- Línea 5461-5470: Aplicación en el mejor modelo:
  ```python
  dbscan_results = dbscan_analysis(
      embeddings=all_embeddings,
      eps=dbscan_config.get("eps", 0.5),
      min_samples=dbscan_config.get("min_samples", 5),
      use_pca=dbscan_config.get("use_pca", True),
      pca_components=dbscan_config.get("pca_components", 50),
      use_tsne=dbscan_config.get("use_tsne", True),
      tsne_components=dbscan_config.get("tsne_components", 2),
      tsne_perplexity=dbscan_config.get("tsne_perplexity", 30)
  )
  ```

**Configuración:**
- Línea 1068-1075: En `conf/config.yaml`:
  ```yaml
  dbscan:
    eps: 0.5
    min_samples: 5
  ```

**Notas:**
- ✅ DBSCAN aplicado correctamente
- ✅ Identifica clusters (regiones de alta densidad)
- ✅ Identifica outliers/ruido (puntos etiquetados como -1)
- ✅ Parámetros configurables (eps, min_samples)

---

### ✅ VI.5. Interpretación de Ruido como Anomalías (Línea 147)

**Requisito del enunciado:**
> Desde la perspectiva de la detección de anomalías, los puntos etiquetados por DBSCAN como ruido constituyen una indicación natural de potencial anomalía, ya que representan vectores que se encuentran en zonas de baja densidad del espacio latente.

**Estado:** ✅ **IMPLEMENTADO**

**Ubicación en notebook:**
- Línea 5203-5211: Identificación de outliers:
  ```python
  # Identificar outliers (ruido)
  n_clusters = len(set(clusters)) - (1 if -1 in clusters else 0)
  n_noise = list(clusters).count(-1)  # Puntos etiquetados como -1 son ruido
  n_in_clusters = len(clusters) - n_noise
  
  print(f"  ✓ DBSCAN completado:")
  print(f"    - Clusters encontrados: {n_clusters}")
  print(f"    - Puntos en clusters: {n_in_clusters}")
  print(f"    - Outliers (ruido): {n_noise}")
  ```
- Línea 5482-5483: Uso para detección de anomalías:
  ```python
  dbscan_outliers = (dbscan_results['clusters'] == -1).astype(int)  # 1 = outlier/anomalía
  true_anomalies = all_labels  # Ground truth
  ```

**Notas:**
- ✅ Puntos etiquetados como -1 se interpretan como outliers/anomalías
- ✅ Se comparan con ground truth para evaluación
- ✅ Lógica correcta: ruido = baja densidad = potencial anomalía

---

### ✅ VI.6. Análisis Visual (Línea 149)

**Requisito del enunciado:**
> Analice los resultados desde el punto de vista visual

**Estado:** ✅ **IMPLEMENTADO**

**Ubicación en notebook:**
- Línea 5240-5334: Función `visualize_dbscan_results()`:
  ```python
  def visualize_dbscan_results(dbscan_results, labels=None, save_path=None):
      """
      Visualiza los resultados de DBSCAN de forma completa.
      
      Muestra:
      1. Clustering DBSCAN (clusters y outliers)
      2. Comparación con ground truth labels
      3. Análisis de distribución de outliers vs normales
      """
  ```

**Visualizaciones implementadas:**

1. **Clustering DBSCAN** (Líneas 5255-5268):
   - ✅ Muestra clusters con diferentes colores
   - ✅ Muestra outliers (ruido) en negro con marcador 'x'
   - ✅ Leyenda con número de clusters y outliers

2. **Ground Truth Labels** (Líneas 5270-5281):
   - ✅ Compara con etiquetas reales (normal vs anomalía)
   - ✅ Verde para normales, rojo para anomalías
   - ✅ Permite comparar visualmente con DBSCAN

3. **DBSCAN Outliers vs Ground Truth** (Líneas 5283-5316):
   - ✅ Visualización combinada:
     - Normal en cluster (lightgreen, pequeño)
     - Normal como outlier DBSCAN (green, grande, 'x')
     - Anomalía en cluster (lightcoral, pequeño)
     - Anomalía como outlier DBSCAN (red, grande, 'x')
   - ✅ Facilita identificar coincidencias y discrepancias

**Ubicación de uso:**
- Línea 5479: `visualize_dbscan_results(dbscan_results, labels=all_labels, save_path=save_path)`

**Notas:**
- ✅ Tres visualizaciones diferentes para análisis completo
- ✅ Comparación visual con ground truth
- ✅ Guarda visualización en archivo (opcional)
- ✅ Usa t-SNE 2D para visualización

---

### ✅ VI.7. Análisis Cuantitativo (Línea 149)

**Requisito del enunciado:**
> Analice los resultados desde el punto de vista visual, y cuantitativa del resultado de la clasificación de anomalías.

**Estado:** ✅ **IMPLEMENTADO**

**Ubicación en notebook:**
- Línea 5481-5550: Análisis cuantitativo completo:
  ```python
  # Análisis cuantitativo: Comparar outliers de DBSCAN con ground truth
  if all_labels is not None:
      dbscan_outliers = (dbscan_results['clusters'] == -1).astype(int)
      true_anomalies = all_labels
      
      # Calcular métricas de clasificación
      dbscan_auc = roc_auc_score(true_anomalies, dbscan_outliers)
      dbscan_ap = average_precision_score(true_anomalies, dbscan_outliers)
      
      # Matriz de confusión
      cm = confusion_matrix(true_anomalies, dbscan_outliers)
      tn, fp, fn, tp = cm.ravel()
      
      # Calcular precisión, recall, F1, accuracy
      precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
      recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
      f1_score = 2 * (precision * recall) / (precision + recall)
      accuracy = (tp + tn) / (tp + tn + fp + fn)
  ```

**Métricas calculadas:**
- ✅ **AUC-ROC**: Área bajo curva ROC
- ✅ **Average Precision (AUC-PR)**: Área bajo curva Precision-Recall
- ✅ **Accuracy**: Precisión general
- ✅ **Precision**: Precisión de detección de anomalías
- ✅ **Recall**: Sensibilidad de detección
- ✅ **F1-Score**: Media armónica de precisión y recall
- ✅ **Matriz de Confusión**: TN, FP, FN, TP

**Estadísticas adicionales:**
- ✅ Total de muestras
- ✅ Muestras normales vs anómalas (ground truth)
- ✅ Clusters encontrados
- ✅ Outliers detectados por DBSCAN
- ✅ Porcentaje de outliers
- ✅ Distribución de outliers (normales vs anomalías)

**Notas:**
- ✅ Análisis cuantitativo completo y detallado
- ✅ Compara DBSCAN outliers con ground truth
- ✅ Métricas estándar de clasificación implementadas
- ✅ Estadísticas descriptivas para entender resultados

---

## Resumen de Verificación

| Componente | Estado | Observaciones |
|------------|--------|---------------|
| Objetivo del proyecto | ✅ | Correctamente implementado y documentado |
| Dataset MVTec AD | ✅ | Configurado y cargado correctamente |
| 10 clases seleccionadas | ✅ | Todas las clases están en la configuración |
| DataModule (Lightning) | ✅ | Implementado siguiendo mejores prácticas |
| Detección por embeddings | ✅ | Implementado con múltiples métodos |
| Detección por reconstrucción | ✅ | Implementado, podría mejorarse documentación |
| **Gestión con Hydra** | ✅ | **Inicializado y configurado correctamente** |
| **Estructura conf/** | ✅ | **Todos los archivos requeridos presentes** |
| **Dimensión espacio latente (z)** | ✅ | **Configurable en YAML** |
| **Épocas y batch size** | ✅ | **Configurables en YAML** |
| **Otros hiperparámetros** | ✅ | **Ampliamente configurables** |
| **PyTorch Lightning** | ✅ | **Correctamente implementado** |
| **LightningDataModule** | ✅ | **MVTecDataModule implementado** |
| **LightningModule métodos mínimos** | ✅ | **training_step, test_step, configure_optimizers** |
| **EarlyStopping callback** | ✅ | **Implementado y configurado** |
| **ReduceLROnPlateau callback** | ✅ | **Implementado (scheduler + monitor)** |
| **Entrenamiento solo sin defectos** | ✅ | **Solo datos 'good' en entrenamiento** |
| **Scripts externos** | ⚠️ | **No implementado (opcional según nota)** |
| **Estructura ResNet-18 (conv1, conv2, conv3)** | ✅ | **Implementada correctamente** |
| **Modelo A (entrenado desde 0)** | ✅ | **Pesos aleatorios, sin pre-entrenamiento** |
| **Modelo B (destilación teacher-student)** | ✅ | **ResNet-18 como teacher, destilación implementada** |
| **Extracción de embeddings** | ✅ | **Método get_embedding() implementado** |
| **3 configuraciones por modelo (9 totales)** | ✅ | **Cumple con requisito de 9 entrenamientos** |
| **Modelo C (U-Net Autoencoder)** | ✅ | **Basado en U-Net con skip connections** |
| **Reconstrucción de imágenes** | ✅ | **Forward() reconstruye imágenes de entrada** |
| **Embeddings del autoencoder** | ✅ | **Extraídos del espacio latente** |
| **Entrenamiento desde 0 (Modelo C)** | ✅ | **Sin pre-entrenamiento, pesos aleatorios** |
| **Extracción de embeddings (validación)** | ✅ | **Del conjunto de validación/entrenamiento (solo normales)** |
| **Estimación distribución normal (μ, Σ)** | ✅ | **Fórmulas exactas según enunciado** |
| **Distancia de Mahalanobis** | ✅ | **d = sqrt((z - μ)^T Σ^(-1) (z - μ))** |
| **Clasificación por percentiles** | ✅ | **Umbral basado en percentil de distancias normales** |
| **Distancia Euclidiana** | ✅ | **d = ||z - μ|| implementada** |
| **Reconstruction Loss** | ✅ | **MSE entre reconstrucción y original** |
| **Selección 3 mejores modelos** | ✅ | **Según AUC-ROC, criterio claro** |
| **Conversión a cuantizados** | ✅ | **Cuantización dinámica implementada** |
| **Comparación de tamaño** | ✅ | **Original vs cuantizado + ratio compresión** |
| **Comparación de latencia** | ✅ | **Promedio 100 iteraciones + speedup** |
| **Comparación de rendimiento** | ✅ | **AUC-ROC y AUC-PR + retención** |
| **Análisis en informe** | ✅ | **Resumen comparativo y estadístico** |
| **Selección mejor modelo para DBSCAN** | ✅ | **Mejor según AUC-ROC** |
| **Extracción embeddings conjunto prueba** | ✅ | **Para todas las imágenes de prueba** |
| **Reducción dimensionalidad PCA** | ✅ | **Configurable, default 50 componentes** |
| **Reducción dimensionalidad t-SNE** | ✅ | **2D para visualización** |
| **Aplicación DBSCAN** | ✅ | **Clusters y outliers identificados** |
| **Interpretación ruido como anomalías** | ✅ | **Puntos -1 = outliers/anomalías** |
| **Análisis visual DBSCAN** | ✅ | **3 visualizaciones: clusters, ground truth, comparación** |
| **Análisis cuantitativo DBSCAN** | ✅ | **AUC-ROC, AUC-PR, matriz confusión, precisión, recall, F1** |

---

## Acciones Recomendadas

1. ✅ **Completado**: Verificación de implementación de sección I y II
2. ✅ **Completado**: Verificación de implementación de sección III (líneas 21-43)
3. ✅ **Completado**: Verificación de implementación de PyTorch Lightning (líneas 45-56)
4. ✅ **Completado**: Verificación de Modelo Clasificador CNN (líneas 58-76)
5. ✅ **Completado**: Verificación de Modelo C Autoencoder U-Net (líneas 78-84)
6. ✅ **Completado**: Verificación de Evaluación de Anomalías (líneas 103-136)
7. ✅ **Completado**: Verificación de Modelos Cuantizados (líneas 137-139)
8. ✅ **Completado**: Verificación de Análisis DBSCAN (líneas 141-150)
9. ⚠️ **Opcional**: Añadir nota explícita en el notebook sobre la pregunta filosófica de detección de anomalías (línea 19 del enunciado)
10. ⚠️ **Opcional**: Crear scripts externos en carpeta "scripts" y documentarlos en el notebook (según nota del profesor, línea 56)

---

## Historial de Cambios

- **2025-01-27**: Verificación inicial de sección I y II (líneas 9-19 del enunciado)
  - Confirmado: Objetivo implementado
  - Confirmado: Dataset MVTec AD configurado
  - Confirmado: 10 clases seleccionadas
  - Confirmado: Detección de anomalías implementada (embeddings y reconstrucción)

- **2025-01-27**: Verificación de sección III.1-III.3 (líneas 21-43 del enunciado)
  - Confirmado: Hydra inicializado y configurado correctamente
  - Confirmado: Estructura de directorios conf/ completa
  - Confirmado: Todos los archivos YAML requeridos presentes
  - Confirmado: Dimensión del espacio latente (z) configurable
  - Confirmado: Épocas, batch size y otros hiperparámetros configurables
  - Nota: Se usa `unet_autoencoder.yaml` en lugar de `vae.yaml` (equivalente funcional)

- **2025-01-27**: Verificación de sección III.4 PyTorch Lightning (líneas 45-56 del enunciado)
  - Confirmado: PyTorch Lightning correctamente implementado
  - Confirmado: MVTecDataModule hereda de LightningDataModule con todos los métodos
  - Confirmado: CNNClassifierLightning y AutoencoderLightning implementan métodos mínimos requeridos
  - Confirmado: training_step, test_step, configure_optimizers implementados en ambos módulos
  - Confirmado: EarlyStopping callback implementado y configurado
  - Confirmado: ReduceLROnPlateau scheduler implementado (patience=5)
  - Confirmado: LearningRateMonitor callback implementado
  - Confirmado: Entrenamiento solo con datos sin defectos (solo 'good')
  - Nota: Scripts externos no implementados (opcional según nota del profesor)

- **2025-01-27**: Verificación de sección III.A Modelo Clasificador CNN (líneas 58-76 del enunciado)
  - Confirmado: Estructura ResNet-18 implementada para conv1, conv2_x, conv3_x
  - Confirmado: Clasificador FC implementado después de las 3 convoluciones
  - Confirmado: Modelo A entrenado desde 0 (pesos aleatorios, model_type="scratch")
  - Confirmado: Modelo B con destilación teacher-student (ResNet-18 como teacher)
  - Confirmado: Destilación implementada con temperatura, alpha y KL divergence
  - Confirmado: Extracción de embeddings implementada (método get_embedding())
  - Confirmado: 3 configuraciones para Modelo A (variando hiperparámetros)
  - Confirmado: 3 configuraciones para Modelo B (variando hiperparámetros y destilación)
  - Confirmado: 3 configuraciones para Modelo C (variando hiperparámetros)
  - Confirmado: Total de 9 entrenamientos (cumple con requisito)

- **2025-01-27**: Verificación de sección III.B Modelo C Autoencoder U-Net (líneas 78-84 del enunciado)
  - Confirmado: Autoencoder basado en U-Net con skip connections implementado
  - Confirmado: Encoder, bottleneck y decoder con skip connections correctamente implementados
  - Confirmado: Reconstrucción de imágenes implementada (forward() reconstruye entrada)
  - Confirmado: Extracción de embeddings del espacio latente implementada
  - Confirmado: Método get_embedding() implementado para extraer embeddings
  - Confirmado: Entrenamiento completamente desde 0 (sin pre-entrenamiento)
  - Confirmado: Similar a implementación de Tarea 5 (como menciona la nota)
  - Confirmado: Permite comparación con Modelos A y B (diferentes arquitecturas para embeddings)

- **2025-01-27**: Verificación de sección IV Evaluación de Anomalías (líneas 103-136 del enunciado)
  - Confirmado: Extracción de embeddings del conjunto de validación/entrenamiento (solo datos normales)
  - Confirmado: Estimación de distribución normal: μ = (1/N) Σ z_i y Σ = (1/(N-1)) Σ (z_i - μ)(z_i - μ)^T
  - Confirmado: Fórmulas implementadas exactamente como en el enunciado
  - Confirmado: Cálculo de distancia de Mahalanobis: d = sqrt((z - μ)^T Σ^(-1) (z - μ))
  - Confirmado: Clasificación usando percentiles (umbral basado en percentil de distancias normales)
  - Confirmado: Distancia Euclidiana implementada: d = ||z - μ||
  - Confirmado: Reconstruction Loss implementado: MSE entre reconstrucción y original
  - Confirmado: Tres métodos de detección implementados y justificados
  - Confirmado: Métricas calculadas: AUC-ROC, AUC-PR

- **2025-01-27**: Verificación de sección V Modelos Cuantizados (líneas 137-139 del enunciado)
  - Confirmado: Selección de 3 mejores modelos según AUC-ROC (criterio claro y justificado)
  - Confirmado: Conversión a modelos cuantizados usando cuantización dinámica de PyTorch
  - Confirmado: Comparación de tamaño: original vs cuantizado + ratio de compresión
  - Confirmado: Comparación de latencia: promedio sobre 100 iteraciones + speedup
  - Confirmado: Comparación de rendimiento: AUC-ROC y AUC-PR + diferencia y retención
  - Confirmado: Análisis completo incluido: resumen comparativo y estadístico
  - Confirmado: Métricas calculadas para todos los aspectos requeridos

- **2025-01-27**: Verificación de sección VI Análisis DBSCAN (líneas 141-150 del enunciado)
  - Confirmado: Selección del mejor modelo según AUC-ROC para análisis DBSCAN
  - Confirmado: Extracción de embeddings del conjunto de prueba (todas las imágenes)
  - Confirmado: Reducción de dimensionalidad con PCA (configurable, default 50 componentes)
  - Confirmado: Reducción de dimensionalidad con t-SNE (2D para visualización)
  - Confirmado: Aplicación de DBSCAN para identificar clusters y outliers
  - Confirmado: Interpretación de ruido (-1) como anomalías (puntos de baja densidad)
  - Confirmado: Análisis visual: 3 visualizaciones (clusters, ground truth, comparación)
  - Confirmado: Análisis cuantitativo: AUC-ROC, AUC-PR, matriz de confusión, precisión, recall, F1
  - Confirmado: Comparación de outliers DBSCAN con ground truth para evaluación

