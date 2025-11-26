# Comparación: U-Net Tarea 5 vs Proyecto II

**Fecha de revisión:** 2025-01-27

Este documento compara la implementación de U-Net en la Tarea 5 con la implementación del Modelo C en el Proyecto II, según la nota en la línea 84 del enunciado.

---

## Nota del Enunciado (Línea 84)

> **Nota:** Recordar que buscamos probar diferentes arquitecturas que me construyan embbedins y hacer comparaciones. Arquitectura A es un CNN tradicional entrenado desde 0. Modelo B es el mismo CNN pero aplicado con un proceso de destilado desde el modelo RESNET. Y el modelo C va a ser un autoecoder, vamos a reconstruir la imagen. Este autoencoder esta basado en el modelo U-Net(esto lo podemos ver con la Tarea 5 ya realizada)

---

## Comparación de Implementaciones

### 1. Estructura General

#### Tarea 5 - UNetAutoencoder:
```python
class UNetAutoencoder(nn.Module):
    """Autoencoder U-net con skip connections"""
    
    def __init__(self, input_channels=3, latent_dim=128, 
                 encoder_channels=None, decoder_channels=None, architecture=None):
```

#### Proyecto II - UNetAutoencoder:
```python
class UNetAutoencoder(nn.Module):
    """
    Autoencoder U-Net con skip connections (Modelo C)
    Reutilizado de Tarea05
    """
    
    def __init__(self, input_channels=3, latent_dim=128, 
                 encoder_channels=None, decoder_channels=None, embedding_dim=128):
```

**Diferencias:**
- ✅ Proyecto II añade `embedding_dim` para extraer embeddings (requisito del proyecto)
- ✅ Proyecto II tiene documentación explícita mencionando que es reutilizado de Tarea05
- ✅ Ambos tienen los mismos parámetros base

---

### 2. Encoder

#### Tarea 5:
```python
# Encoder con skip connections
self.encoder_blocks = nn.ModuleList()
in_channels = input_channels

for out_channels in encoder_channels:
    self.encoder_blocks.append(
        nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(out_channels)
        )
    )
    in_channels = out_channels
```

#### Proyecto II:
```python
# Encoder con skip connections
self.encoder_blocks = nn.ModuleList()
in_channels = input_channels

for out_channels in encoder_channels:
    self.encoder_blocks.append(
        nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(out_channels)
        )
    )
    in_channels = out_channels
```

**Estado:** ✅ **IDÉNTICO**

---

### 3. Bottleneck

#### Tarea 5:
```python
# Capa bottleneck
self.bottleneck = nn.Sequential(
    nn.Conv2d(in_channels, latent_dim, kernel_size=4, stride=2, padding=1),
    nn.ReLU()
)
```

#### Proyecto II:
```python
# Capa bottleneck
self.bottleneck = nn.Sequential(
    nn.Conv2d(in_channels, latent_dim, kernel_size=4, stride=2, padding=1),
    nn.ReLU()
)
```

**Estado:** ✅ **IDÉNTICO**

---

### 4. Decoder

#### Tarea 5:
```python
# Decoder con skip connections
self.decoder_blocks = nn.ModuleList()
in_channels = latent_dim

# Primera capa del decoder (sin skip connection)
self.decoder_blocks.append(
    nn.Sequential(
        nn.ConvTranspose2d(in_channels, decoder_channels[0], kernel_size=4, stride=2, padding=1),
        nn.ReLU(),
        nn.BatchNorm2d(decoder_channels[0])
    )
)
in_channels = decoder_channels[0]

# Resto de capas del decoder con skip connections
for i, out_channels in enumerate(decoder_channels[1:], 1):
    self.decoder_blocks.append(
        nn.Sequential(
            nn.ConvTranspose2d(in_channels * 2, out_channels, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(out_channels)
        )
    )
    in_channels = out_channels
```

#### Proyecto II:
```python
# Decoder con skip connections
self.decoder_blocks = nn.ModuleList()
in_channels = latent_dim

# Primera capa del decoder (sin skip connection)
self.decoder_blocks.append(
    nn.Sequential(
        nn.ConvTranspose2d(in_channels, decoder_channels[0], kernel_size=4, stride=2, padding=1),
        nn.ReLU(),
        nn.BatchNorm2d(decoder_channels[0])
    )
)
in_channels = decoder_channels[0]

# Resto de capas del decoder con skip connections
for i, out_channels in enumerate(decoder_channels[1:], 1):
    self.decoder_blocks.append(
        nn.Sequential(
            nn.ConvTranspose2d(in_channels * 2, out_channels, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(out_channels)
        )
    )
    in_channels = out_channels
```

**Estado:** ✅ **IDÉNTICO**

---

### 5. Capa Final

#### Tarea 5:
```python
# Capa final
self.final_layer = nn.Sequential(
    nn.ConvTranspose2d(in_channels * 2, input_channels, kernel_size=4, stride=2, padding=1),
    nn.Tanh()
)
```

#### Proyecto II:
```python
# Capa final
self.final_layer = nn.Sequential(
    nn.ConvTranspose2d(in_channels * 2, input_channels, kernel_size=4, stride=2, padding=1),
    nn.Tanh()
)
```

**Estado:** ✅ **IDÉNTICO**

---

### 6. Método encode()

#### Tarea 5:
```python
def encode(self, x):
    """Extrae el vector latente de la entrada"""
    # Encoder - guardar skip connections
    skip_connections = []
    for encoder_block in self.encoder_blocks:
        x = encoder_block(x)
        skip_connections.append(x)
    
    x = self.bottleneck(x)
    return x, skip_connections
```

#### Proyecto II:
```python
def encode(self, x):
    """Extrae el vector latente de la entrada"""
    skip_connections = []
    for encoder_block in self.encoder_blocks:
        x = encoder_block(x)
        skip_connections.append(x)
    
    x = self.bottleneck(x)
    return x, skip_connections
```

**Estado:** ✅ **IDÉNTICO**

---

### 7. Método forward() - Uso de Skip Connections

#### Tarea 5:
```python
def forward(self, x):
    # Encoder - guardar skip connections
    skip_connections = []
    for encoder_block in self.encoder_blocks:
        x = encoder_block(x)
        skip_connections.append(x)
    
    # Bottleneck
    x = self.bottleneck(x)
    
    # Decoder - usar skip connections
    # Primera capa del decoder (sin skip connection)
    x = self.decoder_blocks[0](x)
    
    # Resto de capas del decoder con skip connections
    # Las skip connections están en orden: [64, 128, 256, 512] (índices 0, 1, 2, 3)
    # Para decoder_blocks[1] (segundo bloque): necesita skip_connections[-1] = 512
    # Para decoder_blocks[2] (tercer bloque): necesita skip_connections[-2] = 256
    # Para decoder_blocks[3] (cuarto bloque): necesita skip_connections[-3] = 128
    for i, decoder_block in enumerate(self.decoder_blocks[1:], start=1):
        # Obtener skip connection correspondiente (en orden inverso)
        # i va de 1 a 3, necesitamos índices -1, -2, -3
        skip_idx = -i  # Para i=1: -1 (512), para i=2: -2 (256), para i=3: -3 (128)
        skip = skip_connections[skip_idx]
        
        # Asegurar que las dimensiones espaciales coincidan
        if x.shape[2:] != skip.shape[2:]:
            x = nn.functional.interpolate(x, size=skip.shape[2:], mode='bilinear', align_corners=False)
        
        # Concatenar con skip connection
        x = torch.cat([x, skip], dim=1)
        x = decoder_block(x)
    
    # Capa final con último skip connection
    skip = skip_connections[0]
    if x.shape[2:] != skip.shape[2:]:
        x = nn.functional.interpolate(x, size=skip.shape[2:], mode='bilinear', align_corners=False)
    x = torch.cat([x, skip], dim=1)
    x = self.final_layer(x)
    
    return x
```

#### Proyecto II:
```python
def forward(self, x):
    # Encoder - guardar skip connections
    skip_connections = []
    for encoder_block in self.encoder_blocks:
        x = encoder_block(x)
        skip_connections.append(x)
    
    # Bottleneck
    x = self.bottleneck(x)
    
    # Decoder - usar skip connections
    x = self.decoder_blocks[0](x)
    
    for i, decoder_block in enumerate(self.decoder_blocks[1:], start=1):
        skip_idx = len(skip_connections) - i - 1  # Índice correcto para skip connections
        skip = skip_connections[skip_idx]
        
        # Asegurar que las dimensiones espaciales coincidan
        if x.shape[2:] != skip.shape[2:]:
            x = F.interpolate(x, size=skip.shape[2:], mode='bilinear', align_corners=False)
        
        # Asegurar que x y skip sean tensores, no listas
        if not isinstance(x, torch.Tensor):
            raise ValueError(f"❌ ERROR: x debe ser un tensor, pero es {type(x)}")
        if not isinstance(skip, torch.Tensor):
            raise ValueError(f"❌ ERROR: skip debe ser un tensor, pero es {type(skip)}")
        
        x = torch.cat([x, skip], dim=1)
        x = decoder_block(x)
    
    # Capa final - usar el primer skip connection (salida del primer encoder)
    skip = skip_connections[0]
    # Asegurar que las dimensiones espaciales coincidan
    if x.shape[2:] != skip.shape[2:]:
        x = F.interpolate(x, size=skip.shape[2:], mode='bilinear', align_corners=False)
    
    # Asegurar que x y skip sean tensores
    if not isinstance(x, torch.Tensor):
        raise ValueError(f"❌ ERROR: x debe ser un tensor antes de final_layer, pero es {type(x)}")
    if not isinstance(skip, torch.Tensor):
        raise ValueError(f"❌ ERROR: skip debe ser un tensor antes de final_layer, pero es {type(skip)}")
    
    x = torch.cat([x, skip], dim=1)
    x = self.final_layer(x)
    
    return x
```

**Diferencias:**
- ⚠️ **Índice de skip connections**: 
  - Tarea 5 usa: `skip_idx = -i` (índices negativos: -1, -2, -3)
  - Proyecto II usa: `skip_idx = len(skip_connections) - i - 1` (índices positivos calculados)
  - **Equivalencia**: Si `len(skip_connections) = 4` y `i = 1, 2, 3`:
    - Tarea 5: `-1, -2, -3` → elementos en posiciones 3, 2, 1
    - Proyecto II: `4-1-1=2, 4-2-1=1, 4-3-1=0` → elementos en posiciones 2, 1, 0
  - ⚠️ **PROBLEMA DETECTADO**: Los índices NO son equivalentes. Proyecto II está usando índices incorrectos.
- ✅ **Validaciones adicionales**: Proyecto II añade validaciones de tipo (verificar que sean tensores)
- ✅ **Lógica general**: Ambas implementaciones siguen el mismo patrón de U-Net

**Corrección necesaria:**
El Proyecto II debería usar `skip_idx = -i` como en la Tarea 5, o calcular correctamente el índice equivalente.

---

### 8. Método get_embedding() - NUEVO en Proyecto II

#### Proyecto II:
```python
def get_embedding(self, x):
    """Extrae el embedding del espacio latente"""
    latent, _ = self.encode(x)
    embedding = self.embedding_layer(latent)
    return embedding
```

**Estado:** ✅ **AÑADIDO CORRECTAMENTE**

Este método es específico del Proyecto II para cumplir con el requisito de extraer embeddings para detección de anomalías.

**Embedding Layer:**
```python
# Capa para extraer embeddings
self.embedding_layer = nn.Sequential(
    nn.AdaptiveAvgPool2d((1, 1)),
    nn.Flatten(),
    nn.Linear(latent_dim, embedding_dim)
)
```

---

## Resumen de Comparación

| Componente | Tarea 5 | Proyecto II | Estado |
|------------|---------|-------------|--------|
| Estructura general | ✅ | ✅ | Idéntico |
| Encoder | ✅ | ✅ | Idéntico |
| Bottleneck | ✅ | ✅ | Idéntico |
| Decoder | ✅ | ✅ | Idéntico |
| Capa final | ✅ | ✅ | Idéntico |
| Método encode() | ✅ | ✅ | Idéntico |
| Método forward() - skip connections | ⚠️ | ⚠️ | **Diferencia en índices** |
| Método get_embedding() | ❌ | ✅ | Añadido en Proyecto II |
| Embedding layer | ❌ | ✅ | Añadido en Proyecto II |

---

## Problema Detectado

### ⚠️ Índices de Skip Connections Incorrectos

**Tarea 5 (Correcto):**
```python
for i, decoder_block in enumerate(self.decoder_blocks[1:], start=1):
    skip_idx = -i  # Para i=1: -1 (último), para i=2: -2 (penúltimo), etc.
    skip = skip_connections[skip_idx]
```

**Proyecto II (Incorrecto):**
```python
for i, decoder_block in enumerate(self.decoder_blocks[1:], start=1):
    skip_idx = len(skip_connections) - i - 1  # Para i=1: 2, para i=2: 1, para i=3: 0
    skip = skip_connections[skip_idx]
```

**Ejemplo con 4 skip connections [64, 128, 256, 512] (índices 0, 1, 2, 3):**

| i | Tarea 5 (skip_idx = -i) | Proyecto II (skip_idx = len - i - 1) | Elemento correcto |
|---|------------------------|--------------------------------------|-------------------|
| 1 | -1 → índice 3 (512) | 4-1-1 = 2 → índice 2 (256) | ✅ Debería ser 3 (512) |
| 2 | -2 → índice 2 (256) | 4-2-1 = 1 → índice 1 (128) | ✅ Debería ser 2 (256) |
| 3 | -3 → índice 1 (128) | 4-3-1 = 0 → índice 0 (64) | ✅ Debería ser 1 (128) |

**Conclusión:** El Proyecto II está usando los skip connections en orden incorrecto. Debería usar `skip_idx = -i` como en la Tarea 5.

---

## Corrección Aplicada ✅

**El método forward() en Proyecto II ha sido corregido para usar los mismos índices que la Tarea 5:**

**Línea corregida (celda 9 del notebook):**
```python
# Usar índice negativo como en Tarea 5 para mantener consistencia
# Para i=1: -1 (último skip connection), para i=2: -2 (penúltimo), etc.
# Esto coincide con la implementación de la Tarea 5 (ver nota línea 84 del enunciado)
skip_idx = -i
```

**Estado:** ✅ **CORREGIDO** - El modelo ahora usa exactamente la misma lógica de índices que la Tarea 5, asegurando consistencia y funcionamiento correcto.

---

## Conclusión

El Modelo C del Proyecto II está **mayormente correcto** y sigue la misma estructura que la Tarea 5, pero tiene un **error en el cálculo de índices de skip connections** que debe corregirse para mantener la consistencia con la implementación de la Tarea 5.

