"""
Modelos de arquitectura para el Proyecto II
Incluye: BasicBlock, CNNClassifier, UNetAutoencoder, transfer_resnet18_weights
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet18


class BasicBlock(nn.Module):
    """Bloque básico de ResNet (2 convoluciones con skip connection)"""

    def __init__(self, in_channels, out_channels, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class CNNClassifier(nn.Module):
    """
    CNN Clasificador basado en ResNet-18 para las primeras 3 convoluciones
    Modelo A: Desde cero (scratch)
    Modelo B: Con destilación (distilled)
    """

    def __init__(self, num_classes=10, conv1_channels=64, conv2_channels=[64, 64],
                 conv3_channels=[128, 128], fc_hidden=512, dropout=0.5,
                 embedding_dim=256, model_type="scratch"):
        super(CNNClassifier, self).__init__()
        self.model_type = model_type
        self.num_classes = num_classes
        self.embedding_dim = embedding_dim

        # conv1: Primera convolución (similar a ResNet-18)
        self.conv1 = nn.Conv2d(3, conv1_channels, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(conv1_channels)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # conv2_x: Bloques residuales
        self.conv2_x = self._make_layer(conv1_channels, conv2_channels[0], conv2_channels[1], num_blocks=2, stride=1)

        # conv3_x: Bloques residuales
        self.conv3_x = self._make_layer(conv2_channels[-1], conv3_channels[0], conv3_channels[1], num_blocks=2, stride=2)

        # Global Average Pooling
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        # Clasificador
        self.fc = nn.Sequential(
            nn.Linear(conv3_channels[-1], fc_hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(fc_hidden, num_classes)
        )

        # Capa para extraer embeddings (para detección de anomalías)
        self.embedding_layer = nn.Linear(conv3_channels[-1], embedding_dim)

    def _make_layer(self, in_channels, base_channels, out_channels, num_blocks, stride):
        layers = []
        layers.append(BasicBlock(in_channels, base_channels, stride))
        for _ in range(1, num_blocks):
            layers.append(BasicBlock(base_channels, out_channels, stride=1))
        return nn.Sequential(*layers)

    def forward(self, x):
        # conv1
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.maxpool(x)

        # conv2_x
        x = self.conv2_x(x)

        # conv3_x
        x = self.conv3_x(x)

        # Global Average Pooling
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)

        # Embedding para detección de anomalías
        embedding = self.embedding_layer(x)

        # Clasificación
        logits = self.fc(x)

        return logits, embedding

    def get_embedding(self, x):
        """Extrae solo el embedding sin clasificación"""
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.maxpool(x)
        x = self.conv2_x(x)
        x = self.conv3_x(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        embedding = self.embedding_layer(x)
        return embedding


class UNetAutoencoder(nn.Module):
    """
    Autoencoder U-Net con skip connections (Modelo C)
    Reutilizado de Tarea05
    """

    def __init__(self, input_channels=3, latent_dim=128, encoder_channels=None,
                 decoder_channels=None, embedding_dim=128):
        super(UNetAutoencoder, self).__init__()
        self.architecture = "unet_autoencoder"
        self.embedding_dim = embedding_dim

        if encoder_channels is None:
            encoder_channels = [64, 128, 256, 512]
        if decoder_channels is None:
            decoder_channels = [512, 256, 128, 64]

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

        # Capa bottleneck
        self.bottleneck = nn.Sequential(
            nn.Conv2d(in_channels, latent_dim, kernel_size=4, stride=2, padding=1),
            nn.ReLU()
        )

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

        # Capa final
        self.final_layer = nn.Sequential(
            nn.ConvTranspose2d(in_channels * 2, input_channels, kernel_size=4, stride=2, padding=1),
            nn.Tanh()
        )

        # Capa para extraer embeddings
        self.embedding_layer = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(latent_dim, embedding_dim)
        )

    def encode(self, x):
        """Extrae el vector latente de la entrada"""
        skip_connections = []
        for encoder_block in self.encoder_blocks:
            x = encoder_block(x)
            skip_connections.append(x)

        x = self.bottleneck(x)
        return x, skip_connections

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
            # Usar índice negativo como en Tarea 5 para mantener consistencia
            skip_idx = -i
            skip = skip_connections[skip_idx]

            # Asegurar que las dimensiones espaciales coincidan
            if x.shape[2:] != skip.shape[2:]:
                x = F.interpolate(x, size=skip.shape[2:], mode='bilinear', align_corners=False)

            x = torch.cat([x, skip], dim=1)
            x = decoder_block(x)

        # Capa final - usar el primer skip connection (salida del primer encoder)
        skip = skip_connections[0]
        # Asegurar que las dimensiones espaciales coincidan
        if x.shape[2:] != skip.shape[2:]:
            x = F.interpolate(x, size=skip.shape[2:], mode='bilinear', align_corners=False)

        x = torch.cat([x, skip], dim=1)
        x = self.final_layer(x)

        return x

    def get_embedding(self, x):
        """Extrae el embedding del espacio latente"""
        latent, _ = self.encode(x)
        embedding = self.embedding_layer(latent)
        return embedding


def transfer_resnet18_weights(student_model, teacher_model=None):
    """
    Transfiere los pesos de ResNet-18 pre-entrenado a las primeras 3 convoluciones
    del modelo estudiante (Modelo B).
    
    Según el enunciado: "Vamos a aprovechar ya entrenamiento que existen en las 
    primeras 3 capas(conv1, conv2 y conv3 de RESNET), y utilizar la tecnica de teacher-student"
    
    Args:
        student_model: Instancia de CNNClassifier (model_type="distilled")
        teacher_model: ResNet-18 pre-entrenado (opcional, se carga si es None)
    
    Returns:
        student_model: Modelo con pesos transferidos
    """
    if student_model.model_type != "distilled":
        print("⚠️ Advertencia: transfer_resnet18_weights solo debe usarse con model_type='distilled'")
        return student_model
    
    # Cargar ResNet-18 pre-entrenado si no se proporciona
    if teacher_model is None:
        try:
            from torchvision.models import ResNet18_Weights
            teacher_model = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        except:
            teacher_model = resnet18(pretrained=True)
        teacher_model.eval()
    
    print("📥 Transfiriendo pesos de ResNet-18 a las primeras 3 convoluciones del Modelo B...")
    
    # Transferir conv1
    try:
        student_model.conv1.load_state_dict(teacher_model.conv1.state_dict())
        student_model.bn1.load_state_dict(teacher_model.bn1.state_dict())
        print("  ✓ conv1 y bn1 transferidos")
    except Exception as e:
        print(f"  ⚠️ Error transfiriendo conv1: {e}")
    
    # Transferir conv2_x (layer1 en ResNet-18)
    try:
        teacher_layer1 = teacher_model.layer1
        student_conv2_x = student_model.conv2_x
        
        # Transferir cada bloque
        for i, (teacher_block, student_block) in enumerate(zip(teacher_layer1, student_conv2_x)):
            student_block.conv1.load_state_dict(teacher_block.conv1.state_dict())
            student_block.bn1.load_state_dict(teacher_block.bn1.state_dict())
            student_block.conv2.load_state_dict(teacher_block.conv2.state_dict())
            student_block.bn2.load_state_dict(teacher_block.bn2.state_dict())
            if hasattr(teacher_block, 'shortcut') and hasattr(student_block, 'shortcut'):
                if len(teacher_block.shortcut) > 0 and len(student_block.shortcut) > 0:
                    student_block.shortcut.load_state_dict(teacher_block.shortcut.state_dict())
        print("  ✓ conv2_x (layer1) transferido")
    except Exception as e:
        print(f"  ⚠️ Error transfiriendo conv2_x: {e}")
    
    # Transferir conv3_x (layer2 en ResNet-18)
    try:
        teacher_layer2 = teacher_model.layer2
        student_conv3_x = student_model.conv3_x
        
        # Transferir cada bloque
        for i, (teacher_block, student_block) in enumerate(zip(teacher_layer2, student_conv3_x)):
            student_block.conv1.load_state_dict(teacher_block.conv1.state_dict())
            student_block.bn1.load_state_dict(teacher_block.bn1.state_dict())
            student_block.conv2.load_state_dict(teacher_block.conv2.state_dict())
            student_block.bn2.load_state_dict(teacher_block.bn2.state_dict())
            if hasattr(teacher_block, 'shortcut') and hasattr(student_block, 'shortcut'):
                if len(teacher_block.shortcut) > 0 and len(student_block.shortcut) > 0:
                    student_block.shortcut.load_state_dict(teacher_block.shortcut.state_dict())
        print("  ✓ conv3_x (layer2) transferido")
    except Exception as e:
        print(f"  ⚠️ Error transfiriendo conv3_x: {e}")
    
    print("✓ Transferencia de pesos completada")
    return student_model

