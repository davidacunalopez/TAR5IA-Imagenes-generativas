"""
Módulo con las arquitecturas de autoencoders para detección de anomalías
"""
import torch
import torch.nn as nn


class AutoencoderClassic(nn.Module):
    """Autoencoder clásico sin skip connections"""
    
    def __init__(self, input_channels=3, latent_dim=128, encoder_channels=None, decoder_channels=None):
        super(AutoencoderClassic, self).__init__()
        
        if encoder_channels is None:
            encoder_channels = [64, 128, 256, 512]
        if decoder_channels is None:
            decoder_channels = [512, 256, 128, 64]
        
        # Encoder
        encoder_layers = []
        in_channels = input_channels
        
        for out_channels in encoder_channels:
            encoder_layers.extend([
                nn.Conv2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1),
                nn.ReLU(),
                nn.BatchNorm2d(out_channels)
            ])
            in_channels = out_channels
        
        # Capa final del encoder
        encoder_layers.extend([
            nn.Conv2d(in_channels, latent_dim, kernel_size=4, stride=2, padding=1),
            nn.ReLU()
        ])
        
        self.encoder = nn.Sequential(*encoder_layers)
        
        # Decoder
        decoder_layers = []
        in_channels = latent_dim
        
        for out_channels in decoder_channels:
            decoder_layers.extend([
                nn.ConvTranspose2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1),
                nn.ReLU(),
                nn.BatchNorm2d(out_channels)
            ])
            in_channels = out_channels
        
        # Capa final del decoder
        decoder_layers.extend([
            nn.ConvTranspose2d(in_channels, input_channels, kernel_size=4, stride=2, padding=1),
            nn.Tanh()
        ])
        
        self.decoder = nn.Sequential(*decoder_layers)
    
    def encode(self, x):
        """Extrae el vector latente de la entrada"""
        return self.encoder(x)
    
    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded


class UNetAutoencoder(nn.Module):
    """Autoencoder U-net con skip connections"""
    
    def __init__(self, input_channels=3, latent_dim=128, encoder_channels=None, decoder_channels=None):
        super(UNetAutoencoder, self).__init__()
        
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
            # Duplicar canales de entrada para concatenar con skip connection
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
    
    def encode(self, x):
        """Extrae el vector latente de la entrada"""
        # Encoder - guardar skip connections
        skip_connections = []
        for encoder_block in self.encoder_blocks:
            x = encoder_block(x)
            skip_connections.append(x)
        
        # Bottleneck
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
        for i, decoder_block in enumerate(self.decoder_blocks):
            # Concatenar con skip connection correspondiente (en orden inverso)
            skip = skip_connections[-(i+1)]
            # Asegurar que las dimensiones coincidan
            if x.shape[2:] != skip.shape[2:]:
                x = nn.functional.interpolate(x, size=skip.shape[2:], mode='bilinear', align_corners=False)
            x = torch.cat([x, skip], dim=1)
            x = decoder_block(x)
        
        # Capa final con último skip connection
        skip = skip_connections[0]
        if x.shape[2:] != skip.shape[2:]:
            x = nn.functional.interpolate(x, size=skip.shape[2:], mode='bilinear', align_corners=False)
        x = torch.cat([x, skip], dim=1)
        x = self.final_layer(x)
        
        return x

