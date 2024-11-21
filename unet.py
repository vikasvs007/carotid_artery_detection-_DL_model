
import torch
import torch.nn as nn
import torch.nn.functional as F

class UNet(nn.Module):
    def __init__(self, input_channels=1, output_channels=1, depth=4, batch_norm=True, padding=True, up_mode='upconv'):
        super(UNet, self).__init__()

        self.input_channels = input_channels
        self.output_channels = output_channels
        self.depth = depth
        self.batch_norm = batch_norm
        self.padding = padding
        self.up_mode = up_mode

        # Feature map sizes for each level of the U-Net
        self.features = [64 * 2 ** i for i in range(depth)]

        # Encoder
        self.encoders = nn.ModuleList()
        in_channels = input_channels
        for feature in self.features:
            self.encoders.append(self._conv_block(in_channels, feature, batch_norm, padding))
            in_channels = feature

        # Pooling layers
        self.pools = nn.ModuleList([nn.MaxPool2d(kernel_size=2, stride=2) for _ in range(depth)])

        # Bottleneck
        self.bottleneck = self._conv_block(self.features[-1], self.features[-1] * 2, batch_norm, padding)

        # Decoder
        self.decoders = nn.ModuleList()
        self.upsamples = nn.ModuleList()
        for feature in reversed(self.features):
            self.upsamples.append(self._upsample(feature * 2, up_mode))
            self.decoders.append(self._conv_block(feature * 2, feature, batch_norm, padding))

        # Final convolution
        self.final_conv = nn.Conv2d(self.features[0], output_channels, kernel_size=1)

    def forward(self, x):
        # Encoding path
        enc_features = []
        for encoder, pool in zip(self.encoders, self.pools):
            x = encoder(x)
            enc_features.append(x)
            x = pool(x)

        # Bottleneck
        x = self.bottleneck(x)

        # Decoding path
        for upsample, decoder, enc_feature in zip(self.upsamples, self.decoders, reversed(enc_features)):
            x = upsample(x)
            x = torch.cat([x, enc_feature], dim=1)
            x = decoder(x)

        # Final convolution
        x = self.final_conv(x)
        return x

    def _conv_block(self, in_channels, out_channels, batch_norm, padding):
        """Create a convolutional block with optional batch normalization and padding."""
        layers = [
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=int(padding)),
            nn.ReLU(inplace=True)
        ]
        if batch_norm:
            layers.append(nn.BatchNorm2d(out_channels))
        layers.extend([
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=int(padding)),
            nn.ReLU(inplace=True)
        ])
        if batch_norm:
            layers.append(nn.BatchNorm2d(out_channels))
        return nn.Sequential(*layers)

    def _upsample(self, out_channels, up_mode):
        """Create an upsampling layer."""
        if up_mode == 'upconv':
            return nn.ConvTranspose2d(out_channels, out_channels // 2, kernel_size=2, stride=2)
        elif up_mode == 'upsample':
            return nn.Sequential(
                nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
                nn.Conv2d(out_channels, out_channels // 2, kernel_size=1)
            )
        else:
            raise ValueError(f"Invalid up_mode '{up_mode}'. Choose 'upconv' or 'upsample'.")
