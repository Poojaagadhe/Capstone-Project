import torch
import torch.nn as nn


class UNetGenerator(nn.Module):

    def __init__(self, in_channels=1, out_channels=1, features=64):
        super().__init__()

        # Encoder
        self.down1 = nn.Sequential(
            nn.Conv2d(in_channels, features, 4, 2, 1),
            nn.LeakyReLU(0.2)
        )

        self.down2 = nn.Sequential(
            nn.Conv2d(features, features * 2, 4, 2, 1, bias=False),
            nn.InstanceNorm2d(features * 2),
            nn.LeakyReLU(0.2)
        )

        self.down3 = nn.Sequential(
            nn.Conv2d(features * 2, features * 4, 4, 2, 1, bias=False),
            nn.InstanceNorm2d(features * 4),
            nn.LeakyReLU(0.2)
        )

        self.down4 = nn.Sequential(
            nn.Conv2d(features * 4, features * 8, 4, 2, 1, bias=False),
            nn.InstanceNorm2d(features * 8),
            nn.LeakyReLU(0.2)
        )

        # Bottleneck
        self.bottleneck = nn.Sequential(
            nn.Conv2d(features * 8, features * 8, 4, 2, 1),
            nn.ReLU()
        )

        # Decoder
        self.up1 = nn.Sequential(
            nn.ConvTranspose2d(features * 8, features * 8, 4, 2, 1, bias=False),
            nn.InstanceNorm2d(features * 8),
            nn.ReLU(),
            nn.Dropout(0.5)
        )

        self.up2 = nn.Sequential(
            nn.ConvTranspose2d(features * 16, features * 4, 4, 2, 1, bias=False),
            nn.InstanceNorm2d(features * 4),
            nn.ReLU(),
            nn.Dropout(0.5)
        )

        self.up3 = nn.Sequential(
            nn.ConvTranspose2d(features * 8, features * 2, 4, 2, 1, bias=False),
            nn.InstanceNorm2d(features * 2),
            nn.ReLU(),
            nn.Dropout(0.5)
        )

        self.up4 = nn.Sequential(
            nn.ConvTranspose2d(features * 4, features, 4, 2, 1, bias=False),
            nn.InstanceNorm2d(features),
            nn.ReLU()
        )

        self.final = nn.Sequential(
            nn.ConvTranspose2d(features * 2, out_channels, 4, 2, 1),
            nn.Tanh()
        )

    def forward(self, x):

        d1 = self.down1(x)
        d2 = self.down2(d1)
        d3 = self.down3(d2)
        d4 = self.down4(d3)

        bottleneck = self.bottleneck(d4)

        u1 = self.up1(bottleneck)
        u2 = self.up2(torch.cat([u1, d4], dim=1))
        u3 = self.up3(torch.cat([u2, d3], dim=1))
        u4 = self.up4(torch.cat([u3, d2], dim=1))

        output = self.final(torch.cat([u4, d1], dim=1))

        return output
