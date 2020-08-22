import torch.nn as nn

class FaceSwapper(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            # input is 3 x 64 x 64
            nn.Conv2d(3, 64, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. 64 x 32 x 32
            nn.Conv2d(64, 128, 4, 2, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. 128 x 16 x 16
            nn.Conv2d(128, 256, 4, 2, 1, bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. 256 x 8 x 8
            nn.Conv2d(256, 512, 4, 2, 1, bias=False),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. 512 x 4 x 4
            nn.Conv2d(512, 1024, 4, 1, 0, bias=False),
            nn.Sigmoid()
        )

        self.decoderA = nn.Sequential(
            nn.ConvTranspose2d( 1024, 512, 4, 1, 0, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(True),
            # state size. 512 x 4 x 4
            nn.ConvTranspose2d(512, 256, 4, 2, 1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            # state size. 256 x 8 x 8
            nn.ConvTranspose2d(256, 128, 4, 2, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            # state size. 128 x 16 x 16
            nn.ConvTranspose2d(128, 64, 4, 2, 1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            # state size. 64 x 32 x 32
            nn.ConvTranspose2d( 64, 3, 4, 2, 1, bias=False),
            nn.Sigmoid()
            # state size. 3 x 64 x 64
        )

        self.decoderB = nn.Sequential(
            nn.ConvTranspose2d( 1024, 512, 4, 1, 0, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(True),
            # state size. 512 x 4 x 4
            nn.ConvTranspose2d(512, 256, 4, 2, 1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            # state size. 256 x 8 x 8
            nn.ConvTranspose2d(256, 128, 4, 2, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            # state size. 128 x 16 x 16
            nn.ConvTranspose2d(128, 64, 4, 2, 1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            # state size. 64 x 32 x 32
            nn.ConvTranspose2d( 64, 3, 4, 2, 1, bias=False),
            nn.Sigmoid()
            # state size. 3 x 64 x 64
        )

    def forward(self, x, decoder):
        out = self.encoder(x)

        if decoder == "A":
            out = self.decoderA(out)
        else:
            out = self.decoderB(out)

        return out

# Model based in this paper: https://arxiv.org/pdf/1609.04802.pdf

class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride):
        super().__init__()

        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)

        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size, stride, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.activation = nn.PReLU()

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.activation(out)
        out = self.conv2(x)
        out = self.bn2(out)

        return out + x

class UpsampleBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size):
        super().__init__()

        self.main = nn.Sequential(
            nn.Conv2d(in_channels, out_channels * 4, kernel_size, stride=1, padding=1),
            nn.PixelShuffle(2),
            nn.PReLU()
        )

    def forward(self, x):
        return self.main(x)

class Conv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride):
        super().__init__()

        self.main = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU()
        )

    def forward(self, x):
        return self.main(x)

class Unflatten(nn.Module):
    def forward(self, x):
        return x.view(-1, 512, 4, 4)

class FaceSwapperSR(nn.Module):
    def __init__(self):
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(),

            Conv(64, 64, 3, 2),
            Conv(64, 128, 3, 1),
            Conv(128, 128, 3, 2),
            Conv(128, 256, 3, 1),
            Conv(256, 256, 3, 2),
            Conv(256, 512, 3, 1),
            Conv(512, 512, 3, 2),

            nn.Flatten(),

            nn.Linear(8192, 1024),
            nn.LeakyReLU(),
            nn.Linear(1024, 8192),
            nn.Sigmoid()
        )

        self.decoder = nn.Sequential(
            Unflatten(),
            UpsampleBlock(512, 128, 3),
            ResBlock(128, 128, 3, 1),
            UpsampleBlock(128, 64, 3),
            ResBlock(64, 64, 3, 1),
            UpsampleBlock(64, 32, 3),
            ResBlock(32, 32, 3, 1),
            UpsampleBlock(32, 16, 3),

            nn.Conv2d(16, 3, 3, 1, padding=1),         
        )

    def forward(self, x, decoder):
        out = self.encoder(x)

        if decoder == "A":
            out = self.decoderA(out)
        else:
            out = self.decoderB(out)

        return out

import torch
a = torch.randn((64, 3, 64, 64))
m = FaceSwapperSR()
print(m.decoder(m.encoder(a)).shape)