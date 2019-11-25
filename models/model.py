import torch.nn as nn
#help by: https://github.com/udacity/deep-learning-v2-pytorch/blob/master/autoencoder/convolutional-autoencoder/Upsampling_Solution.ipynb

class Autoencoder(nn.Module):
    def __init__(self):
        super(Autoencoder, self).__init__()

        self.encoder = nn.Sequential(
            nn.Conv2d(3, 128, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2, True),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2, True),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2, True),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(512, 1024, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2, True),
            nn.MaxPool2d(2, 2)
        )
        
        self.decoder = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.ConvTranspose2d(1024, 512, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2, True),
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.ConvTranspose2d(512, 256, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2, True),
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.ConvTranspose2d(256, 128, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2, True),
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.ConvTranspose2d(128, 3, kernel_size=3, padding=1),
            nn.Tanh()
        )
        
    def forward(self, x):
        out = self.encoder(x)
        out = self.decoder(out)
        return out