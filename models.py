import torch.nn as nn

class Unflatten(nn.Module):
    
    def forward(self, x):
        return x.view(-1, 512, 4, 4)

class DownBlock(nn.Module):
    def __init__(self, in_features, out_features):
        super(DownBlock, self).__init__()
        
        self.conv1 = nn.Conv2d(in_features, out_features, kernel_size=3, stride=2, padding=1, bias=False)
        self.norm1 = nn.InstanceNorm2d(out_features)#nn.BatchNorm2d(out_features)
        
        self.conv2 = nn.Conv2d(out_features, out_features, kernel_size=3, padding=1, bias=False)
        self.norm2 = nn.InstanceNorm2d(out_features)#nn.BatchNorm2d(out_features)
        
        self.conv_identity = nn.Conv2d(in_features, out_features, kernel_size=1, stride=2, bias=False)
        self.norm_identity = nn.InstanceNorm2d(out_features)#nn.BatchNorm2d(out_features)
        
        self.activation = nn.LeakyReLU(0.1, inplace=True)#relu?
        
    def forward(self, x):
        identity = x
        
        out = self.conv1(x)
        out = self.norm1(out)
        out = self.activation(out)
        
        out = self.conv2(out)
        out = self.norm2(out)
        
        identity = self.conv_identity(identity)
        identity = self.norm_identity(identity)
        
        out += identity
        
        out = self.activation(out)
        
        return out

class UpBlock(nn.Module):
    def __init__(self, in_features, out_features):
        super(UpBlock, self).__init__()
        
        self.conv1 = nn.ConvTranspose2d(in_features, out_features, kernel_size=3, padding=1, bias=False)
        self.norm1 = nn.InstanceNorm2d(out_features)#nn.BatchNorm2d(out_features)
        
        self.conv2 = nn.ConvTranspose2d(out_features, out_features*4, kernel_size=3, padding=1, bias=False)
        self.norm2 = nn.InstanceNorm2d(out_features*4)#nn.BatchNorm2d(out_features*4)
        
        self.conv_identity = nn.ConvTranspose2d(in_features, out_features*4, kernel_size=1, bias=False)
        self.norm_identity = nn.InstanceNorm2d(out_features*4)#nn.BatchNorm2d(out_features*4)
        
        self.activation = nn.LeakyReLU(0.1, inplace=True)#relu?
        
        self.pixel_shuffle = nn.PixelShuffle(2)
        
    def forward(self, x):
        identity = x
        
        out = self.conv1(x)
        out = self.norm1(out)
        out = self.activation(out)
        
        out = self.conv2(out)
        out = self.norm2(out)
        
        identity = self.conv_identity(identity)
        identity = self.norm_identity(identity)
        
        out += identity
        
        out = self.activation(out)
        
        out = self.pixel_shuffle(out)
        
        return out

#https://github.com/deepfakes/faceswap/blob/127d3dbe99b9ae57c5df7d783a2cbb934d7eaad7/plugins/train/model/villain.py#L14
class Autoencoder(nn.Module):
    def __init__(self):
        super(Autoencoder, self).__init__()
        
        self.encoder = nn.Sequential(
            DownBlock(3, 16),
            DownBlock(16, 32),
            DownBlock(32, 64),
            #pixel shuffle
            DownBlock(64, 128),
            #pixel shuggel
            DownBlock(128, 256),
            DownBlock(256, 512),
    
            nn.Flatten(),

            nn.Linear(512 * 4 * 4, 2048),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Linear(2048, 512 * 4 * 4),
            nn.LeakyReLU(0.1, inplace=True),
    
            Unflatten(),

            UpBlock(512, 256),
        )
        
        self.decoder = nn.Sequential(
            
            UpBlock(256, 128),
            UpBlock(128, 64),
            UpBlock(64, 32),
            UpBlock(32, 16),
            UpBlock(16, 3),
        )
        
    def forward(self, x):
        out = self.encoder(x)
        out = self.decoder(out)
            
        return out