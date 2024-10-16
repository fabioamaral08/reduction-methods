import torch
import torch.nn as nn
from torch.nn import functional as F

__all__ = ['ConvAutoencoderModule']
"""
Autoencoder for inputs with shape (Ns, Nc, Nx, Ny) -> (-1, 5, 64, 64)

"""

class Encoder(nn.Module):
    def __init__(self, latent_dim) -> None:
        super(Encoder,self).__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(5,64,kernel_size=3,stride=2, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.Conv2d(64,32,kernel_size=3,stride=2, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.Conv2d(32,16,kernel_size=3,stride=2, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(16),
            nn.Conv2d(16,8,kernel_size=3,stride=2, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(8),
            nn.Flatten(),
            nn.Linear(128,latent_dim*2),
            nn.ReLU(),
            nn.Linear(latent_dim*2,latent_dim)
        )

    def forward(self,x):
        x = self.conv(x)
        return x
class Decoder(nn.Module):
    def __init__(self, latent_dim) -> None:
        super(Decoder,self).__init__()

        self.dense = nn.Sequential(
            nn.Linear(latent_dim,latent_dim*2),
            nn.ReLU(),
            nn.Linear(latent_dim*2, 128),
            nn.ReLU(),
        )
        self.conv1 = nn.Sequential(
            # Reshape
            nn.Conv2d(8,16,kernel_size=3,stride=1, padding='same'),
            nn.ReLU(),
            nn.BatchNorm2d(16),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(16,32,kernel_size=3,stride=1, padding='same'),
            nn.ReLU(),
            nn.BatchNorm2d(32),
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(32,64,kernel_size=3,stride=1, padding='same'),
            nn.ReLU(),
            nn.BatchNorm2d(64),
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(64,5,kernel_size=3,stride=1, padding='same'),
            nn.ReLU(),
            nn.BatchNorm2d(5),
        )

        self.conv5 = nn.Sequential(
            nn.Conv2d(5,5,kernel_size=3,stride=1, padding='same'),
        )

    def forward(self, x):

        x = self.dense(x)
        x = torch.reshape(x,(-1,8,4,4))
        x = self.conv1(x)
        x = F.interpolate(x,size = 8)
        x = self.conv2(x)
        x = F.interpolate(x,size = 16)
        x = self.conv3(x)
        x = F.interpolate(x,size = 32)
        x = self.conv4(x)
        x = F.interpolate(x,size = 64)
        x = self.conv5(x)
        return x
    


class ConvAutoencoderModule(nn.Module):
    def __init__(self, latent_dim, max_in = 1, min_in = 0) -> None:
        super(ConvAutoencoderModule,self).__init__()
        # for normalization 
        self.register_buffer('min', min_in)
        self.register_buffer('input_range', max_in - min_in)
        
        self.encoder = Encoder(latent_dim)
        self.decoder = Decoder(latent_dim)
        
    def forward(self, x):
        # Normalize input:
        x = (x - self.min)/self.input_range

        # run autoencoder 
        latent = self.encoder(x)
        decode = self.decoder(latent)

        # denormalize
        output = decode * self.input_range + self.min
        
        return output