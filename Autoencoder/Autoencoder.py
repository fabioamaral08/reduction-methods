import torch.nn as nn

__all__ = ['AutoencoderModule']

"""
Autoencoder for inputs with shape (Ns, Nc, Nx, Ny) -> (-1, 5, 80, 80)

"""

class AutoencoderModule(nn.Module):
    def __init__(self, n_input, latent_dim) -> None:
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Linear(n_input,8192), # 2^13
            nn.ReLU(),
            nn.Linear(8192,2048), # 2^11
            nn.ReLU(),
            nn.Linear(2048,512), # 2^9
            nn.ReLU(),
            nn.Linear(512,128), # 2^7
            nn.ReLU(),
            nn.Linear(128,32), # 2^5
            nn.ReLU(),
            nn.Linear(32,latent_dim)
        ) 
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim,32),
            nn.ReLU(),
            nn.Linear(32,128),
            nn.ReLU(),
            nn.Linear(128,512),
            nn.ReLU(),
            nn.Linear(512,2048),
            nn.ReLU(),
            nn.Linear(2048, 8192),
            nn.ReLU(),
            nn.Linear(8192,n_input)
        )

    def forward(self, x):
        
        latent = self.encoder(x)
        return self.decoder(latent)