import torch
import torch.nn as nn
__all__ = ['VariationalAutoencoderModule']

"""
Variational Autoencoder for inputs with shape (Ns, Nc, Nx, Ny) -> (-1, 5, Nx)

"""

class VariationalAutoencoderModule(nn.Module):
    def __init__(self, n_input, latent_dim, max_in = 1, min_in = 0) -> None:
        super(VariationalAutoencoderModule,self).__init__()
        # for normalization 
        self.register_buffer('min', min_in)
        self.register_buffer('input_range', max_in - min_in)
        self.encoder = nn.Sequential(
            nn.Linear(n_input,8192), # 2^13
            nn.ReLU(),
            nn.Linear(8192,2048), # 2^11
            nn.ReLU(),
            nn.Linear(2048,512), # 2^9
            nn.ReLU(),
            nn.Flatten(), # Merge the channels
            nn.Linear(512*5,128), # 2^7
            nn.ReLU(),
            nn.Linear(128,32), # 2^5
            nn.ReLU(),
        ) 
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim,32),
            nn.ReLU(),
            nn.Linear(32,128),
            nn.ReLU(),
            nn.Linear(128,512*5),
            nn.ReLU(),
            nn.Unflatten(1,(5,512)), 
            nn.Linear(512,2048),
            nn.ReLU(),
            nn.Linear(2048, 8192),
            nn.ReLU(),
            nn.Linear(8192,n_input)
        )

        self.gen_mu  = nn.Linear(32,latent_dim)
        self.gen_std = nn.Linear(32,latent_dim)

    def encode(self, x):
        # Normalize input:
        x = (x - self.min)/self.input_range

        result = self.encoder(x)
        mu = self.gen_mu(result)
        log_var = self.gen_std(result)
        latent = self.reparametrize(mu, log_var)
        return latent, mu, log_var
    

    def decode(self, latent):
        decode = self.decoder(latent)
        # denormalize
        return decode * self.input_range + self.min
    
    def forward(self, x):
        # run autoencoder 
        latent,mu,log_var = self.encode(x)
        output = self.decode(latent)
        
        return output,mu,log_var
    
    def reparametrize(self, mu, log_var):
        std = torch.exp(0.5 * log_var)
        result = torch.randn_like(std)
        return result * std + mu