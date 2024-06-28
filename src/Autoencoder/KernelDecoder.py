import torch
import torch.nn as nn
__all__ = ['KernelDecoderModule']

"""
Autoencoder for inputs with shape (Ns, Nc, Nx*Ny) -> (-1, 5, Nx)

"""

class KernelDecoderModule(nn.Module):
    # def __init__(self, n_input, latent_dim, num_params = 2, max_in = 1, min_in = 0) -> None:
    def __init__(self, n_input, latent_dim, num_params = 2,max_in = 1, min_in = 0):
        super(KernelDecoderModule,self).__init__()
        # for normalization 
        self.register_buffer('min', min_in)
        self.register_buffer('input_range', max_in - min_in)
        min_hidden = 8
        while min_hidden < latent_dim:
            min_hidden *=2
        self.num_params = num_params
        

        self.decoder = nn.Sequential(
            nn.Linear(latent_dim + num_params,min_hidden),
            nn.ReLU(),
            nn.Linear(min_hidden,min_hidden*4),
            nn.ReLU(),
            nn.Linear(min_hidden*4,min_hidden*8),
            nn.ReLU(),
            nn.Linear(min_hidden*8,min_hidden*80), # 80 = 5 chanels x 16 dimensions
            nn.ReLU(),
            nn.Unflatten(1,(5,min_hidden*16)), 
            nn.Linear(min_hidden*16, min_hidden*32),
            nn.ReLU(),
            nn.Linear(min_hidden*32,n_input)
        )
        

    def decode(self, latent, param):
        # add parameter information
        p = torch.ones((latent.shape[0],self.num_params),device=latent.device)
        for i in range(self.num_params):
            p[...,i] = param[...,i]
        latent = torch.cat((latent,p),-1)

        decode = self.decoder(latent)
        return decode * self.input_range + self.min
    def forward(self, latent, param):
        output = self.decode(latent,param)
        
        return output
