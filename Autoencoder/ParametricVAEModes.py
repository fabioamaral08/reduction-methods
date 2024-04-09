import torch
import torch.nn as nn
__all__ = ['ParametricVAEModesModule']

"""
Autoencoder for inputs with shape (Ns, Nc, Nx*Ny) -> (-1, 5, Nx)

"""

class ParametricVAEModesModule(nn.Module):
    def __init__(self, n_input, latent_dim, num_params = 2, max_in = 1, min_in = 0) -> None:
        super(ParametricVAEModesModule,self).__init__()
        # for normalization 
        self.register_buffer('min', min_in)
        self.register_buffer('input_range', max_in - min_in)

        self.num_params = num_params
        self.latent_dim = latent_dim
        self.encoder = nn.Sequential(
            nn.Linear(n_input + num_params,2048), # 2^11
            nn.ReLU(),
            nn.Linear(2048,512), # 2^9
            nn.ReLU(),
            nn.Linear(512,128), # 2^7
            nn.ReLU(),
            nn.Flatten(), # Merge the channels
            nn.Linear(128*5,32), # 2^5
            nn.ReLU(),
            nn.Linear(32,8), # 2^3
            nn.ReLU(),
        ) 
        self.decoders = [self.create_decoder(num_params, n_input) for _ in range(latent_dim)]

        self.gen_mu  = nn.Linear(8,latent_dim)
        self.gen_std = nn.Linear(8,latent_dim)

    def create_decoder(self, num_params, n_input):
        decoder = nn.Sequential(
        nn.Linear(1 + num_params,16),
        nn.ReLU(),
        nn.Linear(16,32),
        nn.ReLU(),
        nn.Linear(32,64*5),
        nn.ReLU(),
        nn.Unflatten(1,(5,64)), 
        nn.Linear(64,128),
        nn.ReLU(),
        nn.Linear(128, 256),
        nn.ReLU(),
        nn.Linear(256,n_input)
        )
        return decoder
        
    def encode(self, x, param):
        # Normalize input:
        x = (x - self.min)/self.input_range

        # add parameter information
        p = torch.ones((x.shape[0],5,self.num_params),device=x.device)
        for i in range(self.num_params):
            p[...,i] = param[...,i].view(-1,1)
        x = torch.cat((x,p),-1)

        #variational generation
        result = self.encoder(x)
        mu = self.gen_mu(result)
        log_var = self.gen_std(result)
        latent = self.reparametrize(mu, log_var)
        return latent, mu, log_var
    


    def decode_list(self, latent, param):
        # add parameter information
        p = torch.zeros((latent.shape[0],self.num_params),device=latent.device)
        for i in range(self.num_params):
            p[...,i] = param[...,i]
        # latent = torch.cat((latent,p),-1)
        decode = [self.decoders[i](torch.cat((latent[...,i],p),-1)) for i in range(self.latent_dim)]
        return decode
    
    def decode(self, latent, param):
        decode_list = self.decode_list(latent, param)
        decode = torch.stack(decode_list).sum(0)
        # denormalize
        return decode * self.input_range + self.min
    
    def forward(self, x, param):
        # run autoencoder 
        latent,mu,log_var = self.encode(x,param)
        output = self.decode(latent,param)
        
        return output,mu,log_var
    
    def reparametrize(self, mu, log_var):
        std = torch.exp(0.5 * log_var)
        result = torch.randn_like(std)
        return result * std + mu