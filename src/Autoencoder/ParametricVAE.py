import torch
import torch.nn as nn
__all__ = ['ParametricVAEModule']

"""
Autoencoder for inputs with shape (Ns, Nc, Nx*Ny) -> (-1, 5, Nx)

"""

class ParametricVAEModule(nn.Module):
    def __init__(self, n_input, latent_dim, num_params = 2, max_in = 1, min_in = 0, small = True, pred = False) -> None:
        super(ParametricVAEModule,self).__init__()
        # for normalization 
        self.register_buffer('min', min_in)
        self.register_buffer('input_range', max_in - min_in)

        self.num_params = num_params
        if small:
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
            self.decoder = nn.Sequential(
                nn.Linear(latent_dim + num_params,8),
                nn.ReLU(),
                nn.Linear(8,32),
                nn.ReLU(),
                nn.Linear(32,128*5),
                nn.ReLU(),
                nn.Unflatten(1,(5,128)), 
                nn.Linear(128,512),
                nn.ReLU(),
                nn.Linear(512, 2048),
                nn.ReLU(),
                nn.Linear(2048,n_input)
            )

            
            self.gen_mu  = nn.Linear(8,latent_dim)
            self.gen_std = nn.Linear(8,latent_dim)
        else:
            self.encoder = nn.Sequential(
                nn.Linear(n_input + num_params,8192), # 2^13
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
                nn.Linear(latent_dim + num_params,32),
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

        if pred:
            self.predictor = nn.Sequential(
                nn.Linear(latent_dim, 20),
                nn.ReLU(),
                nn.Linear(20, 20),
                nn.ReLU(),
                nn.Linear(20, 20),
                nn.ReLU(),
                nn.Linear(20, 20),
                nn.ReLU(),
                nn.Linear(20, latent_dim),
            )

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
    

    def decode(self, latent, param):
        # add parameter information
        p = torch.ones((latent.shape[0],self.num_params),device=latent.device)
        for i in range(self.num_params):
            p[...,i] = param[...,i]
        latent = torch.cat((latent,p),-1)

        decode = self.decoder(latent)
        # denormalize
        return decode * self.input_range + self.min
    
    def forward(self, x, param):
        # run autoencoder 
        latent,mu,log_var = self.encode(x,param)
        output = self.decode(latent,param)
        
        return output,mu,log_var
    
    def reparametrize(self, mu, log_var):
        std = torch.exp(0.5 * log_var)
        e = torch.randn_like(std)
        return e * std + mu