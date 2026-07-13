import torch
import torch.nn as nn
from itertools import chain, combinations
from torch.autograd.functional import jvp

__all__ = ['SINDyAutoencoderModule', 'loss_sindy_ae']

"""
Convolutional autoencoder for inputs with shape (Ns, Nc, Nx, Ny).
Encoder: n_layers conv blocks (Conv2d -> BatchNorm -> ReLU -> MaxPool) growing
channels by n_filters per layer, followed by a dense projection to the latent space.
Decoder mirrors the encoder with transposed convolutions back to the input resolution.
"""


class SINDyTorch(nn.Module):
    def __init__(self, n_latent, degree, include_bias = False) -> None:
        super().__init__()
        # Materialise as a list — the generator would be exhausted on first iteration otherwise
        self._poly_com = list(self._combinations(n_latent, int(not include_bias), degree, False, include_bias))
        self.n_output_features_ = len(self._poly_com)

        self.coefficients = nn.Parameter(torch.ones(self.n_output_features_, n_latent))
        self.mask = torch.ones_like(self.coefficients, requires_grad=False)
    

    @staticmethod
    def _combinations(
        n_features, min_degree, max_degree, interaction_only, include_bias
    ):
        comb = combinations
        start = max(1, min_degree)
        iter = chain.from_iterable(
            comb(range(n_features), i) for i in range(start, max_degree + 1)
        )
        if include_bias:
            iter = chain(comb(range(n_features), 0), iter)
        return iter
    
    def transform(self, X: torch.Tensor):
        # torch.stack keeps gradients flowing; in-place assignment into a zeros tensor would not
        # prod over an empty index (bias term) gives 1.0 per sample, which is correct
        cols = [X[:, list(combo)].prod(dim=-1) for combo in self._poly_com]
        return torch.stack(cols, dim=-1)
    

    def forward(self, X):
        Theta = self.transform(X)
        return Theta.matmul(self.coefficients * self.mask)

    def threshold(self, eps):
        self.mask = 1. * self.coefficients > eps
        self.coefficients = self.coefficients * self.mask



class _ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
        )

    def forward(self, x):
        return self.block(x)


class _SINDyEncoder(nn.Module):
    def __init__(self, in_channels, input_size, n_filters, n_layers, latent_dim) -> None:
        super().__init__()

        h, w = input_size
        channels = in_channels
        convs = []
        for i in range(n_layers):
            out_channels = n_filters * (i + 1)
            convs.append(_ConvBlock(channels, out_channels))
            channels, h, w = out_channels, h // 2, w // 2

        self.convs = nn.ModuleList(convs)
        self.pool = nn.MaxPool2d(kernel_size=2, ceil_mode=True)
        self.flatten = nn.Flatten()
        self.dense = nn.Linear(channels * h * w, latent_dim)

    def forward(self, x):
        for conv in self.convs:
            x = conv(x)
            x = self.pool(x)

        x = self.flatten(x)
        x = self.dense(x)
        return x


class _SINDyDecoder(nn.Module):
    def __init__(self, n_filters, n_layers, latent_dim, old_shape, output_channels) -> None:
        super().__init__()

        self.old_shape = old_shape  # (C, H, W)
        c, h, w = old_shape

        self.dense = nn.Sequential(
            nn.Linear(latent_dim, c * h * w),
            nn.ReLU(),
        )

        channels = c
        deconvs = []
        for i in range(n_layers, 0, -1):
            out_channels = n_filters * i
            deconvs.append(nn.Sequential(
                nn.ConvTranspose2d(channels, out_channels, kernel_size=3, stride=2, padding=1, output_padding=1, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(),
            ))
            channels = out_channels

        self.deconvs = nn.ModuleList(deconvs)
        self.deconv_final = nn.Conv2d(channels, output_channels, kernel_size=3, padding=1)

    def forward(self, x):
        x = self.dense(x)
        x = x.view(-1, *self.old_shape)

        for deconv in self.deconvs:
            x = deconv(x)

        return self.deconv_final(x)


class SINDyAutoencoderModule(nn.Module):
    def __init__(self, n_filters, n_layers, latent_dim, input_shape, degree = 2, include_bias=False) -> None:
        super().__init__()

        self.n_filters = n_filters
        self.n_layers = n_layers
        self.latent_dim = latent_dim
        self.input_shape = input_shape

        channels, h, w = input_shape

        self.encoder = _SINDyEncoder(channels, (h, w), n_filters, n_layers, latent_dim)
        self.sindy = SINDyTorch(latent_dim, degree, include_bias)
        old_shape = (n_filters * n_layers, h // 2 ** n_layers, w // 2 ** n_layers)
        self.decoder = _SINDyDecoder(n_filters, n_layers, latent_dim, old_shape, channels)

    def forward(self, x):
        latent = self.encoder(x)
        return self.decoder(latent)
    
def kernel_fenep(x:torch.Tensor, y:torch.Tensor, L2:float):
    trace = x[:,0] * y[:,0] + x[:,2] * y[:,2] + 2* x[:,1] * y[:,1]
    return L2 * trace / (L2 - trace)

def kernel_loss (x:torch.Tensor, y:torch.Tensor, L2:float):
    return (kernel_fenep(x,x,L2) + kernel_fenep(y,y,L2) - 2 * kernel_fenep(x,y,L2))

def loss_sindy_ae(
    x: torch.Tensor,
    cae: SINDyAutoencoderModule,
    x_dot: torch.Tensor,
    lambda1: float = 1.0,
    lambda2: float = 1.0,
    lambda3: float = 1e-4,
    rec_energy = False,
    L2:float | None = None,
) -> torch.Tensor:
    """
    Champion et al. (2019) SINDy-autoencoder loss.

    Terms (matching the paper's formula):
      L_rec   = ||x - ψ(z)||²
      L_dxdt  = ||ẋ - (∇_z ψ(z)) Θ(z)Ξ||²   (SINDy loss in x-space)
      L_dzdt  = ||(∇_x z) ẋ - Θ(z)Ξ||²        (SINDy loss in z-space)
      L_reg   = λ₃ ||Ξ||₁
    """
    
    mse = nn.MSELoss()
    if L2 is None and not rec_energy:
        raise ValueError('L2 must be assigned a float if using rec_energy= True')

    # JVP of encoder: z = φ(x),  dz/dt_true = J_φ(x) · ẋ
    z, dzdt_true = jvp(cae.encoder, (x,), (x_dot,), create_graph=True)

    # SINDy prediction: dz/dt_hat = Θ(z) Ξ
    dzdt = cae.sindy(z)

    # JVP of decoder: x_hat = ψ(z),  dx_hat/dt = J_ψ(z) · dz/dt_hat
    x_hat, dxdt_hat = jvp(cae.decoder, (z,), (dzdt,), create_graph=True)

    loss_rec  = kernel_loss(x, x_hat,L2) if rec_energy else mse(x, x_hat)
    loss_dxdt = mse(x_dot, dxdt_hat)
    loss_dzdt = mse(dzdt_true, dzdt)
    loss_reg  = cae.sindy.coefficients.abs().mean()

    return loss_rec + lambda1 * loss_dxdt + lambda2 * loss_dzdt + lambda3 * loss_reg
