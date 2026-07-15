import torch
import torch.nn as nn
from itertools import chain, combinations
from torch.autograd.functional import jvp
from typing import Sequence

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
        self.register_buffer('mask', torch.ones_like(self.coefficients))
    

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
        with torch.no_grad():
            self.mask = (1. * self.coefficients > eps).to(self.coefficients.dtype)
            self.coefficients.mul_(self.mask)



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
            # matches nn.MaxPool2d(kernel_size=2, ceil_mode=True) below, which rounds up
            channels, h, w = out_channels, (h + 1) // 2, (w + 1) // 2

        self.out_channels, self.out_h, self.out_w = channels, h, w
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
    def __init__(self, n_filters, n_layers, latent_dim, old_shape, output_channels, upsample_mode="deconv") -> None:
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
            if upsample_mode == "resize_conv":
                # nearest-upsample + stride-1 conv avoids the checkerboard artifacts that
                # ConvTranspose2d(kernel_size=3, stride=2) produces (uneven kernel overlap)
                block = nn.Sequential(
                    nn.Upsample(scale_factor=2, mode="nearest"),
                    nn.Conv2d(channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False),
                    nn.BatchNorm2d(out_channels),
                    nn.ReLU(),
                )
            elif upsample_mode == "deconv":
                block = nn.Sequential(
                    nn.ConvTranspose2d(channels, out_channels, kernel_size=3, stride=2, padding=1, output_padding=1, bias=False),
                    nn.BatchNorm2d(out_channels),
                    nn.ReLU(),
                )
            else:
                raise ValueError(f"Unknown upsample_mode: {upsample_mode!r}")
            deconvs.append(block)
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
    def __init__(self, n_filters, n_layers, latent_dim, input_shape, degree = 2, include_bias=False, decoder_upsample="deconv") -> None:
        super().__init__()

        self.n_filters = n_filters
        self.n_layers = n_layers
        self.latent_dim = latent_dim
        self.input_shape = input_shape

        channels, h, w = input_shape

        self.encoder = _SINDyEncoder(channels, (h, w), n_filters, n_layers, latent_dim)
        self.sindy = SINDyTorch(latent_dim, degree, include_bias)
        old_shape = (self.encoder.out_channels, self.encoder.out_h, self.encoder.out_w)
        self.decoder = _SINDyDecoder(n_filters, n_layers, latent_dim, old_shape, channels, upsample_mode=decoder_upsample)

    def forward(self, x):
        latent = self.encoder(x)
        return self.decoder(latent)
    
def kernel_fenep(x:torch.Tensor, y:torch.Tensor, L2:float):
    trace = x[:,0] * y[:,0] + x[:,2] * y[:,2] + 2* x[:,1] * y[:,1]
    return L2 * trace / (L2 - trace)

def kernel_loss (x:torch.Tensor, y:torch.Tensor, L2:float):
    return (kernel_fenep(x,x,L2) + kernel_fenep(y,y,L2) - 2 * kernel_fenep(x,y,L2))

def trace_const(x:torch.Tensor, L2:float, margin:float = 1e-1):
    trace_hat = x[:,0]**2 + x[:,2]**2 + 2*x[:,1]**2
    return torch.relu(trace_hat - (L2 - margin)).pow(2).mean()

def loss_sindy_ae(
    x: torch.Tensor,
    cae: SINDyAutoencoderModule,
    x_dot: torch.Tensor,
    weights: Sequence[float] = (1.0, 1.0, 1e-4, 1.0, 1.0),
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

    weights: [lambda_dxdt, lambda_dzdt, lambda_reg, lambda_energy, lambda_barrier]
    lambda_energy and lambda_barrier are only applied when rec_energy=True.
    """
    lambda1, lambda2, lambda3, lambda4, lambda5 = weights
    mse = nn.MSELoss()
    if L2 is None and not rec_energy:
        raise ValueError('L2 must be assigned a float if using rec_energy= True')

    # JVP of encoder: z = φ(x),  dz/dt_true = J_φ(x) · ẋ
    z, dzdt_true = jvp(cae.encoder, (x,), (x_dot,), create_graph=True)

    # SINDy prediction: dz/dt_hat = Θ(z) Ξ
    dzdt = cae.sindy(z)

    # JVP of decoder: x_hat = ψ(z),  dx_hat/dt = J_ψ(z) · dz/dt_hat
    x_hat, dxdt_hat = jvp(cae.decoder, (z,), (dzdt,), create_graph=True)

    
    loss_rec  = mse(x, x_hat)
    loss_dxdt = mse(x_dot, dxdt_hat)
    loss_dzdt = mse(dzdt_true, dzdt)
    loss_reg  = cae.sindy.coefficients.abs().mean()
    loss_sum = loss_rec + lambda1 * loss_dxdt + lambda2 * loss_dzdt + lambda3 * loss_reg
    if rec_energy:
        loss_energy = torch.relu(kernel_loss(x, x_hat,L2)).mean()
        loss_barrier = trace_const(x_hat, L2)
        loss_sum += lambda4 * loss_energy + lambda5 * loss_barrier
    return loss_sum
