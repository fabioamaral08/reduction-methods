# from . import Autoencoder
# from . import ConvAutoencoder

from .Autoencoder import AutoencoderModule
from .ConvAutoencoder import ConvAutoencoderModule
from .VariationalAutoencoder import VariationalAutoencoderModule
from .ParametricAutoencoder import ParametricAutoencoderModule
from .ParametricVAE import ParametricVAEModule
from .ParametricVAEModes import ParametricVAEModesModule
from .VAE_Transformer import VAE_Transformer

__all__ = ['AutoencoderModule', 'ConvAutoencoderModule', 'VariationalAutoencoderModule','ParametricVAEModule', 'ParametricAutoencoderModule', 'ParametricVAEModesModule','VAE_Transfomer']
# __all__.extend(Autoencoder.__all__)
# __all__.extend(ConvAutoencoder.__all__)