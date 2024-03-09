# from . import Autoencoder
# from . import ConvAutoencoder

from .Autoencoder import AutoencoderModule
from .ConvAutoencoder import ConvAutoencoderModule
from .VariationalAutoencoder import VariationalAutoencoderModule

__all__ = ['AutoencoderModule', 'ConvAutoencoderModule', 'VariationalAutoencoderModule']
# __all__.extend(Autoencoder.__all__)
# __all__.extend(ConvAutoencoder.__all__)