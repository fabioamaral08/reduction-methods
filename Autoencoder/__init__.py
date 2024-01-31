# from . import Autoencoder
# from . import ConvAutoencoder

from .Autoencoder import AutoencoderModule
from .ConvAutoencoder import ConvAutoencoderModule

__all__ = ['AutoencoderModule', 'ConvAutoencoderModule']
# __all__.extend(Autoencoder.__all__)
# __all__.extend(ConvAutoencoder.__all__)