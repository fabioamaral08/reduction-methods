# from . import Autoencoder
# from . import ConvAutoencoder

from .Autoencoder import *
from .ConvAutoencoder import *
from .VariationalAutoencoder import *
from .ParametricAutoencoder import *
from .ParametricVAE import *
from .ParametricVAEModes import *
from .VAE_Transformer import *
from .ParametricBVAE import *

# __all__ = ['AutoencoderModule', 'ConvAutoencoderModule', 'VariationalAutoencoderModule','ParametricVAEModule', 'ParametricAutoencoderModule', 'ParametricVAEModesModule','VAE_Transfomer']
# __all__.extend(Autoencoder.__all__)
# __all__.extend(ConvAutoencoder.__all__)