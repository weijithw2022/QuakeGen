import torch
import torch.nn as nn

class QuakeNetEncoder(nn.Module):
    """
    Encoder network for QuakeNet
    Processes E-N-Z seismic data in a time-series format.
    """

   