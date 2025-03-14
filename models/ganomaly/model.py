import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.parallel
from network import Encoder, Decoder, Discriminator, Generator

def l1_loss(input, target):
    return torch.mean(torch.abs(input - target))

def l2_loss(input, target):
    return torch.mean(torch.pow((input-target), 2))


   