import torch
import torch.nn as nn
import torch.nn.parallel
import torch.nn.functional as F

class PhaseShuffle(nn.Module):
    # Phase shuffling module
    def __init__(self, shift_factor):
        super(PhaseShuffle, self).__init__()
        self.shift_factor = shift_factor

    def forward(self, x):
        if self.shift_factor == 0:
            return x
        batch_size, channels, samples = x.size()
        shift_values = torch.randint(-self.shift_factor, self.shift_factor + 1, (batch_size,), device=x.device)  
        # Initialize output tensor
        shuffled_x = torch.zeros_like(x)

        for i in range(batch_size):
            shift = shift_values[i].item()
            if shift > 0:
                shuffled_x[i, :, shift:] = x[i, :, :-shift]  # Shift right
            elif shift < 0:
                shuffled_x[i, :, :shift] = x[i, :, -shift:]  # Shift left
            else:
                shuffled_x[i] = x[i]  # No shift

        return shuffled_x

class Encoder(nn.Module):
    # WaveGAN Discriminator 
    def __init__(self, input_size, input_channels, base_channels, kernel_size,  stride, padding, alpha, phase_shuffle, latent_dim,  num_gpus=1, num_extra_layers=0, add_final_conv=True):
        super(Encoder, self).__init__()
        self.ngpu = num_gpus

        self.main = nn.Sequential(
            nn.Conv1d(input_channels, base_channels, kernel_size, stride, padding, bias=False),
            nn.LeakyReLU(alpha, inplace=True),
            PhaseShuffle(phase_shuffle),

            nn.Conv1d(base_channels, base_channels * 2, kernel_size, stride, padding, bias=False),
            nn.LeakyReLU(alpha, inplace=True),
            PhaseShuffle(phase_shuffle),
            
            nn.Conv1d(base_channels * 2, base_channels * 4, kernel_size, stride, padding, bias=False),
            nn.LeakyReLU(alpha, inplace=True),
            PhaseShuffle(phase_shuffle),

            nn.Conv1d(base_channels * 4, base_channels * 16, kernel_size, stride, padding, bias=False),
            nn.LeakyReLU(alpha, inplace=True), 
            PhaseShuffle(phase_shuffle),

            nn.Conv1d(base_channels * 16, latent_dim, kernel_size, stride, padding, bias=False),
            nn.LeakyReLU(alpha, inplace=True),

            nn.Flatten(),
            nn.Linear(base_channels * 16 * 16, 1)
        )

        def forward(self, x):
            return self.main(x)


