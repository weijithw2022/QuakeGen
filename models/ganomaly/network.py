import torch
import torch.nn as nn
import torch.nn.parallel
import torch.nn.functional as F

class PhaseShuffle(nn.Module):
    def __init__(self, shift_factor):
        super(PhaseShuffle, self).__init__()
        self.shift_factor = shift_factor

    def forward(self, x):
        if self.shift_factor == 0:
            return x

        batch_size, channels, samples = x.size()
        shift_values = torch.randint(-self.shift_factor, self.shift_factor + 1, (batch_size,), device=x.device)
        shuffled_x = torch.zeros_like(x)

        for i in range(batch_size):
            shift = shift_values[i].item()
            if shift > 0:
                shuffled_x[i, :, shift:] = x[i, :, :-shift]  # Shift right
                shuffled_x[i, :, :shift] = x[i, :, 0:shift]  # Pad left with edge values
            elif shift < 0:
                shuffled_x[i, :, :shift] = x[i, :, -shift:]  # Shift left
                shuffled_x[i, :, shift:] = x[i, :, -1].unsqueeze(-1).expand(channels, -shift)  # Fix padding
            else:
                shuffled_x[i] = x[i]  # No shift

        return shuffled_x

class Encoder(nn.Module):
    # WaveGAN Discriminator 
    def __init__(self, input_size, input_channels, base_channels, kernel_size,  stride, padding, alpha, latent_dim,  shuffle_factor, num_gpus=1, num_extra_layers=0, add_final_conv=True):
        super(Encoder, self).__init__()
        self.ngpu = num_gpus
        self.shuffle_factor = shuffle_factor

        self.leaky_relu = nn.LeakyReLU(alpha, inplace=True)
        self.conv1 = nn.Conv1d(input_channels, base_channels, kernel_size, stride, padding, bias=False)

        self.conv2 = nn.Conv1d(base_channels, 2 * base_channels, kernel_size, stride, padding, bias=False)
        self.batchnorm2 = nn.BatchNorm1d(2 *base_channels)

        self.conv3 = nn.Conv1d(2 * base_channels, 4 * base_channels, kernel_size, stride, padding, bias=False)
        self.batchnorm3 = nn.BatchNorm1d(4 * base_channels)

        self.conv4 = nn.Conv1d(4 * base_channels, 8 * base_channels, kernel_size, stride, padding, bias=False)
        self.batchnorm4 = nn.BatchNorm1d(8 * base_channels)
        #  input_size // (stride ** 4) * 8 * base_channels
        self.fc = nn.Linear(8 * base_channels * 12, 1)
    
    def phaseshuffle(self, x, shuffle_factor):
        if shuffle_factor == 0:
            return x  # No phase shuffle applied

        batch_size, channels, seq_len = x.shape
        phase = torch.randint(-shuffle_factor, shuffle_factor + 1, (1,)).item()  # Random shift in range [-rad, rad]

        # Compute left and right padding
        pad_l = max(phase, 0)
        pad_r = max(-phase, 0)

        # Apply reflection padding
        x_padded = F.pad(x, (pad_l, pad_r), mode='reflect')

        # Slice the tensor to maintain original sequence length
        x_shuffled = x_padded[:, :, pad_r: pad_r + seq_len]
        
        return x_shuffled
    
    def forward(self, x):
        x = self.leaky_relu(self.conv1(x))
        x = self.phaseshuffle(x, self.shuffle_factor)
        x = self.leaky_relu(self.batchnorm2(self.conv2(x)))
        x = self.phaseshuffle(x, self.shuffle_factor)
        x = self.leaky_relu(self.batchnorm3(self.conv3(x)))
        x = self.phaseshuffle(x, self.shuffle_factor)
        x = self.leaky_relu(self.batchnorm4(self.conv4(x)))
        x = x.view(x.size(0), -1)
        # x = torch.reshape(x, (x.size(0), -1))
        x = self.fc(x)
        return x

class Decoder(nn.Module):
    # WaveGAN Generator
    def __init__(self, latent_dim, base_channels, output_channels, kernel_size, stride, padding, phase_shuffle, num_gpus=1, num_extra_layers=0, add_final_conv=True):
        super(Decoder, self).__init__()
        self.ngpu = num_gpus
        self.latent_dim = latent_dim
        
        self.main = nn.Sequential(
            nn.Linear(latent_dim, 8 * base_channels * 12),
            Reshape(8 * base_channels, 12),

            nn.ReLU(True),
            nn.ConvTranspose1d(8 * base_channels, 4 * base_channels, kernel_size, stride, padding=2, output_padding=0, bias=False),

            nn.ReLU(True),
            nn.ConvTranspose1d(4 * base_channels, 2 * base_channels, kernel_size, stride, padding=2, output_padding=1, bias=False),

            nn.ReLU(True),
            nn.ConvTranspose1d(2 * base_channels, base_channels, kernel_size, stride, padding=3,output_padding=1, bias=False),

            nn.ReLU(True),
            nn.ConvTranspose1d(base_channels, output_channels, kernel_size, stride, padding=2, output_padding=1, bias=False),

            nn.Tanh()
        )
    
    def forward(self, x):
        return self.main(x)