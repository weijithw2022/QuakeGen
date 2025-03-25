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

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)

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
        self.fc = nn.Linear(8 * base_channels * 12, latent_dim)
    
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
        print("After FC Shape: ", x.shape)
        return x

class Decoder(nn.Module):
    # WaveGAN Generator
    def __init__(self, latent_dim, base_channels, output_channels, kernel_size, stride, padding, num_gpus=1, num_extra_layers=0, add_final_conv=True):
        super(Decoder, self).__init__()
        self.ngpu = num_gpus
        self.latent_dim = latent_dim

        self.relu = nn.ReLU(True)
        self.tanh = nn.Tanh()

        self.fc = nn.Linear(latent_dim, 8 * base_channels * 12)
        self.batchnorm1 = nn.BatchNorm1d(8 * base_channels)

        self.deconv1 = nn.ConvTranspose1d(8 * base_channels, 4 * base_channels, kernel_size, stride, padding=2, output_padding=0, bias=False)
        self.batchnorm2 = nn.BatchNorm1d(4 * base_channels)

        self.deconv2 = nn.ConvTranspose1d(4 * base_channels, 2 * base_channels, kernel_size, stride, padding=2, output_padding=1, bias=False)
        self.batchnorm3 = nn.BatchNorm1d(2 * base_channels)

        self.deconv3 = nn.ConvTranspose1d(2 * base_channels, base_channels, kernel_size, stride, padding=3, output_padding=1, bias=False)
        self.batchnorm4 = nn.BatchNorm1d(base_channels)

        self.deconv4 = nn.ConvTranspose1d(base_channels, output_channels, kernel_size, stride, padding=2, output_padding=1, bias=False)
    
    
    def forward(self, x):
        x = self.fc(x)
        x = x.view(x.size(0), -1, 12)   
        x = self.relu(self.batchnorm1(x))
        x = self.relu(self.batchnorm2(self.deconv1(x)))
        x = self.relu(self.batchnorm3(self.deconv2(x)))
        x = self.relu(self.batchnorm4(self.deconv3(x)))
        x = self.tanh(self.deconv4(x))
        return x
    
class Discriminator(nn.Module):
    def __init__(self, input_size, input_channels, base_channels, kernel_size,  stride, padding, alpha, latent_dim,  shuffle_factor):
        super(Discriminator, self).__init__()
        encoder = Encoder(input_size, input_channels, base_channels, kernel_size,  stride, padding, alpha, latent_dim,  shuffle_factor)
        # Only features(Extract all layers except the last one)
        self.features = nn.Sequential(*list(encoder.children())[:-1])
        # Classifier(Extract the last layer for classification)
        self.classifier = nn.Sequential(list(encoder.children())[-1], nn.Sigmoid())
    
    def forward(self, x):
        features = self.features(x)
        classifier = self.classifier(features.view(features.size(0), -1))
        return features, classifier

class Generator(nn.Module):
    def __init__(self, input_size, input_channels, output_channels, base_channels, kernel_size,  stride, padding, alpha, latent_dim,  shuffle_factor):
        super(Generator, self).__init__()
        self.encoder = Encoder(input_size, input_channels, base_channels, kernel_size,  stride, padding, alpha, latent_dim,  shuffle_factor)
        self.decoder = Decoder(latent_dim, base_channels, output_channels, kernel_size, stride, padding)
        
    def forward(self, x):
        dense_rep_input = self.encoder(x)
        generated_wave = self.decoder(dense_rep_input)
        dense_rep_output = self.encoder(generated_wave)
        return generated_wave, dense_rep_input, dense_rep_output

class WGanomaly(nn.Module):
    """
    Encoder network for QuakeNet
    Processes E-N-Z seismic data in a time-series format.
    """
    def __init__(self, input_size, input_channels, base_channels, kernel_size, stride, padding, alpha, latent_dim, shuffle_factor, wadv, wcon, wenc, num_gpus=1, num_extra_layers=0, add_final_conv=True):
        super(WGanomaly, self).__init__()
        self.ngpu = num_gpus
        self.latent_dim = latent_dim
        self.shuffle_factor = shuffle_factor
        self.wadv = wadv
        self.wcon = wcon
        self.wenc = wenc
        self.l2_loss = nn.MSELoss()
        self.l1_loss = nn.L1Loss()

        self.encoder = Encoder(input_size, input_channels, base_channels, kernel_size, stride, padding, alpha, latent_dim, shuffle_factor)
        self.decoder = Decoder(latent_dim, base_channels, input_channels, kernel_size, stride, padding, num_gpus, num_extra_layers, add_final_conv)
        self.discriminator = Discriminator(input_size, input_channels, base_channels, kernel_size, stride, padding, alpha, latent_dim, shuffle_factor)
        self.generator = Generator(latent_dim, base_channels, input_channels, kernel_size, stride, padding, num_gpus, num_extra_layers, add_final_conv)
        # self.generator.apply(weights_init)
        # self.discriminator.apply(weights_init)

    def forward(self, x):
        # Encoder-Decoder Process
        generated_wave, dense_rep_input, dense_rep_output = self.generator(x)
        # Discriminator Process
        features_real, classifier_real = self.discriminator(x)
        # Generated Wave Discriminator Process
        features_generated, classifier_generated = self.discriminator(generated_wave)
        return {
            'generated_wave': generated_wave,
            'dense_rep_input': dense_rep_input,
            'dense_rep_output': dense_rep_output,
            'features_real': features_real,
            'classifier_real': classifier_real,
            'features_generated': features_generated,
            'classifier_generated': classifier_generated
        }

    def compute_loss(self, x, outputs):
        adv_loss = self.l2_loss(outputs["features_real"], outputs["features_generated"])
        con_loss = self.l1_loss(x, outputs["generated_wave"])
        enc_loss = self.l2_loss(outputs["dense_rep_input"], outputs["dense_rep_output"])

        """  # Normalize weights dynamically
            wadv = 1.0 / (adv_loss.detach() + 1e-8)
            wcon = 1.0 / (con_loss.detach() + 1e-8)
            wenc = 1.0 / (enc_loss.detach() + 1e-8)

            # Scale to keep relative balance
            sum_weights = wadv + wcon + wenc
            wadv /= sum_weights
            wcon /= sum_weights
            wenc /= sum_weights
        """

        loss = self.wadv*adv_loss + self.wcon*con_loss + self.wenc*enc_loss

        return{
            'loss': loss,
            'adv_loss': adv_loss,
            'con_loss': con_loss,
            'enc_loss': enc_loss
        }