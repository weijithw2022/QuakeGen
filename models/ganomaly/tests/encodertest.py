import torch
import torch.nn as nn
import torch.nn.functional as F
import pytest
from network import Encoder 

@pytest.fixture
def sample_input():
    batch_size = 4
    input_channels = 3
    input_size = 3000 
    return torch.randn(batch_size, input_channels, input_size)

@pytest.fixture
def encoder():
    return Encoder(
        input_size=3000,
        input_channels=3,
        base_channels=8,
        kernel_size=7,
        stride=4,
        padding=3,
        alpha=0.2,
        latent_dim=3,
        shuffle_factor=2,
        num_gpus=1,
        num_extra_layers=0
    )

def test_encoder_initialization(encoder):
    """Test if the Encoder model initializes correctly."""
    assert isinstance(encoder, Encoder), "Encoder failed to initialize"

def test_forward_pass(encoder, sample_input):
    """Test if the Encoder produces an output of the correct shape."""
    output = encoder(sample_input)
    assert output.shape == (sample_input.shape[0], 3), f"Unexpected output shape: {output.shape}"

def test_phase_shuffle_shape_preservation(encoder, sample_input):
    """Test that phase shuffle does not change shape."""
    shuffled = encoder.phaseshuffle(sample_input, shuffle_factor=2)
    assert shuffled.shape == sample_input.shape, "Phase shuffle changed input shape"

def test_encoder_on_cuda(encoder, sample_input):
    """Test if the model runs on CUDA if available."""
    if torch.cuda.is_available():
        encoder.cuda()
        sample_input = sample_input.cuda()
        output = encoder(sample_input)
        assert output.is_cuda, "Model failed to run on CUDA"

def test_gradient_flow(encoder, sample_input):
    """Ensure gradients flow properly."""
    sample_input.requires_grad = True
    output = encoder(sample_input)
    output.mean().backward()
    assert sample_input.grad is not None, "Gradients did not flow properly"

if __name__ == "__main__":
    pytest.main()
