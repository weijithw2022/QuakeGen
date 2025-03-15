import pytest
import torch
import torch.nn as nn
from network import Decoder  

@pytest.fixture
def decoder():
    """Fixture to initialize the Decoder model."""
    latent_dim = 100
    base_channels = 8
    output_channels = 3
    kernel_size = 7
    stride = 4
    padding = 3
    
    model = Decoder(latent_dim, base_channels, output_channels, kernel_size, stride, padding)
    return model

@pytest.fixture
def sample_input():
    """Fixture to generate a random latent vector input."""
    batch_size = 4
    latent_dim = 100
    return torch.randn(batch_size, latent_dim)

def test_forward_pass(decoder, sample_input):
    """Test whether the Decoder's forward pass produces the expected output shape."""
    output = decoder(sample_input)
    
    assert isinstance(output, torch.Tensor), "Output is not a torch tensor"
    assert output.shape[0] == sample_input.shape[0], "Batch size mismatch in output"
    assert output.shape[1] == 3, "Output channel mismatch"  # Should match `output_channels`
    print(f"Output shape: {output.shape}")

def test_gradient_flow(decoder, sample_input):
    """Ensure gradients flow properly during backpropagation."""
    sample_input.requires_grad = True
    output = decoder(sample_input)
    
    loss = output.mean()
    loss.backward()
    
    assert sample_input.grad is not None, "Gradients did not flow properly"

def test_device_compatibility(decoder, sample_input):
    """Check if the model can run on both CPU and GPU."""
    if torch.cuda.is_available():
        device = torch.device("cuda")
        decoder.cuda()
        sample_input = sample_input.to(device)
        
        output = decoder(sample_input)
        assert output.device.type == "cuda", "Decoder output is not on CUDA"
    
    else:
        print("CUDA not available, skipping GPU test.")
