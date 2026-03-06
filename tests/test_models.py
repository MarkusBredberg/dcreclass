# tests/test_models.py
import torch
from dcreclass.models import CNN, DualScatterSqueezeNet, DualCSN

def test_cnn_output_shape():
    x = torch.randn(4, 1, 128, 128)       # batch of 4 images
    model = CNN(input_shape=(1, 128, 128), num_classes=2)
    assert model(x).shape == (4, 2)

def test_dualssn_output_shape():
    img  = torch.randn(4, 1, 128, 128)
    scat = torch.randn(4, 81, 32, 32)     # scattering coeff shape
    model = DualScatterSqueezeNet(img_shape=(1,128,128), scat_shape=(81,32,32), num_classes=2)
    assert model(img, scat).shape == (4, 2)
    
def test_dualcsn_output_shape():
    img  = torch.randn(4, 1, 128, 128)
    scat = torch.randn(4, 81, 32, 32)     # scattering coeff shape
    model = DualCSN(img_shape=(1,128,128), scat_shape=(81,32,32), num_classes=2)
    assert model(img, scat).shape == (4, 2)