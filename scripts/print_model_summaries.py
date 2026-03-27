#!/usr/bin/env python
"""Print model summaries (structure + parameter counts) for RAW data classifiers."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import torch
from torchsummary import summary
from dcreclass.models import ImageCNN, SimpleScatterNet, DualCNNSqueezeNet, DualScatterSqueezeNet

# Input shapes for RAW data (images downsampled to 128x128, 1 channel)
IMG_SHAPE  = (1, 128, 128)
# Scattering coefficients: J=2, L=12, order=2 → 169 channels, spatial 128/2^J = 32
SCAT_SHAPE = (169, 32, 32)

device = "cpu"

models = [
    ("ImageCNN",         ImageCNN(input_shape=IMG_SHAPE, num_classes=2)),
    ("SimpleScatterNet", SimpleScatterNet(input_shape=IMG_SHAPE, num_classes=2)),
    ("DualCSN",          DualCNNSqueezeNet(input_shape=IMG_SHAPE, num_classes=2)),
    ("DualSSN",          DualScatterSqueezeNet(img_shape=IMG_SHAPE, scat_shape=SCAT_SHAPE, num_classes=2)),
]

for name, model in models:
    print(f"\n{'='*70}")
    print(f"  {name}  |  input: {'img' if name != 'DualSSN' else 'img + scat'}")
    print(f"{'='*70}")
    if name == "DualSSN":
        summary(model, input_size=[IMG_SHAPE, SCAT_SHAPE], device=device)
    else:
        summary(model, input_size=IMG_SHAPE, device=device)
