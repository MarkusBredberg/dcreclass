# models/__init__.py
# Expose model classes for import by training/evaluation scripts.
# Usage: from dcreclass.models import CNN, DualScatterSqueezeNet

from .classifiers import CNN, ImageCNN, ScatterNet, DualCNNSqueezeNet, DualScatterSqueezeNet