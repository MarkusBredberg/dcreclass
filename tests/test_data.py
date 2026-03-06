# tests/test_data.py
import torch
from dcreclass.data import get_classes

def test_get_classes_returns_list():
    classes = get_classes()
    assert isinstance(classes, list)
    assert all('tag' in c and 'description' in c for c in classes)