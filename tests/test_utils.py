# tests/test_utils.py
import torch
from dcreclass.utils import fold_T_axis

def test_fold_T_axis_shape():
    x = torch.randn(4, 3, 128, 128)
    out = fold_T_axis(x)
    assert out.ndim == 4             # should still be 4D
    assert out.shape[0] == 4         # batch size unchanged