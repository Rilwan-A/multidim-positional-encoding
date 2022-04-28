import time

import numpy as np
import torch

from positional_encodings import *

def test_torch_1d_correct_shape():
    p_enc_1d = PositionalEncoding1D(10)
    x = torch.zeros((1, 6, 10))
    assert p_enc_1d(x).shape == (1, 6, 10)

    p_enc_1d = PositionalEncodingPermute1D(10)
    x = torch.zeros((1, 10, 6))
    assert p_enc_1d(x).shape == (1, 10, 6)


def test_torch_2d_correct_shape():
    p_enc_2d = PositionalEncoding2D(170)
    y = torch.zeros((1, 1, 1024, 170))
    assert p_enc_2d(y).shape == (1, 1, 1024, 170)

    p_enc_2d = PositionalEncodingPermute2D(169)
    y = torch.zeros((1, 169, 1, 1024))
    assert p_enc_2d(y).shape == (1, 169, 1, 1024)


def test_torch_3d_correct_shape():
    p_enc_3d = PositionalEncoding3D(125)
    z = torch.zeros((3, 5, 6, 4, 125))
    assert p_enc_3d(z).shape == (3, 5, 6, 4, 125)

    p_enc_3d = PositionalEncodingPermute3D(11)
    z = torch.zeros((7, 11, 5, 6, 4))
    assert p_enc_3d(z).shape == (7, 11, 5, 6, 4)


def test_torch_summer():
    model_with_sum = Summer(PositionalEncoding2D(125))
    model_wo_sum = PositionalEncoding2D(125)
    z = torch.rand(3, 5, 6, 125)
    assert (
        np.sum(np.abs((model_wo_sum(z) + z).numpy() - model_with_sum(z).numpy()))
        < 0.0001
    ), "The summer is not working properly!"


def test_torch_1D_cache():
    p_enc_1d = PositionalEncoding1D(10)
    x = torch.zeros((1, 6, 10))
    y = torch.zeros((1, 7, 10))

    assert not p_enc_1d.cached_penc
    assert p_enc_1d(x).shape == (1, 6, 10)
    assert p_enc_1d.cached_penc.shape == (1, 6, 10)

    assert p_enc_1d(y).shape == (1, 7, 10)
    assert p_enc_1d.cached_penc.shape == (1, 7, 10)
