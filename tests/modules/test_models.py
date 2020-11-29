import torch

from modules import model_zoo


def test_models():
    seq = torch.randn(1, 30, 64)

    model_ab = model_zoo.get_model('anchor-based', base_model='attention',
                                   num_feature=64, num_hidden=16,
                                   anchor_scales=(4, 8, 16), num_head=8)
    pred_cls, pred_loc = model_ab(seq)
    assert pred_cls.shape == (30, 3)
    assert pred_loc.shape == (30, 3, 2)

    model_ab = model_zoo.get_model('anchor-free', base_model='attention',
                                   num_feature=64, num_hidden=16, num_head=8)
    pred_cls, pred_loc, pred_ctr = model_ab(seq)
    assert pred_cls.shape == (30,)
    assert pred_loc.shape == (30, 2)
    assert pred_ctr.shape == (30,)
