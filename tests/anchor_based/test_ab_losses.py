import math

import torch

from anchor_based import losses


def test_calc_cls_loss():
    pred = torch.tensor([
        [0.4, 0.6],
        [0.0, 0.4],
        [0.9, 0.8],
        [0.3, 0.2],
    ], dtype=torch.float32)
    test = torch.tensor([
        [1, -1],
        [0, 0],
        [1, 0],
        [-1, 0]
    ], dtype=torch.long)
    out = losses.calc_cls_loss(pred, test).item()
    ans = (-math.log(.4) - math.log(.9) - math.log(1 - .6) - math.log(1 - .3)) / 4
    assert math.isclose(out, ans, abs_tol=1e-5)


def test_calc_loc_loss():
    pred = torch.tensor([[0, 1],
                         [2, 3],
                         [4, 5]], dtype=torch.float32).unsqueeze(1)
    test = torch.tensor([[0.5, 2],
                         [-1, 5],
                         [4, 5]], dtype=torch.float32).unsqueeze(1)
    cls_label = torch.tensor([1, 1, 0], dtype=torch.long).unsqueeze(1)

    l1 = losses.calc_loc_loss(pred, test, cls_label, use_smooth=False).item()
    ans_l1 = (0.5 + 1 + 3 + 2) / 4
    assert math.isclose(l1, ans_l1, abs_tol=1e-5)

    smoothl1 = losses.calc_loc_loss(pred, test, cls_label, use_smooth=True).item()
    ans_smoothl1 = (0.125 + 0.5 + 2.5 + 1.5) / 4
    assert math.isclose(smoothl1, ans_smoothl1, abs_tol=1e-5)
