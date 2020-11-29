import math

import torch
from torch.nn import functional as F

from anchor_free import losses


def test_focal_loss():
    alpha = 0.25
    gamma = 2
    pred = torch.tensor([[0.6, 0.4], [0.3, 0.7]], dtype=torch.float32)
    test = torch.tensor([0, 1], dtype=torch.long)
    output = losses.focal_loss(pred, test)

    answer = 0
    # first sample
    alpha_t = torch.tensor([alpha, 1 - alpha], dtype=torch.float32)
    p_t = torch.tensor([0.6, 1 - 0.4], dtype=torch.float32)
    loss = - alpha_t * (1 - p_t).pow(gamma) * p_t.log()
    answer += loss.sum()

    # second sample
    alpha_t = torch.tensor([1 - alpha, alpha], dtype=torch.float32)
    p_t = torch.tensor([1 - 0.3, 0.7], dtype=torch.float32)
    loss = - alpha_t * (1 - p_t).pow(gamma) * p_t.log()
    answer += loss.sum()

    assert torch.isclose(output, answer)


def test_focal_loss_with_logits():
    pred = torch.tensor([[-1, 2], [3, 4]], dtype=torch.float32)
    test = torch.tensor([0, 1], dtype=torch.long)

    a = losses.focal_loss_with_logits(pred, test)
    b = losses.focal_loss(pred.sigmoid(), test)
    assert torch.isclose(a, b)


def test_iou_offset():
    offset_a = torch.tensor([[1, 1], [3, 2]], dtype=torch.float32)
    offset_b = torch.tensor([[4, 2], [2, 5]], dtype=torch.float32)
    output = losses.iou_offset(offset_a, offset_b)
    answer = torch.tensor([2 / 6, 4 / 8], dtype=torch.float32)
    assert torch.isclose(output, answer).all()


def test_calc_cls_loss():
    pred = torch.tensor([0.4, 0.7], dtype=torch.float32)
    test = torch.tensor([0, 1], dtype=torch.long)

    output = losses.calc_cls_loss(pred, test, kind='focal')
    answer = losses.focal_loss(torch.tensor(
        [[0.6, 0.4], [0.3, 0.7]]), test, reduction='sum') / test.sum()
    assert torch.isclose(output, answer)

    output = losses.calc_cls_loss(pred, test, kind='cross-entropy')
    answer = torch.tensor((-math.log(1 - 0.4) - math.log(0.7)) / 2,
                          dtype=torch.float32)
    assert torch.isclose(output, answer)


def test_calc_loc_loss():
    pred = torch.tensor([[1, 1], [3, 2], [8, 7]], dtype=torch.float32)
    test = torch.tensor([[4, 2], [2, 5], [6, 9]], dtype=torch.float32)
    loc_weight = torch.tensor([1, 1, 0], dtype=torch.float32)

    output = losses.calc_loc_loss(pred, test, loc_weight, kind='soft-iou')
    iou = torch.tensor([2 / 6, 4 / 8], dtype=torch.float32)
    answer = -torch.log(iou + 1e-8).mean()
    assert torch.isclose(output, answer)

    output = losses.calc_loc_loss(pred, test, loc_weight, kind='smooth-l1')
    answer = F.smooth_l1_loss(pred[:2], test[:2])
    assert torch.isclose(output, answer)


def test_calc_ctr_loss():
    pred = torch.tensor([0.4, 0.6, 0.3, 0.8], dtype=torch.float32)
    test = torch.tensor([0.8, 0.1, 0.6, 0.4], dtype=torch.float32)
    pos_mask = torch.tensor([1, 0, 1, 0], dtype=torch.bool)
    output = losses.calc_ctr_loss(pred, test, pos_mask)

    pred = torch.tensor([0.4, 0.3], dtype=torch.float32)
    test = torch.tensor([0.8, 0.6], dtype=torch.float32)
    answer = F.binary_cross_entropy(pred, test)
    assert torch.isclose(output, answer)
