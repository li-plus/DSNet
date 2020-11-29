import torch
from torch.nn import functional as F


def calc_loc_loss(pred_loc: torch.Tensor,
                  test_loc: torch.Tensor,
                  cls_label: torch.Tensor,
                  use_smooth: bool = True
                  ) -> torch.Tensor:
    """Compute location regression loss only on positive samples.

    :param pred_loc: Predicted bbox offsets. Sized [N, S, 2].
    :param test_loc: Ground truth bbox offsets. Sized [N, S, 2].
    :param cls_label: Class labels where the 1 marks the positive samples. Sized
        [N, S].
    :param use_smooth: If true, use smooth L1 loss. Otherwise, use L1 loss.
    :return: Scalar loss value.
    """
    pos_idx = cls_label.eq(1).unsqueeze(-1).repeat((1, 1, 2))

    pred_loc = pred_loc[pos_idx]
    test_loc = test_loc[pos_idx]

    if use_smooth:
        loc_loss = F.smooth_l1_loss(pred_loc, test_loc)
    else:
        loc_loss = (pred_loc - test_loc).abs().mean()

    return loc_loss


def calc_cls_loss(pred: torch.Tensor, test: torch.Tensor) -> torch.Tensor:
    """Compute classification loss.

    :param pred: Predicted confidence (0-1). Sized [N, S].
    :param test: Class label where 1 marks positive, -1 marks negative, and 0
        marks ignored. Sized [N, S].
    :return: Scalar loss value.
    """
    pred = pred.view(-1)
    test = test.view(-1)

    pos_idx = test.eq(1).nonzero().squeeze(-1)
    pred_pos = pred[pos_idx].unsqueeze(-1)
    pred_pos = torch.cat([1 - pred_pos, pred_pos], dim=-1)
    gt_pos = torch.ones(pred_pos.shape[0], dtype=torch.long, device=pred.device)
    loss_pos = F.nll_loss(pred_pos.log(), gt_pos)

    neg_idx = test.eq(-1).nonzero().squeeze(-1)
    pred_neg = pred[neg_idx].unsqueeze(-1)
    pred_neg = torch.cat([1 - pred_neg, pred_neg], dim=-1)
    gt_neg = torch.zeros(pred_neg.shape[0], dtype=torch.long,
                         device=pred.device)
    loss_neg = F.nll_loss(pred_neg.log(), gt_neg)

    loss = (loss_pos + loss_neg) * 0.5
    return loss
