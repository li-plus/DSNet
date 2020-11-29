import torch
from torch.nn import functional as F


def calc_cls_loss(pred: torch.Tensor,
                  test: torch.Tensor,
                  kind: str = 'focal'
                  ) -> torch.Tensor:
    """Compute classification loss on both positive and negative samples.

    :param pred: Predicted class. Sized [N, S].
    :param test: Class label where 1 marks positive, -1 marks negative, and 0
        marks ignored. Sized [N, S].
    :param kind: Loss type. Choose from (focal, cross-entropy).
    :return: Scalar loss value.
    """
    test = test.type(torch.long)
    num_pos = test.sum()

    pred = pred.unsqueeze(-1)
    pred = torch.cat([1 - pred, pred], dim=-1)

    if kind == 'focal':
        loss = focal_loss(pred, test, reduction='sum')
    elif kind == 'cross-entropy':
        loss = F.nll_loss(pred.log(), test)
    else:
        raise ValueError(f'Invalid loss type {kind}')

    loss = loss / num_pos
    return loss


def iou_offset(offset_a: torch.Tensor,
               offset_b: torch.Tensor,
               eps: float = 1e-8
               ) -> torch.Tensor:
    """Compute IoU offsets between multiple offset pairs.

    :param offset_a: Offsets of N positions. Sized [N, 2].
    :param offset_b: Offsets of N positions. Sized [N, 2].
    :param eps: Small floating value to prevent division by zero.
    :return: IoU values of N positions. Sized [N].
    """
    left_a, right_a = offset_a[:, 0], offset_a[:, 1]
    left_b, right_b = offset_b[:, 0], offset_b[:, 1]

    length_a = left_a + right_a
    length_b = left_b + right_b

    intersect = torch.min(left_a, left_b) + torch.min(right_a, right_b)
    intersect[intersect < 0] = 0
    union = length_a + length_b - intersect
    union[union <= 0] = eps

    iou = intersect / union
    return iou


def calc_loc_loss(pred_loc: torch.Tensor,
                  test_loc: torch.Tensor,
                  cls_label: torch.Tensor,
                  kind: str = 'soft-iou',
                  eps: float = 1e-8
                  ) -> torch.Tensor:
    """Compute soft IoU loss for regression only on positive samples.

    :param pred_loc: Predicted offsets. Sized [N, 2].
    :param test_loc: Ground truth offsets. Sized [N, 2].
    :param cls_label: Class label specifying positive samples.
    :param kind: Loss type. Choose from (soft-iou, smooth-l1).
    :param eps: Small floating value to prevent division by zero.
    :return: Scalar loss value.
    """
    cls_label = cls_label.type(torch.bool)
    pred_loc = pred_loc[cls_label]
    test_loc = test_loc[cls_label]

    if kind == 'soft-iou':
        iou = iou_offset(pred_loc, test_loc)
        loss = -torch.log(iou + eps).mean()
    elif kind == 'smooth-l1':
        loss = F.smooth_l1_loss(pred_loc, test_loc)
    else:
        raise ValueError(f'Invalid loss type {kind}')

    return loss


def calc_ctr_loss(pred, test, pos_mask):
    pos_mask = pos_mask.type(torch.bool)

    pred = pred[pos_mask]
    test = test[pos_mask]

    loss = F.binary_cross_entropy(pred, test)
    return loss


def one_hot_embedding(labels: torch.Tensor, num_classes: int) -> torch.Tensor:
    """Embedding labels to one-hot form.

    :param labels: Class labels. Sized [N].
    :param num_classes: Number of classes.
    :return: One-hot encoded labels. sized [N, #classes].
    """
    eye = torch.eye(num_classes, device=labels.device)
    return eye[labels]


def focal_loss(x: torch.Tensor,
               y: torch.Tensor,
               alpha: float = 0.25,
               gamma: float = 2,
               reduction: str = 'sum'
               ) -> torch.Tensor:
    """Compute focal loss for binary classification.
        FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)

    :param x: Predicted confidence. Sized [N, D].
    :param y: Ground truth label. Sized [N].
    :param alpha: Alpha parameter in focal loss.
    :param gamma: Gamma parameter in focal loss.
    :param reduction: Aggregation type. Choose from (sum, mean, none).
    :return: Scalar loss value.
    """
    _, num_classes = x.shape

    t = one_hot_embedding(y, num_classes)

    # p_t = p if t > 0 else 1-p
    p_t = x * t + (1 - x) * (1 - t)
    # alpha_t = alpha if t > 0 else 1-alpha
    alpha_t = alpha * t + (1 - alpha) * (1 - t)
    # FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)
    fl = -alpha_t * (1 - p_t).pow(gamma) * p_t.log()

    if reduction == 'sum':
        fl = fl.sum()
    elif reduction == 'mean':
        fl = fl.mean()
    elif reduction == 'none':
        pass
    else:
        raise ValueError(f'Invalid reduction mode {reduction}')

    return fl


def focal_loss_with_logits(x, y, reduction='sum'):
    """Compute focal loss with logits input"""
    return focal_loss(x.sigmoid(), y, reduction=reduction)
