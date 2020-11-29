import numpy as np

from helpers import bbox_helper


def get_loc_label(target: np.ndarray) -> np.ndarray:
    """Generate location offset label from ground truth summary.

    :param target: Ground truth summary. Sized [N].
    :return: Location offset label in LR format. Sized [N, 2].
    """
    seq_len, = target.shape

    bboxes = bbox_helper.seq2bbox(target)
    offsets = bbox2offset(bboxes, seq_len)

    return offsets


def get_ctr_label(target: np.ndarray,
                  offset: np.ndarray,
                  eps: float = 1e-8
                  ) -> np.ndarray:
    """Generate centerness label for ground truth summary.

    :param target: Ground truth summary. Sized [N].
    :param offset: LR offset corresponding to target. Sized [N, 2].
    :param eps: Small floating value to prevent division by zero.
    :return: Centerness label. Sized [N].
    """
    target = np.asarray(target, dtype=np.bool)
    ctr_label = np.zeros(target.shape, dtype=np.float32)

    offset_left, offset_right = offset[target, 0], offset[target, 1]
    ctr_label[target] = np.minimum(offset_left, offset_right) / (
        np.maximum(offset_left, offset_right) + eps)

    return ctr_label


def bbox2offset(bboxes: np.ndarray, seq_len: int) -> np.ndarray:
    """Convert LR bounding boxes to LR offsets.

    :param bboxes: LR bounding boxes.
    :param seq_len: Sequence length N.
    :return: LR offsets. Sized [N, 2].
    """
    pos_idx = np.arange(seq_len, dtype=np.float32)
    offsets = np.zeros((seq_len, 2), dtype=np.float32)

    for lo, hi in bboxes:
        bbox_pos = pos_idx[lo:hi]
        offsets[lo:hi] = np.vstack((bbox_pos - lo, hi - 1 - bbox_pos)).T

    return offsets


def offset2bbox(offsets: np.ndarray) -> np.ndarray:
    """Convert LR offsets to bounding boxes.

    :param offsets: LR offsets. Sized [N, 2].
    :return: Bounding boxes corresponding to offsets. Sized [N, 2].
    """
    offset_left, offset_right = offsets[:, 0], offsets[:, 1]
    seq_len, _ = offsets.shape
    indices = np.arange(seq_len)
    bbox_left = indices - offset_left
    bbox_right = indices + offset_right + 1
    bboxes = np.vstack((bbox_left, bbox_right)).T
    return bboxes
