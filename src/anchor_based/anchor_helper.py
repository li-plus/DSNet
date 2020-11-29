from typing import List, Tuple

import numpy as np

from helpers import bbox_helper


def get_anchors(seq_len: int, scales: List[int]) -> np.ndarray:
    """Generate all multi-scale anchors for a sequence in center-width format.

    :param seq_len: Sequence length.
    :param scales: List of bounding box widths.
    :return: All anchors in center-width format.
    """
    anchors = np.zeros((seq_len, len(scales), 2), dtype=np.int32)
    for pos in range(seq_len):
        for scale_idx, scale in enumerate(scales):
            anchors[pos][scale_idx] = [pos, scale]
    return anchors


def get_pos_label(anchors: np.ndarray,
                  targets: np.ndarray,
                  iou_thresh: float
                  ) -> Tuple[np.ndarray, np.ndarray]:
    """Generate positive samples for training.

    :param anchors: List of CW anchors
    :param targets: List of CW target bounding boxes
    :param iou_thresh: If IoU between a target bounding box and any anchor is
        higher than this threshold, the target is regarded as a positive sample.
    :return: Class and location offset labels
    """
    seq_len, num_scales, _ = anchors.shape
    anchors = np.reshape(anchors, (seq_len * num_scales, 2))

    loc_label = np.zeros((seq_len * num_scales, 2))
    cls_label = np.zeros(seq_len * num_scales, dtype=np.int32)

    for target in targets:
        target = np.tile(target, (seq_len * num_scales, 1))
        iou = bbox_helper.iou_cw(anchors, target)
        pos_idx = np.where(iou > iou_thresh)
        cls_label[pos_idx] = 1
        loc_label[pos_idx] = bbox2offset(target[pos_idx], anchors[pos_idx])

    loc_label = loc_label.reshape((seq_len, num_scales, 2))
    cls_label = cls_label.reshape((seq_len, num_scales))

    return cls_label, loc_label


def get_neg_label(cls_label: np.ndarray, num_neg: int) -> np.ndarray:
    """Generate random negative samples.

    :param cls_label: Class labels including only positive samples.
    :param num_neg: Number of negative samples.
    :return: Label with original positive samples (marked by 1), negative
        samples (marked by -1), and ignored samples (marked by 0)
    """
    seq_len, num_scales = cls_label.shape
    cls_label = cls_label.copy().reshape(-1)
    cls_label[cls_label < 0] = 0  # reset negative samples

    neg_idx, = np.where(cls_label == 0)
    np.random.shuffle(neg_idx)
    neg_idx = neg_idx[:num_neg]

    cls_label[neg_idx] = -1
    cls_label = np.reshape(cls_label, (seq_len, num_scales))
    return cls_label


def offset2bbox(offsets: np.ndarray, anchors: np.ndarray) -> np.ndarray:
    """Convert predicted offsets to CW bounding boxes.

    :param offsets: Predicted offsets.
    :param anchors: Sequence anchors.
    :return: Predicted bounding boxes.
    """
    offsets = offsets.reshape(-1, 2)
    anchors = anchors.reshape(-1, 2)

    offset_center, offset_width = offsets[:, 0], offsets[:, 1]
    anchor_center, anchor_width = anchors[:, 0], anchors[:, 1]

    # Tc = Oc * Aw + Ac
    bbox_center = offset_center * anchor_width + anchor_center
    # Tw = exp(Ow) * Aw
    bbox_width = np.exp(offset_width) * anchor_width

    bbox = np.vstack((bbox_center, bbox_width)).T
    return bbox


def bbox2offset(bboxes: np.ndarray, anchors: np.ndarray) -> np.ndarray:
    """Convert bounding boxes to offset labels.

    :param bboxes: List of CW bounding boxes.
    :param anchors: List of CW anchors.
    :return: Offsets labels for training.
    """
    bbox_center, bbox_width = bboxes[:, 0], bboxes[:, 1]
    anchor_center, anchor_width = anchors[:, 0], anchors[:, 1]

    # Oc = (Tc - Ac) / Aw
    offset_center = (bbox_center - anchor_center) / anchor_width
    # Ow = ln(Tw / Aw)
    offset_width = np.log(bbox_width / anchor_width)

    offset = np.vstack((offset_center, offset_width)).T
    return offset
