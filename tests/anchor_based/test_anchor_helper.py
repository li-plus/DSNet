import unittest

import numpy as np

from anchor_based import anchor_helper


class TestAnchorHelper(unittest.TestCase):
    def setUp(self):
        self.seq_len = 5
        self.anchor_scales = [2, 4]
        self.num_scales = len(self.anchor_scales)
        self.anchors = np.array([[[0, 2], [0, 4]],
                                 [[1, 2], [1, 4]],
                                 [[2, 2], [2, 4]],
                                 [[3, 2], [3, 4]],
                                 [[4, 2], [4, 4]]], dtype=np.int32)
        self.targets = np.array([[3, 2], [0.5, 1]], dtype=np.float32)

        self.iou_thresh = 0.499
        self.pos_cls = np.array([[1, 0],
                                 [1, 0],
                                 [0, 1],
                                 [1, 1],
                                 [0, 1]], dtype=np.int32)
        self.loc_label = np.array([
            [[0.25, np.log(0.5)], [0, 0]],
            [[-0.25, np.log(0.5)], [0, 0]],
            [[0, 0], [0.25, np.log(0.5)]],
            [[0, 0], [0, np.log(0.5)]],
            [[0, 0], [-0.25, np.log(0.5)]],
        ], dtype=np.float32)

        self.num_neg = 3
        self.pred_bboxes = np.array([[[0.5, 1], [0, 0]],
                                     [[0.5, 1], [0, 0]],
                                     [[0, 0], [3, 2]],
                                     [[3, 2], [3, 2]],
                                     [[0, 0], [3, 2]]], dtype=np.float32)

    def test_get_anchors(self):
        out = anchor_helper.get_anchors(self.seq_len, self.anchor_scales)
        assert np.isclose(self.anchors, out).all()

    def test_get_pos_label(self):
        out_cls, out_loc = anchor_helper.get_pos_label(self.anchors, self.targets, self.iou_thresh)
        assert np.isclose(self.pos_cls, out_cls).all()
        assert np.isclose(self.loc_label, out_loc).all()

    def test_get_neg_label(self):
        cls_label = anchor_helper.get_neg_label(self.pos_cls, self.num_neg)
        assert (cls_label == -1).sum() == self.num_neg
        assert ((cls_label == 1) == (self.pos_cls == 1)).all()

    def test_offset2bbox(self):
        bboxes = anchor_helper.offset2bbox(self.loc_label, self.anchors)
        bboxes = bboxes.reshape((self.seq_len, self.num_scales, 2))
        bboxes = np.expand_dims(self.pos_cls, -1) * bboxes
        assert np.isclose(bboxes, self.pred_bboxes).all()
