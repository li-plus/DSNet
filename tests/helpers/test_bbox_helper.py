import numpy as np

from helpers import bbox_helper


def test_lr2cw():
    lr_bbox = np.array([[1, 3], [2, 7], [19, 50]])
    output = bbox_helper.lr2cw(lr_bbox)
    answer = np.array([[2, 2], [4.5, 5], [34.5, 31]])
    assert np.isclose(output, answer).all()

    lr_bbox = np.array([[1.25, 2.75], [1.485, 3.123]])
    output = bbox_helper.lr2cw(lr_bbox)
    answer = np.array([[2, 1.5], [2.304, 1.638]])
    assert np.isclose(output, answer).all()


def test_cw2lr():
    cw_bbox = np.array([[2, 8], [6, 7]])
    output = bbox_helper.cw2lr(cw_bbox)
    answer = np.array([[-2, 6], [2.5, 9.5]])
    assert np.isclose(output, answer).all()

    cw_bbox = np.array([[1.524, 9.428], [4.518, 1.025]])
    output = bbox_helper.cw2lr(cw_bbox)
    answer = np.array([[-3.19, 6.238], [4.0055, 5.0305]])
    assert np.isclose(output, answer).all()


def test_seq2bbox():
    sequence = np.array([0, 1, 1, 1, 1, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 1, 1, 1])
    output = bbox_helper.seq2bbox(sequence)
    answer = [[1, 5], [8, 10], [15, 18]]
    assert np.isclose(output, answer).all()

    assert not bbox_helper.seq2bbox(np.array([0, 0, 0])).any()
    assert not bbox_helper.seq2bbox(np.array([])).any()


class TestIou(object):

    def setup(self):
        self.anchor_lr = np.array(
            [[1, 5], [1, 5], [1, 5], [1, 5], [1, 5]], dtype=np.float32)
        self.target_lr = np.array(
            [[1, 5], [0, 6], [2, 4], [3, 8], [8, 9]], dtype=np.float32)

        self.anchor_cw = bbox_helper.lr2cw(self.anchor_lr)
        self.target_cw = bbox_helper.lr2cw(self.target_lr)

        self.answer = np.array([1, 4 / 6, 2 / 4, 2 / 7, 0])

    def test_iou_lr(self):
        output = bbox_helper.iou_lr(self.anchor_lr, self.target_lr)
        assert np.isclose(output, self.answer).all()

    def test_iou_cw(self):
        output = bbox_helper.iou_cw(self.anchor_cw, self.target_cw)
        assert np.isclose(output, self.answer).all()


def test_nms():
    scores = np.array([0.9, 0.8, 0.7, 0.6])
    bboxes = np.array([[1, 5], [2, 4], [4, 8], [5, 9]])
    keep_scores, keep_bboxes = bbox_helper.nms(scores, bboxes, 0.5)

    ans_scores = [0.9, 0.7]
    ans_bboxes = [[1, 5], [4, 8]]
    assert np.isclose(keep_scores, ans_scores).all()
    assert np.isclose(keep_bboxes, ans_bboxes).all()
