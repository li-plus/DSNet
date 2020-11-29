import numpy as np

from anchor_free import anchor_free_helper


def test_get_loc_label():
    inputs = np.array([0, 0, 0, 1, 1, 1, 1, 0, 0, 1, 1, 1, 0])
    output = anchor_free_helper.get_loc_label(inputs)
    answer = np.array([[0, 0],
                       [0, 0],
                       [0, 0],
                       [0, 3],
                       [1, 2],
                       [2, 1],
                       [3, 0],
                       [0, 0],
                       [0, 0],
                       [0, 2],
                       [1, 1],
                       [2, 0],
                       [0, 0]])
    assert np.isclose(output, answer).all()


def test_get_ctr_label():
    target = np.array([0, 0, 0, 1, 1, 1, 1, 0, 0, 1, 1, 1, 0])
    offset = anchor_free_helper.get_loc_label(target)
    output = anchor_free_helper.get_ctr_label(target, offset)
    answer = np.array([0, 0, 0, 0, 0.5, 0.5, 0, 0, 0, 0, 1, 0, 0])
    assert np.isclose(output, answer).all()

    target = np.array([0, 0, 0])
    offset = anchor_free_helper.get_loc_label(target)
    output = anchor_free_helper.get_ctr_label(target, offset)
    answer = np.zeros(target.size, dtype=np.float32)
    assert np.isclose(output, answer).all()

    target = np.array([])
    offset = anchor_free_helper.get_loc_label(target)
    output = anchor_free_helper.get_ctr_label(target, offset)
    answer = np.zeros(target.size, dtype=np.float32)
    assert np.isclose(output, answer).all()


def test_bbox2offset():
    bboxes = np.array([[3, 7], [9, 12]])
    output = anchor_free_helper.bbox2offset(bboxes, 13)
    answer = np.array([[0, 0],
                       [0, 0],
                       [0, 0],
                       [0, 3],
                       [1, 2],
                       [2, 1],
                       [3, 0],
                       [0, 0],
                       [0, 0],
                       [0, 2],
                       [1, 1],
                       [2, 0],
                       [0, 0]])
    assert np.isclose(output, answer).all()


def test_offset2bbox():
    offset = np.array([[1, 2], [3, 4]], dtype=np.float32)
    output = anchor_free_helper.offset2bbox(offset)
    answer = np.array([[-1, 3], [-2, 6]], dtype=np.float32)
    assert np.isclose(output, answer).all()
