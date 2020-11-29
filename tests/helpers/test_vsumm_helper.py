import math

import numpy as np

from helpers import vsumm_helper


def test_knapsack():
    # datasets from:
    # https://people.sc.fsu.edu/~jburkardt/datasets/knapsack_01/knapsack_01.html
    values = [92, 57, 49, 68, 60, 43, 67, 84, 87, 72]
    weights = [23, 31, 29, 44, 53, 38, 63, 85, 89, 82]
    capacity = 165
    output = vsumm_helper.knapsack(values, weights, capacity)
    output = np.array(output)
    answer, = np.where(np.array([1, 1, 1, 1, 0, 1, 0, 0, 0, 0]) > 0.5)
    assert (output == answer).all()

    values = [825594, 1677009, 1676628, 1523970, 943972, 97426, 69666, 1296457,
              1679693, 1902996, 1844992,
              1049289, 1252836, 1319836, 953277, 2067538, 675367, 853655,
              1826027, 65731, 901489, 577243, 466257, 369261]
    weights = [382745, 799601, 909247, 729069, 467902, 44328, 34610, 698150,
               823460, 903959, 853665,
               551830, 610856, 670702, 488960, 951111, 323046, 446298, 931161,
               31385, 496951, 264724, 224916, 169684]
    capacity = 6404180
    output = vsumm_helper.knapsack(values, weights, capacity)
    output = np.array(output)
    answer, = np.where(np.array(
        [1, 1, 0, 1, 1, 1, 0, 0, 0, 1, 1, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 1, 1,
         1]) > 0.5)
    assert (output == answer).all()


def test_f1_score():
    pred = np.array([0, 1, 1, 0, 1], dtype=np.bool)
    test = np.array([1, 1, 0, 1, 1], dtype=np.bool)
    f1 = vsumm_helper.f1_score(pred, test)
    assert math.isclose(f1, 4 / 7)
