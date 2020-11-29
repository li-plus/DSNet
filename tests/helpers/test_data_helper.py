import random
from unittest import mock

import h5py

from helpers import data_helper


class MockH5pyFile(object):
    def __init__(self, path, *args, **kwargs):
        self.path = path


# @mock.patch.object(h5py, 'File', MockH5pyFile)
# def test_get_datasets():
#     splits = [{
#         'train_keys': [f'stub{i % 10}/video_{i}' for i in range(100)],
#         'test_keys': [f'stub{i % 5}/video_{i}' for i in range(50)]
#     }]
#     datasets = data_helper.get_datasets(splits)
#     for path, dataset in datasets.items():
#         assert isinstance(dataset, MockH5pyFile)
#         assert dataset.path == path


def test_average_meter():
    num = 100
    xs = [random.randint(0, 100) for _ in range(num)]
    ys = [random.randint(0, 100) for _ in range(num)]

    avg_meter = data_helper.AverageMeter('x', 'y')
    assert avg_meter.x == 0.0
    assert avg_meter.y == 0.0

    for x, y in zip(xs, ys):
        avg_meter.update(x=x, y=y)

    assert avg_meter.x == sum(xs) / num
    assert avg_meter.y == sum(ys) / num
