from unittest import mock

import numpy as np

import train


def mock_get_arguments():
    ARGS = ('anchor-based --max-epoch 2 --device cpu '
            '--splits ../splits/tvsum.yml ../splits/summe.yml')
    parser = train.init_helper.get_parser()
    args = parser.parse_args(ARGS.split())
    return args


class MockH5pyFile(object):
    def __init__(self, *args, **kwargs):
        pass

    def __getitem__(self, key):
        seq_len = 64
        rate = 15  # down-sampling factor
        num_users = 4
        num_features = 1024

        seq = np.random.random((seq_len, num_features))
        gtscore = np.random.random(seq_len)
        n_frames = np.int32(seq_len * rate)
        nfps = np.full(seq_len // 4, 4)
        end_time = np.cumsum(nfps)
        begin_time = np.hstack((0, end_time[:-1]))
        cps = np.vstack((begin_time, end_time - 1)).T
        picks = np.arange(0, seq_len) * rate
        user_summary = np.random.random((num_users, seq_len * rate)) > 0.5

        video_file = {
            'features': seq.astype(np.float32),
            'gtscore': gtscore.astype(np.float32),
            'change_points': cps.astype(np.int32),
            'n_frames': n_frames.astype(np.int32),
            'n_frame_per_seg': nfps.astype(np.int32),
            'picks': picks.astype(np.int32),
            'user_summary': user_summary.astype(np.int32)
        }
        return video_file


def mock_load_yaml(path):
    path = str(path)
    assert 'tvsum' in path or 'summe' in path
    splits = [{
        'train_keys': [f'stub{i % 10}/video_{i}' for i in range(100)],
        'test_keys': [f'stub{i % 5}/video_{i}' for i in range(50)]
    } for _ in range(2)]
    return splits


def mock_dump_yaml(obj, path):
    pass


@mock.patch.object(train.init_helper, 'get_arguments', mock_get_arguments)
@mock.patch.object(train.data_helper.h5py, 'File', MockH5pyFile)
@mock.patch.object(train.data_helper, 'load_yaml', mock_load_yaml)
@mock.patch.object(train.data_helper, 'dump_yaml', mock_dump_yaml)
def test_train():
    train.main()
