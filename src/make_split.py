import argparse
import random
from pathlib import Path

import h5py
import yaml


def make_random_splits(keys, num_test, num_splits):
    splits = []
    for _ in range(num_splits):
        random.shuffle(keys)
        test_keys = keys[:num_test]
        train_keys = list(set(keys) - set(test_keys))
        splits.append({
            'train_keys': train_keys,
            'test_keys': test_keys
        })
    return splits


def make_cross_val_splits(keys, num_videos, num_test):
    random.shuffle(keys)
    splits = []
    for i in range(0, num_videos, num_test):
        test_keys = keys[i: i + num_test]
        train_keys = list(set(keys) - set(test_keys))
        splits.append({
            'train_keys': train_keys,
            'test_keys': test_keys
        })
    return splits


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, required=True,
                        help='Path to h5 dataset')
    parser.add_argument('--extra-datasets', type=str, nargs='+', default=[],
                        help='Extra datasets to append to train set')
    parser.add_argument('--save-path', type=str, required=True,
                        help='Path to save generated splits')
    parser.add_argument('--num-splits', type=int, default=5,
                        help='How many splits to generate')
    parser.add_argument('--train-ratio', type=float, default=0.8,
                        help='Percentage of training data')
    parser.add_argument('--method', type=str, default='random',
                        choices=['random', 'cross'],
                        help='Random selection or cross validation')
    args = parser.parse_args()

    dataset = h5py.File(args.dataset, 'r')
    keys = list(dataset.keys())
    keys = [str(Path(args.dataset) / key) for key in keys]

    extra_keys = []
    for extra_dataset_path in args.extra_datasets:
        extra_dataset_path = Path(extra_dataset_path)
        extra_dataset = h5py.File(str(extra_dataset_path))
        extra_dataset_keys = list(extra_dataset.keys())
        extra_dataset_keys = [str(extra_dataset_path / key) for key in
                              extra_dataset_keys]
        extra_keys += extra_dataset_keys

    num_videos = len(keys)
    num_train = round(num_videos * args.train_ratio)
    num_test = num_videos - num_train

    if args.method == 'random':
        splits = make_random_splits(keys, num_test, args.num_splits)
    elif args.method == 'cross':
        splits = make_cross_val_splits(keys, num_videos, num_test)
    else:
        raise ValueError(f'Invalid method {args.method}')

    # append extra keys
    if extra_keys:
        for split in splits:
            split['train_keys'] += extra_keys
            random.shuffle(split['train_keys'])

    # save splits
    save_path = Path(args.save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    with open(save_path, 'w') as f:
        yaml.dump(splits, f)


if __name__ == '__main__':
    main()
