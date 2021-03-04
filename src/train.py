import logging
from pathlib import Path

from anchor_based.train import train as train_anchor_based
from anchor_free.train import train as train_anchor_free
from helpers import init_helper, data_helper

logger = logging.getLogger()

TRAINER = {
    'anchor-based': train_anchor_based,
    'anchor-free': train_anchor_free
}


def get_trainer(model_type):
    assert model_type in TRAINER
    return TRAINER[model_type]


def main():
    args = init_helper.get_arguments()

    init_helper.init_logger(args.model_dir, args.log_file)
    init_helper.set_random_seed(args.seed)

    logger.info(vars(args))

    model_dir = Path(args.model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)
    data_helper.get_ckpt_dir(model_dir).mkdir(parents=True, exist_ok=True)

    trainer = get_trainer(args.model)

    data_helper.dump_yaml(vars(args), model_dir / 'args.yml')

    for split_path in args.splits:
        split_path = Path(split_path)
        splits = data_helper.load_yaml(split_path)

        results = {}
        stats = data_helper.AverageMeter('fscore')

        for split_idx, split in enumerate(splits):
            logger.info(f'Start training on {split_path.stem}: split {split_idx}')
            ckpt_path = data_helper.get_ckpt_path(model_dir, split_path, split_idx)
            fscore = trainer(args, split, ckpt_path)
            stats.update(fscore=fscore)
            results[f'split{split_idx}'] = float(fscore)

        results['mean'] = float(stats.fscore)
        data_helper.dump_yaml(results, model_dir / f'{split_path.stem}.yml')

        logger.info(f'Training done on {split_path.stem}. F-score: {stats.fscore:.4f}')


if __name__ == '__main__':
    main()
