#!/usr/bin/env bash

set -e

cd $(dirname $(dirname $(realpath $0)))/src

# anchor based: attn, lstm, bilstm, gcn
python train.py anchor-based --max-epoch 2 --splits ../splits/tvsum.yml ../splits/summe.yml
python evaluate.py anchor-based --splits ../splits/tvsum.yml ../splits/summe.yml

python train.py anchor-based --max-epoch 2 --splits ../splits/tvsum.yml ../splits/summe.yml --base-model lstm
python evaluate.py anchor-based --splits ../splits/tvsum.yml ../splits/summe.yml --base-model lstm

python train.py anchor-based --max-epoch 2 --splits ../splits/tvsum.yml ../splits/summe.yml --base-model bilstm
python evaluate.py anchor-based --splits ../splits/tvsum.yml ../splits/summe.yml --base-model bilstm

python train.py anchor-based --max-epoch 2 --splits ../splits/tvsum.yml ../splits/summe.yml --base-model gcn
python evaluate.py anchor-based --splits ../splits/tvsum.yml ../splits/summe.yml --base-model gcn

# anchor free: attn, lstm, bilstm, gcn
python train.py anchor-free --max-epoch 2 --splits ../splits/tvsum.yml ../splits/summe.yml --nms-thresh 0.4
python evaluate.py anchor-free --splits ../splits/tvsum.yml ../splits/summe.yml --nms-thresh 0.4

python train.py anchor-free --max-epoch 2 --splits ../splits/tvsum.yml ../splits/summe.yml --nms-thresh 0.4 --base-model lstm
python evaluate.py anchor-free --splits ../splits/tvsum.yml ../splits/summe.yml --nms-thresh 0.4 --base-model lstm

python train.py anchor-free --max-epoch 2 --splits ../splits/tvsum.yml ../splits/summe.yml --nms-thresh 0.4 --base-model bilstm
python evaluate.py anchor-free --splits ../splits/tvsum.yml ../splits/summe.yml --nms-thresh 0.4 --base-model bilstm

python train.py anchor-free --max-epoch 2 --splits ../splits/tvsum.yml ../splits/summe.yml --nms-thresh 0.4 --base-model gcn
python evaluate.py anchor-free --splits ../splits/tvsum.yml ../splits/summe.yml --nms-thresh 0.4 --base-model gcn

# splits: tvsum-aug, tvsum-trans, summe-aug, summe-trans
python train.py anchor-based --max-epoch 2 --splits ../splits/tvsum_aug.yml
python evaluate.py anchor-based --splits ../splits/tvsum_aug.yml

python train.py anchor-based --max-epoch 2 --splits ../splits/tvsum_trans.yml
python evaluate.py anchor-based --splits ../splits/tvsum_trans.yml

python train.py anchor-based --max-epoch 2 --splits ../splits/summe_aug.yml
python evaluate.py anchor-based --splits ../splits/summe_aug.yml

python train.py anchor-based --max-epoch 2 --splits ../splits/summe_trans.yml
python evaluate.py anchor-based --splits ../splits/summe_trans.yml

# make split
rm -rf ../splits_custom/

python make_split.py --save-path ../splits_custom/tvsum.yml \
    --dataset ../datasets/eccv16_dataset_tvsum_google_pool5.h5

python make_split.py --save-path ../splits_custom/tvsum_aug.yml \
    --dataset ../datasets/eccv16_dataset_tvsum_google_pool5.h5 \
    --extra-datasets ../datasets/eccv16_dataset_summe_google_pool5.h5 \
                     ../datasets/eccv16_dataset_ovp_google_pool5.h5 \
                     ../datasets/eccv16_dataset_youtube_google_pool5.h5

python make_split.py --save-path ../splits_custom/tvsum_trans.yml \
    --dataset ../datasets/eccv16_dataset_tvsum_google_pool5.h5 \
    --extra-datasets ../datasets/eccv16_dataset_summe_google_pool5.h5 \
                     ../datasets/eccv16_dataset_ovp_google_pool5.h5 \
                     ../datasets/eccv16_dataset_youtube_google_pool5.h5 \
    --train-ratio 0

# make shots
python make_shots.py --dataset ../datasets/eccv16_dataset_ovp_google_pool5.h5
python make_shots.py --dataset ../datasets/eccv16_dataset_youtube_google_pool5.h5
