from typing import Iterable, List

import numpy as np
from ortools.algorithms.pywrapknapsack_solver import KnapsackSolver


def f1_score(pred: np.ndarray, test: np.ndarray) -> float:
    """Compute F1-score on binary classification task.

    :param pred: Predicted binary label. Sized [N].
    :param test: Ground truth binary label. Sized [N].
    :return: F1-score value.
    """
    assert pred.shape == test.shape
    pred = np.asarray(pred, dtype=np.bool)
    test = np.asarray(test, dtype=np.bool)
    overlap = (pred & test).sum()
    if overlap == 0:
        return 0.0
    precision = overlap / pred.sum()
    recall = overlap / test.sum()
    f1 = 2 * precision * recall / (precision + recall)
    return float(f1)


def knapsack(values: Iterable[int],
             weights: Iterable[int],
             capacity: int
             ) -> List[int]:
    """Solve 0/1 knapsack problem using dynamic programming.

    :param values: Values of each items. Sized [N].
    :param weights: Weights of each items. Sized [N].
    :param capacity: Total capacity of the knapsack.
    :return: List of packed item indices.
    """
    knapsack_solver = KnapsackSolver(
        KnapsackSolver.KNAPSACK_DYNAMIC_PROGRAMMING_SOLVER, 'test'
    )

    values = list(values)
    weights = list(weights)
    capacity = int(capacity)

    knapsack_solver.Init(values, [weights], [capacity])
    knapsack_solver.Solve()
    packed_items = [x for x in range(0, len(weights))
                    if knapsack_solver.BestSolutionContains(x)]

    return packed_items


def downsample_summ(summ: np.ndarray) -> np.ndarray:
    """Down-sample the summary by 15 times"""
    return summ[::15]


def get_keyshot_summ(pred: np.ndarray,
                     cps: np.ndarray,
                     n_frames: int,
                     nfps: np.ndarray,
                     picks: np.ndarray,
                     proportion: float = 0.15
                     ) -> np.ndarray:
    """Generate keyshot-based video summary i.e. a binary vector.

    :param pred: Predicted importance scores.
    :param cps: Change points, 2D matrix, each row contains a segment.
    :param n_frames: Original number of frames.
    :param nfps: Number of frames per segment.
    :param picks: Positions of subsampled frames in the original video.
    :param proportion: Max length of video summary compared to original length.
    :return: Generated keyshot-based summary.
    """
    assert pred.shape == picks.shape
    picks = np.asarray(picks, dtype=np.int32)

    # Get original frame scores from downsampled sequence
    frame_scores = np.zeros(n_frames, dtype=np.float32)
    for i in range(len(picks)):
        pos_lo = picks[i]
        pos_hi = picks[i + 1] if i + 1 < len(picks) else n_frames
        frame_scores[pos_lo:pos_hi] = pred[i]

    # Assign scores to video shots as the average of the frames.
    seg_scores = np.zeros(len(cps), dtype=np.int32)
    for seg_idx, (first, last) in enumerate(cps):
        scores = frame_scores[first:last + 1]
        seg_scores[seg_idx] = int(1000 * scores.mean())

    # Apply knapsack algorithm to find the best shots
    limits = int(n_frames * proportion)
    packed = knapsack(seg_scores, nfps, limits)

    # Get key-shot based summary
    summary = np.zeros(n_frames, dtype=np.bool)
    for seg_idx in packed:
        first, last = cps[seg_idx]
        summary[first:last + 1] = True

    return summary


def bbox2summary(seq_len: int,
                 pred_cls: np.ndarray,
                 pred_bboxes: np.ndarray,
                 change_points: np.ndarray,
                 n_frames: int,
                 nfps: np.ndarray,
                 picks: np.ndarray
                 ) -> np.ndarray:
    """Convert predicted bounding boxes to summary"""
    score = np.zeros(seq_len, dtype=np.float32)
    for bbox_idx in range(len(pred_bboxes)):
        lo, hi = pred_bboxes[bbox_idx, 0], pred_bboxes[bbox_idx, 1]
        score[lo:hi] = np.maximum(score[lo:hi], [pred_cls[bbox_idx]])

    pred_summ = get_keyshot_summ(score, change_points, n_frames, nfps, picks)
    return pred_summ


def get_summ_diversity(pred_summ: np.ndarray,
                       features: np.ndarray
                       ) -> float:
    """Evaluate diversity of the generated summary.

    :param pred_summ: Predicted down-sampled summary. Sized [N, F].
    :param features: Normalized down-sampled video features. Sized [N, F].
    :return: Diversity value.
    """
    assert len(pred_summ) == len(features)
    pred_summ = np.asarray(pred_summ, dtype=np.bool)
    pos_features = features[pred_summ]

    if len(pos_features) < 2:
        return 0.0

    diversity = 0.0
    for feat in pos_features:
        diversity += (feat * pos_features).sum() - (feat * feat).sum()

    diversity /= len(pos_features) * (len(pos_features) - 1)
    return diversity


def get_summ_f1score(pred_summ: np.ndarray,
                     test_summ: np.ndarray,
                     eval_metric: str = 'avg'
                     ) -> float:
    """Compare predicted summary with ground truth summary (keyshot-based).

    :param pred_summ: Predicted binary label of N frames. Sized [N].
    :param test_summ: Ground truth binary labels of U users. Sized [U, N].
    :param eval_metric: Evaluation method. Choose from (max, avg).
    :return: F1-score value.
    """
    pred_summ = np.asarray(pred_summ, dtype=np.bool)
    test_summ = np.asarray(test_summ, dtype=np.bool)
    _, n_frames = test_summ.shape

    if pred_summ.size > n_frames:
        pred_summ = pred_summ[:n_frames]
    elif pred_summ.size < n_frames:
        pred_summ = np.pad(pred_summ, (0, n_frames - pred_summ.size))

    f1s = [f1_score(user_summ, pred_summ) for user_summ in test_summ]

    if eval_metric == 'avg':
        final_f1 = np.mean(f1s)
    elif eval_metric == 'max':
        final_f1 = np.max(f1s)
    else:
        raise ValueError(f'Invalid eval metric {eval_metric}')

    return float(final_f1)
