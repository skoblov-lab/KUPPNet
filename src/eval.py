from itertools import starmap, groupby
from typing import Iterable, Mapping, Union, Text

import numpy as np
from fn import F

from src.structures import NetInput, Stats


def evaluate(predictions: Iterable[np.ndarray],
             inp: NetInput,
             hparams: Mapping[str, Union[str, int, float]],
             cli_params: Mapping[str, Union[str, int, float]],
             **kwargs):
    ts = cli_params['threshold'] if cli_params['threshold'] is not None else hparams['threshold']
    ids_ord = {x: i for i, x in enumerate(inp.ids)}
    groups = parse_true_cls(cli_params['input_cls'], ids_ord)
    y_true = [np.zeros(shape=x.shape) for x in predictions]
    for seq_ord, cls_pos in groups:
        y_true[seq_ord][cls_pos] = 1
    y_true, y_pred = map(
        np.concatenate,
        [y_true, predictions])
    assert len(y_true) == len(y_pred)
    return compute_stats(y_true, y_pred, ts)


def parse_true_cls(file_path: Text, ids_ord: Mapping[Text, int]):
    process_lines = (F(map, lambda x: x.strip().split())
                     >> (starmap, lambda x1, x2: (x1, int(x2)))
                     >> F(sorted, key=lambda x: ids_ord[x[0]])
                     >> (groupby, lambda x: x[0])
                     >> (map, lambda x: (ids_ord[x[0]], x[1])))
    with open(file_path) as f:
        grouped_cls = process_lines(f)
    return grouped_cls


def cls_to_array(cls: Iterable[int], array: np.ndarray):
    cls_array = np.zeros(shape=array.shape)
    cls_array[np.array(list(cls))] = 1
    return cls_array


def compute_stats(y_true: np.ndarray, y_pred: np.ndarray, ts: float = None) \
        -> Stats:
    """

    :param y_true:
    :param y_pred:
    :param ts:
    :return:
    """
    labels_true = y_true.copy().astype(np.int32)
    if ts:
        labels_pred = np.zeros(shape=y_pred.shape, dtype=np.int32)
        labels_pred[y_pred >= ts] = 1
    else:
        labels_pred = y_pred.round()

    negative_true = np.equal(labels_true, 0).astype(np.float32)
    positive_true = np.equal(labels_true, 1).astype(np.float32)
    negative_pred = np.equal(labels_pred, 0).astype(np.float32)
    positive_pred = np.equal(labels_pred, 1).astype(np.float32)

    true_negatives = (negative_true * negative_pred).sum()
    true_positives = (positive_true * positive_pred).sum()
    false_negatives = (positive_true * negative_pred).sum()
    false_positives = (positive_pred * negative_true).sum()

    accuracy = (true_positives + true_negatives) / len(y_true)
    fpr = false_positives / (false_positives + true_negatives)
    fnr = false_negatives / (true_positives + false_negatives)
    precision = true_positives / (true_positives + false_positives)
    recall = true_positives / (true_positives + false_negatives)
    f1 = 2 * precision * recall / (precision + recall)
    specificity = true_negatives / (true_negatives + false_positives)

    return Stats(accuracy, fnr, fpr, precision, recall, f1, specificity)


if __name__ == '__main__':
    raise RuntimeError
