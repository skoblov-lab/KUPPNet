from functools import partial
from itertools import starmap, groupby
from typing import Iterable, Mapping, Union, Text, List, Dict

import numpy as np
from fn import F

from src.structures import NetInput, Stats
from src.predict import predict


def predict_eval_dump(model,
                      inp: NetInput,
                      hparams: Mapping[str, Union[str, int, float]],
                      cli_params: Mapping[str, Union[str, int, float]]):
    bs = cli_params['batch_size'] if cli_params['batch_size'] is not None else hparams['batch_size']
    ws = hparams['window_size']
    predictions = list(predict(model, inp, batch_size=bs, window_size=ws))
    eval_and_dump(predictions, inp, hparams, cli_params)


def eval_and_dump(predictions: Iterable[np.ndarray],
                  inp: NetInput,
                  hparams: Mapping[str, Union[str, int, float]],
                  cli_params: Mapping[str, Union[str, int, float]]):
    ts = cli_params['threshold'] if cli_params['threshold'] is not None else hparams['threshold']
    mode = cli_params['eval_output_mode']

    def dump_stats():
        stats = evaluate(predictions, inp, hparams, cli_params, ts)
        print("Model {} results for sequences in {} and true classes in {}".format(
            cli_params['model'], cli_params['input_seqs'], cli_params['input_cls']))
        print(stats, file=cli_params['output_file'])

    def dump_tsv():
        # TODO: find a better structure for pred_true pairs
        pred_positions = map(lambda x: zip(np.where(x > ts)[0], x[x > ts]), predictions)
        y_pred = ((id_, ((pos, cls) for pos, cls in pos_cls)) for id_, pos_cls in zip(inp.ids, pred_positions))
        y_true = parse_true_cls(cli_params['input_cls'], ids_ord={x: i for i, x in enumerate(inp.ids)})
        print("id", "pos", "prob", "true_class", "prediction", sep='\t', file=cli_params['output_file'])
        for id_, pos in y_pred:
            for p, s in pos:
                if id_ in y_true and p in y_true[id_]:
                    print(id_, p, s, 1, 1, file=cli_params['output_file'], sep='\t')
                else:
                    print(id_, p, s, 0, 1, file=cli_params['output_file'], sep='\t')

    if mode == 'stats_only':
        dump_stats()
    elif mode == 'tsv_only':
        dump_tsv()
    else:
        dump_stats()
        print('#' * 10, file=cli_params['output_file'])
        dump_tsv()


def evaluate(predictions: Iterable[np.ndarray],
             inp: NetInput,
             hparams: Mapping[str, Union[str, int, float]],
             cli_params: Mapping[str, Union[str, int, float]],
             threshold: float):
    ids_ord = {x: i for i, x in enumerate(inp.ids)}
    groups = parse_true_cls(cli_params['input_cls'], ids_ord)
    y_true = [np.zeros(shape=x.shape) for x in predictions]
    for seq_ord, cls_pos in groups.items():
        y_true[seq_ord][cls_pos] = 1
    y_true, y_pred = map(
        np.concatenate,
        [y_true, predictions])
    assert len(y_true) == len(y_pred)
    return compute_stats(y_true, y_pred, threshold)


def parse_true_cls(file_path: Text, ids_ord) -> Dict[int, np.ndarray]:
    process_lines = (F(map, lambda x: x.strip().split())
                     >> (map, lambda x: (x[0], int(x[1])))
                     # >> F(sorted, key=lambda x: ids_ord[x[0]])
                     >> (partial(groupby, key=lambda x: x[0]))
                     >> (map, lambda x: (ids_ord[x[0]], x[1])))
    with open(file_path) as f:
        grouped_cls = {g: np.array(list(x for _, x in gg), dtype=np.int32) for g, gg in process_lines(f)}
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
