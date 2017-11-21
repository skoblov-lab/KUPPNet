from itertools import chain, filterfalse, tee
from numbers import Integral
from typing import Iterable, Optional, Mapping, Text

import numpy as np
from functools import reduce

from src.structures import NetInput, Interval, Seq


def predict_and_dump(inp: NetInput, model: "keras model", hparams: Mapping, cli_params: Mapping):
    """
    Higher-order function which uses "predict" function,
    extracts valid predictions and then dumps them into
    cli_params['output_file'] (if it's None will print to stdout)
    :param inp: NetInput with joined sequences, negatives and masks
    :param model: compiled model to be used for predictions
    :param hparams: default parameters for compiled model
    :param cli_params: parameters provided through CLI interface
    :return:
    """

    def prepare_seq(seq: Seq, pos: Iterable[int]) -> Optional[Text]:
        if not pos:
            return None
        seq_mut = list(seq.data.raw)
        for p in pos:
            seq_mut[p] = seq_mut[p].lower()
        return '\n'.join(map(lambda x: ''.join(x), (seq_mut[i:i + 80] for i in range(0, len(seq_mut), 80))))

    def prepare(to_dump):
        if cli_params['prediction_output_mode'] == 'tsv':
            return chain.from_iterable(
                ("\t".join([id_, str(n + 1), str(s)]) for n, s in zip(pos, sc)) for id_, _, pos, sc in to_dump)
        seqs = filterfalse(lambda x: x[1] is None, ((id_, prepare_seq(seq, pos)) for id_, seq, pos, _ in to_dump))
        return ('\n'.join(('>' + id_, seq)) for id_, seq in seqs)

    ts = cli_params['threshold'] if cli_params['threshold'] is not None else hparams['threshold']
    bs = cli_params['batch_size'] if cli_params['batch_size'] is not None else hparams['batch_size']
    predictions1, predictions2 = tee(predict(model, inp, batch_size=bs))
    valid_predictions = zip(inp.ids, inp.seqs,
                            (np.where(p >= ts)[0] for p in predictions1),
                            (p[p >= ts] for p in predictions2))
    prepared = filterfalse(lambda x: x is None, prepare(valid_predictions))
    for line in prepared:
        print(line, file=cli_params['output_file'])


def predict(model, inp: NetInput, batch_size: Integral = 100) \
        -> Iterable[np.ndarray]:
    """
    Predicts classes in merged array of sequences provided in inp
    Merges windowed predictions using intervals created by rolling_window function
    :param model: compiled keras model.
    :param inp: input to model.
    Must contain:
        1) joined (sequences)
        2) masks (False-padded rolled sequences)
        3) negative (ones for any potentially positive class)
        4) rolled sequences intervals
    :param batch_size:
    :return:
    """

    def acc_len(current, to_add):
        return current + [current[-1] + len(to_add)]

    split_intervals = reduce(acc_len, inp.rolled_seqs[1:], [len(inp.rolled_seqs[0])])

    predictions = model.predict(
        [inp.joined, inp.masks[:, :, None], inp.negative[:, :, None]],
        batch_size=batch_size)

    predictions = np.split(predictions, np.array(split_intervals)[:-1])
    predictions = (_merge(a, ints) for a, ints in zip(predictions, inp.rolled_seqs))
    return predictions


def _merge(array: np.ndarray, intervals: Iterable[Interval], num_classes: Optional[Integral] = None) \
        -> np.ndarray:
    # TODO: docs
    """

    :param array:
    :param intervals:
    :param num_classes:
    :return:
    """
    denom = (np.zeros(shape=(max(int_.stop for int_ in intervals), num_classes)) if num_classes
             else np.zeros(shape=(max(int_.stop for int_ in intervals),)))
    merged = (np.zeros(shape=(max(int_.stop for int_ in intervals), num_classes)) if num_classes
              else np.zeros(shape=(max(int_.stop for int_ in intervals),)))
    for int_, arr in zip(intervals, array.reshape(array.shape[:-1])):
        merged[int_.start:int_.stop] += arr[:int_.stop - int_.start]
        denom[int_.start:int_.stop] += 1
    return merged / denom


if __name__ == '__main__':
    raise RuntimeError
