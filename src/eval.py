from itertools import groupby, starmap, chain
from typing import Iterable, Mapping, Union, Generator, Tuple, Optional

import numpy as np
from fn import F
from functools import partial
from io import TextIOWrapper

from src.predict import predict
from src.structures import NetInput, Stats, Site


# TODO: test eval mode


def eval_and_dump(model: Optional,
                  predictions: Optional[Union[Iterable[np.ndarray]], Iterable[str]],
                  inp: NetInput,
                  hparams: Mapping[str, Union[str, int, float, TextIOWrapper]],
                  cli_params: Mapping[str, Union[str, int, float, TextIOWrapper]]) -> None:
    # TODO: docs
    """

    :param model:
    :param predictions:
    :param inp:
    :param hparams:
    :param cli_params:
    :return:
    """
    stats, sites = evaluate(model, predictions, inp, hparams, cli_params)
    dump(cli_params, stats, sites)


def evaluate(model: Optional,
             predictions: Optional[Union[Iterable[np.ndarray]], Iterable[str]],
             inp: NetInput,
             hparams: Mapping[str, Union[str, int, float, TextIOWrapper]],
             cli_params: Mapping[str, Union[str, int, float, TextIOWrapper]]) \
        -> Tuple[Optional[Stats], Optional[Iterable[Site]]]:
    # TODO: docs
    """

    :param model:
    :param predictions:
    :param inp:
    :param hparams:
    :param cli_params:
    :return:
    """
    bs = cli_params['batch_size'] if cli_params['batch_size'] is not None else hparams['batch_size']
    ws = hparams['window_size']
    ts = cli_params['threshold'] if cli_params['threshold'] is not None else hparams['threshold']
    mode = cli_params['eval_output_mode']

    if model is None:
        if isinstance(predictions, Iterable[str]):
            predictions = parse_cls(predictions)
        else:
            predictions = zip(inp.ids, predictions)
    else:
        predictions = zip(inp.ids, predict(model, inp, batch_size=bs, window_size=ws))
    true_cls = map_onto_pos(inp=inp, classes=parse_cls(cli_params['input_cls']))

    y_true = [x[y > 0] for (_, x), y in zip(true_cls, inp.negative)]
    y_pred = [x[y > 0] for (_, x), y in zip(predictions, inp.negative)]

    stats = (compute_stats(y_true=np.concatenate(y_true), y_pred=np.concatenate(y_pred), ts=ts)
             if mode == 'full' or mode == 'stats_only' else None)
    sites = compile_sites(inp, y_true, y_pred) if mode == 'full' or mode == 'tsv_only' else None
    return stats, sites


def compile_sites(inp, y_true, y_pred):
    # TODO: docs
    """

    :param inp:
    :param y_true:
    :param y_pred:
    :return:
    """
    positions = (np.where(y > 0)[0] for y in inp.negative)

    def comp_site(id_, cls_true, cls_pred, pos):
        site = [id_, pos, 0, 0]
        if cls_pred:
            site[2] = 1
        if cls_true:
            site[3] = 1
        return Site(*site)

    sites = chain.from_iterable(
        ((id_, pos, p, t) for pos, p, t in zip(pp, yp, yt))
        for id_, pp, yp, yt in zip(inp.ids, positions, y_pred, y_true))
    return starmap(comp_site, sites)


def dump(cli_params: Mapping[str, Union[str, int, float, TextIOWrapper]],
         stats: Optional[Stats],
         sites: Iterable[Site]):
    # TODO: docs
    """

    :param cli_params:
    :param stats:
    :param sites:
    :return:
    """
    mode = cli_params['eval_output_mode']

    def dump_tsv():
        for site in sites:
            print(*site, sep='\t', file=cli_params['output_file'])

    def dump_stats():
        print(stats, file=cli_params['output_file'])

    print('KUPPNet model {} evaluation.\nSites are taken from {}\nSeqs are taken from {}'.format(
        cli_params['model'],
        cli_params['input_cls'],
        cli_params['input_seqs']))

    if mode == 'stats_only':
        dump_stats()
    elif mode == 'tsv_only':
        dump_tsv()
    else:
        dump_stats()
        print('#' * 15, file=cli_params['output_file'])
        dump_tsv()


def map_onto_pos(inp: NetInput,
                 classes: Generator[Tuple[str, np.ndarray], None, None],
                 offset: int = 0):
    # TODO: docs
    """

    :param inp:
    :param classes:
    :param offset:
    :return:
    """
    ids_ord = {x: i for i, x in enumerate(inp.ids)}
    template = np.zeros(shape=inp.joined.shape)
    for id_, positions in classes:
        template[ids_ord[id_]][positions - offset] = 1
    return template


def parse_cls(id_cls_pairs: Iterable[str]) \
        -> Generator[Tuple[str, np.ndarray]]:
    """
    Parses classes provided either by means of str with id-pos pairs
    separated by " "-like symbol
    or id-Iterable pairs (id and all true classes for this id)
    :param id_cls_pairs:
    :return: Generator yielding tuple with ids and Generator with positions of true classes
    """
    parse_lines = (F(map, lambda x: x.strip().split())
                   >> (map, lambda x: (x[0], int(x[1])))
                   >> (partial(groupby, key=lambda x: x[0])))
    return ((g, np.ndarray([x for _, x in gg])) for g, gg in parse_lines(id_cls_pairs))


def compute_stats(y_true: np.ndarray, y_pred: np.ndarray, ts: float = None) \
        -> Stats:
    # TODO: docs
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
