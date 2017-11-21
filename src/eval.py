from itertools import groupby, starmap, chain
from typing import Iterable, Mapping, Union, Generator, Tuple, Optional, Sequence, MutableSequence

import numpy as np
from io import TextIOWrapper

from src.predict import predict
from src.structures import NetInput, Stats, Site


# TODO: test eval mode


def eval_and_dump(model: Optional,
                  inp: NetInput,
                  hparams: Mapping[str, Union[str, int, float, TextIOWrapper]],
                  cli_params: Mapping[str, Union[str, int, float, TextIOWrapper]]) \
        -> None:
    # TODO: docs
    """

    :param model:
    :param inp:
    :param hparams:
    :param cli_params:
    :return:
    """
    stats, sites = evaluate(model, cli_params['predictions'], inp, hparams, cli_params)
    dump(cli_params, stats, sites)


def evaluate(model: Optional,
             predictions: Optional[Union[Iterable[np.ndarray], TextIOWrapper]],
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
    ts = cli_params['threshold'] if cli_params['threshold'] is not None else hparams['threshold']
    mode = cli_params['eval_output_mode']

    templates = [np.zeros(shape=(len(x.data.raw),)) for x in inp.seqs]
    templates_bool = [t.astype(bool) for t in templates]
    for e, s in enumerate(inp.seqs):
        for i in (16, 17, 20):
            templates_bool[e][s.data.encoded == i] = True

    if model is None:
        offset = 1
        if isinstance(predictions, TextIOWrapper):
            pred = parse_cls(predictions, ts)
        else:
            pred = zip(inp.ids, predictions)
    else:
        offset = 0
        pred = zip(inp.ids, (np.where(p > ts)[0] for p in predict(model, inp, batch_size=bs)))
    ids_ord = {x: i for i, x in enumerate(inp.ids)}
    pred = map_onto_pos(ids_ord=ids_ord, templates=templates, masks=templates_bool,
                        classes=pred, offset=offset)
    true_cls = map_onto_pos(ids_ord=ids_ord, templates=templates, masks=templates_bool,
                            classes=parse_cls(cli_params['input_cls']))
    stats = (compute_stats(y_true=np.concatenate(true_cls), y_pred=np.concatenate(pred))
             if mode == 'full' or mode == 'stats_only' else None)
    sites = compile_sites(inp, true_cls, pred) if mode == 'full' or mode == 'tsv_only' else None
    return stats, sites


def compile_sites(inp, y_true, y_pred):
    # TODO: docs
    """

    :param inp:
    :param y_true:
    :param y_pred:
    :return:
    """
    positions = (np.where(y > 0)[0] + 1 for y in inp.negative)

    def comp_site(id_, pos, cls_pred, cls_true):
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


def map_onto_pos(ids_ord: Mapping[str, int],
                 classes: Generator[Tuple[str, np.ndarray], None, None],
                 templates: MutableSequence[np.ndarray],
                 masks: Sequence[np.ndarray],
                 offset: int = 1):
    # TODO: docs
    """

    :param ids_ord:
    :param classes:
    :param templates:
    :param masks:
    :param offset:
    :return:
    """
    templates_cp = [a.copy() for a in templates]
    for id_, positions in classes:
        pos = ids_ord[id_]
        templates_cp[pos][positions - offset] = 1
    for i, temp in enumerate(templates_cp):
        templates_cp[i] = templates_cp[i][masks[i]]
    return templates_cp


def parse_cls(id_cls_pairs: Iterable[str], ts: int = None) \
        -> Generator[Tuple[str, np.ndarray], None, None]:
    """
    Parses classes provided either by means of str with id-pos pairs
    separated by " "-like symbol
    or id-Iterable pairs (id and all true classes for this id)
    :param id_cls_pairs:
    :param ts:
    :return: Generator yielding tuple with ids and AA positions of true positive classes
    """
    lines = map(lambda x: x.strip().split(), id_cls_pairs)
    lines = (filter(lambda x: x[2] >= ts, map(lambda x: (x[0], int(x[1]), float(x[2])), lines)) if ts
             else map(lambda x: (x[0], int(x[1])), lines))
    lines = groupby(lines, lambda x: x[0])
    return ((g, np.array([x[1] for x in gg])) for g, gg in lines)


def compute_stats(y_true: np.ndarray, y_pred: np.ndarray, ts: float = None) \
        -> Stats:
    # TODO: docs
    """
    :param y_true:
    :param y_pred:
    :param ts:
    :return:
    """
    if ts:
        labels_pred = np.zeros(shape=y_pred.shape, dtype=np.int32)
        labels_pred[y_pred >= ts] = 1
    else:
        labels_pred = y_pred.round()

    negative_true = np.equal(y_true, 0).astype(np.float32)
    positive_true = np.equal(y_true, 1).astype(np.float32)
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
