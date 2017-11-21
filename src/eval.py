from itertools import groupby, starmap, chain
from typing import Iterable, Mapping, Union, Generator, Tuple, Optional, Sequence, MutableSequence

import numpy as np
from io import TextIOWrapper

from src.predict import predict
from src.structures import NetInput, Stats, Site


def eval_and_dump(model: Optional,
                  inp: NetInput,
                  hparams: Mapping[str, Union[str, int, float, TextIOWrapper]],
                  cli_params: Mapping[str, Union[str, int, float, TextIOWrapper]]) \
        -> None:
    """
    Calls evaluate and dumps it's results using dump function.
    If no model is provided, will evaluate predictions read from file.
    :param model: Optional compiled model.
    :param inp: NetInput (output of prepare_input function)
    :param hparams: hparams loaded from json file
    :param cli_params: kwargs provided through CLI
    :return: Returns nothing. If no output file is provided, will dump results to stdout
    """
    stats, sites = evaluate(model, cli_params['predictions'], inp, hparams, cli_params)
    dump(cli_params, stats, sites)


def evaluate(model: Optional,
             predictions: Optional[Union[Iterable[np.ndarray], TextIOWrapper]],
             inp: NetInput,
             hparams: Mapping[str, Union[str, int, float, TextIOWrapper]],
             cli_params: Mapping[str, Union[str, int, float, TextIOWrapper]]) \
        -> Tuple[Optional[Stats], Optional[Iterable[Site]]]:
    """
    Evaluates model performance in terms of various statistics:
    F1-score, precision, recall, false positive rate,
    false negative rate, accuracy and specificity.
    Besides evaluation aggregates predictions according to id
    and returns phosphosites Iterable over Site with
    id-position-predicted_class-true-class for each Site.
    :param model: If model is passed, predicts phosphosites using this model.
    If model is None, reads predictions from file.
    :param predictions: Either opened file in tsv-like format
    (such as an output of predict_and_dump function)
    or Iterable over Tuples with id and positions of
    predicted positive sites.
    :param inp: NetInput
    :param hparams: hparams loaded from json file
    :param cli_params: kwargs provided through CLI
    :return: Depending on eval_output_mode specified in cli_params returns
    one of:
    1) Stats, Iterable[Sites] --> full
    2) Stats, None --> stats_only
    3) None, Iterable[Sites] --> tsv only
    """
    bs = cli_params['batch_size'] if cli_params['batch_size'] is not None else hparams['batch_size']
    ts = cli_params['threshold'] if cli_params['threshold'] is not None else hparams['threshold']
    mode = cli_params['eval_output_mode']

    # prepare templates and masks (with all possible positive classes) for evaluation
    templates = [np.zeros(shape=(len(x.data.raw),)) for x in inp.seqs]
    templates_bool = [t.astype(bool) for t in templates]
    for e, s in enumerate(inp.seqs):
        for i in (16, 17, 20):
            templates_bool[e][s.data.encoded == i] = True

    # parse predictions from file or predict de novo if model is not None
    if model is None:
        offset = 1
        if isinstance(predictions, TextIOWrapper):
            pred = parse_cls(predictions, ts)
        else:
            pred = zip(inp.ids, predictions)
    else:
        offset = 0
        pred = zip(inp.ids, (np.where(p > ts)[0] for p in predict(model, inp, batch_size=bs)))

    # correct ids' order
    ids_ord = {x: i for i, x in enumerate(inp.ids)}

    # prepare predictions to be used in compute_stats
    pred = map_onto_pos(ids_ord=ids_ord, templates=templates, masks=templates_bool,
                        classes=pred, offset=offset)
    true_cls = map_onto_pos(ids_ord=ids_ord, templates=templates, masks=templates_bool,
                            classes=parse_cls(cli_params['input_cls']))

    # compute stats and compile sites
    stats = (compute_stats(y_true=np.concatenate(true_cls), y_pred=np.concatenate(pred))
             if mode == 'full' or mode == 'stats_only' else None)
    sites = compile_sites(inp, true_cls, pred, templates_bool) if mode == 'full' or mode == 'tsv_only' else None

    return stats, sites


def compile_sites(inp: NetInput,
                  y_true: Iterable[np.ndarray],
                  y_pred: Iterable[np.ndarray],
                  masks: Iterable[np.ndarray]):
    """
    Prepares sites to be dumped in tsv file
    :param inp: NetInput
    :param y_true: True known classes mapped on templates
    :param y_pred: True predicted classes mapped on templates
    :param masks: boolean numpy arrays with
    True placed at positions of any class that
    could be positive
    :return: Iterable over Sites
    """
    positions = (np.where(y > 0)[0] + 1 for y in masks)

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
    """
    Prints evaluation results (to stdout if no output_file is provided).
    If call chain is in sequence kuppnet-->eval_and_dump-->dump
    2 things are guaranteed:
    1) Either stats or sites are provided
    2) cli_params['eval_output_mode'] is not None
    :param cli_params:
    :param stats: Stats
    :param sites: Iterable over Sites
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
                 classes: Union[Iterable[Tuple[str, np.ndarray]], Generator[Tuple[str, np.ndarray], None, None]],
                 templates: MutableSequence[np.ndarray],
                 masks: Sequence[np.ndarray],
                 offset: int = 1) -> Iterable[np.ndarray]:
    """
    Maps true true positive classes onto templates
    :param ids_ord: Correct order o ids from 0 to len(inp.seqs)
    :param classes: True positive classes positions
    in form of pairs id-positions
    :param templates: 0-templates in correct order
    where each template has len of corresponding sequence
    :param masks: boolean numpy arrays with
    True placed at positions of any class that
    could be positive
    :param offset: offset for positions
    (default is 1 since AA numeration usually starts with 1)
    :return: numpy arrays with 1's places onto positions
    provided in classes with bool mask applied to each template
    leaving only relevant classes
    """
    templates_cp = [a.copy() for a in templates]
    for id_, positions in classes:
        pos = ids_ord[id_]
        templates_cp[pos][positions - offset] = 1
    for i, temp in enumerate(templates_cp):
        templates_cp[i] = templates_cp[i][masks[i]]
    return templates_cp


def parse_cls(id_cls: Iterable[str], ts: int = None) \
        -> Generator[Tuple[str, np.ndarray], None, None]:
    """
    Parses entries with " "-like separated id-position-(optional score).
    Positions are provided for true positive classes only
    :param id_cls: Iterator over opened file where each line
    contains either id-position or id-position-score
    :param ts: if threshold is passed will only select sited >= ts.
    ts values are assumed to be placed at 2'd position if starting from 0
    :return: Generator yielding tuple with ids and AA positions of true positive classes
    """
    lines = map(lambda x: x.strip().split(), id_cls)
    lines = (filter(lambda x: x[2] >= ts, map(lambda x: (x[0], int(x[1]), float(x[2])), lines)) if ts
             else map(lambda x: (x[0], int(x[1])), lines))
    lines = groupby(lines, lambda x: x[0])
    return ((g, np.array([x[1] for x in gg])) for g, gg in lines)


def compute_stats(y_true: np.ndarray, y_pred: np.ndarray, ts: float = None) \
        -> Stats:
    """
    Computes stats comparing binary classes in y_true and y_pred.
    :param y_true: True classes
    :param y_pred: Predicted classes
    :param ts: if provided, classes with score >= ts
    will be assigned 1, < ts will be assigned 0
    :return: Stats (namedtuple) with accuracy, fnr, fpr, precision, recall, f1, specificity
    :raises: ValueError if y_true and y_pred don't have the same length
    """
    if len(y_pred) != len(y_pred):
        raise ValueError('True and Predicted classes must have the same length')
    if ts:
        labels_pred = np.zeros(shape=y_pred.shape, dtype=np.int32)
        labels_pred[y_pred >= ts] = 1
        labels_pred[y_pred < ts] = 0
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
