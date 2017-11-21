from itertools import chain
from typing import Iterable, Tuple, Mapping, List, Any, Union, Sequence, Optional, Text, Generator

import numpy as np
from Bio.SeqIO.FastaIO import SimpleFastaParser
from Bio.SeqUtils import IUPACData
from functools import reduce, partial
from io import TextIOWrapper
from pyrsistent import pvector, v

from src.structures import Interval, NetInput, Seq


def prepare_input(fasta_file_handle: TextIOWrapper,
                  hparams: Mapping[str, Union[str, int, float]],
                  cli_params: Mapping[str, Union[str, int, float]]) -> NetInput:
    """
    Prepares an ready-to-use input for model.
    :param fasta_file_handle:  opened fasta file handle
    :param hparams:
    :param cli_params:
    Must contain value for 'mode' key;
    one of 3 modes:
    1) predict -- means we do not need/know True classes
    2) eval -- means True classes are to be in the same form as INPUT.seqs for the purpose of evaluation.
    Network input stays the same (since True classes are meant to be provided using separate file)
    -- hence, function's output for 'predict' and 'eval' are the same as well
    3) train -- means True classes are to be rolled over and passed
    as input in the same form as INPUT.rolled_seqs
    :return: namedtuple with all the data needed to run the model
    in any of the 3 modes specified above
    """
    w_size = hparams['window_size']
    w_step = cli_params['window_step'] if cli_params['window_step'] is not None else hparams['window_step']
    seq_maxlen = hparams['seq_maxlen']
    mode = cli_params['mode']
    if mode == 'predict' or mode == 'eval':
        return NetInput(*prepare_eval(fasta_file_handle, w_size,
                                      w_step, seq_maxlen))
    elif mode == 'train':
        pass
    else:
        raise ValueError("did not understand mode: should be either 'eval' or 'train'")


def validate_window(window_size: int, window_step: int) -> Optional[bool]:
    if window_step or window_size:
        if not (window_step and window_size):
            raise ValueError('Provide either window_size and window_step or neither of them')
        return True


def prepare_eval(fasta_handle: TextIOWrapper,
                 window_size: Optional[int] = None,
                 window_step: Optional[int] = None,
                 maxlen: Optional[int] = None):
    """
    Prepares input for prediction/evaluation
    :param fasta_handle:
    :param window_size:
    :param window_step:
    :param maxlen:
    :return:
    """

    def create_negative(array: np.ndarray, classes: Iterable = (16, 17, 20)) \
            -> np.ndarray:
        """
        Masks all the classes not belonging to _classes_ variable
        (i.e. negative classes -- not in {Ser, Thr, Tyr})
        with zeros.
        Classes able to be positive classes are assigned ones.
        :param array: input array with encoded sequence
        :param classes: true positive classes
        :return: numpy array (binary-encoded sequence)
        with negative classes as zeros and positive classes as ones
        """
        neg = np.zeros(shape=array.shape, dtype=np.float32)
        for i, cls in enumerate(classes, start=1):
            neg[array == cls] = 1.0
        return neg

    def prepare_x(sequences: Iterable[Interval[np.ndarray]], join_len: int) \
            -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Prepares encoded sequences to be used in a model:
        merges into one array, calculates masks and negatives (see create_negative docs)
        :param sequences: numerically encoded sequences
        :param join_len: see join function docs
        :return: joined sequences, masks with False-padded tails and negative
        (np array  with zeros everywhere except for classes able to be positive)
        """
        joined_x, masks = join((s.data for s in chain.from_iterable(sequences)), join_len)
        negative = create_negative(joined_x)
        return joined_x, masks, negative

    fasta = list(SimpleFastaParser(fasta_handle))
    ids = [x[0] for x in fasta]
    seqs = encode_seqs([x[1] for x in fasta], {x: i for i, x in enumerate(
        sorted(IUPACData.protein_letters_1to3.keys()), start=1)})

    if validate_window(window_size, window_step):
        roll = partial(roll_window, window_size=window_size, window_step=window_step, stop_at=window_size // 2)
        rolled = [roll(a) for a in (s.data.encoded for s in seqs)]
        joined_x_, masks_, negative_ = prepare_x(rolled, join_len=window_size)
        rolled = [[Interval(x.start, x.stop) for x in r] for r in rolled]
        return ids, seqs, rolled, None, joined_x_, masks_, negative_

    joined_x_, masks_, negative_ = prepare_x(seqs, join_len=maxlen)
    return ids, seqs, None, None, joined_x_, masks_, negative_


def join(arrays: Union[Sequence[np.ndarray], Iterable[np.ndarray], np.ndarray],
         arrays_len: Optional[int] = None,
         array_maxlen: Optional[int] = None,
         dtype: np.dtype = np.int32) \
        -> Tuple[np.ndarray, np.ndarray]:
    """
    Joins arrays across 0th dimension;
    pads missing tails with zeros.
    In case of one array just pads it's tail with zeros.
    :param arrays: arrays to be joined.
    Warning: if it's an Iterator and arrays_len is not provided, it'll be evaluated.
    :param arrays_len: number of arrays.
    :param array_maxlen: maximum sample length in arrays. If not provided, function will attempt to calculate it.
    :param dtype: desired dtype of output.
    :return: tuple of joined arrays and boolean mask with 0-tails as False.
    :raises ValueError: if length < maximum length in arrays.
    :raises TypeError: if len(arrays) fails (if arrays_len is not provided and arrays is not an Iterator).
    """
    if isinstance(arrays, np.ndarray):
        arrays_len = 1
        arrays = [arrays]
    elif isinstance(arrays, Iterable) or isinstance(arrays, Generator) and not arrays_len:
        arrays = list(arrays)
        arrays_len = len(arrays)
    else:
        try:
            arrays_len = len(arrays)
        except TypeError:
            print(arrays, type(arrays))
            raise TypeError("Can't calculate arrays length. Check type of 'arrays' variable")

    lenmax = max(len(x) for x in arrays)
    if array_maxlen:
        if lenmax > array_maxlen:
            raise ValueError('Length is smaller than maxlen of arrays')
        length = array_maxlen
    else:
        length = lenmax

    joined = np.zeros((arrays_len, length), dtype=dtype)
    masks = np.zeros((arrays_len, length), dtype=bool)

    for i, arr in enumerate(arrays):
        joined[i, :len(arr)] = arr
        masks[i, :len(arr)] = True

    return joined, masks


def mask_false_cls(array: np.ndarray,
                   target_classes: Iterable[int] = (16, 17, 20)):
    tmp = np.zeros(shape=array.shape, dtype=np.bool)
    for i in target_classes:
        tmp[array == i] = True
    return tmp


def encode_seqs(seqs: Sequence[Text],
                classes_mapping: Mapping[str, int]) \
        -> List[Interval[np.ndarray]]:
    """
    Numerically encodes sequences
    :param seqs: Sequence of texts
    :param classes_mapping: Mapping between textual and integral representations of characters in seqs.
    If for char x no corresponding value is found in classes_mapping it will be assigned 0
    :return: list of encoded arrays
    """

    def encode(seq: Text) -> np.ndarray:
        return np.array([classes_mapping[x] if x in classes_mapping else 0 for x in seq], dtype=np.int32)

    borders = _segment_borders(seqs)
    encoded_seqs = [Interval(start, stop, Seq(encode(seq), seq)) for (start, stop), seq in zip(borders, seqs)]
    return encoded_seqs


def _segment_borders(texts: Iterable[Text]) -> List[Tuple[int, int]]:
    # TODO docs
    """
    Returns a list of cumulative start/stop positions for segments in `texts`.
    :param texts: a list of strings
    :return: list of (start position, stop position)
    >>> _segment_borders(['amino acid', 'is any']) == [(0, 10), (10, 16)]
    True
    """

    def aggregate_boundaries(boundaries: pvector, text):
        return (
            boundaries + [(boundaries[-1][1], boundaries[-1][1] + len(text))]
            if boundaries else v((0, len(text)))
        )

    return list(reduce(aggregate_boundaries, texts, v()))


def roll_window(array: Union[Sequence[Any], np.ndarray], window_size: int, window_step: int, stop_at=0) \
        -> Sequence[Interval[np.ndarray]]:
    """
    rolls window over an array of objects (numbers, letters, etc.)
    :param array: array of objects
    :param window_size:
    :param window_step:
    :param stop_at: left-hand position to stop a window at;
    by default = window_size - 1 which means if will produce output of arrays with len = window_size
    if you want to guarantee that all array values will be in an output, use stop_at = window_size // 2
    :return: tuple of intervals array had been split at and list of arrays original array had been split to
    >>> a = np.arange(20)
    >>> roll_window(a, 10, 3)
    ([(0, 10), (3, 13), (6, 16), (9, 19)],
     [array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9]),
      array([ 3,  4,  5,  6,  7,  8,  9, 10, 11, 12]),
      array([ 6,  7,  8,  9, 10, 11, 12, 13, 14, 15]),
      array([ 9, 10, 11, 12, 13, 14, 15, 16, 17, 18])])
    >>> roll_window(a, 10, 3, 5)
    ([(0, 10), (3, 13), (6, 16), (9, 19), (12, 22)],
     [array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9]),
      array([ 3,  4,  5,  6,  7,  8,  9, 10, 11, 12]),
      array([ 6,  7,  8,  9, 10, 11, 12, 13, 14, 15]),
      array([ 9, 10, 11, 12, 13, 14, 15, 16, 17, 18]),
      array([12, 13, 14, 15, 16, 17, 18, 19])])
    """
    if not stop_at:
        stop_at = window_size - 1
    intervals = [(i, window_size + i) for i in range(0, len(array) - stop_at, window_step)]
    if not intervals:
        return [Interval(0, window_size, np.array(array, dtype=np.int32))]
    return [Interval(start, stop, array[start:stop]) for start, stop in intervals]


if __name__ == '__main__':
    raise RuntimeError
