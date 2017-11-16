import sys
from typing import TypeVar, Container, Generic, Optional, Sequence, NamedTuple, Text

import numpy as np

_slots_supported = (sys.version_info >= (3, 6, 2) or
                    (3, 5, 3) <= sys.version_info < (3, 6))
T = TypeVar("T")

Stats = NamedTuple('stats', [
    ('accuracy', float),
    ('fnr', float),
    ('fpr', float),
    ('precision', float),
    ('recall', float),
    ('f1_score', float),
    ('specificity', float)])


Site = NamedTuple('site', [
    ('id', str),
    ('pos', int),
    ('pred', int),
    ('true', int)])


class Interval(Container, Generic[T]):
    if _slots_supported:
        __slots__ = ("start", "stop", "data")

    def __init__(self, start: int, stop: int, data: Optional[T] = None):
        self.start = start
        self.stop = stop
        self.data = data

    def __contains__(self, item: T) -> bool:
        return False if self.data is None or item is None else self.data == item

    def __iter__(self):
        return iter(range(self.start, self.stop))

    def __eq__(self, other: "Interval"):
        return (self.start, self.stop, self.data) == (other.start, other.stop, other.data)

    def __hash__(self):
        return hash((self.start, self.stop, self.data))

    def __len__(self):
        return self.stop - self.start

    def __bool__(self):
        return bool(len(self))

    def __and__(self, other: "Interval"):
        # TODO docs
        first, second = sorted([self, other], key=lambda iv: iv.start)
        return type(self)(first.start, second.stop, [first.data, second.data])

    def __repr__(self):
        return "{}(start={}, stop={}, data={})".format(type(self).__name__,
                                                       self.start,
                                                       self.stop,
                                                       self.data)

    def reload(self, value: T):
        return type(self)(self.start, self.stop, value)


Seq = NamedTuple('seq', ([('encoded', np.ndarray), ('raw', Text)]))

NetInput = NamedTuple('input', [
    ('ids', Sequence[str]),
    ('seqs', Sequence[Seq]),
    ('rolled_seqs', Optional[Sequence[Sequence[Interval[np.ndarray]]]]),
    ('rolled_cls', Optional[Sequence[Sequence[Interval[np.ndarray]]]]),
    ('joined', np.ndarray),
    ('masks', np.ndarray),
    ('negative', np.ndarray)])

if __name__ == '__main__':
    raise RuntimeError
