from src.structures import Interval, NetInput
import numpy as np
from typing import Iterable, Union, Text, Optional
from io import TextIOWrapper


def parse_predictions(predictions: Iterable[np.ndarray], inp: NetInput, treshold: float):
    valid_predictions = zip(inp.ids, (p[p >= treshold] for p in predictions))



def dump_predictions(to_dump, file: Optional[Union[TextIOWrapper, Text]] = None):
    def dump():
        pass
    pass


if __name__ == '__main__':
    raise RuntimeError
