import numpy as np

# typing
from typing import Sequence


def parse_rangelist(s: str | range | Sequence[int]):
    """parse a comma-separated list of integers or ranges of integers

    Examples
    --------
        >>> parse_rangelist('1, 2, 5-7, 8-10, 11, 11')
        [1, 2, 5, 6, 7, 8, 9, 10, 11]
        >>> parse_rangelist(range(1, 10))
        [1, 2, 3, 4, 5, 6, 7, 8, 9]
    """
    if isinstance(s, Sequence) and not isinstance(s, str):
        return list(s)
    segments = [seg.lstrip() for seg in s.strip().split(',')]
    li = []
    for seg in segments:
        if '-' in seg:
            start, end = seg.split('-')
            li.extend(range(int(start), int(end) + 1))
        else:
            li.append(int(seg))
    return list(dict.fromkeys(li))  # remove duplicates


def seedmosh(seed, size=None):
    return np.random.default_rng(seed).integers(0, 2**32, size=size)


def filter_seeds(*seeds, reverse=True) -> list[int] | None:
    seeds = flatten_iter([seeds])
    seeds: list[int | None]
    seeds = [int(s) for s in seeds if s is not None]
    if not all(s >= 0 for s in seeds):
        raise ValueError("All seeds must be >= 0")
    if reverse:
        seeds.reverse()
    return seeds or None


def flatten_iter(it):
    new = []
    for i in it:
        try:
            new.extend(flatten_iter(i))
        except TypeError:
            new.append(i)
    return new
