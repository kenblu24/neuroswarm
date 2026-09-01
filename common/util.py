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
