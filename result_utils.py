"""Utilities to convert integer/compact result encodings into project format.

Provides:
- int_to_result(code) -> 'V' or 'D'
- int_to_ranking(code) -> ranking string like 'D6'
- normalize_result_dict(d) -> dict with 'Result' and 'Ranking' normalized

This is intentionally conservative: if the input is already a string it will
be normalized; integer codes map by a simple index-based table derived from
the project's `kaart` keys.
"""
from typing import Any, Dict, List

# Ranking keys (same order as used in ai2/ai3 `kaart`)
RANKING_KEYS: List[str] = [
    'A', 'B0', 'B2', 'B4', 'B6',
    'C0', 'C2', 'C4', 'C6',
    'D0', 'D2', 'D4', 'D6',
    'E0', 'E2', 'E4', 'E6',
    'NG'
]


def int_to_result(code: Any) -> str:
    """Map an integer/boolean/string code to 'V' (win) or 'D' (loss).

    Rules (defaults that can be extended):
    - 1, '1', True, 'W', 'V', 'win' -> 'V'
    - 0, -1, '0', False, 'L', 'D', 'loss' -> 'D'
    - If unknown, return input uppercased if it's 'V' or 'D', otherwise 'D'.
    """
    if isinstance(code, bool):
        return 'V' if code else 'D'
    if isinstance(code, (int,)):
        return 'V' if code == 1 else 'D'
    if code is None:
        return 'D'
    s = str(code).strip().upper()
    if s in ('1', 'W', 'WIN', 'V'):
        return 'V'
    if s in ('0', '-1', 'L', 'LOSS', 'D'):
        return 'D'
    # fallback: if it's single char V/D keep it, otherwise default to 'D'
    return s if s in ('V', 'D') else 'D'


def int_to_ranking(code: Any) -> str:
    """Map an integer or string code to a ranking string.

    - If `code` is int and within range -> return RANKING_KEYS[code]
    - If `code` is string and looks like a ranking (e.g., 'D6', 'c0'),
      return uppercased normalized form.
    - Otherwise return 'NG'.
    """
    if isinstance(code, int):
        if 0 <= code < len(RANKING_KEYS):
            return RANKING_KEYS[code]
        return 'NG'
    if code is None:
        return 'NG'
    s = str(code).upper().strip()
    # quick validation: letter + optional digit(s)
    if s in RANKING_KEYS:
        return s
    # allow 'A' -> 'A', 'B' -> 'B0' fallback
    if len(s) == 1 and s + '0' in RANKING_KEYS:
        return s + '0'
    return 'NG'


def normalize_result_dict(d: Dict[str, Any], *,
                          result_field: str = 'Result',
                          ranking_field: str = 'Ranking',
                          result_int_field: str = 'ResultInt',
                          ranking_int_field: str = 'RankingInt') -> Dict[str, str]:
    """Return a new dict with normalized 'Result' and 'Ranking' strings.

    The function will look for existing fields in this order:
    1. If `result_field` is present and string-like, normalize it.
    2. Else if `result_int_field` present, map it via int_to_result().
    Ranking similar: check `ranking_field` then `ranking_int_field`.
    """
    out: Dict[str, str] = {}

    # Result
    if result_field in d and d[result_field] is not None:
        out['Result'] = int_to_result(d[result_field])
    elif result_int_field in d:
        out['Result'] = int_to_result(d[result_int_field])
    else:
        out['Result'] = 'D'

    # Ranking
    if ranking_field in d and d[ranking_field] is not None:
        out['Ranking'] = int_to_ranking(d[ranking_field])
    elif ranking_int_field in d:
        out['Ranking'] = int_to_ranking(d[ranking_int_field])
    else:
        out['Ranking'] = 'NG'

    return out


if __name__ == '__main__':
    # Quick examples
    examples = [
        ({'ResultInt': 1, 'RankingInt': 12}, "int codes -> win, D6"),
        ({'ResultInt': 0, 'RankingInt': 2}, "int codes -> loss, B2"),
        ({'Result': 'W', 'Ranking': 'd6'}, "string codes -> win, D6"),
        ({'Result': 'L', 'Ranking': None}, "string loss, unknown rank"),
        ({'Result': True, 'RankingInt': 0}, "bool win, rank A"),
    ]

    for item, note in examples:
        print(f"Input: {item}  ({note}) ->", normalize_result_dict(item))
