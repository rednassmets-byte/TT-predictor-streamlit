# setup
from typing import Optional, Dict, Any
from pyvttl.vttl_api import VttlApi
from pyvttl.province import Province

# Do not call the API at import time in example code. Instantiate in real runs.
api: Optional[VttlApi] = None
naam = 'Sander smets'


def get_Results(club: str = None, name_search: str = None, season: int = None):
    """Call the API and return members for a club.

    Note: requires `api` to be initialized (VttlApi instance).
    """
    global api
    if api is None:
        raise RuntimeError("VttlApi client is not initialized. Initialize `api` before calling get_Results().")

    return api.getMembers(
        club=club,
        season=season,
        name_search=name_search,
        extended_information=None,
        ranking_points_information=None,
        with_results=True,
        with_opponent_ranking_evaluation=None,
    )


kaart = {'A': [0, 0], 'B0': [0, 0], 'B2': [0, 0], 'B4': [0, 0], 'B6': [0, 0],
        'C0': [0, 0], 'C2': [0, 0], 'C4': [0, 0], 'C6': [0, 0],
        'D0': [0, 0], 'D2': [0, 0], 'D4': [0, 0], 'D6': [0, 0],
        'E0': [0, 0], 'E2': [0, 0], 'E4': [0, 0], 'E6': [0, 0],
        'NG': [0, 0]}
# Examples of valid ways to update the counts in `kaart`:
# Increment wins for ranking 'A' (index 0)
kaart['A'][0] += 1
# Increment losses for ranking 'A' (index 1)
kaart['A'][1] += 1
print(kaart)