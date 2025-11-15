#etup

from typing import Optional
from pyvttl.vttl_api import VttlApi

# Instantiate the API client properly
api: Optional[VttlApi] = VttlApi()  # Replace with actual initialization if needed
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


# Basic structure of the kaart dictionary
kaart = {
    'A': [0, 0], 'B0': [0, 0], 'B2': [0, 0], 'B4': [0, 0], 'B6': [0, 0],
    'C0': [0, 0], 'C2': [0, 0], 'C4': [0, 0], 'C6': [0, 0],
    'D0': [0, 0], 'D2': [0, 0], 'D4': [0, 0], 'D6': [0, 0],
    'E0': [0, 0], 'E2': [0, 0], 'E4': [0, 0], 'E6': [0, 0],
    'NG': [0, 0]
}

# Example updates to kaart
kaart['A'][0] += 1  # win
kaart['A'][1] += 1  # loss

results = get_Results(club='A-182', name_search='Sander Smets', season=26)

# Debug: Print the structure
print("Type of results:", type(results))
print("Length of results:", len(results) if isinstance(results, list) else "N/A")
print("\nFirst result:")
print(results[0] if results else "Empty")

if results and isinstance(results, list):
    print("\nKeys in first result:")
    print(results[0].keys() if isinstance(results[0], dict) else "Not a dictionary")
    
    # If there's a 'results' or 'Matches' field, print that too
    if 'results' in results[0]:
        print("\nFirst match in results:")
        print(results[0]['results'][0] if results[0]['results'] else "No matches")

# Process results and update kaart
if isinstance(results, list):
    for member in results:
        # Each member may have a 'results' or 'Matches' field
        matches = member.get('results') or member.get('Matches') or member.get('Results') or []
        
        for match in matches:
            # Extract division and match result
            division = match.get('ranking') or match.get('Ranking') or match.get('division')
            score_for = match.get('SetFor') or match.get('SetsFor') or match.get('set_for') or 0
            score_against = match.get('SetAgainst') or match.get('SetsAgainst') or match.get('set_against') or 0

            # Coerce scores to integers
            try:
                sf = int(score_for)
                sa = int(score_against)
            except (TypeError, ValueError):
                continue

            if division in kaart:
                if sf > sa:
                    kaart[division][0] += 1  # Win
                elif sf < sa:
                    kaart[division][1] += 1  # Loss

print(kaart)