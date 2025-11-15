#etup

from typing import Optional
from pyvttl.vttl_api import VttlApi

# Instantiate the API client properly
api: Optional[VttlApi] = VttlApi()  # Replace with actual initialization if needed



    


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



kaart = {
    'A': [0, 0], 'B0': [0, 0], 'B2': [0, 0], 'B4': [0, 0], 'B6': [0, 0],
    'C0': [0, 0], 'C2': [0, 0], 'C4': [0, 0], 'C6': [0, 0],
    'D0': [0, 0], 'D2': [0, 0], 'D4': [0, 0], 'D6': [0, 0],
    'E0': [0, 0], 'E2': [0, 0], 'E4': [0, 0], 'E6': [0, 0],
    'NG': [0, 0]
}


def extract_rank_and_results(results):
    """Extract rank and result from the member entries."""
    extracted_data = []
    for entry in results['MemberEntries']:
        for result in entry['ResultEntries']:
            extracted_data.append({
                'Rank': result['Ranking'],
                'Result': result['Result']
            })
    return extracted_data



def update_kaart_with_results(kaart, rank_and_results):
    """Update the kaart dictionary based on the rank and results."""
    for entry in rank_and_results:
        rank = entry['Rank']
        result = entry['Result']
        
        if result == 'V':  # Win
            if rank in kaart:
                kaart[rank][0] += 1  # Increment win count
        elif result == 'D':  # Loss
            if rank in kaart:
                kaart[rank][1] += 1  # Increment loss count

def extract_rank_(results):
    
    
    for entry in results['MemberEntries']:
        rank = entry.get('Ranking')  # Assuming 'Ranking' is directly in MemberEntry
        
    return rank


def print_kaart(kaart):
    """Print the kaart dictionary in a formatted way."""
    for division, scores in kaart.items():
        print(f"{division}:  {scores[0]},  {scores[1]}")



naam = str(input("Naam speler: "))
club = str(input("Club bv A182: "))
seizoen = int(input("Seizoen:   bv 26 "))
results_data = get_Results(club=club, name_search=naam, season=seizoen)
results = get_Results(club=club, name_search=naam, season=seizoen)
extracted_data = extract_rank_and_results(results_data)
kaart = {
    'A ': [0, 0], 'B0': [0, 0], 'B2': [0, 0], 'B4': [0, 0], 'B6': [0, 0],
    'C0': [0, 0], 'C2': [0, 0], 'C4': [0, 0], 'C6': [0, 0],
    'D0': [0, 0], 'D2': [0, 0], 'D4': [0, 0], 'D6': [0, 0],
    'E0': [0, 0], 'E2': [0, 0], 'E4': [0, 0], 'E6': [0, 0],
    'NG': [0, 0]
}  
extract_rank_and_results(results)
update_kaart_with_results(kaart,extracted_data)
print_kaart(kaart)
print(f"seizoen: 20{seizoen-1}-20{seizoen}")
#print(extract_rank_(results))
print(api.getMembers(
        club=club,
        season=seizoen,
        name_search=naam,
        extended_information=None,
        ranking_points_information=None,
        with_results=None,
        with_opponent_ranking_evaluation=None,
    ))