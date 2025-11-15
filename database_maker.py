#etup
import pdfplumber
import pandas as pd
from typing import Optional
from pyvttl.vttl_api import VttlApi

# Instantiate the API client with credentials
api: Optional[VttlApi] = VttlApi(username='sandersmets', password='Imdc1234')

# Lees de PDF en extraheer alle tabellen met pdfplumber
all_tables = []
with pdfplumber.open("Club-API.pdf") as pdf:
    for page in pdf.pages:
        tables = page.extract_tables()
        for table in tables:
            df = pd.DataFrame(table)
            all_tables.append(df)

# Combineer alle tabellen in één DataFrame
df = pd.concat(all_tables, ignore_index=True)

# Sla de DataFrame op als CSV
df.to_csv("club_data.csv", index=False, encoding='utf-8')

# Lees de CSV terug en maak een mapping van clubnr naar provincie en naam
df_read = pd.read_csv("club_data.csv", encoding='utf-8', header=1)
club_to_province = {row['Clubnr.']: row['Provincie'] for _, row in df_read.iterrows()}
club_to_name = {row['Clubnr.']: row['Club'] for _, row in df_read.iterrows()}

def get_province_for_club(club_code):
    """Return the province for a given club code, e.g., 'A182' -> 'Antwerpen'."""
    return club_to_province.get(club_code, 'Niet gevonden')

def get_club_name_for_club(club_code):
    """Return the club name for a given club code, e.g., 'A182' -> 'TTC Nodo'."""
    return club_to_name.get(club_code, 'Niet gevonden')

def get_Results(club: str = None, name_search: str = None, season: int = None,):
    """Call the API and return members for a club."""
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

def extract_rank(information):
    """Extract the Ranking from each MemberEntry."""
    ranks = []
    for member in information.MemberEntries:
        rank = getattr(member, 'Ranking', None)
        ranks.append(rank)
    return ranks[0] if ranks else None  # Return first rank or None

def extract_club(results):
    """Extract the Club from each MemberEntry."""
    clubs = []
    for member in results.MemberEntries:
        club = getattr(member, 'Club', None)
        clubs.append(club)
    return clubs

def print_kaart(kaart):
    """Print the kaart dictionary in a formatted way."""
    for division, scores in kaart.items():
        print(f"{division}:  {scores[0]},  {scores[1]}")

def get_information(club: str = None, naam: str = None, seizoen: int = None):
    return api.getMembers(
        club=club,
        season=seizoen,
        name_search=naam,
        extended_information=True,
        ranking_points_information=None,
        with_results=None,
        with_opponent_ranking_evaluation=None,
    )

def extract_category(information):
    categories = []
    for member in information.MemberEntries:
        category = getattr(member, 'Category', None)
        categories.append(category)
    return categories[0] if categories else None  # Return first category or None

def extract_unique_index(information):
    unique_indices = []
    for member in information.MemberEntries:
        unique_index = getattr(member, 'UniqueIndex', None)
        unique_indices.append(unique_index)
    return unique_indices[0] if unique_indices else None  # Return first unique index or None

def get_data(club: str = None, name: str = None, season: int = 24):
    naam = str(name)
    club = str(club)
    seizoen = int(season)
    
    try:
        results_data = get_Results(club=club, name_search=naam, season=seizoen)
        extracted_data = extract_rank_and_results(results_data)
        information = get_information(club=club, naam=naam, seizoen=seizoen)
        
        kaart = {
            'A': [0, 0], 'B0': [0, 0], 'B2': [0, 0], 'B4': [0, 0], 'B6': [0, 0],
            'C0': [0, 0], 'C2': [0, 0], 'C4': [0, 0], 'C6': [0, 0],
            'D0': [0, 0], 'D2': [0, 0], 'D4': [0, 0], 'D6': [0, 0],
            'E0': [0, 0], 'E2': [0, 0], 'E4': [0, 0], 'E6': [0, 0],
            'NG': [0, 0]
        }
        
        update_kaart_with_results(kaart, extracted_data)
        
        # Return data with single values instead of lists
        return {
            'current_ranking': extract_rank(information),
            'category': extract_category(information),
            'province': get_province_for_club(club),
            'club_name': get_club_name_for_club(club),
            'unique_index': extract_unique_index(information),
            'kaart': kaart,
            'season': f"20{seizoen-1}-20{seizoen}"
        }
        
    except Exception as e:
        print(f"Error fetching data: {e}")
        return None

def get_memberlist(club: str = None, season: int = None):
    """Call the API and return members for a club."""
    global api
    if api is None:
        raise RuntimeError("VttlApi client is not initialized. Initialize `api` before calling get_memberlist().")
    return api.getMembers(
        club=club,
        season=season,
        name_search=None,
        extended_information=None,
        ranking_points_information=None,
        with_results=None,
        with_opponent_ranking_evaluation=None,
    )

if __name__ == "__main__":
    lijst = get_memberlist(club="A182", season=24)
    print(lijst)
