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

# Voorbeeld: print de mapping voor A182


# Definieer een functie om provincie op te halen voor een club
def get_province_for_club(club_code):
    """Return the province for a given club code, e.g., 'A182' -> 'Antwerpen'."""
    return club_to_province.get(club_code, 'Niet gevonden')

# Definieer een functie om clubnaam op te halen voor een club
def get_club_name_for_club(club_code):
    """Return the club name for a given club code, e.g., 'A182' -> 'TTC Nodo'."""
    return club_to_name.get(club_code, 'Niet gevonden')


def get_Results(club: str = None, name_search: str = None, season: int = None,):
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

def extract_rank(results):
    """Extract the Ranking from each MemberEntry."""
    ranks = []
    for member in results.MemberEntries:
        rank = getattr(member, 'Ranking', None)
        ranks.append(rank)
    return ranks

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
    return categories

def extract_unique_index(information):
    unique_indices = []
    for member in information.MemberEntries:
        unique_index = getattr(member, 'UniqueIndex', None)
        unique_indices.append(unique_index)
    return unique_indices

def extract_names(information):
    names = []
    for member in information.MemberEntries:
        first_name = getattr(member, 'FirstName', '')
        last_name = getattr(member, 'LastName', '')
        full_name = f"{first_name} {last_name}".strip()
        names.append(full_name)
    return names

def get_data(club: str = None, name: str = None, season: int = 25):
    naam = str(name)
    club = str(club)
    seizoen = int(season)
    results_data = get_Results(club=club, name_search=naam, season=seizoen)
    extracted_data = extract_rank_and_results(results_data)
    information = get_information(club=club, naam=naam, seizoen=seizoen)
    information_next_season = get_information(club=club, naam=naam, seizoen=seizoen+1)
    kaart = {
        'A': [0, 0], 'B0': [0, 0], 'B2': [0, 0], 'B4': [0, 0], 'B6': [0, 0],
        'C0': [0, 0], 'C2': [0, 0], 'C4': [0, 0], 'C6': [0, 0],
        'D0': [0, 0], 'D2': [0, 0], 'D4': [0, 0], 'D6': [0, 0],
        'E0': [0, 0], 'E2': [0, 0], 'E4': [0, 0], 'E6': [0, 0],
        'NG': [0, 0]
    }
    update_kaart_with_results(kaart, extracted_data)
    return {
        'name': naam,
        'season': f"20{seizoen-1}-20{seizoen}",
        'current_ranking': extract_rank(information),
        'next_ranking': extract_rank(information_next_season),
        'category': extract_category(information),
        'province': get_province_for_club(club),
        'club_name': get_club_name_for_club(club),
        'unique_index': extract_unique_index(information),
        'kaart': kaart
    }
# Example usage


## ai database



def get_information_club(club: str = None, seizoen: int = None):
    return api.getMembers(
        club=club,
        season=seizoen,
        extended_information=True,
        ranking_points_information=None,
        with_results=None,
        with_opponent_ranking_evaluation=None,
    )

def create_club_csv(club: str, season: int):
    """Create a CSV file with member data for a given club and season."""
    club_info = get_information_club(club=club, seizoen=season)
    names = extract_names(club_info)
    all_data = []
    for name in names:
        try:
            data = get_data(club=club, name=name, season=season)
            all_data.append(data)
        except Exception as e:
            print(f"Error fetching data for {name}: {e}")
            continue
    df = pd.DataFrame(all_data)
    filename = f"club_members_data_{club}_{season}.csv"
    df.to_csv(filename, index=False, encoding='utf-8')
    print(f"Data saved to {filename}")

# Example usage

import os




def append_to_main_file_batch(data_list, filename="club_members_main_data.csv"):
    """Append multiple data entries to the main file in batch."""

    if not data_list:
        return 0

    # Create DataFrame from all data
    df_new = pd.DataFrame(data_list)

    # Convert lists to strings for deduplication
    df_new['unique_index'] = df_new['unique_index'].apply(lambda x: str(x[0]) if isinstance(x, list) and x else str(x))
    df_new['season'] = df_new['season'].astype(str)

    # Check if file exists
    if os.path.exists(filename):
        # Read existing data
        df_existing = pd.read_csv(filename, encoding='utf-8')
        # Convert lists to strings for deduplication
        df_existing['unique_index'] = df_existing['unique_index'].apply(lambda x: str(x) if not isinstance(x, str) else x)
        df_existing['season'] = df_existing['season'].astype(str)
        # Append new data
        df_combined = pd.concat([df_existing, df_new], ignore_index=True)
        # Remove duplicates based on unique_index and season
        df_combined = df_combined.drop_duplicates(subset=['unique_index', 'season'], keep='last')
    else:
        df_combined = df_new

    # Save combined data
    df_combined.to_csv(filename, index=False, encoding='utf-8')
    return len(data_list)

def add_data_for_club_batch(club: str, season: int):
    """Add data for all members of a club in a given season to the main CSV file in batch."""
    club_info = get_information_club(club=club, seizoen=season)
    names = extract_names(club_info)

    all_data = []
    for name in names:
        try:
            data = get_data(club=club, name=name, season=season)
            all_data.append(data)
            print(f"Processed data for {name}")
        except Exception as e:
            print(f"Error fetching data for {name}: {e}")
            continue

    # Add all data in one batch
    added_count = append_to_main_file_batch(all_data)
    print(f"Successfully added {added_count} records to club_members_main_data.csv")

# Example usage
if __name__ == "__main__":
    add_data_for_club_batch(club="A182", season=25)
