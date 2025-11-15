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

print("CSV-bestand succesvol aangemaakt: club_data.csv")

# Lees de CSV terug en maak een mapping van clubnr naar provincie
df_read = pd.read_csv("club_data.csv", encoding='utf-8', header=1)
club_to_province = {row['Clubnr.']: row['Provincie'] for _, row in df_read.iterrows()}

# Voorbeeld: print de mapping voor A182


# Definieer een functie om provincie op te halen voor een club
def get_province_for_club(club_code):
    """Return the province for a given club code, e.g., 'A182' -> 'Antwerpen'."""
    return club_to_province.get(club_code, 'Niet gevonden')


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
naam = str("Jef Frans")
club = str(("A182"))
seizoen = int(("25"))
results_data = get_Results(club=club, name_search=naam, season=seizoen)
results = get_Results(club=club, name_search=naam, season=seizoen)
extracted_data = extract_rank_and_results(results_data)
information = get_information(club=club, naam=naam, seizoen=seizoen)
information_next_season = get_information(club=club, naam=naam, seizoen=seizoen+1)
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
print(f"Huidig klassement:{extract_rank(information)}")
print(f"Volgend Klassment:{extract_rank(information_next_season)}")
print(f"Categorie huidig seizoen:{extract_category(information)}") 
print(f"Provincie: {get_province_for_club(club)}")