import pandas as pd
from database_maker import get_information, extract_elo
import ast
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock

# Load club data to create reverse mapping
try:
    df_clubs = pd.read_csv("club_data.csv", encoding='utf-8', header=1)
    club_name_to_code = {row['Club']: row['Clubnr.'] for _, row in df_clubs.iterrows()}
except FileNotFoundError:
    print("Warning: club_data.csv not found. Using default club code.")
    club_name_to_code = {}

# Read the CSV file
df = pd.read_csv("club_members_with_next_ranking.csv")

# Add elo column if it doesn't exist
if 'elo' not in df.columns:
    df['elo'] = None

print(f"Processing {len(df)} members with parallel requests...")

# Lock for thread-safe dataframe updates
df_lock = Lock()
processed_count = [0]

def process_member(idx, row):
    """Process a single member and return the result."""
    try:
        name = row['name']
        season_str = row['season']
        club_name = row['club_name']
        
        club = club_name_to_code.get(club_name, "A182")
        season_year = int(season_str.split('-')[1])
        season = season_year % 100
        
        unique_index_str = row['unique_index']
        try:
            parsed = ast.literal_eval(str(unique_index_str))
            unique_index = parsed[0] if isinstance(parsed, list) else parsed
        except:
            unique_index = None
        
        # Get information from API
        info = get_information(naam=name, club=club, seizoen=season)
        elo = extract_elo(info)
        
        with df_lock:
            processed_count[0] += 1
            print(f"[{processed_count[0]}/{len(df)}] {name}: ELO={elo}")
        
        return idx, elo, None
        
    except Exception as e:
        with df_lock:
            processed_count[0] += 1
            print(f"[{processed_count[0]}/{len(df)}] Error: {name} - {e}")
        return idx, None, str(e)

# Use ThreadPoolExecutor for parallel processing
with ThreadPoolExecutor(max_workers=10) as executor:
    futures = {executor.submit(process_member, idx, row): idx for idx, row in df.iterrows()}
    
    for future in as_completed(futures):
        idx, elo, error = future.result()
        if elo is not None:
            df.at[idx, 'elo'] = elo

# Save the updated CSV
df.to_csv("club_members_with_next_ranking.csv", index=False)
print("\nDone! CSV file updated with ELO information.")
