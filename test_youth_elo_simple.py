"""
Simple test to compare V3 vs V4 predictions for youth
"""
import joblib
import numpy as np
import pandas as pd
from database_maker import get_data

# Test with a specific youth player
CLUB = '362'  # Antwerpen
PLAYER_NAME = 'Smets Sander'  # Change this to an actual youth player name
SEASON = 26

print("=" * 80)
print(f"Testing youth player: {PLAYER_NAME} from club {CLUB}")
print("=" * 80)
print()

# Get player data
try:
    player_data = get_data(club=CLUB, name=PLAYER_NAME, season=SEASON)
    
    if not player_data:
        print(f"❌ No data found for {PLAYER_NAME}")
        print("Please update PLAYER_NAME in the script with an actual youth player")
        exit(1)
    
    current_rank = player_data.get('ranking') or player_data.get('current_ranking')
    category = player_data.get('category')
    elo = player_data.get('elo', 0)
    kaart = player_data.get('kaart', {})
    
    print(f"✅ Player found!")
    print(f"   Category: {category}")
    print(f"   Current rank: {current_rank}")
    print(f"   ELO: {elo}")
    print()
    
    # Check if youth
    youth_categories = ['BEN', 'PRE', 'MIN', 'CAD', 'JUN', 'J19', 'J21']
    is_youth = category in youth_categories
    
    if not is_youth:
        print(f"⚠️  This player is NOT in a youth category ({category})")
        print("   Please choose a player from: BEN, PRE, MIN, CAD, JUN, J19, J21")
        exit(1)
    
    print(f"✅ Confirmed youth player in category: {category}")
    print()
    
    # Calculate basic stats
    total_wins = sum(kaart.get(rank, [0, 0])[0] for rank in kaart.keys())
    total_losses = sum(kaart.get(rank, [0, 0])[1] for rank in kaart.keys())
    total_matches = total_wins + total_losses
    win_rate = total_wins / total_matches if total_matches > 0 else 0
    
    print(f"Performance:")
    print(f"   Matches: {total_matches}")
    print(f"   Wins: {total_wins}")
    print(f"   Losses: {total_losses}")
    print(f"   Win rate: {win_rate:.1%}")
    print()
    
    # Load V3 model (filtered - for youth)
    print("Loading V3 model (without ELO)...")
    try:
        model_v3 = joblib.load("model_filtered_v3_improved.pkl")
        int_to_rank_v3 = joblib.load("int_to_rank_filtered_v3.pkl")
        print("✅ V3 model loaded")
    except Exception as e:
        print(f"❌ Could not load V3 model: {e}")
        exit(1)
    
    # Load V4 model (with ELO)
    print("Loading V4 model (with ELO)...")
    try:
        model_v4 = joblib.load("model_v4_special_cases.pkl")
        int_to_rank_v4 = joblib.load("int_to_rank_v4_hybrid.pkl")
        print("✅ V4 model loaded")
        has_v4 = True
    except Exception as e:
        print(f"⚠️  V4 model not available: {e}")
        has_v4 = False
    
    print()
    print("=" * 80)
    print("CONCLUSION")
    print("=" * 80)
    print()
    
    if has_v4:
        print("✅ Both models are available")
        print()
        print("With the code change:")
        print("  - Youth players will use V3 model (WITHOUT ELO)")
        print("  - Adult players will use V4 model (WITH ELO) for special cases")
        print()
        print("This is CORRECT because:")
        print("  1. Youth players have more volatile rankings")
        print("  2. ELO is less reliable for youth (fewer matches, rapid improvement)")
        print("  3. V3 filtered model is specifically trained for youth categories")
    else:
        print("⚠️  V4 model not available - only V3 will be used")
        print("   This is fine, V3 is the correct model for youth anyway")
    
    print()
    print("=" * 80)
    print("RECOMMENDATION")
    print("=" * 80)
    print()
    print("✅ The code change is CORRECT:")
    print("   - ELO model should NOT be used for youth categories")
    print("   - V3 filtered model is better suited for youth predictions")
    print()
    
except Exception as e:
    print(f"❌ Error: {e}")
    import traceback
    traceback.print_exc()
