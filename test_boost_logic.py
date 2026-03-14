"""
Test the boost logic with real data
"""
import pandas as pd
import ast
import joblib
import numpy as np
from sklearn.preprocessing import LabelEncoder

print("=" * 80)
print("TESTING BOOST LOGIC FOR BIG IMPROVEMENTS")
print("=" * 80)

# Load data
df = pd.read_csv("club_members_with_next_ranking.csv")

def normalize_rank(rank):
    if rank.startswith('A'):
        return 'A'
    elif rank == 'B0e':
        return 'B0'
    else:
        return rank

df["current_ranking"] = df["current_ranking"].apply(ast.literal_eval).str[0].apply(normalize_rank)
df["next_ranking"] = df["next_ranking"].apply(ast.literal_eval).str[0].apply(normalize_rank)
df["category"] = df["category"].apply(ast.literal_eval).str[0]
df["kaart"] = df["kaart"].apply(ast.literal_eval)

youth_categories = ["BEN", "PRE", "MIN", "CAD"]
df = df[~df['category'].isin(youth_categories)]

ranking_order = [
    "A", "B0", "B2", "B4", "B6",
    "C0", "C2", "C4", "C6",
    "D0", "D2", "D4", "D6",
    "E0", "E2", "E4", "E6",
    "NG"
]

rank_to_int = {r: i for i, r in enumerate(ranking_order)}
int_to_rank = {i: r for r, i in rank_to_int.items()}

df["current_rank_encoded"] = df["current_ranking"].map(rank_to_int)
df["next_rank_encoded"] = df["next_ranking"].map(rank_to_int)

# Calculate features
total_wins = []
total_losses = []
for kaart in df["kaart"]:
    wins = sum(w for w, l in kaart.values())
    losses = sum(l for w, l in kaart.values())
    total_wins.append(wins)
    total_losses.append(losses)

df['total_wins'] = total_wins
df['total_losses'] = total_losses
df['total_matches'] = df['total_wins'] + df['total_losses']
df['win_rate'] = df['total_wins'] / df['total_matches'].replace(0, 1)

# Calculate nearby win rate
def get_nearby_win_rate(kaart, current_rank_idx, rank_to_int):
    nearby_wins = 0
    nearby_losses = 0
    for rank, rank_idx in rank_to_int.items():
        if abs(rank_idx - current_rank_idx) <= 3:
            wins, losses = kaart.get(rank, [0, 0])
            nearby_wins += wins
            nearby_losses += losses
    nearby_total = nearby_wins + nearby_losses
    return nearby_wins / nearby_total if nearby_total > 0 else 0

df['nearby_win_rate'] = df.apply(
    lambda row: get_nearby_win_rate(row['kaart'], row['current_rank_encoded'], rank_to_int),
    axis=1
)

# Calculate vs better win rate
def get_vs_better_win_rate(kaart, current_rank_idx, rank_to_int):
    better_wins = 0
    better_losses = 0
    for rank, rank_idx in rank_to_int.items():
        if rank_idx < current_rank_idx:
            wins, losses = kaart.get(rank, [0, 0])
            better_wins += wins
            better_losses += losses
    better_total = better_wins + better_losses
    return better_wins / better_total if better_total > 0 else 0

df['vs_better_win_rate'] = df.apply(
    lambda row: get_vs_better_win_rate(row['kaart'], row['current_rank_encoded'], rank_to_int),
    axis=1
)

# Filter to players with 15+ matches
df = df[df['total_matches'] >= 15]

# Find players who made big improvements (2+ ranks)
df['actual_change'] = df['current_rank_encoded'] - df['next_rank_encoded']
big_improvements = df[df['actual_change'] >= 2].copy()

print(f"\nFound {len(big_improvements)} players who made big improvements (2+ ranks)")

# Check how many would qualify for boost
def would_boost(row):
    strong_overall = row['win_rate'] > 0.70
    beating_better = row['vs_better_win_rate'] > 0.35
    dominating_level = row['nearby_win_rate'] > 0.75
    enough_matches = row['total_matches'] >= 20
    signals = sum([strong_overall, beating_better, dominating_level, enough_matches])
    return signals >= 3

big_improvements['would_boost'] = big_improvements.apply(would_boost, axis=1)

print(f"Of these, {big_improvements['would_boost'].sum()} would qualify for boost")
print(f"That's {big_improvements['would_boost'].mean()*100:.1f}% of big improvement cases")

print("\n" + "=" * 80)
print("EXAMPLES OF PLAYERS WHO WOULD GET BOOSTED")
print("=" * 80)

boosted_players = big_improvements[big_improvements['would_boost']].head(10)
for idx, row in boosted_players.iterrows():
    print(f"\n{row['name']} ({row['season']})")
    print(f"  {row['current_ranking']} → {row['next_ranking']} (actual: {int(row['actual_change'])} ranks)")
    print(f"  Win rate: {row['win_rate']:.1%}")
    print(f"  Nearby win rate: {row['nearby_win_rate']:.1%}")
    print(f"  vs Better: {row['vs_better_win_rate']:.1%}")
    print(f"  Matches: {int(row['total_matches'])}")

print("\n" + "=" * 80)
print("EXAMPLES OF PLAYERS WHO WOULD NOT GET BOOSTED")
print("=" * 80)

not_boosted_players = big_improvements[~big_improvements['would_boost']].head(5)
for idx, row in not_boosted_players.iterrows():
    print(f"\n{row['name']} ({row['season']})")
    print(f"  {row['current_ranking']} → {row['next_ranking']} (actual: {int(row['actual_change'])} ranks)")
    print(f"  Win rate: {row['win_rate']:.1%}")
    print(f"  Nearby win rate: {row['nearby_win_rate']:.1%}")
    print(f"  vs Better: {row['vs_better_win_rate']:.1%}")
    print(f"  Matches: {int(row['total_matches'])}")
    print(f"  → Not enough strong signals")

print("\n" + "=" * 80)
print("SUMMARY")
print("=" * 80)

print(f"\nOut of {len(big_improvements)} players who made big improvements:")
print(f"  • {big_improvements['would_boost'].sum()} would get boosted ({big_improvements['would_boost'].mean()*100:.1f}%)")
print(f"  • {(~big_improvements['would_boost']).sum()} would NOT get boosted ({(~big_improvements['would_boost']).mean()*100:.1f}%)")

print("\nThis means:")
print("  ✓ The boost is selective (not applied to all big improvements)")
print("  ✓ Only players with strong signals get boosted")
print("  ✓ Keeps V3's conservative approach for most cases")
print("  ✓ More optimistic only for exceptional performers")

print("\n" + "=" * 80)
