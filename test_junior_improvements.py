"""
Test the junior-specific improvements
"""
import pandas as pd
import ast
import joblib
import numpy as np

print("=" * 80)
print("TESTING JUNIOR IMPROVEMENTS")
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

# Filter for non-youth
youth_categories = ["BEN", "PRE", "MIN", "CAD"]
df = df[~df['category'].isin(youth_categories)].copy()

# Load model
model = joblib.load("model_v3_improved.pkl")
category_encoder = joblib.load("category_encoder_v3.pkl")
feature_cols = joblib.load("feature_cols_v3.pkl")
rank_to_int = joblib.load("rank_to_int_v3.pkl")
int_to_rank = joblib.load("int_to_rank_v3.pkl")
ranking_order = joblib.load("ranking_order_v3.pkl")

# Prepare features (abbreviated version - just for testing)
df["current_rank_encoded"] = df["current_ranking"].map(rank_to_int)
df["next_rank_encoded"] = df["next_ranking"].map(rank_to_int)

# Calculate basic features
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

# Filter for JUN/J19
jun_j19 = df[df['category'].isin(['JUN', 'J19'])].copy()

print(f"\nAnalyzing {len(jun_j19)} JUN/J19 players")

# Count how many would benefit from each rule
rule1_candidates = 0  # JUN high performers
rule2_candidates = 0  # J19 decline predictions
rule3_candidates = 0  # Active high performers

for idx, row in jun_j19.iterrows():
    kaart = row['kaart']
    current_idx = row['current_rank_encoded']
    category = row['category']
    total_matches = row['total_matches']
    win_rate = row['win_rate']
    
    # Calculate nearby win rate
    nearby_wins = 0
    nearby_losses = 0
    for rank in ranking_order:
        rank_idx = rank_to_int[rank]
        if abs(rank_idx - current_idx) <= 3:
            wins, losses = kaart.get(rank, [0, 0])
            nearby_wins += wins
            nearby_losses += losses
    nearby_total = nearby_wins + nearby_losses
    nearby_win_rate = nearby_wins / nearby_total if nearby_total > 0 else 0
    
    # Calculate vs better win rate
    better_wins = 0
    better_losses = 0
    for rank in ranking_order:
        rank_idx = rank_to_int[rank]
        if rank_idx < current_idx:
            wins, losses = kaart.get(rank, [0, 0])
            better_wins += wins
            better_losses += losses
    better_total = better_wins + better_losses
    vs_better_win_rate = better_wins / better_total if better_total > 0 else 0
    
    # Check rules
    # RULE 1: JUN high performers
    if category == 'JUN':
        if nearby_win_rate > 0.70 and vs_better_win_rate > 0.25 and total_matches >= 20:
            # Check if actually improved
            if row['next_rank_encoded'] < row['current_rank_encoded']:
                rule1_candidates += 1
    
    # RULE 2: J19 decline predictions (would need model prediction to test properly)
    if category == 'J19':
        if row['next_rank_encoded'] > row['current_rank_encoded']:  # Actually declined
            if nearby_win_rate > 0.35:  # But not terrible
                rule2_candidates += 1
    
    # RULE 3: Active high performers
    if total_matches >= 40 and win_rate > 0.65 and nearby_win_rate > 0.65:
        if row['next_rank_encoded'] < row['current_rank_encoded']:  # Actually improved
            rule3_candidates += 1

print(f"\n{'='*80}")
print("RULE IMPACT ANALYSIS")
print(f"{'='*80}")

print(f"\nRULE 1 (JUN high performers):")
print(f"  Candidates who actually improved: {rule1_candidates}")
print(f"  These players would get boosted predictions")

print(f"\nRULE 2 (J19 decline conservatism):")
print(f"  J19 players who declined but had OK performance: {rule2_candidates}")
print(f"  These predictions would be kept at same rank")

print(f"\nRULE 3 (Active high performers):")
print(f"  Very active high performers who improved: {rule3_candidates}")
print(f"  These players would get boosted predictions")

total_helped = rule1_candidates + rule3_candidates
print(f"\n{'='*80}")
print(f"ESTIMATED IMPROVEMENT")
print(f"{'='*80}")
print(f"\nTotal players who would benefit: {total_helped}")
print(f"Percentage of JUN/J19: {(total_helped / len(jun_j19)) * 100:.1f}%")
print(f"\nExpected accuracy improvement: +{(total_helped / len(jun_j19)) * 100 * 0.5:.1f} percentage points")
print(f"(Assuming 50% of these would have been wrong without the rules)")

print(f"\n{'='*80}")
print("NEXT STEPS")
print(f"{'='*80}")
print("\n1. The rules have been added to app.py")
print("2. Test the app with JUN/J19 players")
print("3. Monitor if predictions improve")
print("4. Adjust thresholds if needed")
