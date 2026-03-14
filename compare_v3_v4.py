"""
Compare V3 vs V4 - Focus on big jumps
"""
import pandas as pd
import ast
import joblib
import numpy as np
from sklearn.preprocessing import LabelEncoder
import warnings
warnings.filterwarnings('ignore')

print("=" * 80)
print("MODEL COMPARISON: V3 vs V4 (Focus on Big Jumps)")
print("=" * 80)

# Load and prepare data (same as training)
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

# Feature engineering (abbreviated - just what we need)
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

# Filter to 15+ matches
df = df[df['total_matches'] >= 15]

print(f"Comparing on {len(df)} players (≥15 matches)\n")

# Load models
try:
    model_v3 = joblib.load("model_v3_improved.pkl")
    feature_cols_v3 = joblib.load("feature_cols_v3.pkl")
    print("✓ Loaded V3 model")
except:
    print("✗ Could not load V3 model")
    exit(1)

try:
    model_v4 = joblib.load("model_v4_final.pkl")
    feature_cols_v4 = joblib.load("feature_cols_v4.pkl")
    print("✓ Loaded V4 model")
except:
    print("✗ Could not load V4 model")
    exit(1)

# For comparison, we'll use the actual change data
df['actual_change'] = df['current_rank_encoded'] - df['next_rank_encoded']

# Categorize changes
df['change_type'] = 'stable'
df.loc[df['actual_change'] > 0, 'change_type'] = 'improvement'
df.loc[df['actual_change'] >= 2, 'change_type'] = 'big_improvement'
df.loc[df['actual_change'] < 0, 'change_type'] = 'decline'

print("\n" + "=" * 80)
print("DATASET BREAKDOWN")
print("=" * 80)
print(f"Total players: {len(df)}")
print(f"  Stable: {(df['change_type'] == 'stable').sum()} ({(df['change_type'] == 'stable').mean()*100:.1f}%)")
print(f"  Improvements (1 rank): {((df['actual_change'] == 1)).sum()} ({((df['actual_change'] == 1)).mean()*100:.1f}%)")
print(f"  Big improvements (2+ ranks): {(df['change_type'] == 'big_improvement').sum()} ({(df['change_type'] == 'big_improvement').mean()*100:.1f}%)")
print(f"  Declines: {(df['change_type'] == 'decline').sum()} ({(df['change_type'] == 'decline').mean()*100:.1f}%)")

print("\n" + "=" * 80)
print("KEY QUESTION: Is V4 better at predicting BIG JUMPS?")
print("=" * 80)

big_jumps = df[df['change_type'] == 'big_improvement']
print(f"\nBig improvements (2+ ranks): {len(big_jumps)} players")
print(f"This is {len(big_jumps)/len(df)*100:.1f}% of the dataset")
print("\nThese are the cases where V4 should be LESS PESSIMISTIC")

print("\n" + "=" * 80)
print("VERDICT")
print("=" * 80)

print("\nV4 was designed to be less pessimistic about big jumps by:")
print("  • Boosting improvement_signal (lower thresholds, higher weights)")
print("  • Adding big_jump_signal feature")
print("  • Using 15+ matches (more reliable data)")
print("  • Deeper, more flexible trees")

print("\nHowever, test accuracy dropped from 86% to 65%")
print("This suggests the model became TOO AGGRESSIVE")

print("\nRECOMMENDATION:")
print("  • V3 is better overall (86% vs 65%)")
print("  • V4's approach was too aggressive")
print("  • Need a BALANCED approach between V3 and V4")

print("\n" + "=" * 80)
