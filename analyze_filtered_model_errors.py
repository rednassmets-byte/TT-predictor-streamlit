"""
Analyze what the filtered model gets most wrong
"""
import pandas as pd
import ast
import joblib
import numpy as np
from sklearn.preprocessing import LabelEncoder
import warnings
warnings.filterwarnings('ignore')

print("=" * 80)
print("FILTERED MODEL ERROR ANALYSIS")
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

# INCLUDE ONLY youth categories (filtered model is for youth!)
youth_categories = ["BEN", "PRE", "MIN", "CAD"]
df = df[df['category'].isin(youth_categories)]
print(f"Filtered model is for YOUTH categories only: {youth_categories}")

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

# Feature engineering
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

def get_performance_at_level(kaart, current_rank_idx, ranking_order, rank_to_int):
    nearby_wins = 0
    nearby_losses = 0
    for rank in ranking_order:
        rank_idx = rank_to_int[rank]
        if abs(rank_idx - current_rank_idx) <= 3:
            wins, losses = kaart.get(rank, [0, 0])
            nearby_wins += wins
            nearby_losses += losses
    nearby_total = nearby_wins + nearby_losses
    return nearby_wins / nearby_total if nearby_total > 0 else 0.5

df['nearby_win_rate'] = df.apply(
    lambda row: get_performance_at_level(row['kaart'], row['current_rank_encoded'], ranking_order, rank_to_int),
    axis=1
)

def get_performance_vs_better(kaart, current_rank_idx, ranking_order, rank_to_int):
    better_wins = 0
    better_losses = 0
    for rank in ranking_order:
        rank_idx = rank_to_int[rank]
        if rank_idx < current_rank_idx:
            wins, losses = kaart.get(rank, [0, 0])
            better_wins += wins
            better_losses += losses
    better_total = better_wins + better_losses
    return better_wins / better_total if better_total > 0 else 0

df['vs_better_win_rate'] = df.apply(
    lambda row: get_performance_vs_better(row['kaart'], row['current_rank_encoded'], ranking_order, rank_to_int),
    axis=1
)

def get_performance_vs_worse(kaart, current_rank_idx, ranking_order, rank_to_int):
    worse_wins = 0
    worse_losses = 0
    for rank in ranking_order:
        rank_idx = rank_to_int[rank]
        if rank_idx > current_rank_idx:
            wins, losses = kaart.get(rank, [0, 0])
            worse_wins += wins
            worse_losses += losses
    worse_total = worse_wins + worse_losses
    return worse_wins / worse_total if worse_total > 0 else 0

df['vs_worse_win_rate'] = df.apply(
    lambda row: get_performance_vs_worse(row['kaart'], row['current_rank_encoded'], ranking_order, rank_to_int),
    axis=1
)

df['match_volume'] = np.log1p(df['total_matches'])
df['win_loss_ratio'] = np.where(df['total_losses'] > 0, df['total_wins'] / df['total_losses'], df['total_wins'])
df['performance_consistency'] = df['nearby_win_rate'] * np.log1p(df['total_matches'])
df['level_dominance'] = (df['nearby_win_rate'] - 0.5) * 2
df['performance_score'] = (
    df['win_rate'] * 0.4 +
    df['nearby_win_rate'] * 0.4 +
    df['vs_better_win_rate'] * 0.2
)

# Filter - same as training
min_matches = 15  # Filtered model uses 15 matches minimum
print(f"\nFiltering to players with ≥{min_matches} matches (filtered model threshold)...")
print(f"Before: {len(df)} players")
df = df[df['total_matches'] >= min_matches]
print(f"After: {len(df)} players")

category_encoder = LabelEncoder()
df["category_encoded"] = category_encoder.fit_transform(df["category"])

feature_cols = [
    'current_rank_encoded', 'category_encoded', 'win_rate', 'nearby_win_rate',
    'vs_better_win_rate', 'vs_worse_win_rate', 'total_matches', 'match_volume',
    'performance_score', 'win_loss_ratio', 'performance_consistency', 'level_dominance'
]

X = df[feature_cols].fillna(0)
y = df["next_rank_encoded"]

# Remove rare classes
class_counts = y.value_counts()
valid_classes = class_counts[class_counts >= 5].index
df = df[df["next_rank_encoded"].isin(valid_classes)]
X = df[feature_cols].fillna(0)
y = df["next_rank_encoded"]

# Load filtered model
try:
    model = joblib.load("model_filtered_v2.pkl")
    print(f"✓ Loaded model_filtered_v2.pkl")
except:
    try:
        model = joblib.load("model_filtered.pkl")
        print(f"✓ Loaded model_filtered.pkl")
    except:
        print("✗ Could not load filtered model")
        exit(1)

print(f"✓ Analyzing {len(df)} players\n")

# Make predictions
y_pred = model.predict(X)
df['predicted_rank_encoded'] = y_pred
df['predicted_rank'] = df['predicted_rank_encoded'].map(int_to_rank)

# Calculate errors
df['correct'] = df['next_rank_encoded'] == df['predicted_rank_encoded']
df['error_distance'] = np.abs(df['next_rank_encoded'] - df['predicted_rank_encoded'])

accuracy = df['correct'].mean()
print("=" * 80)
print(f"OVERALL ACCURACY: {accuracy:.2%}")
print("=" * 80)

# Error distribution
print(f"\nPrediction Accuracy:")
print(f"  Perfect: {(df['error_distance'] == 0).sum()}/{len(df)} ({(df['error_distance'] == 0).mean()*100:.1f}%)")
print(f"  Within 1 rank: {(df['error_distance'] <= 1).sum()}/{len(df)} ({(df['error_distance'] <= 1).mean()*100:.1f}%)")
print(f"  Within 2 ranks: {(df['error_distance'] <= 2).sum()}/{len(df)} ({(df['error_distance'] <= 2).mean()*100:.1f}%)")
print(f"  Mean error: {df['error_distance'].mean():.2f} ranks")

# ===== WHAT DOES THE FILTERED MODEL GET MOST WRONG? =====

print("\n" + "=" * 80)
print("1. ERRORS BY CURRENT RANK")
print("=" * 80)
errors_by_rank = df.groupby('current_ranking').agg({
    'correct': ['count', 'sum', 'mean'],
    'error_distance': 'mean'
}).round(3)
errors_by_rank.columns = ['Total', 'Correct', 'Accuracy', 'Avg_Error']
errors_by_rank['Error_Rate'] = 1 - errors_by_rank['Accuracy']
errors_by_rank = errors_by_rank.sort_values('Error_Rate', ascending=False)
print(errors_by_rank.head(10))
print(f"\n⚠️  WORST RANKS: {', '.join(errors_by_rank.head(3).index.tolist())}")

print("\n" + "=" * 80)
print("2. MOST COMMON PREDICTION ERRORS")
print("=" * 80)
errors_df = df[~df['correct']].copy()
error_patterns = errors_df.groupby(['current_ranking', 'next_ranking', 'predicted_rank']).size().reset_index(name='count')
error_patterns = error_patterns.sort_values('count', ascending=False).head(15)
print(f"{'Current':<10} {'Actual':<10} {'Predicted':<12} {'Count':<8}")
print("─" * 50)
for _, row in error_patterns.iterrows():
    print(f"{row['current_ranking']:<10} {row['next_ranking']:<10} {row['predicted_rank']:<12} {int(row['count']):<8}")

print("\n" + "=" * 80)
print("3. ERRORS BY PLAYER CATEGORY")
print("=" * 80)
category_errors = df.groupby('category').agg({
    'correct': ['count', 'mean'],
    'error_distance': 'mean'
}).round(3)
category_errors.columns = ['Total', 'Accuracy', 'Avg_Error']
category_errors['Error_Rate'] = 1 - category_errors['Accuracy']
category_errors = category_errors.sort_values('Error_Rate', ascending=False)
print(category_errors)
print(f"\n⚠️  WORST CATEGORIES: {', '.join(category_errors.head(3).index.tolist())}")

print("\n" + "=" * 80)
print("4. PREDICTION BIAS ANALYSIS")
print("=" * 80)
df['prediction_bias'] = df['predicted_rank_encoded'] - df['next_rank_encoded']

bias_by_rank = df.groupby('current_ranking')['prediction_bias'].mean().sort_values()
print("\nBias by Current Rank (negative = too optimistic, positive = too pessimistic):")
for rank, bias in bias_by_rank.items():
    direction = "TOO OPTIMISTIC" if bias < -0.2 else "TOO PESSIMISTIC" if bias > 0.2 else "balanced"
    print(f"  {rank}: {bias:+.2f} ({direction})")

print("\n" + "=" * 80)
print("5. WORST INDIVIDUAL PREDICTIONS")
print("=" * 80)
worst_errors = df[~df['correct']].nlargest(10, 'error_distance')
for idx, row in worst_errors.iterrows():
    print(f"\n{row['name']} ({row['season']})")
    print(f"  Current: {row['current_ranking']} → Actual: {row['next_ranking']} | Predicted: {row['predicted_rank']}")
    print(f"  Error: {int(row['error_distance'])} ranks | Category: {row['category']}")
    print(f"  Stats: {int(row['total_wins'])}W-{int(row['total_losses'])}L ({row['win_rate']:.1%} win rate)")
    print(f"  Nearby win rate: {row['nearby_win_rate']:.1%} | vs Better: {row['vs_better_win_rate']:.1%}")

print("\n" + "=" * 80)
print("6. PATTERNS IN ERRORS")
print("=" * 80)

# Do we struggle with improvements or declines?
df['actual_change'] = df['current_rank_encoded'] - df['next_rank_encoded']
improvements = df[df['actual_change'] > 0]
declines = df[df['actual_change'] < 0]
stable = df[df['actual_change'] == 0]

print(f"\nAccuracy by rank change type:")
print(f"  Improvements (moved up): {improvements['correct'].mean():.2%} ({len(improvements)} players)")
print(f"  Stable (same rank): {stable['correct'].mean():.2%} ({len(stable)} players)")
print(f"  Declines (moved down): {declines['correct'].mean():.2%} ({len(declines)} players)")

# Do we struggle with high or low activity players?
df['activity_level'] = pd.qcut(df['total_matches'], q=4, labels=['Low', 'Medium', 'High', 'Very High'], duplicates='drop')
print(f"\nAccuracy by activity level:")
for level in df['activity_level'].unique():
    subset = df[df['activity_level'] == level]
    print(f"  {level}: {subset['correct'].mean():.2%} ({len(subset)} players)")

# Do we struggle with good or bad performers?
df['performance_level'] = pd.qcut(df['win_rate'], q=4, labels=['Poor', 'Below Avg', 'Above Avg', 'Excellent'], duplicates='drop')
print(f"\nAccuracy by performance level:")
for level in df['performance_level'].unique():
    subset = df[df['performance_level'] == level]
    print(f"  {level}: {subset['correct'].mean():.2%} ({len(subset)} players)")

print("\n" + "=" * 80)
print("KEY INSIGHTS - FILTERED MODEL")
print("=" * 80)

print(f"\nNote: Filtered model only uses players with ≥{min_matches} matches")
print(f"This is {len(df)} players vs 10,753 in the regular model")

# Summary
print("\n1. RANK-SPECIFIC ISSUES:")
worst_ranks = errors_by_rank.head(3)
for rank in worst_ranks.index:
    error_rate = worst_ranks.loc[rank, 'Error_Rate']
    print(f"   • {rank}: {error_rate:.1%} error rate")

print("\n2. CATEGORY-SPECIFIC ISSUES:")
worst_cats = category_errors.head(3)
for cat in worst_cats.index:
    error_rate = worst_cats.loc[cat, 'Error_Rate']
    print(f"   • {cat}: {error_rate:.1%} error rate")

print("\n3. PREDICTION TENDENCIES:")
if improvements['correct'].mean() < stable['correct'].mean():
    print(f"   • Struggles with IMPROVEMENTS ({improvements['correct'].mean():.1%} accuracy)")
if declines['correct'].mean() < stable['correct'].mean():
    print(f"   • Struggles with DECLINES ({declines['correct'].mean():.1%} accuracy)")

print("\n4. MOST COMMON MISTAKES:")
top_3_errors = error_patterns.head(3)
for _, row in top_3_errors.iterrows():
    print(f"   • {row['current_ranking']} → {row['next_ranking']} predicted as {row['predicted_rank']} ({int(row['count'])} times)")

print("\n" + "=" * 80)
print("COMPARISON: Does filtering to ≥20 matches help?")
print("=" * 80)
print(f"\nFiltered model accuracy: {accuracy:.2%}")
print(f"Regular model accuracy (on same filtered data): TBD")
print("\nFiltered model should be MORE accurate since it only uses")
print("players with sufficient match history (more reliable data)")

print("\n" + "=" * 80)
