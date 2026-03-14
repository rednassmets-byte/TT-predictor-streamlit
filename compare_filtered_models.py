"""
Compare Filtered V2 vs V3 models
"""
import pandas as pd
import ast
import joblib
import numpy as np
from sklearn.preprocessing import LabelEncoder
import warnings
warnings.filterwarnings('ignore')

print("=" * 80)
print("FILTERED MODEL COMPARISON: V2 vs V3")
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

# Filter to youth categories
youth_categories = ["BEN", "PRE", "MIN", "CAD"]
df = df[df['category'].isin(youth_categories)]

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

# V3 features
df['is_unranked'] = (df['current_ranking'] == 'NG').astype(int)
df['is_entry_rank'] = (df['current_ranking'] == 'E6').astype(int)
df['breakthrough_signal'] = (
    (df['win_rate'] - 0.6).clip(0, 1) * 2 +
    df['vs_better_win_rate'] * 2 +
    (df['nearby_win_rate'] - 0.7).clip(0, 1) * 1.5
)
df['activity_volatility'] = np.tanh(df['total_matches'] / 50) * df['win_rate']
df['poor_performer_signal'] = (
    (0.4 - df['win_rate']).clip(0, 1) * 2 +
    (0.3 - df['nearby_win_rate']).clip(0, 1) * 1.5
)
df['is_excellent'] = ((df['win_rate'] > 0.7) & (df['total_matches'] >= 20)).astype(int)
df['is_youngest'] = df['category'].isin(['MIN', 'PRE']).astype(int)
df['is_oldest'] = df['category'].isin(['BEN', 'CAD']).astype(int)

def get_ranked_opponent_ratio(kaart):
    total = sum(w + l for w, l in kaart.values())
    ng_matches = kaart.get('NG', [0, 0])
    ng_total = ng_matches[0] + ng_matches[1]
    ranked_matches = total - ng_total
    return ranked_matches / total if total > 0 else 0

df['ranked_opponent_ratio'] = df['kaart'].apply(get_ranked_opponent_ratio)
df['ng_win_rate'] = df.apply(
    lambda row: row['win_rate'] if row['current_ranking'] == 'NG' else 0,
    axis=1
)
df['ng_has_many_matches'] = ((df['current_ranking'] == 'NG') & (df['total_matches'] >= 20)).astype(int)
df['e6_ready_to_advance'] = (
    (df['current_ranking'] == 'E6') & 
    (df['nearby_win_rate'] > 0.6) & 
    (df['total_matches'] >= 15)
).astype(int)
df['win_rate_capped'] = df['win_rate'].clip(0.1, 0.9)
df['nearby_win_rate_capped'] = df['nearby_win_rate'].clip(0.1, 0.9)
df['youth_match_reliability'] = np.tanh(df['total_matches'] / 40)

# Filter
min_matches = 15
df = df[df['total_matches'] >= min_matches]

category_encoder = LabelEncoder()
df["category_encoded"] = category_encoder.fit_transform(df["category"])

# V2 features
feature_cols_v2 = [
    'current_rank_encoded', 'category_encoded', 'win_rate', 'nearby_win_rate',
    'vs_better_win_rate', 'vs_worse_win_rate', 'total_matches', 'match_volume',
    'performance_score', 'win_loss_ratio', 'performance_consistency', 'level_dominance'
]

# V3 features
feature_cols_v3 = [
    'current_rank_encoded', 'category_encoded', 'win_rate_capped', 'nearby_win_rate_capped',
    'vs_better_win_rate', 'vs_worse_win_rate', 'total_matches', 'match_volume',
    'win_loss_ratio', 'performance_consistency', 'level_dominance',
    'is_unranked', 'is_entry_rank', 'breakthrough_signal', 'activity_volatility',
    'poor_performer_signal', 'is_excellent', 'is_youngest', 'is_oldest',
    'ranked_opponent_ratio', 'ng_win_rate', 'ng_has_many_matches',
    'e6_ready_to_advance', 'youth_match_reliability'
]

X_v2 = df[feature_cols_v2].fillna(0)
X_v3 = df[feature_cols_v3].fillna(0)
y = df["next_rank_encoded"]

# Remove rare classes
class_counts = y.value_counts()
valid_classes = class_counts[class_counts >= 5].index
df = df[df["next_rank_encoded"].isin(valid_classes)]
X_v2 = df[feature_cols_v2].fillna(0)
X_v3 = df[feature_cols_v3].fillna(0)
y = df["next_rank_encoded"]

# Load models
model_v2 = joblib.load("model_filtered_v2.pkl")
model_v3 = joblib.load("model_filtered_v3_improved.pkl")

print(f"Comparing on {len(df)} youth players\n")

# Make predictions
y_pred_v2 = model_v2.predict(X_v2)
y_pred_v3 = model_v3.predict(X_v3)

df['pred_v2'] = y_pred_v2
df['pred_v3'] = y_pred_v3
df['correct_v2'] = y == y_pred_v2
df['correct_v3'] = y == y_pred_v3
df['actual_change'] = df['current_rank_encoded'] - df['next_rank_encoded']

print("=" * 80)
print("OVERALL ACCURACY")
print("=" * 80)
print(f"V2: {df['correct_v2'].mean():.2%}")
print(f"V3: {df['correct_v3'].mean():.2%}")
print(f"Difference: {(df['correct_v3'].mean() - df['correct_v2'].mean())*100:+.2f}%")

print("\n" + "=" * 80)
print("ACCURACY BY RANK CHANGE TYPE")
print("=" * 80)

improvements = df[df['actual_change'] > 0]
stable = df[df['actual_change'] == 0]
declines = df[df['actual_change'] < 0]

print(f"\nImprovements ({len(improvements)} players):")
print(f"  V2: {improvements['correct_v2'].mean():.2%}")
print(f"  V3: {improvements['correct_v3'].mean():.2%}")
print(f"  Difference: {(improvements['correct_v3'].mean() - improvements['correct_v2'].mean())*100:+.2f}%")

print(f"\nStable ({len(stable)} players):")
print(f"  V2: {stable['correct_v2'].mean():.2%}")
print(f"  V3: {stable['correct_v3'].mean():.2%}")
print(f"  Difference: {(stable['correct_v3'].mean() - stable['correct_v2'].mean())*100:+.2f}%")

if len(declines) > 0:
    print(f"\nDeclines ({len(declines)} players):")
    print(f"  V2: {declines['correct_v2'].mean():.2%}")
    print(f"  V3: {declines['correct_v3'].mean():.2%}")
    print(f"  Difference: {(declines['correct_v3'].mean() - declines['correct_v2'].mean())*100:+.2f}%")

print("\n" + "=" * 80)
print("ACCURACY BY PROBLEMATIC AREAS")
print("=" * 80)

ng_players = df[df['is_unranked'] == 1]
e6_players = df[df['is_entry_rank'] == 1]
excellent = df[df['is_excellent'] == 1]

print(f"\nNG (unranked) players ({len(ng_players)} players):")
print(f"  V2: {ng_players['correct_v2'].mean():.2%}")
print(f"  V3: {ng_players['correct_v3'].mean():.2%}")
print(f"  Difference: {(ng_players['correct_v3'].mean() - ng_players['correct_v2'].mean())*100:+.2f}%")

print(f"\nE6 (entry rank) players ({len(e6_players)} players):")
print(f"  V2: {e6_players['correct_v2'].mean():.2%}")
print(f"  V3: {e6_players['correct_v3'].mean():.2%}")
print(f"  Difference: {(e6_players['correct_v3'].mean() - e6_players['correct_v2'].mean())*100:+.2f}%")

if len(excellent) > 0:
    print(f"\nExcellent performers ({len(excellent)} players):")
    print(f"  V2: {excellent['correct_v2'].mean():.2%}")
    print(f"  V3: {excellent['correct_v3'].mean():.2%}")
    print(f"  Difference: {(excellent['correct_v3'].mean() - excellent['correct_v2'].mean())*100:+.2f}%")

print("\n" + "=" * 80)
print("ACCURACY BY CATEGORY")
print("=" * 80)

for cat in ['MIN', 'PRE', 'CAD', 'BEN']:
    cat_df = df[df['category'] == cat]
    if len(cat_df) > 0:
        print(f"\n{cat} ({len(cat_df)} players):")
        print(f"  V2: {cat_df['correct_v2'].mean():.2%}")
        print(f"  V3: {cat_df['correct_v3'].mean():.2%}")
        print(f"  Difference: {(cat_df['correct_v3'].mean() - cat_df['correct_v2'].mean())*100:+.2f}%")

print("\n" + "=" * 80)
print("VERDICT")
print("=" * 80)

# Weighted score focusing on problematic areas
v2_score = (
    improvements['correct_v2'].mean() * 0.3 +
    ng_players['correct_v2'].mean() * 0.25 +
    e6_players['correct_v2'].mean() * 0.25 +
    stable['correct_v2'].mean() * 0.2
)

v3_score = (
    improvements['correct_v3'].mean() * 0.3 +
    ng_players['correct_v3'].mean() * 0.25 +
    e6_players['correct_v3'].mean() * 0.25 +
    stable['correct_v3'].mean() * 0.2
)

print(f"\nWeighted Score (30% improvements, 25% NG, 25% E6, 20% stable):")
print(f"  V2: {v2_score:.2%}")
print(f"  V3: {v3_score:.2%}")
print(f"  Difference: {(v3_score - v2_score)*100:+.2f}%")

if v3_score > v2_score:
    print("\n✓ V3 IS BETTER - Use model_filtered_v3_improved.pkl")
else:
    print("\n✗ V2 IS BETTER - Stick with model_filtered_v2.pkl")
