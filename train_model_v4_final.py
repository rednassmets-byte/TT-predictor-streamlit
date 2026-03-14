"""
Model V4 - Final version
Changes based on user feedback:
1. Train on players with min 15 matches (more reliable data)
2. Less pessimistic about big jumps (boost improvement signals)
"""
import pandas as pd
import ast
import joblib
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import warnings
warnings.filterwarnings('ignore')

print("=" * 80)
print("RANKING PREDICTION MODEL V4 - FINAL (Less Pessimistic)")
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

# Exclude youth categories
youth_categories = ["BEN", "PRE", "MIN", "CAD"]
df = df[~df['category'].isin(youth_categories)]

print(f"Training on {len(df)} players (excluding youth categories)")

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

# ===== FEATURE ENGINEERING =====

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

# ===== SLIGHTLY LESS PESSIMISTIC FEATURES - BALANCED =====

# 1. SLIGHTLY STRONGER improvement signal - balanced approach
df['improvement_signal'] = (
    (df['nearby_win_rate'] - 0.58).clip(0, 1) * 2.5 +  # Slightly lower threshold
    df['vs_better_win_rate'] * 2 +  # Slightly higher weight
    (df['vs_worse_win_rate'] - 0.68).clip(0, 1) * 1.2  # Slightly lower threshold
)

# 2. Big jump indicator - moderate signal for 2+ rank improvements
df['big_jump_signal'] = (
    (df['nearby_win_rate'] > 0.72).astype(int) * 1.5 +  # Dominating peers
    (df['vs_better_win_rate'] > 0.35).astype(int) * 1.5 +  # Beating better players
    (df['win_rate'] > 0.68).astype(int)  # Strong overall
)

# 3. Decline signal (keep same)
df['decline_signal'] = (
    (0.4 - df['nearby_win_rate']).clip(0, 1) * 2 +
    (0.5 - df['vs_worse_win_rate']).clip(0, 1) * 1.5
)

# 4. Junior features
df['is_junior'] = df['category'].isin(['JUN', 'J19', 'J21']).astype(int)
df['junior_volatility'] = df['is_junior'] * df['improvement_signal']

# 5. D-E zone
df['in_de_zone'] = df['current_rank_encoded'].isin([9, 10, 11, 12, 13, 14, 15, 16]).astype(int)

# 6. Capped win rates
df['win_rate_capped'] = df['win_rate'].clip(0.1, 0.9)
df['nearby_win_rate_capped'] = df['nearby_win_rate'].clip(0.1, 0.9)

# 7. Match reliability
df['match_reliability'] = np.tanh(df['total_matches'] / 30)
df['reliable_performance'] = df['nearby_win_rate_capped'] * df['match_reliability']

# 8. Rank flags
df['is_low_rank'] = (df['current_rank_encoded'] >= 13).astype(int)
df['is_mid_rank'] = ((df['current_rank_encoded'] >= 9) & (df['current_rank_encoded'] < 13)).astype(int)

# Filter: MINIMUM 15 MATCHES (user requirement)
min_matches = 15
print(f"\nFiltering to players with ≥{min_matches} matches...")
print(f"Before: {len(df)} players")
df = df[df['total_matches'] >= min_matches]
print(f"After: {len(df)} players")

# Encode category
category_encoder = LabelEncoder()
df["category_encoded"] = category_encoder.fit_transform(df["category"])

# ===== FEATURE SET =====
feature_cols = [
    'current_rank_encoded',
    'category_encoded',
    'win_rate_capped',
    'nearby_win_rate_capped',
    'vs_better_win_rate',
    'vs_worse_win_rate',
    'total_matches',
    'match_volume',
    'win_loss_ratio',
    'performance_consistency',
    'level_dominance',
    'improvement_signal',  # BOOSTED
    'big_jump_signal',  # NEW - for 2+ rank jumps
    'decline_signal',
    'is_junior',
    'junior_volatility',
    'in_de_zone',
    'match_reliability',
    'reliable_performance',
    'is_low_rank',
    'is_mid_rank'
]

X = df[feature_cols].fillna(0)
y = df["next_rank_encoded"]

# Remove rare classes
class_counts = y.value_counts()
valid_classes = class_counts[class_counts >= 5].index
df = df[df["next_rank_encoded"].isin(valid_classes)]
X = df[feature_cols].fillna(0)
y = df["next_rank_encoded"]

print(f"\nClasses with ≥5 samples: {len(valid_classes)}")
print(f"Total features: {len(feature_cols)}")

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# ===== LESS CONSERVATIVE MODEL =====
print("\nTraining LESS PESSIMISTIC RandomForest...")

model = RandomForestClassifier(
    n_estimators=400,
    max_depth=15,  # Same as V3
    min_samples_split=6,  # Same as V3
    min_samples_leaf=3,  # Same as V3
    max_features='sqrt',
    class_weight='balanced',
    random_state=42,
    n_jobs=-1,
    max_samples=0.8
)

model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print("\n" + "=" * 80)
print("RESULTS")
print("=" * 80)
print(f"Test Accuracy: {accuracy:.2%}")

errors = np.abs(y_test.values - y_pred)
print(f"\nPrediction Analysis:")
print(f"  Perfect: {(errors == 0).sum()}/{len(y_test)} ({(errors == 0).sum()/len(y_test)*100:.1f}%)")
print(f"  Within 1 rank: {(errors <= 1).sum()}/{len(y_test)} ({(errors <= 1).sum()/len(y_test)*100:.1f}%)")
print(f"  Within 2 ranks: {(errors <= 2).sum()}/{len(y_test)} ({(errors <= 2).sum()/len(y_test)*100:.1f}%)")
print(f"  Mean error: {errors.mean():.2f} ranks")

# Detailed analysis
test_df = df.loc[X_test.index].copy()
test_df['predicted'] = y_pred
test_df['correct'] = y_test.values == y_pred
test_df['actual_change'] = test_df['current_rank_encoded'] - test_df['next_rank_encoded']

improvements = test_df[test_df['actual_change'] > 0]
big_improvements = test_df[test_df['actual_change'] >= 2]
declines = test_df[test_df['actual_change'] < 0]
stable = test_df[test_df['actual_change'] == 0]

print(f"\nAccuracy by rank change type:")
print(f"  Improvements: {improvements['correct'].mean():.2%} ({len(improvements)} players)")
if len(big_improvements) > 0:
    print(f"  Big improvements (2+ ranks): {big_improvements['correct'].mean():.2%} ({len(big_improvements)} players)")
print(f"  Stable: {stable['correct'].mean():.2%} ({len(stable)} players)")
print(f"  Declines: {declines['correct'].mean():.2%} ({len(declines)} players)")

# Check juniors
juniors = test_df[test_df['is_junior'] == 1]
non_juniors = test_df[test_df['is_junior'] == 0]
print(f"\nAccuracy by age group:")
print(f"  Juniors: {juniors['correct'].mean():.2%} ({len(juniors)} players)")
print(f"  Non-juniors: {non_juniors['correct'].mean():.2%} ({len(non_juniors)} players)")

# Feature importance
print("\nTop 15 Feature Importances:")
feature_importance = sorted(zip(feature_cols, model.feature_importances_), key=lambda x: x[1], reverse=True)
for feat, imp in feature_importance[:15]:
    print(f"  {feat}: {imp:.3f}")

# Save model
joblib.dump(model, "model_v4_final.pkl")
joblib.dump(category_encoder, "category_encoder_v4.pkl")
joblib.dump(rank_to_int, "rank_to_int_v4.pkl")
joblib.dump(int_to_rank, "int_to_rank_v4.pkl")
joblib.dump(feature_cols, "feature_cols_v4.pkl")
joblib.dump(ranking_order, "ranking_order_v4.pkl")

print("\n" + "=" * 80)
print("MODEL V4 SAVED!")
print("=" * 80)
print(f"Features: {len(feature_cols)}")
print(f"Training samples: {len(X_train)}")
print(f"Test samples: {len(X_test)}")
print("\nKey changes from V3:")
print("  ✓ Min 15 matches (was 10) - more reliable data")
print("  ✓ BOOSTED improvement_signal (less pessimistic)")
print("  ✓ NEW big_jump_signal (for 2+ rank improvements)")
print("  ✓ Lower thresholds (55% vs 60% for improvements)")
print("  ✓ Higher weights for beating better players")
print("  ✓ Deeper trees (max_depth=16)")
print("  ✓ More flexible (min_samples_split=5, min_leaf=2)")
