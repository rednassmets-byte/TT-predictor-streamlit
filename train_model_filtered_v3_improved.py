"""
Filtered model V3 - Improved for youth categories (BEN/PRE/MIN/CAD)
Based on error analysis findings:
1. Better handling of E6 rank (entry-level)
2. Better NG (unranked) player predictions
3. Better detection of excellent performers (big jumps)
4. Handle high-activity player volatility
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
print("FILTERED MODEL V3 - IMPROVED FOR YOUTH CATEGORIES")
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

# Filter to youth categories only
youth_categories = ["BEN", "PRE", "MIN", "CAD"]
df = df[df['category'].isin(youth_categories)]

print(f"Training on {len(df)} youth players")

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

# ===== BASIC FEATURES =====
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

# ===== NEW FEATURES FOR YOUTH IMPROVEMENT =====

# 1. NG (unranked) player flag - special handling needed
df['is_unranked'] = (df['current_ranking'] == 'NG').astype(int)

# 2. E6 (entry rank) flag - problematic area
df['is_entry_rank'] = (df['current_ranking'] == 'E6').astype(int)

# 3. Youth breakthrough signal - detecting big jumps
# High win rate + beating better players = ready for big jump
df['breakthrough_signal'] = (
    (df['win_rate'] - 0.6).clip(0, 1) * 2 +  # Strong overall performance
    df['vs_better_win_rate'] * 2 +  # Beating better players (key for youth)
    (df['nearby_win_rate'] - 0.7).clip(0, 1) * 1.5  # Dominating peers
)

# 4. Activity volatility - high activity youth are more volatile
df['activity_volatility'] = np.tanh(df['total_matches'] / 50) * df['win_rate']

# 5. Poor performer signal - likely to stay or decline
df['poor_performer_signal'] = (
    (0.4 - df['win_rate']).clip(0, 1) * 2 +
    (0.3 - df['nearby_win_rate']).clip(0, 1) * 1.5
)

# 6. Excellent performer flag - might make big jumps
df['is_excellent'] = ((df['win_rate'] > 0.7) & (df['total_matches'] >= 20)).astype(int)

# 7. Category-specific features (younger = more volatile)
df['is_youngest'] = df['category'].isin(['MIN', 'PRE']).astype(int)
df['is_oldest'] = df['category'].isin(['BEN', 'CAD']).astype(int)

# 8. Match quality indicator - for NG players
# NG players with many matches against ranked players are more likely to get ranked
def get_ranked_opponent_ratio(kaart):
    """Ratio of matches against ranked (non-NG) players"""
    total = sum(w + l for w, l in kaart.values())
    ng_matches = kaart.get('NG', [0, 0])
    ng_total = ng_matches[0] + ng_matches[1]
    ranked_matches = total - ng_total
    return ranked_matches / total if total > 0 else 0

df['ranked_opponent_ratio'] = df['kaart'].apply(get_ranked_opponent_ratio)

# 9. NG-specific features
df['ng_win_rate'] = df.apply(
    lambda row: row['win_rate'] if row['current_ranking'] == 'NG' else 0,
    axis=1
)
df['ng_has_many_matches'] = ((df['current_ranking'] == 'NG') & (df['total_matches'] >= 20)).astype(int)

# 10. Readiness to advance from E6
df['e6_ready_to_advance'] = (
    (df['current_ranking'] == 'E6') & 
    (df['nearby_win_rate'] > 0.6) & 
    (df['total_matches'] >= 15)
).astype(int)

# 11. Capped win rates to avoid overconfidence
df['win_rate_capped'] = df['win_rate'].clip(0.1, 0.9)
df['nearby_win_rate_capped'] = df['nearby_win_rate'].clip(0.1, 0.9)

# 12. Match reliability for youth (need more matches than adults)
df['youth_match_reliability'] = np.tanh(df['total_matches'] / 40)  # Stricter than adults

# Filter: at least 15 matches
min_matches = 15
print(f"\nFiltering to players with ≥{min_matches} matches...")
print(f"Before: {len(df)} players")
df = df[df['total_matches'] >= min_matches]
print(f"After: {len(df)} players")

# Encode category
category_encoder = LabelEncoder()
df["category_encoded"] = category_encoder.fit_transform(df["category"])

# ===== ENHANCED FEATURE SET FOR YOUTH =====
feature_cols = [
    # Original features
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
    # New youth-specific features
    'is_unranked',
    'is_entry_rank',
    'breakthrough_signal',
    'activity_volatility',
    'poor_performer_signal',
    'is_excellent',
    'is_youngest',
    'is_oldest',
    'ranked_opponent_ratio',
    'ng_win_rate',
    'ng_has_many_matches',
    'e6_ready_to_advance',
    'youth_match_reliability'
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

# ===== IMPROVED MODEL FOR YOUTH =====
print("\nTraining improved RandomForest for youth...")

model = RandomForestClassifier(
    n_estimators=400,  # More trees for stability
    max_depth=15,  # Deeper to capture youth volatility patterns
    min_samples_split=6,  # More flexible
    min_samples_leaf=3,  # Smaller leaves
    max_features='sqrt',
    class_weight='balanced',
    random_state=42,
    n_jobs=-1,
    max_samples=0.8  # Bootstrap sampling
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

# Check problematic areas
ng_players = test_df[test_df['is_unranked'] == 1]
e6_players = test_df[test_df['is_entry_rank'] == 1]
excellent = test_df[test_df['is_excellent'] == 1]
improvements = test_df[test_df['actual_change'] > 0]

print(f"\nAccuracy by problematic areas:")
if len(ng_players) > 0:
    print(f"  NG (unranked): {ng_players['correct'].mean():.2%} ({len(ng_players)} players)")
if len(e6_players) > 0:
    print(f"  E6 (entry rank): {e6_players['correct'].mean():.2%} ({len(e6_players)} players)")
if len(excellent) > 0:
    print(f"  Excellent performers: {excellent['correct'].mean():.2%} ({len(excellent)} players)")
if len(improvements) > 0:
    print(f"  Improvements: {improvements['correct'].mean():.2%} ({len(improvements)} players)")

# Category breakdown
print(f"\nAccuracy by category:")
for cat in df['category'].unique():
    cat_test = test_df[test_df['category'] == cat]
    if len(cat_test) > 0:
        print(f"  {cat}: {cat_test['correct'].mean():.2%} ({len(cat_test)} players)")

# Feature importance
print("\nTop 15 Feature Importances:")
feature_importance = sorted(zip(feature_cols, model.feature_importances_), key=lambda x: x[1], reverse=True)
for feat, imp in feature_importance[:15]:
    print(f"  {feat}: {imp:.3f}")

# Save model
joblib.dump(model, "model_filtered_v3_improved.pkl")
joblib.dump(category_encoder, "category_encoder_filtered_v3.pkl")
joblib.dump(rank_to_int, "rank_to_int_filtered_v3.pkl")
joblib.dump(int_to_rank, "int_to_rank_filtered_v3.pkl")
joblib.dump(feature_cols, "feature_cols_filtered_v3.pkl")
joblib.dump(ranking_order, "ranking_order_filtered_v3.pkl")

print("\n" + "=" * 80)
print("FILTERED MODEL V3 SAVED!")
print("=" * 80)
print(f"Features: {len(feature_cols)} (up from 12)")
print(f"Training samples: {len(X_train)}")
print(f"Test samples: {len(X_test)}")
print("\nKey improvements for youth:")
print("  ✓ NG (unranked) player handling")
print("  ✓ E6 (entry rank) special features")
print("  ✓ Breakthrough signal for big jumps")
print("  ✓ Excellent performer detection")
print("  ✓ Activity volatility handling")
print("  ✓ Category-specific features (younger vs older)")
print("  ✓ Match quality indicators")
print("  ✓ Deeper trees (max_depth=15)")
print("  ✓ More trees (n_estimators=400)")
