"""
Filtered model V2 for youth categories (BEN/PRE/MIN/CAD)
Same simplified approach as regular V2
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
print("FILTERED MODEL V2 - YOUTH CATEGORIES (BEN/PRE/MIN/CAD)")
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

# Calculate simple features
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

# Performance at your level
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

# Performance vs better players
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

df['match_volume'] = np.log1p(df['total_matches'])
df['performance_score'] = (
    df['win_rate'] * 0.4 +
    df['nearby_win_rate'] * 0.4 +
    df['vs_better_win_rate'] * 0.2
)

# Add more predictive features
# 1. Performance vs worse players (should be high if ready to advance)
def get_performance_vs_worse(kaart, current_rank_idx, ranking_order, rank_to_int):
    worse_wins = 0
    worse_losses = 0
    for rank in ranking_order:
        rank_idx = rank_to_int[rank]
        if rank_idx > current_rank_idx:  # Worse rank (higher index)
            wins, losses = kaart.get(rank, [0, 0])
            worse_wins += wins
            worse_losses += losses
    worse_total = worse_wins + worse_losses
    return worse_wins / worse_total if worse_total > 0 else 0

df['vs_worse_win_rate'] = df.apply(
    lambda row: get_performance_vs_worse(row['kaart'], row['current_rank_encoded'], ranking_order, rank_to_int),
    axis=1
)

# 2. Win streak indicator (recent momentum)
df['win_loss_ratio'] = np.where(df['total_losses'] > 0, df['total_wins'] / df['total_losses'], df['total_wins'])

# 3. Consistency score (how stable is performance)
df['performance_consistency'] = df['nearby_win_rate'] * np.log1p(df['total_matches'])

# 4. Dominance at current level (beating peers consistently)
df['level_dominance'] = (df['nearby_win_rate'] - 0.5) * 2  # Scale -1 to 1

# Filter: at least 15 matches for more stable predictions
min_matches = 15
print(f"\nFiltering to players with ≥{min_matches} matches...")
print(f"Before: {len(df)} players")
df = df[df['total_matches'] >= min_matches]
print(f"After: {len(df)} players")

# Encode category
category_encoder = LabelEncoder()
df["category_encoded"] = category_encoder.fit_transform(df["category"])

# Enhanced feature set (12 features - balanced between simple and informative)
feature_cols = [
    'current_rank_encoded',
    'category_encoded',
    'win_rate',
    'nearby_win_rate',
    'vs_better_win_rate',
    'vs_worse_win_rate',  # NEW: Should dominate worse players
    'total_matches',
    'match_volume',
    'performance_score',
    'win_loss_ratio',  # NEW: Momentum indicator
    'performance_consistency',  # NEW: Stability
    'level_dominance'  # NEW: Dominance at current level
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

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Train model with more trees for stability
print("\nTraining stable RandomForest...")

model = RandomForestClassifier(
    n_estimators=300,  # More trees = more stable predictions
    max_depth=12,  # Slightly deeper for youth patterns
    min_samples_split=8,  # Allow more flexibility
    min_samples_leaf=4,  # Smaller leaves for better fit
    max_features='sqrt',
    class_weight='balanced',
    random_state=42,
    n_jobs=-1
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
print(f"  Mean error: {errors.mean():.2f} ranks")

# Check bias
prediction_changes = y_test.values - y_pred
print(f"\nPrediction Bias: {prediction_changes.mean():.2f} ranks")
if prediction_changes.mean() > 0.1:
    print("  ⚠️  Model is TOO PESSIMISTIC")
elif prediction_changes.mean() < -0.1:
    print("  ⚠️  Model is TOO OPTIMISTIC")
else:
    print("  ✓ Model is well-calibrated")

# Save model
joblib.dump(model, "model_filtered_v2.pkl")
joblib.dump(category_encoder, "category_encoder_filtered_v2.pkl")
joblib.dump(rank_to_int, "rank_to_int_filtered_v2.pkl")
joblib.dump(int_to_rank, "int_to_rank_filtered_v2.pkl")
joblib.dump(feature_cols, "feature_cols_filtered_v2.pkl")
joblib.dump(ranking_order, "ranking_order_filtered_v2.pkl")

print("\n" + "=" * 80)
print("FILTERED MODEL V2 SAVED!")
print("=" * 80)
print(f"Features: {len(feature_cols)} (enhanced from 8)")
print(f"Training samples: {len(X_train)}")
print(f"Test samples: {len(X_test)}")
print("\nNew features added:")
print("  ✓ vs_worse_win_rate (dominance over weaker players)")
print("  ✓ win_loss_ratio (momentum)")
print("  ✓ performance_consistency (stability)")
print("  ✓ level_dominance (peer dominance)")
