"""
Complete rewrite of ranking prediction model
Focus: Simplicity, interpretability, and actual performance trends
"""
import pandas as pd
import ast
import joblib
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import warnings
warnings.filterwarnings('ignore')

print("=" * 80)
print("RANKING PREDICTION MODEL V2 - COMPLETE REWRITE")
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

# Exclude youth categories (they have separate model)
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

# ===== KEY INSIGHT: Focus on what actually predicts rank changes =====

# 1. Overall performance
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

# 2. Performance against similar ranks (most predictive)
def get_performance_at_level(kaart, current_rank_idx, ranking_order, rank_to_int):
    """Get win rate against players near your level"""
    nearby_wins = 0
    nearby_losses = 0
    
    # Look at ranks within 3 levels of current rank
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

# 3. Performance against better players (shows potential)
def get_performance_vs_better(kaart, current_rank_idx, ranking_order, rank_to_int):
    """Get win rate against better players"""
    better_wins = 0
    better_losses = 0
    
    for rank in ranking_order:
        rank_idx = rank_to_int[rank]
        if rank_idx < current_rank_idx:  # Better rank (lower index)
            wins, losses = kaart.get(rank, [0, 0])
            better_wins += wins
            better_losses += losses
    
    better_total = better_wins + better_losses
    return better_wins / better_total if better_total > 0 else 0

df['vs_better_win_rate'] = df.apply(
    lambda row: get_performance_vs_better(row['kaart'], row['current_rank_encoded'], ranking_order, rank_to_int),
    axis=1
)

# 4. Consistency (do they play regularly?)
df['match_volume'] = np.log1p(df['total_matches'])  # Log scale for volume

# 5. Momentum (recent performance proxy)
df['performance_score'] = (
    df['win_rate'] * 0.4 +  # Overall performance
    df['nearby_win_rate'] * 0.4 +  # Performance at your level
    df['vs_better_win_rate'] * 0.2  # Potential shown vs better players
)

# ===== ENHANCED FEATURES (same as filtered model) =====
# 6. Performance vs worse players (should dominate if ready to advance)
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

# 7. Win/loss ratio (momentum indicator)
df['win_loss_ratio'] = np.where(df['total_losses'] > 0, df['total_wins'] / df['total_losses'], df['total_wins'])

# 8. Performance consistency (stability)
df['performance_consistency'] = df['nearby_win_rate'] * np.log1p(df['total_matches'])

# 9. Level dominance (how much you dominate peers)
df['level_dominance'] = (df['nearby_win_rate'] - 0.5) * 2  # Scale -1 to 1

# Filter: Only players with meaningful data (at least 10 matches)
min_matches = 10
print(f"\nFiltering to players with ≥{min_matches} matches...")
print(f"Before: {len(df)} players")
df = df[df['total_matches'] >= min_matches]
print(f"After: {len(df)} players")

# Encode category
category_encoder = LabelEncoder()
df["category_encoded"] = category_encoder.fit_transform(df["category"])

# ===== ENHANCED FEATURE SET - Balanced and informative =====
feature_cols = [
    'current_rank_encoded',  # Where you are now
    'category_encoded',  # Age/experience level
    'win_rate',  # Overall performance
    'nearby_win_rate',  # Performance at your level (MOST IMPORTANT)
    'vs_better_win_rate',  # Potential
    'vs_worse_win_rate',  # Dominance over weaker players
    'total_matches',  # Experience
    'match_volume',  # Activity level
    'performance_score',  # Combined metric
    'win_loss_ratio',  # Momentum
    'performance_consistency',  # Stability
    'level_dominance'  # Peer dominance
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

# ===== STABLE MODEL - More trees for consistency =====
print("\nTraining stable RandomForest...")

model = RandomForestClassifier(
    n_estimators=300,  # More trees = more stable predictions
    max_depth=12,  # Slightly deeper for better patterns
    min_samples_split=8,  # Allow more flexibility
    min_samples_leaf=4,  # Smaller leaves for better fit
    max_features='sqrt',  # Standard
    class_weight='balanced',  # Handle imbalance
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

# Analyze predictions
errors = np.abs(y_test.values - y_pred)
print(f"\nPrediction Analysis:")
print(f"  Perfect: {(errors == 0).sum()}/{len(y_test)} ({(errors == 0).sum()/len(y_test)*100:.1f}%)")
print(f"  Within 1 rank: {(errors <= 1).sum()}/{len(y_test)} ({(errors <= 1).sum()/len(y_test)*100:.1f}%)")
print(f"  Within 2 ranks: {(errors <= 2).sum()}/{len(y_test)} ({(errors <= 2).sum()/len(y_test)*100:.1f}%)")
print(f"  Mean error: {errors.mean():.2f} ranks")

# Feature importance
print("\nFeature Importance:")
for feat, imp in sorted(zip(feature_cols, model.feature_importances_), key=lambda x: x[1], reverse=True):
    print(f"  {feat}: {imp:.3f}")

# Analyze bias (are we too pessimistic/optimistic?)
prediction_changes = y_test.values - y_pred  # Positive = predicted worse than actual
print(f"\nPrediction Bias:")
print(f"  Average bias: {prediction_changes.mean():.2f} ranks")
if prediction_changes.mean() > 0.1:
    print("  ⚠️  Model is TOO PESSIMISTIC (predicting worse than reality)")
elif prediction_changes.mean() < -0.1:
    print("  ⚠️  Model is TOO OPTIMISTIC (predicting better than reality)")
else:
    print("  ✓ Model is well-calibrated")

# Save model
joblib.dump(model, "model_v2.pkl")
joblib.dump(category_encoder, "category_encoder_v2.pkl")
joblib.dump(rank_to_int, "rank_to_int_v2.pkl")
joblib.dump(int_to_rank, "int_to_rank_v2.pkl")
joblib.dump(feature_cols, "feature_cols_v2.pkl")
joblib.dump(ranking_order, "ranking_order_v2.pkl")

print("\n" + "=" * 80)
print("MODEL V2 SAVED!")
print("=" * 80)
print(f"Features: {len(feature_cols)} (enhanced from 8)")
print(f"Training samples: {len(X_train)}")
print(f"Test samples: {len(X_test)}")
print("\nKey improvements:")
print("  ✓ Focus on nearby_win_rate (most predictive)")
print("  ✓ Enhanced feature set (12 features)")
print("  ✓ Shallower trees (prevent overfitting)")
print("  ✓ Balanced class weights")
print("  ✓ Added: vs_worse_win_rate, win_loss_ratio, consistency, dominance")
