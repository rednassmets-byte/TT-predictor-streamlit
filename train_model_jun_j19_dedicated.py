"""
Dedicated model for JUN and J19 categories
These categories have different dynamics than adult players:
- Higher volatility
- Faster progression
- More likely to make big jumps
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
print("JUN/J19 DEDICATED MODEL")
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

# ONLY JUN and J19
df = df[df['category'].isin(['JUN', 'J19'])]

print(f"Training on {len(df)} JUN/J19 players")

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

# Basic stats
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

# Standard features
df['win_loss_ratio'] = df['total_wins'] / df['total_losses'].replace(0, 1)
df['match_volume'] = np.log1p(df['total_matches'])
df['performance_score'] = (
    df['win_rate'] * 0.4 + 
    df['nearby_win_rate'] * 0.4 + 
    df['vs_better_win_rate'] * 0.2
)
df['performance_consistency'] = df['nearby_win_rate'] * np.log1p(df['total_matches'])
df['level_dominance'] = (df['nearby_win_rate'] - 0.5) * 2

# JUNIOR-SPECIFIC FEATURES (more aggressive)
df['win_rate_capped'] = np.clip(df['win_rate'], 0.1, 0.9)
df['nearby_win_rate_capped'] = np.clip(df['nearby_win_rate'], 0.1, 0.9)

# More aggressive improvement signal for juniors
df['improvement_signal'] = (
    np.maximum(0, df['nearby_win_rate'] - 0.55) * 3 +  # Lower threshold, higher weight
    df['vs_better_win_rate'] * 2.5 +  # Higher weight
    np.maximum(0, df['vs_worse_win_rate'] - 0.65) * 1.5  # Lower threshold
)

# More sensitive decline signal
df['decline_signal'] = (
    np.maximum(0, 0.45 - df['nearby_win_rate']) * 2.5 +  # Higher threshold
    np.maximum(0, 0.55 - df['vs_worse_win_rate']) * 2  # Higher threshold
)

# Junior volatility (always 1 for this model, but keep for consistency)
df['is_junior'] = 1
df['junior_volatility'] = df['improvement_signal']

# Rank zone features
df['in_de_zone'] = df['current_rank_encoded'].isin([9, 10, 11, 12, 13, 14, 15, 16]).astype(int)
df['match_reliability'] = np.tanh(df['total_matches'] / 25)  # Lower threshold for juniors
df['reliable_performance'] = df['nearby_win_rate_capped'] * df['match_reliability']
df['is_low_rank'] = (df['current_rank_encoded'] >= 13).astype(int)
df['is_mid_rank'] = ((df['current_rank_encoded'] >= 9) & (df['current_rank_encoded'] < 13)).astype(int)

# ADDITIONAL JUNIOR FEATURES
# 1. Breakthrough potential (high win rate + beating better players)
df['breakthrough_potential'] = (
    np.maximum(0, df['win_rate'] - 0.6) * 3 +
    df['vs_better_win_rate'] * 3 +
    np.maximum(0, df['nearby_win_rate'] - 0.7) * 2
)

# 2. Activity level (juniors who play more tend to improve faster)
df['high_activity'] = (df['total_matches'] >= 30).astype(int)
df['activity_boost'] = df['high_activity'] * df['improvement_signal']

# 3. Dominance at current level
df['dominating_level'] = (df['nearby_win_rate'] > 0.75).astype(int)
df['dominance_boost'] = df['dominating_level'] * 2

# 4. Struggling signal (for decline prediction)
df['struggling'] = ((df['nearby_win_rate'] < 0.4) & (df['total_matches'] >= 15)).astype(int)

# 5. Momentum (recent performance indicator - approximated by vs_better performance)
df['has_momentum'] = (df['vs_better_win_rate'] > 0.3).astype(int)
df['momentum_boost'] = df['has_momentum'] * df['improvement_signal']

# Encode category
category_encoder = LabelEncoder()
df['category_encoded'] = category_encoder.fit_transform(df['category'])

# Feature list
feature_cols = [
    'current_rank_encoded',
    'category_encoded',
    'win_rate',
    'nearby_win_rate',
    'vs_better_win_rate',
    'vs_worse_win_rate',
    'total_matches',
    'match_volume',
    'performance_score',
    'win_loss_ratio',
    'performance_consistency',
    'level_dominance',
    'win_rate_capped',
    'nearby_win_rate_capped',
    'improvement_signal',
    'decline_signal',
    'is_junior',
    'junior_volatility',
    'in_de_zone',
    'match_reliability',
    'reliable_performance',
    'is_low_rank',
    'is_mid_rank',
    # Junior-specific features
    'breakthrough_potential',
    'high_activity',
    'activity_boost',
    'dominating_level',
    'dominance_boost',
    'struggling',
    'has_momentum',
    'momentum_boost'
]

X = df[feature_cols]
y = df['next_rank_encoded']

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

print(f"\nTraining set: {len(X_train)} players")
print(f"Test set: {len(X_test)} players")

# Train model with adjusted parameters for juniors
model = RandomForestClassifier(
    n_estimators=200,  # More trees
    max_depth=15,  # Deeper trees for more complex patterns
    min_samples_split=5,  # Lower to capture junior volatility
    min_samples_leaf=2,  # Lower to capture rare cases
    class_weight='balanced',  # Handle imbalanced classes
    random_state=42,
    n_jobs=-1
)

print("\nTraining JUN/J19 dedicated model...")
model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print(f"\n{'='*80}")
print(f"MODEL PERFORMANCE")
print(f"{'='*80}")
print(f"Overall Accuracy: {accuracy:.2%}")

# Detailed analysis
test_df = X_test.copy()
test_df['actual'] = y_test
test_df['predicted'] = y_pred
test_df['correct'] = (test_df['actual'] == test_df['predicted']).astype(int)
test_df['current_rank_encoded'] = X_test['current_rank_encoded']
test_df['category_encoded'] = X_test['category_encoded']

# Map back to categories
category_mapping = dict(zip(category_encoder.transform(category_encoder.classes_), category_encoder.classes_))
test_df['category'] = test_df['category_encoded'].map(category_mapping)

# By category
print(f"\nAccuracy by category:")
for cat in ['JUN', 'J19']:
    cat_data = test_df[test_df['category'] == cat]
    if len(cat_data) > 0:
        print(f"  {cat}: {cat_data['correct'].mean():.2%} ({len(cat_data)} players)")

# By change type
improvements = test_df[test_df['predicted'] < test_df['current_rank_encoded']]
declines = test_df[test_df['predicted'] > test_df['current_rank_encoded']]
stable = test_df[test_df['predicted'] == test_df['current_rank_encoded']]

print(f"\nAccuracy by prediction type:")
print(f"  Improvements: {improvements['correct'].mean():.2%} ({len(improvements)} players)")
print(f"  Declines: {declines['correct'].mean():.2%} ({len(declines)} players)")
print(f"  Stable: {stable['correct'].mean():.2%} ({len(stable)} players)")

# Feature importance
feature_importance = pd.DataFrame({
    'feature': feature_cols,
    'importance': model.feature_importances_
}).sort_values('importance', ascending=False)

print(f"\nTop 10 Most Important Features:")
for idx, row in feature_importance.head(10).iterrows():
    print(f"  {row['feature']}: {row['importance']:.4f}")

# Save model
print(f"\n{'='*80}")
print("SAVING MODEL")
print(f"{'='*80}")

joblib.dump(model, "model_jun_j19.pkl")
joblib.dump(category_encoder, "category_encoder_jun_j19.pkl")
joblib.dump(feature_cols, "feature_cols_jun_j19.pkl")
joblib.dump(rank_to_int, "rank_to_int_jun_j19.pkl")
joblib.dump(int_to_rank, "int_to_rank_jun_j19.pkl")
joblib.dump(ranking_order, "ranking_order_jun_j19.pkl")

print("✓ Saved model_jun_j19.pkl")
print("✓ Saved category_encoder_jun_j19.pkl")
print("✓ Saved feature_cols_jun_j19.pkl")
print("✓ Saved rank_to_int_jun_j19.pkl")
print("✓ Saved int_to_rank_jun_j19.pkl")
print("✓ Saved ranking_order_jun_j19.pkl")

print(f"\n{'='*80}")
print("JUN/J19 DEDICATED MODEL COMPLETE")
print(f"{'='*80}")
print("\nKey improvements for juniors:")
print("  ✓ More aggressive improvement signals")
print("  ✓ Lower thresholds for detecting potential")
print("  ✓ Breakthrough potential feature")
print("  ✓ Activity-based boosting")
print("  ✓ Momentum detection")
print("  ✓ Dominance at level detection")
print("  ✓ Better decline signal sensitivity")
