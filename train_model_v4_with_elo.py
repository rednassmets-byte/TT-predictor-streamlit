"""
Model V4 - With ELO Rating
Key improvements over V3:
1. Includes ELO rating as a feature
2. ELO-based performance metrics
3. Better prediction accuracy with ELO data
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
print("RANKING PREDICTION MODEL V4 - WITH ELO")
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

# Handle ELO - convert to numeric, filter out bad data
df["elo"] = pd.to_numeric(df["elo"], errors='coerce')

# Filter out players with missing or zero ELO (bad data quality)
print(f"Total players before ELO filter: {len(df)}")
print(f"Players with ELO = 0 or missing: {df['elo'].isna().sum() + (df['elo'] == 0).sum()}")

# Only keep players with valid ELO (> 0)
df = df[df['elo'] > 0].copy()

print(f"Total players after ELO filter: {len(df)}")
print(f"ELO stats: min={df['elo'].min()}, max={df['elo'].max()}, median={df['elo'].median()}, mean={df['elo'].mean():.1f}")

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

df['match_volume'] = np.log1p(df['total_matches'])
df['win_loss_ratio'] = np.where(df['total_losses'] > 0, df['total_wins'] / df['total_losses'], df['total_wins'])
df['performance_consistency'] = df['nearby_win_rate'] * np.log1p(df['total_matches'])
df['level_dominance'] = (df['nearby_win_rate'] - 0.5) * 2

# V3 features
df['win_rate_capped'] = df['win_rate'].clip(0.1, 0.9)
df['nearby_win_rate_capped'] = df['nearby_win_rate'].clip(0.1, 0.9)

df['improvement_signal'] = (
    (df['nearby_win_rate'] - 0.6).clip(0, 1) * 2 +
    df['vs_better_win_rate'] * 1.5 +
    (df['vs_worse_win_rate'] - 0.7).clip(0, 1)
)

df['decline_signal'] = (
    (0.4 - df['nearby_win_rate']).clip(0, 1) * 2 +
    (0.5 - df['vs_worse_win_rate']).clip(0, 1) * 1.5
)

df['is_junior'] = df['category'].isin(['JUN', 'J19', 'J21']).astype(int)
df['junior_volatility'] = df['is_junior'] * df['improvement_signal']
df['in_de_zone'] = df['current_rank_encoded'].isin([9, 10, 11, 12, 13, 14, 15, 16]).astype(int)
df['match_reliability'] = np.tanh(df['total_matches'] / 30)
df['reliable_performance'] = df['nearby_win_rate_capped'] * df['match_reliability']
df['is_low_rank'] = (df['current_rank_encoded'] >= 13).astype(int)
df['is_mid_rank'] = ((df['current_rank_encoded'] >= 9) & (df['current_rank_encoded'] < 13)).astype(int)

# ===== NEW V4 ELO FEATURES (REDUCED TO PREVENT OVERFITTING) =====
print("\nAdding ELO-based features...")

# Use log scale for ELO to reduce its dominance
df['elo_log'] = np.log1p(df['elo'])

# ELO vs rank consistency - does ELO match the current rank?
# Expected ELO ranges per rank (approximate, based on actual data)
rank_elo_map = {
    'A': 1800, 'B0': 1600, 'B2': 1500, 'B4': 1400, 'B6': 1300,
    'C0': 1200, 'C2': 1100, 'C4': 1000, 'C6': 900,
    'D0': 800, 'D2': 750, 'D4': 700, 'D6': 650,
    'E0': 600, 'E2': 550, 'E4': 500, 'E6': 450,
    'NG': 400
}

df['expected_elo'] = df['current_ranking'].map(rank_elo_map)
df['elo_rank_diff'] = df['elo'] - df['expected_elo']

# Simplified ELO features - only use difference from expected
# Positive = ELO higher than rank suggests (potential for improvement)
# Negative = ELO lower than rank suggests (potential for decline)
df['elo_advantage'] = (df['elo_rank_diff'] / 100).clip(-5, 5)  # Normalize and cap

print(f"Players with ELO above expected rank: {(df['elo_rank_diff'] > 50).sum()}")
print(f"Players with ELO below expected rank: {(df['elo_rank_diff'] < -50).sum()}")

# Encode category
category_encoder = LabelEncoder()
df['category_encoded'] = category_encoder.fit_transform(df['category'])

# Select features - REDUCED ELO features to prevent overfitting
feature_cols = [
    'current_rank_encoded', 'category_encoded',
    'win_rate', 'nearby_win_rate', 'vs_better_win_rate', 'vs_worse_win_rate',
    'total_matches', 'match_volume', 'win_loss_ratio',
    'performance_consistency', 'level_dominance',
    'win_rate_capped', 'nearby_win_rate_capped',
    'improvement_signal', 'decline_signal',
    'is_junior', 'junior_volatility', 'in_de_zone',
    'match_reliability', 'reliable_performance',
    'is_low_rank', 'is_mid_rank',
    # V4 ELO features (REDUCED - only 2 features instead of 8)
    'elo_log',  # Log-scaled ELO (less dominant)
    'elo_advantage'  # Difference from expected ELO for rank
]

X = df[feature_cols]
y = df['next_rank_encoded']

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

print(f"\nTraining set: {len(X_train)} players")
print(f"Test set: {len(X_test)} players")

# Train model with reduced ELO influence
print("\nTraining RandomForest with reduced ELO features...")

# Use same hyperparameters as V3 for fair comparison
model = RandomForestClassifier(
    n_estimators=400,
    max_depth=15,
    min_samples_split=5,
    min_samples_leaf=2,
    max_features=0.7,  # Use 70% of features per tree to reduce ELO dominance
    random_state=42,
    n_jobs=-1
)

model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print(f"\n{'='*80}")
print(f"MODEL V4 WITH ELO - ACCURACY: {accuracy:.4f}")
print(f"{'='*80}")

# Feature importance
feature_importance = pd.DataFrame({
    'feature': feature_cols,
    'importance': model.feature_importances_
}).sort_values('importance', ascending=False)

print("\nTop 15 Most Important Features:")
print(feature_importance.head(15).to_string(index=False))

# Save model and encoders
joblib.dump(model, "model_v4_with_elo.pkl")
joblib.dump(category_encoder, "category_encoder_v4.pkl")
joblib.dump(feature_cols, "feature_cols_v4.pkl")
joblib.dump(int_to_rank, "int_to_rank_v4.pkl")
joblib.dump(rank_to_int, "rank_to_int_v4.pkl")
joblib.dump(ranking_order, "ranking_order_v4.pkl")

print("\nModel V4 with ELO saved successfully!")
print("Files: model_v4_with_elo.pkl, category_encoder_v4.pkl, feature_cols_v4.pkl")
print("       int_to_rank_v4.pkl, rank_to_int_v4.pkl, ranking_order_v4.pkl")

# Compare with V3
print("\n" + "="*80)
print("COMPARISON WITH V3 MODEL")
print("="*80)

try:
    v3_model = joblib.load("model_v3_improved.pkl")
    v3_feature_cols = joblib.load("feature_cols_v3.pkl")
    
    # Test V3 on same test set (without ELO features)
    X_test_v3 = X_test[v3_feature_cols]
    y_pred_v3 = v3_model.predict(X_test_v3)
    accuracy_v3 = accuracy_score(y_test, y_pred_v3)
    
    print(f"V3 Model Accuracy: {accuracy_v3:.4f}")
    print(f"V4 Model Accuracy: {accuracy:.4f}")
    print(f"Improvement: {(accuracy - accuracy_v3)*100:.2f}%")
    
except Exception as e:
    print(f"Could not compare with V3: {e}")

print("\n" + "="*80)
print("TRAINING COMPLETE")
print("="*80)
