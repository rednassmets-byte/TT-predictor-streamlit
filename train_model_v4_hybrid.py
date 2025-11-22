"""
Model V4 - Hybrid Approach
- V3 model for normal predictions (best overall accuracy)
- ELO-enhanced model for:
  1. NG (unranked) players - ELO helps predict initial rank
  2. Players making big jumps (2+ rank improvement)
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
print("RANKING PREDICTION MODEL V4 - HYBRID (V3 + ELO for special cases)")
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

# Handle ELO
df["elo"] = pd.to_numeric(df["elo"], errors='coerce')

# Exclude youth categories
youth_categories = ["BEN", "PRE", "MIN", "CAD"]
df = df[~df['category'].isin(youth_categories)]

print(f"Total players: {len(df)}")
print(f"Players with valid ELO (>0): {(df['elo'] > 0).sum()}")

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

# Calculate rank change
df['rank_change'] = df['current_rank_encoded'] - df['next_rank_encoded']  # Positive = improvement

# Identify special cases
df['is_ng'] = (df['current_ranking'] == 'NG').astype(int)
df['big_jump'] = (df['rank_change'] >= 2).astype(int)  # Improved by 2+ ranks
df['is_special_case'] = ((df['is_ng'] == 1) | (df['big_jump'] == 1)).astype(int)

print(f"\nSpecial cases:")
print(f"  NG players: {df['is_ng'].sum()}")
print(f"  Big jumps (2+ ranks): {df['big_jump'].sum()}")
print(f"  Total special cases: {df['is_special_case'].sum()}")
print(f"  Normal cases: {(df['is_special_case'] == 0).sum()}")

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

# ELO features (only for special cases model)
df['elo_valid'] = (df['elo'] > 0).astype(int)
df['elo_log'] = np.where(df['elo'] > 0, np.log1p(df['elo']), 0)

rank_elo_map = {
    'A': 1800, 'B0': 1600, 'B2': 1500, 'B4': 1400, 'B6': 1300,
    'C0': 1200, 'C2': 1100, 'C4': 1000, 'C6': 900,
    'D0': 800, 'D2': 750, 'D4': 700, 'D6': 650,
    'E0': 600, 'E2': 550, 'E4': 500, 'E6': 450,
    'NG': 400
}

df['expected_elo'] = df['current_ranking'].map(rank_elo_map)
df['elo_advantage'] = np.where(df['elo'] > 0, (df['elo'] - df['expected_elo']) / 100, 0).clip(-5, 5)

# Encode category
category_encoder = LabelEncoder()
df['category_encoded'] = category_encoder.fit_transform(df['category'])

# ===== TRAIN TWO MODELS =====

# Model 1: Special cases (NG + big jumps) WITH ELO
print("\n" + "="*80)
print("TRAINING MODEL 1: SPECIAL CASES (NG + Big Jumps) WITH ELO")
print("="*80)

df_special = df[df['is_special_case'] == 1].copy()
# Only use players with valid ELO for special cases model
df_special = df_special[df_special['elo_valid'] == 1].copy()

print(f"Special cases with valid ELO: {len(df_special)}")

feature_cols_special = [
    'current_rank_encoded', 'category_encoded',
    'win_rate', 'nearby_win_rate', 'vs_better_win_rate', 'vs_worse_win_rate',
    'total_matches', 'match_volume', 'win_loss_ratio',
    'performance_consistency', 'level_dominance',
    'win_rate_capped', 'nearby_win_rate_capped',
    'improvement_signal', 'decline_signal',
    'is_junior', 'junior_volatility', 'in_de_zone',
    'match_reliability', 'reliable_performance',
    'is_low_rank', 'is_mid_rank',
    # ELO features for special cases
    'elo_log', 'elo_advantage'
]

X_special = df_special[feature_cols_special]
y_special = df_special['next_rank_encoded']

X_train_sp, X_test_sp, y_train_sp, y_test_sp = train_test_split(
    X_special, y_special, test_size=0.2, random_state=42  # No stratify for special cases (some classes too small)
)

model_special = RandomForestClassifier(
    n_estimators=300,
    max_depth=12,
    min_samples_split=5,
    min_samples_leaf=2,
    max_features=0.7,
    random_state=42,
    n_jobs=-1
)

model_special.fit(X_train_sp, y_train_sp)
y_pred_sp = model_special.predict(X_test_sp)
accuracy_special = accuracy_score(y_test_sp, y_pred_sp)

print(f"\nSpecial Cases Model Accuracy: {accuracy_special:.4f}")

# Model 2: Normal cases (use V3 features, no ELO)
print("\n" + "="*80)
print("TRAINING MODEL 2: NORMAL CASES (V3 features, no ELO)")
print("="*80)

df_normal = df[df['is_special_case'] == 0].copy()

print(f"Normal cases: {len(df_normal)}")

feature_cols_normal = [
    'current_rank_encoded', 'category_encoded',
    'win_rate', 'nearby_win_rate', 'vs_better_win_rate', 'vs_worse_win_rate',
    'total_matches', 'match_volume', 'win_loss_ratio',
    'performance_consistency', 'level_dominance',
    'win_rate_capped', 'nearby_win_rate_capped',
    'improvement_signal', 'decline_signal',
    'is_junior', 'junior_volatility', 'in_de_zone',
    'match_reliability', 'reliable_performance',
    'is_low_rank', 'is_mid_rank'
]

X_normal = df_normal[feature_cols_normal]
y_normal = df_normal['next_rank_encoded']

X_train_nm, X_test_nm, y_train_nm, y_test_nm = train_test_split(
    X_normal, y_normal, test_size=0.2, random_state=42, stratify=y_normal
)

model_normal = RandomForestClassifier(
    n_estimators=400,
    max_depth=15,
    min_samples_split=5,
    min_samples_leaf=2,
    max_features='sqrt',
    random_state=42,
    n_jobs=-1
)

model_normal.fit(X_train_nm, y_train_nm)
y_pred_nm = model_normal.predict(X_test_nm)
accuracy_normal = accuracy_score(y_test_nm, y_pred_nm)

print(f"\nNormal Cases Model Accuracy: {accuracy_normal:.4f}")

# ===== EVALUATE HYBRID APPROACH =====
print("\n" + "="*80)
print("HYBRID MODEL EVALUATION")
print("="*80)

# Combine predictions
all_predictions = []
all_actuals = []

# Special cases predictions
all_predictions.extend(y_pred_sp)
all_actuals.extend(y_test_sp)

# Normal cases predictions
all_predictions.extend(y_pred_nm)
all_actuals.extend(y_test_nm)

hybrid_accuracy = accuracy_score(all_actuals, all_predictions)

print(f"\nHybrid Model Overall Accuracy: {hybrid_accuracy:.4f}")
print(f"  Special cases accuracy: {accuracy_special:.4f} ({len(y_test_sp)} players)")
print(f"  Normal cases accuracy: {accuracy_normal:.4f} ({len(y_test_nm)} players)")

# Compare with V3
print("\n" + "="*80)
print("COMPARISON WITH V3 MODEL")
print("="*80)

try:
    v3_model = joblib.load("model_v3_improved.pkl")
    v3_feature_cols = joblib.load("feature_cols_v3.pkl")
    
    # Test V3 on combined test set
    X_test_combined = pd.concat([X_test_sp[v3_feature_cols], X_test_nm])
    y_test_combined = list(y_test_sp) + list(y_test_nm)
    
    y_pred_v3 = v3_model.predict(X_test_combined)
    accuracy_v3 = accuracy_score(y_test_combined, y_pred_v3)
    
    print(f"V3 Model Accuracy: {accuracy_v3:.4f}")
    print(f"V4 Hybrid Accuracy: {hybrid_accuracy:.4f}")
    print(f"Difference: {(hybrid_accuracy - accuracy_v3)*100:.2f}%")
    
except Exception as e:
    print(f"Could not compare with V3: {e}")

# Save models
joblib.dump(model_special, "model_v4_special_cases.pkl")
joblib.dump(model_normal, "model_v4_normal_cases.pkl")
joblib.dump(feature_cols_special, "feature_cols_v4_special.pkl")
joblib.dump(feature_cols_normal, "feature_cols_v4_normal.pkl")
joblib.dump(category_encoder, "category_encoder_v4_hybrid.pkl")
joblib.dump(int_to_rank, "int_to_rank_v4_hybrid.pkl")
joblib.dump(rank_to_int, "rank_to_int_v4_hybrid.pkl")
joblib.dump(ranking_order, "ranking_order_v4_hybrid.pkl")

print("\n" + "="*80)
print("MODELS SAVED")
print("="*80)
print("Special cases model: model_v4_special_cases.pkl")
print("Normal cases model: model_v4_normal_cases.pkl")
print("Feature columns: feature_cols_v4_special.pkl, feature_cols_v4_normal.pkl")
print("Encoders: category_encoder_v4_hybrid.pkl")
print("Mappings: int_to_rank_v4_hybrid.pkl, rank_to_int_v4_hybrid.pkl, ranking_order_v4_hybrid.pkl")
