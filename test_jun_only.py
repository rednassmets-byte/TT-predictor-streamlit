"""
Test JUN-only model vs Regular model on JUN players only
"""
import pandas as pd
import ast
import joblib
import numpy as np
from sklearn.metrics import accuracy_score

print("=" * 80)
print("TESTING JUN-ONLY MODEL VS REGULAR MODEL ON JUN PLAYERS")
print("=" * 80)

# Load data
df = pd.read_csv("Antwerpen_Data_15-25.csv")
df = df[df['season'].isin(['2023-2024', '2024-2025'])]

def normalize_rank(rank):
    if isinstance(rank, str) and rank.startswith('A'):
        return 'A'
    elif rank == 'B0e':
        return 'B0'
    else:
        return rank

df["current_ranking"] = df["current_ranking"].apply(ast.literal_eval).str[0].apply(normalize_rank)
df["next_ranking"] = df["next_ranking"].apply(ast.literal_eval).str[0].apply(normalize_rank)
df["category"] = df["category"].apply(ast.literal_eval).str[0]
df["kaart"] = df["kaart"].apply(ast.literal_eval)

ranking_order = [
    "A", "B0", "B2", "B4", "B6",
    "C0", "C2", "C4", "C6",
    "D0", "D2", "D4", "D6",
    "E0", "E2", "E4", "E6",
    "NG"
]

rank_to_int = {r: i for i, r in enumerate(ranking_order)}
df["current_rank_encoded"] = df["current_ranking"].map(rank_to_int)
df["next_rank_encoded"] = df["next_ranking"].map(rank_to_int)

# Extract kaart features
def extract_kaart_features_fast(kaart_list):
    all_features = []
    for kaart in kaart_list:
        features = {}
        for rank in ranking_order:
            wins, losses = kaart.get(rank, [0, 0])
            features[f"{rank}_wins"] = wins
            features[f"{rank}_losses"] = losses
        all_features.append(features)
    return pd.DataFrame(all_features)

kaart_features_df = extract_kaart_features_fast(df["kaart"].tolist())
df = pd.concat([df.reset_index(drop=True), kaart_features_df], axis=1)

# Feature engineering
def create_advanced_features_fast(df, ranking_order):
    win_cols = [f"{r}_wins" for r in ranking_order]
    loss_cols = [f"{r}_losses" for r in ranking_order]
    
    wins_array = df[win_cols].values
    losses_array = df[loss_cols].values
    totals_array = wins_array + losses_array
    
    for i, rank in enumerate(ranking_order):
        df[f"{rank}_total"] = totals_array[:, i]
        df[f"{rank}_win_rate"] = np.divide(
            wins_array[:, i], totals_array[:, i], 
            out=np.zeros_like(wins_array[:, i], dtype=float),
            where=totals_array[:, i] > 0
        )
    
    df['total_wins'] = wins_array.sum(axis=1)
    df['total_losses'] = losses_array.sum(axis=1)
    df['total_matches'] = df['total_wins'] + df['total_losses']
    df['overall_win_rate'] = np.divide(
        df['total_wins'], df['total_matches'],
        out=np.zeros_like(df['total_wins'], dtype=float),
        where=df['total_matches'] > 0
    )
    df['performance_consistency'] = df['overall_win_rate'] * np.log1p(df['total_matches'])
    df['recent_performance'] = df['overall_win_rate'] * np.minimum(df['total_matches'] / 10, 1)
    
    current_ranks = df['current_rank_encoded'].values
    df['rank_progression_potential'] = df['overall_win_rate'] * (1 - current_ranks / len(ranking_order))
    
    return df

df = create_advanced_features_fast(df, ranking_order)

# Add opponent strength features
def add_opponent_strength_features(df, ranking_order, rank_to_int):
    for rank in ranking_order:
        rank_idx = rank_to_int[rank]
        wins = df[f"{rank}_wins"]
        losses = df[f"{rank}_losses"]
        total = wins + losses
        opponent_weight = 1 - (rank_idx / len(ranking_order))
        df[f"{rank}_weighted_performance"] = (wins - losses) * opponent_weight * total
    return df

df = add_opponent_strength_features(df, ranking_order, rank_to_int)

# Add performance features
higher_ranks = ['A', 'B0', 'B2', 'B4']
lower_ranks = ['D0', 'D2', 'D4', 'E0', 'E2']
df['performance_vs_higher'] = df[[f"{r}_win_rate" for r in higher_ranks if f"{r}_win_rate" in df.columns]].mean(axis=1)
df['performance_vs_lower'] = df[[f"{r}_win_rate" for r in lower_ranks if f"{r}_win_rate" in df.columns]].mean(axis=1)
df['win_loss_ratio'] = np.divide(df['total_wins'], df['total_losses'], 
                                  out=np.zeros_like(df['total_wins'], dtype=float),
                                  where=df['total_losses'] > 0)

# Filter to JUN only with at least 15 matches
df_jun = df[(df["category"] == "JUN") & (df['total_matches'] >= 15)].copy()
df_jun = df_jun[~df_jun["next_rank_encoded"].isna()].copy()

print(f"\nTest set: {len(df_jun)} JUN players with ≥15 matches")

# Test JUN model
jun_model = joblib.load("model_jun.pkl")
jun_encoder = joblib.load("category_encoder_jun.pkl")
jun_features = joblib.load("feature_cols_jun.pkl")
jun_scaler = joblib.load("scaler_jun.pkl")

df_jun["category_encoded_jun"] = jun_encoder.transform(df_jun["category"])
for col in jun_features:
    if col not in df_jun.columns:
        df_jun[col] = 0

X_jun = df_jun[jun_features].fillna(0)
y_true = df_jun["next_rank_encoded"]
X_jun_scaled = jun_scaler.transform(X_jun)
y_pred_jun = jun_model.predict(X_jun_scaled)

jun_accuracy = accuracy_score(y_true, y_pred_jun)
errors_jun = np.abs(y_true - y_pred_jun)

# Test Regular model
regular_model = joblib.load("model.pkl")
regular_encoder = joblib.load("category_encoder.pkl")
regular_features = joblib.load("feature_cols.pkl")
regular_scaler = joblib.load("scaler.pkl")

df_jun["category_encoded_reg"] = regular_encoder.transform(df_jun["category"])
for col in regular_features:
    if col not in df_jun.columns:
        df_jun[col] = 0

X_reg = df_jun[regular_features].fillna(0)
X_reg_scaled = regular_scaler.transform(X_reg)
y_pred_reg = regular_model.predict(X_reg_scaled)

reg_accuracy = accuracy_score(y_true, y_pred_reg)
errors_reg = np.abs(y_true - y_pred_reg)

# Results
print("\n" + "=" * 80)
print("RESULTS")
print("=" * 80)

print(f"\nJUN-only Model:")
print(f"  Accuracy: {jun_accuracy:.2%}")
print(f"  Perfect predictions: {(errors_jun == 0).sum()}/{len(y_true)} ({(errors_jun == 0).sum()/len(y_true)*100:.1f}%)")
print(f"  Within 1 rank: {(errors_jun <= 1).sum()}/{len(y_true)} ({(errors_jun <= 1).sum()/len(y_true)*100:.1f}%)")
print(f"  Mean error: {errors_jun.mean():.2f} ranks")

print(f"\nRegular Model:")
print(f"  Accuracy: {reg_accuracy:.2%}")
print(f"  Perfect predictions: {(errors_reg == 0).sum()}/{len(y_true)} ({(errors_reg == 0).sum()/len(y_true)*100:.1f}%)")
print(f"  Within 1 rank: {(errors_reg <= 1).sum()}/{len(y_true)} ({(errors_reg <= 1).sum()/len(y_true)*100:.1f}%)")
print(f"  Mean error: {errors_reg.mean():.2f} ranks")

diff = jun_accuracy - reg_accuracy
print(f"\n" + "=" * 80)
print(f"DIFFERENCE: {diff:+.2%}")
if diff > 0:
    print(f"✅ JUN-only model is BETTER by {diff:.2%}")
else:
    print(f"❌ JUN-only model is WORSE by {abs(diff):.2%}")
print("=" * 80)
