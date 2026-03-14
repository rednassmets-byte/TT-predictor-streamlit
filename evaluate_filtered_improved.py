import pandas as pd
import ast
import joblib
import numpy as np
from sklearn.metrics import confusion_matrix

# Load data for seasons 24 and 25
df = pd.read_csv("Antwerpen_Data_15-25.csv")

# Filter for seasons 24 and 25
df = df[df['season'].isin(['2023-2024', '2024-2025'])]

# Preprocess data
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
int_to_rank = {i: r for r, i in rank_to_int.items()}

df["current_rank_encoded"] = df["current_ranking"].map(rank_to_int)
df["next_rank_encoded"] = df["next_ranking"].map(rank_to_int)

# Expand kaart
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

# Advanced feature engineering (same as training)
def create_advanced_features_fast(df, ranking_order):
    win_cols = [f"{r}_wins" for r in ranking_order]
    loss_cols = [f"{r}_losses" for r in ranking_order]
    
    wins_array = df[win_cols].values
    losses_array = df[loss_cols].values
    totals_array = wins_array + losses_array
    
    for i, rank in enumerate(ranking_order):
        total_col = f"{rank}_total"
        win_rate_col = f"{rank}_win_rate"
        
        df[total_col] = totals_array[:, i]
        df[win_rate_col] = np.divide(
            wins_array[:, i], 
            totals_array[:, i], 
            out=np.zeros_like(wins_array[:, i], dtype=float),
            where=totals_array[:, i] > 0
        )
    
    df['total_wins'] = wins_array.sum(axis=1)
    df['total_losses'] = losses_array.sum(axis=1)
    df['total_matches'] = df['total_wins'] + df['total_losses']
    
    total_matches = df['total_matches'].values
    df['overall_win_rate'] = np.divide(
        df['total_wins'], 
        total_matches, 
        out=np.zeros_like(df['total_wins'], dtype=float),
        where=total_matches > 0
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

# Add opponent quality features
higher_ranks = ['A', 'B0', 'B2', 'B4']
lower_ranks = ['D0', 'D2', 'D4', 'E0', 'E2']

df['performance_vs_higher'] = df[[f"{r}_win_rate" for r in higher_ranks if f"{r}_win_rate" in df.columns]].mean(axis=1)
df['performance_vs_lower'] = df[[f"{r}_win_rate" for r in lower_ranks if f"{r}_win_rate" in df.columns]].mean(axis=1)
df['win_loss_ratio'] = np.divide(df['total_wins'], df['total_losses'], 
                                  out=np.zeros_like(df['total_wins'], dtype=float),
                                  where=df['total_losses'] > 0)

# Average opponent rank
opponent_ranks = []
for idx, row in df.iterrows():
    total_opponents = 0
    weighted_rank = 0
    for rank in ranking_order:
        matches = row[f"{rank}_total"]
        if matches > 0:
            weighted_rank += rank_to_int[rank] * matches
            total_opponents += matches
    avg_opponent = weighted_rank / total_opponents if total_opponents > 0 else row['current_rank_encoded']
    opponent_ranks.append(avg_opponent)

df['avg_opponent_rank'] = opponent_ranks

# Toughest opponent beaten
toughest_beaten = []
for idx, row in df.iterrows():
    toughest = len(ranking_order)
    for rank in ranking_order:
        if row[f"{rank}_wins"] > 0:
            toughest = min(toughest, rank_to_int[rank])
    toughest_beaten.append(toughest)

df['toughest_opponent_beaten'] = toughest_beaten

# Win rate std
for idx, row in df.iterrows():
    rates = []
    for rank in ranking_order:
        if row[f"{rank}_total"] > 0:
            rates.append(row[f"{rank}_win_rate"])
    df.at[idx, 'win_rate_std'] = np.std(rates) if len(rates) > 1 else 0

# Recent form
df['recent_form'] = df['overall_win_rate'] * np.minimum(df['total_matches'] / 20, 1)

# Filter players with at least 15 matches
min_matches = 15
df_with_matches = df[df['total_matches'] >= min_matches].copy()
print(f"Filtering players with at least {min_matches} matches: {len(df)} → {len(df_with_matches)} players")

# Filter to youth categories only
youth_categories = ["BEN", "PRE", "MIN", "CAD"]
df_eval = df_with_matches[df_with_matches["category"].isin(youth_categories)].copy()
print(f"Filtering to youth categories: {len(df_with_matches)} → {len(df_eval)} players")

# Load improved filtered model
print("\n=== IMPROVED FILTERED MODEL EVALUATION ===")
category_encoder = joblib.load("category_encoder_filtered_improved.pkl")
feature_cols = joblib.load("feature_cols_filtered_improved.pkl")
scaler = joblib.load("scaler_filtered_improved.pkl")
model = joblib.load("model_filtered_improved.pkl")

# Encode category
df_eval["category_encoded"] = category_encoder.transform(df_eval["category"])

# Ensure all feature columns exist
for col in feature_cols:
    if col not in df_eval.columns:
        df_eval[col] = 0

X = df_eval[feature_cols].fillna(0)
y = df_eval["next_rank_encoded"]

# Apply scaling
X_scaled = scaler.transform(X)

# Predict
y_pred = model.predict(X_scaled)

# Overall accuracy
accuracy = (y_pred == y).mean() * 100
print(f"Improved Filtered Model Accuracy on Seasons 24-25 (≥{min_matches} matches): {accuracy:.2f}%")

# Breakdown by category
for cat in youth_categories:
    cat_mask = df_eval["category"] == cat
    if cat_mask.sum() > 0:
        cat_accuracy = (y_pred[cat_mask] == y[cat_mask]).mean() * 100
        print(f"  - {cat} category: {cat_accuracy:.2f}% ({cat_mask.sum()} players)")

# Accuracy excluding outliers
valid_mask = ~y.isna()
y_clean = y[valid_mask].values
y_pred_clean = y_pred[valid_mask]

rank_diffs = np.abs(y_clean - y_pred_clean)
non_outlier_mask = rank_diffs <= 3
accuracy_no_outliers = (y_pred_clean[non_outlier_mask] == y_clean[non_outlier_mask]).mean() * 100
outliers_removed = (~non_outlier_mask).sum()

print(f"\nAccuracy excluding outliers (≤3 ranks): {accuracy_no_outliers:.2f}%")
print(f"Outliers removed: {outliers_removed} ({outliers_removed/len(y_clean)*100:.2f}%)")

# Compare with original filtered model
print("\n=== COMPARISON WITH ORIGINAL FILTERED MODEL ===")
try:
    original_model = joblib.load("model_filtered.pkl")
    original_encoder = joblib.load("category_encoder_filtered.pkl")
    original_features = joblib.load("feature_cols_filtered.pkl")
    original_scaler = joblib.load("scaler_filtered.pkl")
    
    df_eval["category_encoded_orig"] = original_encoder.transform(df_eval["category"])
    
    for col in original_features:
        if col not in df_eval.columns:
            df_eval[col] = 0
    
    X_orig = df_eval[original_features].fillna(0)
    X_orig_scaled = original_scaler.transform(X_orig)
    y_pred_orig = original_model.predict(X_orig_scaled)
    
    accuracy_orig = (y_pred_orig == y).mean() * 100
    print(f"Original Filtered Model Accuracy: {accuracy_orig:.2f}%")
    print(f"Improved Filtered Model Accuracy: {accuracy:.2f}%")
    print(f"Difference: {accuracy - accuracy_orig:+.2f}%")
    
    if accuracy > accuracy_orig:
        print("✅ Improved model is BETTER!")
    elif accuracy < accuracy_orig:
        print("❌ Improved model is WORSE")
    else:
        print("➖ Same performance")
        
except Exception as e:
    print(f"Could not load original filtered model for comparison: {e}")

print("\n=== SUMMARY ===")
print(f"Total samples: {len(df_eval)}")
print(f"Improved filtered model accuracy: {accuracy:.2f}%")
print(f"Accuracy without outliers: {accuracy_no_outliers:.2f}%")
print(f"Features used: {len(feature_cols)}")
