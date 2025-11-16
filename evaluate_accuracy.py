import pandas as pd
import ast
import joblib
import numpy as np

# Load data for seasons 24 and 25
df = pd.read_csv("Antwerpen_Data_15-25.csv")

# Filter for seasons 24 and 25
df = df[df['season'].isin(['2023-2024', '2024-2025'])]

# Preprocess data as in train_model.py
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
    "A",
    "B0", "B2", "B4", "B6",
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
def extract_kaart_features(kaart):
    f = {}
    for rank in ranking_order:
        wins, losses = kaart.get(rank, [0, 0])
        f[f"{rank}_wins"] = wins
        f[f"{rank}_losses"] = losses
    return f

df = pd.concat([df, df["kaart"].apply(extract_kaart_features).apply(pd.Series)], axis=1)

# Advanced feature engineering (same as in train_model_filtered.py)
def create_advanced_features(df, ranking_order):
    # Win rates per rank
    for rank in ranking_order:
        wins_col = f"{rank}_wins"
        losses_col = f"{rank}_losses"
        total_col = f"{rank}_total"
        win_rate_col = f"{rank}_win_rate"
        
        df[total_col] = df[wins_col] + df[losses_col]
        df[win_rate_col] = np.where(
            df[total_col] > 0, 
            df[wins_col] / df[total_col], 
            0
        )
    
    # Overall statistics
    win_cols = [f"{r}_wins" for r in ranking_order]
    loss_cols = [f"{r}_losses" for r in ranking_order]
    
    df['total_wins'] = df[win_cols].sum(axis=1)
    df['total_losses'] = df[loss_cols].sum(axis=1)
    df['total_matches'] = df['total_wins'] + df['total_losses']
    df['overall_win_rate'] = np.where(
        df['total_matches'] > 0,
        df['total_wins'] / df['total_matches'],
        0
    )
    
    # Performance metrics
    df['performance_consistency'] = df['overall_win_rate'] * np.log1p(df['total_matches'])
    
    # Recent performance
    df['recent_performance'] = df['overall_win_rate'] * np.minimum(df['total_matches'] / 10, 1)
    
    # Rank progression features
    current_ranks = df['current_rank_encoded'].values
    df['rank_progression_potential'] = df['overall_win_rate'] * (1 - current_ranks / len(ranking_order))
    
    return df

df = create_advanced_features(df, ranking_order)

# REGULAR MODEL EVALUATION
print("=== REGULAR MODEL EVALUATION ===")
category_encoder_regular = joblib.load("category_encoder.pkl")
feature_cols_regular = joblib.load("feature_cols.pkl")
scaler_regular = joblib.load("scaler.pkl")

# Encode category
df["category_encoded"] = category_encoder_regular.transform(df["category"])

# Ensure all feature columns exist in df
for col in feature_cols_regular:
    if col not in df.columns:
        df[col] = 0

X_regular = df[feature_cols_regular].fillna(0)
y_regular = df["next_rank_encoded"]

# Apply scaling
X_regular_scaled = scaler_regular.transform(X_regular)

# Load regular model
model_regular = joblib.load("model.pkl")

# Predict with regular model
y_pred_regular = model_regular.predict(X_regular_scaled)

# Overall accuracy
accuracy_regular = (y_pred_regular == y_regular).mean() * 100
print(f"Regular Model Accuracy on Seasons 24-25: {accuracy_regular:.2f}%")

# Breakdown by category type
junior_mask = df["category"].isin(["JUN", "J19"])
other_mask = ~junior_mask

if junior_mask.sum() > 0:
    junior_accuracy = (y_pred_regular[junior_mask] == y_regular[junior_mask]).mean() * 100
    print(f"  - JUN/J19 categories: {junior_accuracy:.2f}% ({junior_mask.sum()} players)")

if other_mask.sum() > 0:
    other_accuracy = (y_pred_regular[other_mask] == y_regular[other_mask]).mean() * 100
    print(f"  - Other categories: {other_accuracy:.2f}% ({other_mask.sum()} players)")

# FILTERED MODEL EVALUATION
print("\n=== FILTERED MODEL EVALUATION ===")
allowed_categories = ["BEN", "PRE", "MIN", "CAD"]
df_filtered = df[df["category"].isin(allowed_categories)]

if not df_filtered.empty:
    # Load filtered model assets
    category_encoder_filtered = joblib.load("category_encoder_filtered.pkl")
    feature_cols_filtered = joblib.load("feature_cols_filtered.pkl")
    scaler_filtered = joblib.load("scaler_filtered.pkl")
    rank_to_int_filtered = joblib.load("rank_to_int_filtered.pkl")
    int_to_rank_filtered = joblib.load("int_to_rank_filtered.pkl")
    
    # Apply filtered category encoding
    df_filtered = df_filtered.copy()
    df_filtered["category_encoded"] = category_encoder_filtered.transform(df_filtered["category"])
    
    # Ensure all filtered feature columns exist
    for col in feature_cols_filtered:
        if col not in df_filtered.columns:
            df_filtered[col] = 0
    
    X_filtered = df_filtered[feature_cols_filtered].fillna(0)
    y_filtered = df_filtered["next_rank_encoded"]
    
    # Apply scaling (CRITICAL!)
    X_filtered_scaled = scaler_filtered.transform(X_filtered)
    
    # Load filtered model
    model_filtered = joblib.load("model_filtered.pkl")
    
    # Predict with filtered model
    y_pred_filtered = model_filtered.predict(X_filtered_scaled)
    
    # Accuracy for filtered model
    accuracy_filtered = (y_pred_filtered == y_filtered).mean() * 100
    print(f"Filtered Model Accuracy on Seasons 24-25 (BEN/PRE/MIN/CAD): {accuracy_filtered:.2f}%")
    
    # Additional stats
    print(f"Number of samples in filtered evaluation: {len(df_filtered)}")
    print(f"Categories in filtered evaluation: {df_filtered['category'].unique()}")
    
    # Show prediction distribution
    print(f"\nActual class distribution:")
    actual_dist = pd.Series(y_filtered).value_counts().sort_index()
    for idx, count in actual_dist.items():
        print(f"  {int_to_rank_filtered.get(idx, idx)}: {count}")
    
    print(f"\nPredicted class distribution:")
    pred_dist = pd.Series(y_pred_filtered).value_counts().sort_index()
    for idx, count in pred_dist.items():
        print(f"  {int_to_rank_filtered.get(idx, idx)}: {count}")
else:
    print("No data for filtered categories in seasons 24-25.")

# SUMMARY
print("\n=== SUMMARY ===")
print(f"Total samples in seasons 24-25: {len(df)}")
print(f"Samples in filtered categories: {len(df_filtered)}")
print(f"Regular model accuracy: {accuracy_regular:.2f}%")
if not df_filtered.empty:
    print(f"Filtered model accuracy: {accuracy_filtered:.2f}%")