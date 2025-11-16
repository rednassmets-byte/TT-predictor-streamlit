import pandas as pd
import ast
import joblib
import numpy as np
from sklearn.metrics import confusion_matrix

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

# Add opponent strength features (same as in train_model_filtered.py)
def add_opponent_strength_features(df, ranking_order, rank_to_int):
    """Calculate weighted performance against different rank levels"""
    for rank in ranking_order:
        rank_idx = rank_to_int[rank]
        wins = df[f"{rank}_wins"]
        losses = df[f"{rank}_losses"]
        total = wins + losses
        
        # Weight by opponent strength (lower rank number = stronger)
        opponent_weight = 1 - (rank_idx / len(ranking_order))
        df[f"{rank}_weighted_performance"] = (wins - losses) * opponent_weight * total
    
    return df

df = add_opponent_strength_features(df, ranking_order, rank_to_int)

# Add performance vs higher/lower ranks
higher_ranks = ['A', 'B0', 'B2', 'B4']
lower_ranks = ['D0', 'D2', 'D4', 'E0', 'E2']

df['performance_vs_higher'] = df[[f"{r}_win_rate" for r in higher_ranks if f"{r}_win_rate" in df.columns]].mean(axis=1)
df['performance_vs_lower'] = df[[f"{r}_win_rate" for r in lower_ranks if f"{r}_win_rate" in df.columns]].mean(axis=1)

# Win/loss ratio
df['win_loss_ratio'] = np.divide(df['total_wins'], df['total_losses'], 
                                  out=np.zeros_like(df['total_wins'], dtype=float),
                                  where=df['total_losses'] > 0)

# REGULAR MODEL EVALUATION
print("=== REGULAR MODEL EVALUATION ===")

# Filter players with at least 15 matches
min_matches = 15
df_with_matches = df[df['total_matches'] >= min_matches].copy()
print(f"Filtering players with at least {min_matches} matches: {len(df)} → {len(df_with_matches)} players")

category_encoder_regular = joblib.load("category_encoder.pkl")
feature_cols_regular = joblib.load("feature_cols.pkl")
scaler_regular = joblib.load("scaler.pkl")

# Encode category
df_with_matches["category_encoded"] = category_encoder_regular.transform(df_with_matches["category"])

# Ensure all feature columns exist in df
for col in feature_cols_regular:
    if col not in df_with_matches.columns:
        df_with_matches[col] = 0

X_regular = df_with_matches[feature_cols_regular].fillna(0)
y_regular = df_with_matches["next_rank_encoded"]

# Apply scaling
X_regular_scaled = scaler_regular.transform(X_regular)

# Load regular model
model_regular = joblib.load("model.pkl")

# Predict with regular model
y_pred_regular = model_regular.predict(X_regular_scaled)

# Overall accuracy
accuracy_regular = (y_pred_regular == y_regular).mean() * 100
print(f"Regular Model Accuracy on Seasons 24-25 (≥{min_matches} matches): {accuracy_regular:.2f}%")

# Breakdown by category type
junior_mask = df_with_matches["category"].isin(["JUN", "J19"])
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
df_filtered = df_with_matches[df_with_matches["category"].isin(allowed_categories)]

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

# CONFUSION ANALYSIS - REGULAR MODEL
print("\n=== REGULAR MODEL: MOST COMMON ERRORS ===")

# Remove NaN values before confusion matrix
valid_mask = ~y_regular.isna()
y_regular_clean = y_regular[valid_mask].values
y_pred_regular_clean = y_pred_regular[valid_mask]

# Calculate confusion matrix
cm_regular = confusion_matrix(y_regular_clean, y_pred_regular_clean)
present_classes_regular = sorted(set(y_regular_clean) | set(y_pred_regular_clean))

# Find biggest errors (off by 2+ ranks)
errors = []
for i, actual_class in enumerate(present_classes_regular):
    for j, pred_class in enumerate(present_classes_regular):
        if i != j and cm_regular[i][j] > 0:
            rank_diff = abs(actual_class - pred_class)
            if rank_diff >= 2:  # Off by 2 or more ranks
                errors.append({
                    'actual': int_to_rank[actual_class],
                    'predicted': int_to_rank[pred_class],
                    'count': cm_regular[i][j],
                    'rank_diff': rank_diff
                })

# Sort by count and rank difference
errors_df = pd.DataFrame(errors).sort_values(['rank_diff', 'count'], ascending=[False, False])
if not errors_df.empty:
    print("\nTop 10 worst predictions (off by 2+ ranks):")
    print(errors_df.head(10).to_string(index=False))
    
    # Show examples for top 3 error types
    print("\n=== EXAMPLES OF WORST PREDICTIONS ===")
    for idx, row in errors_df.head(3).iterrows():
        actual_rank = row['actual']
        pred_rank = row['predicted']
        
        # Find examples in the data
        mask = (df_with_matches['current_ranking'] == actual_rank) & (y_pred_regular == rank_to_int[pred_rank])
        examples = df_with_matches[mask].head(3)
        
        if not examples.empty:
            print(f"\n{actual_rank} → {pred_rank} (off by {int(row['rank_diff'])} ranks, {int(row['count'])} cases):")
            for i, (_, player) in enumerate(examples.iterrows(), 1):
                name = player.get('name', 'Unknown')
                category = player.get('category', 'Unknown')
                total_wins = player.get('total_wins', 0)
                total_losses = player.get('total_losses', 0)
                total_matches = total_wins + total_losses
                win_rate = (total_wins / total_matches * 100) if total_matches > 0 else 0
                
                print(f"  {i}. {name} ({category}) - {int(total_wins)}W/{int(total_losses)}L ({win_rate:.1f}% win rate)")
else:
    print("No major errors (all predictions within 1 rank)")

# CONFUSION ANALYSIS - FILTERED MODEL
if not df_filtered.empty:
    print("\n=== FILTERED MODEL: MOST COMMON ERRORS ===")
    
    # Remove NaN values
    valid_mask_filtered = ~y_filtered.isna()
    y_filtered_clean = y_filtered[valid_mask_filtered].values
    y_pred_filtered_clean = y_pred_filtered[valid_mask_filtered]
    
    cm_filtered = confusion_matrix(y_filtered_clean, y_pred_filtered_clean)
    present_classes_filtered = sorted(set(y_filtered_clean) | set(y_pred_filtered_clean))
    
    errors_filtered = []
    for i, actual_class in enumerate(present_classes_filtered):
        for j, pred_class in enumerate(present_classes_filtered):
            if i != j and cm_filtered[i][j] > 0:
                rank_diff = abs(actual_class - pred_class)
                if rank_diff >= 2:
                    errors_filtered.append({
                        'actual': int_to_rank_filtered[actual_class],
                        'predicted': int_to_rank_filtered[pred_class],
                        'count': cm_filtered[i][j],
                        'rank_diff': rank_diff
                    })
    
    errors_filtered_df = pd.DataFrame(errors_filtered).sort_values(['rank_diff', 'count'], ascending=[False, False])
    if not errors_filtered_df.empty:
        print("\nTop 10 worst predictions (off by 2+ ranks):")
        print(errors_filtered_df.head(10).to_string(index=False))
        
        # Show examples for top 3 error types
        print("\n=== EXAMPLES OF WORST PREDICTIONS (FILTERED) ===")
        for idx, row in errors_filtered_df.head(3).iterrows():
            actual_rank = row['actual']
            pred_rank = row['predicted']
            
            # Find examples in the filtered data
            mask = (df_filtered['current_ranking'] == actual_rank) & (y_pred_filtered == rank_to_int_filtered[pred_rank])
            examples = df_filtered[mask].head(3)
            
            if not examples.empty:
                print(f"\n{actual_rank} → {pred_rank} (off by {int(row['rank_diff'])} ranks, {int(row['count'])} cases):")
                for i, (_, player) in enumerate(examples.iterrows(), 1):
                    name = player.get('name', 'Unknown')
                    category = player.get('category', 'Unknown')
                    total_wins = player.get('total_wins', 0)
                    total_losses = player.get('total_losses', 0)
                    total_matches = total_wins + total_losses
                    win_rate = (total_wins / total_matches * 100) if total_matches > 0 else 0
                    
                    print(f"  {i}. {name} ({category}) - {int(total_wins)}W/{int(total_losses)}L ({win_rate:.1f}% win rate)")
    else:
        print("No major errors (all predictions within 1 rank)")

# SUMMARY
print("\n=== SUMMARY ===")
print(f"Total samples in seasons 24-25: {len(df)}")
print(f"Samples with ≥{min_matches} matches: {len(df_with_matches)}")
print(f"Samples in filtered categories: {len(df_filtered)}")
print(f"Regular model accuracy (≥{min_matches} matches): {accuracy_regular:.2f}%")
if not df_filtered.empty:
    print(f"Filtered model accuracy (≥{min_matches} matches): {accuracy_filtered:.2f}%")