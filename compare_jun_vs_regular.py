"""
Compare JUN/J19 specialized model vs Regular model on JUN/J19 players
"""
import pandas as pd
import ast
import joblib
import numpy as np
from sklearn.metrics import accuracy_score, classification_report

print("=" * 80)
print("COMPARING JUN/J19 MODEL VS REGULAR MODEL ON JUN/J19 PLAYERS")
print("=" * 80)

# Load data for seasons 24 and 25
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
int_to_rank = {i: r for r, i in rank_to_int.items()}

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

# Add performance features
higher_ranks = ['A', 'B0', 'B2', 'B4']
lower_ranks = ['D0', 'D2', 'D4', 'E0', 'E2']

df['performance_vs_higher'] = df[[f"{r}_win_rate" for r in higher_ranks if f"{r}_win_rate" in df.columns]].mean(axis=1)
df['performance_vs_lower'] = df[[f"{r}_win_rate" for r in lower_ranks if f"{r}_win_rate" in df.columns]].mean(axis=1)
df['win_loss_ratio'] = np.divide(df['total_wins'], df['total_losses'], 
                                  out=np.zeros_like(df['total_wins'], dtype=float),
                                  where=df['total_losses'] > 0)

# Filter to players with at least 15 matches
min_matches = 15
df = df[df['total_matches'] >= min_matches]

# Filter to JUN/J19 categories only
jun_categories = ["JUN", "J19"]
df_jun = df[df["category"].isin(jun_categories)].copy()

# Remove rows with NaN in next_rank_encoded
df_jun = df_jun[~df_jun["next_rank_encoded"].isna()].copy()

print(f"\nTest set: {len(df_jun)} JUN/J19 players with ≥{min_matches} matches")
print(f"  JUN: {len(df_jun[df_jun['category'] == 'JUN'])} players")
print(f"  J19: {len(df_jun[df_jun['category'] == 'J19'])} players")

# ===== TEST 1: JUN/J19 SPECIALIZED MODEL =====
print("\n" + "=" * 80)
print("TEST 1: JUN/J19 SPECIALIZED MODEL")
print("=" * 80)

try:
    jun_model = joblib.load("model_jun.pkl")
    jun_encoder = joblib.load("category_encoder_jun.pkl")
    jun_features = joblib.load("feature_cols_jun.pkl")
    jun_scaler = joblib.load("scaler_jun.pkl")
    
    # Encode category
    df_jun["category_encoded_jun"] = jun_encoder.transform(df_jun["category"])
    
    # Ensure all features exist
    for col in jun_features:
        if col not in df_jun.columns:
            df_jun[col] = 0
    
    X_jun = df_jun[jun_features].fillna(0)
    y_true = df_jun["next_rank_encoded"]
    
    # Scale and predict
    X_jun_scaled = jun_scaler.transform(X_jun)
    y_pred_jun = jun_model.predict(X_jun_scaled)
    
    # Calculate accuracy
    jun_accuracy = accuracy_score(y_true, y_pred_jun)
    
    # Calculate errors
    errors_jun = np.abs(y_true - y_pred_jun)
    perfect_jun = (errors_jun == 0).sum()
    within_1_jun = (errors_jun <= 1).sum()
    within_2_jun = (errors_jun <= 2).sum()
    outliers_jun = (errors_jun > 3).sum()
    
    print(f"✓ JUN/J19 Model loaded successfully")
    print(f"\nAccuracy: {jun_accuracy:.2%}")
    print(f"Perfect predictions: {perfect_jun}/{len(y_true)} ({perfect_jun/len(y_true)*100:.1f}%)")
    print(f"Within 1 rank: {within_1_jun}/{len(y_true)} ({within_1_jun/len(y_true)*100:.1f}%)")
    print(f"Within 2 ranks: {within_2_jun}/{len(y_true)} ({within_2_jun/len(y_true)*100:.1f}%)")
    print(f"Outliers (>3 ranks): {outliers_jun}/{len(y_true)} ({outliers_jun/len(y_true)*100:.1f}%)")
    print(f"Mean absolute error: {errors_jun.mean():.2f} ranks")
    
    jun_success = True
except Exception as e:
    print(f"✗ Error loading JUN/J19 model: {e}")
    jun_success = False

# ===== TEST 2: REGULAR MODEL =====
print("\n" + "=" * 80)
print("TEST 2: REGULAR MODEL (for comparison)")
print("=" * 80)

try:
    regular_model = joblib.load("model.pkl")
    regular_encoder = joblib.load("category_encoder.pkl")
    regular_features = joblib.load("feature_cols.pkl")
    regular_scaler = joblib.load("scaler.pkl")
    
    # Encode category
    df_jun["category_encoded_reg"] = regular_encoder.transform(df_jun["category"])
    
    # Ensure all features exist
    for col in regular_features:
        if col not in df_jun.columns:
            df_jun[col] = 0
    
    X_reg = df_jun[regular_features].fillna(0)
    
    # Scale and predict
    X_reg_scaled = regular_scaler.transform(X_reg)
    y_pred_reg = regular_model.predict(X_reg_scaled)
    
    # Calculate accuracy
    reg_accuracy = accuracy_score(y_true, y_pred_reg)
    
    # Calculate errors
    errors_reg = np.abs(y_true - y_pred_reg)
    perfect_reg = (errors_reg == 0).sum()
    within_1_reg = (errors_reg <= 1).sum()
    within_2_reg = (errors_reg <= 2).sum()
    outliers_reg = (errors_reg > 3).sum()
    
    print(f"✓ Regular Model loaded successfully")
    print(f"\nAccuracy: {reg_accuracy:.2%}")
    print(f"Perfect predictions: {perfect_reg}/{len(y_true)} ({perfect_reg/len(y_true)*100:.1f}%)")
    print(f"Within 1 rank: {within_1_reg}/{len(y_true)} ({within_1_reg/len(y_true)*100:.1f}%)")
    print(f"Within 2 ranks: {within_2_reg}/{len(y_true)} ({within_2_reg/len(y_true)*100:.1f}%)")
    print(f"Outliers (>3 ranks): {outliers_reg}/{len(y_true)} ({outliers_reg/len(y_true)*100:.1f}%)")
    print(f"Mean absolute error: {errors_reg.mean():.2f} ranks")
    
    reg_success = True
except Exception as e:
    print(f"✗ Error loading Regular model: {e}")
    reg_success = False

# ===== COMPARISON =====
if jun_success and reg_success:
    print("\n" + "=" * 80)
    print("COMPARISON SUMMARY")
    print("=" * 80)
    
    diff = jun_accuracy - reg_accuracy
    
    print(f"\nAccuracy:")
    print(f"  JUN/J19 Model: {jun_accuracy:.2%}")
    print(f"  Regular Model: {reg_accuracy:.2%}")
    print(f"  Difference: {diff:+.2%}")
    
    print(f"\nPerfect Predictions:")
    print(f"  JUN/J19 Model: {perfect_jun} ({perfect_jun/len(y_true)*100:.1f}%)")
    print(f"  Regular Model: {perfect_reg} ({perfect_reg/len(y_true)*100:.1f}%)")
    print(f"  Difference: {perfect_jun - perfect_reg:+d}")
    
    print(f"\nOutliers (>3 ranks):")
    print(f"  JUN/J19 Model: {outliers_jun} ({outliers_jun/len(y_true)*100:.1f}%)")
    print(f"  Regular Model: {outliers_reg} ({outliers_reg/len(y_true)*100:.1f}%)")
    print(f"  Difference: {outliers_jun - outliers_reg:+d}")
    
    print(f"\nMean Absolute Error:")
    print(f"  JUN/J19 Model: {errors_jun.mean():.2f} ranks")
    print(f"  Regular Model: {errors_reg.mean():.2f} ranks")
    print(f"  Difference: {errors_jun.mean() - errors_reg.mean():+.2f}")
    
    print("\n" + "=" * 80)
    print("VERDICT")
    print("=" * 80)
    
    if diff > 0.02:  # More than 2% better
        print(f"✅ JUN/J19 specialized model is BETTER by {diff:.2%}")
        print("   Recommendation: Keep using JUN/J19 model for JUN/J19 players")
    elif diff < -0.02:  # More than 2% worse
        print(f"❌ JUN/J19 specialized model is WORSE by {abs(diff):.2%}")
        print("   Recommendation: Use Regular model for JUN/J19 players instead")
    else:
        print(f"➖ Models perform similarly (difference: {diff:+.2%})")
        print("   Recommendation: Either model is fine, specialized model adds complexity")
    
    # Breakdown by category
    print("\n" + "=" * 80)
    print("BREAKDOWN BY CATEGORY")
    print("=" * 80)
    
    for cat in ["JUN", "J19"]:
        cat_mask = df_jun["category"] == cat
        if cat_mask.sum() > 0:
            jun_cat_acc = accuracy_score(y_true[cat_mask], y_pred_jun[cat_mask])
            reg_cat_acc = accuracy_score(y_true[cat_mask], y_pred_reg[cat_mask])
            
            print(f"\n{cat} ({cat_mask.sum()} players):")
            print(f"  JUN/J19 Model: {jun_cat_acc:.2%}")
            print(f"  Regular Model: {reg_cat_acc:.2%}")
            print(f"  Difference: {jun_cat_acc - reg_cat_acc:+.2%}")

print("\n" + "=" * 80)
