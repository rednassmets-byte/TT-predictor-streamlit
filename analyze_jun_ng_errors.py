"""
Analyze errors specifically for JUN, J19, and NG players
"""
import pandas as pd
import ast
import joblib
import numpy as np

print("=" * 80)
print("JUN/J19 AND NG ERROR ANALYSIS")
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

# Filter for non-youth (where regular model is used)
youth_categories = ["BEN", "PRE", "MIN", "CAD"]
df_regular = df[~df['category'].isin(youth_categories)].copy()

print(f"\nTotal non-youth players: {len(df_regular)}")

# Load regular model
model = joblib.load("model_v3_improved.pkl")
category_encoder = joblib.load("category_encoder_v3.pkl")
feature_cols = joblib.load("feature_cols_v3.pkl")
rank_to_int = joblib.load("rank_to_int_v3.pkl")
int_to_rank = joblib.load("int_to_rank_v3.pkl")
ranking_order = joblib.load("ranking_order_v3.pkl")

# Prepare features (same as training)
df_regular["current_rank_encoded"] = df_regular["current_ranking"].map(rank_to_int)
df_regular["next_rank_encoded"] = df_regular["next_ranking"].map(rank_to_int)

# Calculate features
total_wins = []
total_losses = []
for kaart in df_regular["kaart"]:
    wins = sum(w for w, l in kaart.values())
    losses = sum(l for w, l in kaart.values())
    total_wins.append(wins)
    total_losses.append(losses)

df_regular['total_wins'] = total_wins
df_regular['total_losses'] = total_losses
df_regular['total_matches'] = df_regular['total_wins'] + df_regular['total_losses']
df_regular['win_rate'] = df_regular['total_wins'] / df_regular['total_matches'].replace(0, 1)

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

df_regular['nearby_win_rate'] = df_regular.apply(
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

df_regular['vs_better_win_rate'] = df_regular.apply(
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

df_regular['vs_worse_win_rate'] = df_regular.apply(
    lambda row: get_performance_vs_worse(row['kaart'], row['current_rank_encoded'], ranking_order, rank_to_int),
    axis=1
)

# Additional features
df_regular['win_loss_ratio'] = df_regular['total_wins'] / df_regular['total_losses'].replace(0, 1)
df_regular['match_volume'] = np.log1p(df_regular['total_matches'])
df_regular['performance_score'] = (
    df_regular['win_rate'] * 0.4 + 
    df_regular['nearby_win_rate'] * 0.4 + 
    df_regular['vs_better_win_rate'] * 0.2
)
df_regular['performance_consistency'] = df_regular['nearby_win_rate'] * np.log1p(df_regular['total_matches'])
df_regular['level_dominance'] = (df_regular['nearby_win_rate'] - 0.5) * 2

# V3 features
df_regular['win_rate_capped'] = np.clip(df_regular['win_rate'], 0.1, 0.9)
df_regular['nearby_win_rate_capped'] = np.clip(df_regular['nearby_win_rate'], 0.1, 0.9)

df_regular['improvement_signal'] = (
    np.maximum(0, df_regular['nearby_win_rate'] - 0.6) * 2 +
    df_regular['vs_better_win_rate'] * 1.5 +
    np.maximum(0, df_regular['vs_worse_win_rate'] - 0.7)
)

df_regular['decline_signal'] = (
    np.maximum(0, 0.4 - df_regular['nearby_win_rate']) * 2 +
    np.maximum(0, 0.5 - df_regular['vs_worse_win_rate']) * 1.5
)

df_regular['is_junior'] = df_regular['category'].isin(['JUN', 'J19', 'J21']).astype(int)
df_regular['junior_volatility'] = df_regular['is_junior'] * df_regular['improvement_signal']
df_regular['in_de_zone'] = df_regular['current_rank_encoded'].isin([9, 10, 11, 12, 13, 14, 15, 16]).astype(int)
df_regular['match_reliability'] = np.tanh(df_regular['total_matches'] / 30)
df_regular['reliable_performance'] = df_regular['nearby_win_rate_capped'] * df_regular['match_reliability']
df_regular['is_low_rank'] = (df_regular['current_rank_encoded'] >= 13).astype(int)
df_regular['is_mid_rank'] = ((df_regular['current_rank_encoded'] >= 9) & (df_regular['current_rank_encoded'] < 13)).astype(int)

# Encode category
df_regular['category_encoded'] = category_encoder.transform(df_regular['category'])

# Make predictions
X = df_regular[feature_cols]
predictions = model.predict(X)
df_regular['predicted_rank_encoded'] = predictions
df_regular['predicted_rank'] = df_regular['predicted_rank_encoded'].map(int_to_rank)
df_regular['correct'] = (df_regular['predicted_rank_encoded'] == df_regular['next_rank_encoded']).astype(int)

# Analyze JUN/J19
print("\n" + "=" * 80)
print("JUN/J19 ANALYSIS")
print("=" * 80)

jun_j19 = df_regular[df_regular['category'].isin(['JUN', 'J19'])]
print(f"\nTotal JUN/J19 players: {len(jun_j19)}")
print(f"Overall accuracy: {jun_j19['correct'].mean():.2%}")

# By category
for cat in ['JUN', 'J19']:
    cat_data = jun_j19[jun_j19['category'] == cat]
    if len(cat_data) > 0:
        print(f"\n{cat}:")
        print(f"  Players: {len(cat_data)}")
        print(f"  Accuracy: {cat_data['correct'].mean():.2%}")
        
        # By change type
        improvements = cat_data[cat_data['next_rank_encoded'] < cat_data['current_rank_encoded']]
        declines = cat_data[cat_data['next_rank_encoded'] > cat_data['current_rank_encoded']]
        stable = cat_data[cat_data['next_rank_encoded'] == cat_data['current_rank_encoded']]
        
        print(f"  Improvements: {improvements['correct'].mean():.2%} ({len(improvements)} players)")
        print(f"  Declines: {declines['correct'].mean():.2%} ({len(declines)} players)")
        print(f"  Stable: {stable['correct'].mean():.2%} ({len(stable)} players)")

# Analyze NG players
print("\n" + "=" * 80)
print("NG (UNRANKED) ANALYSIS")
print("=" * 80)

ng_players = df_regular[df_regular['current_ranking'] == 'NG']
print(f"\nTotal NG players: {len(ng_players)}")
if len(ng_players) > 0:
    print(f"Overall accuracy: {ng_players['correct'].mean():.2%}")
    
    # What ranks do they move to?
    print("\nActual next ranks for NG players:")
    print(ng_players['next_ranking'].value_counts().sort_index())
    
    print("\nPredicted next ranks for NG players:")
    print(ng_players['predicted_rank'].value_counts().sort_index())
    
    # Errors
    ng_errors = ng_players[ng_players['correct'] == 0]
    print(f"\nNG prediction errors: {len(ng_errors)}")
    if len(ng_errors) > 0:
        print("\nMost common errors:")
        error_summary = ng_errors.groupby(['predicted_rank', 'next_ranking']).size().sort_values(ascending=False).head(10)
        for (pred, actual), count in error_summary.items():
            print(f"  Predicted {pred}, Actually {actual}: {count} times")

# Compare JUN/J19 vs other categories
print("\n" + "=" * 80)
print("COMPARISON: JUN/J19 vs OTHER CATEGORIES")
print("=" * 80)

other_cats = df_regular[~df_regular['category'].isin(['JUN', 'J19'])]
print(f"\nJUN/J19 accuracy: {jun_j19['correct'].mean():.2%} ({len(jun_j19)} players)")
print(f"Other categories accuracy: {other_cats['correct'].mean():.2%} ({len(other_cats)} players)")
print(f"Difference: {(jun_j19['correct'].mean() - other_cats['correct'].mean()) * 100:.2f} percentage points")

print("\n" + "=" * 80)
print("RECOMMENDATIONS")
print("=" * 80)

print("\n1. JUN/J19 ISSUES:")
if jun_j19['correct'].mean() < other_cats['correct'].mean():
    print("   ✗ JUN/J19 accuracy is LOWER than other categories")
    print("   → Consider training a separate model for JUN/J19")
    print("   → Add more junior-specific features")
    print("   → Increase junior_volatility weight")
else:
    print("   ✓ JUN/J19 accuracy is comparable to other categories")

print("\n2. NG (UNRANKED) ISSUES:")
if len(ng_players) > 0 and ng_players['correct'].mean() < 0.7:
    print("   ✗ NG predictions are challenging")
    print("   → Consider rule-based logic for NG players")
    print("   → Use win_rate and total_matches to predict first rank")
    print("   → NG with >70% win rate → E4 or better")
    print("   → NG with 50-70% win rate → E6")
    print("   → NG with <50% win rate → stay NG")
else:
    print("   ✓ NG predictions are reasonable")

print("\n" + "=" * 80)
