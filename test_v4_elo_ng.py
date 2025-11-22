"""
Test V4 ELO model specifically on NG (unranked) players
Compare with V3 model performance
"""
import pandas as pd
import ast
import joblib
import numpy as np
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

print("=" * 80)
print("TESTING V4 ELO MODEL ON NG (UNRANKED) PLAYERS")
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
df["elo"] = pd.to_numeric(df["elo"], errors='coerce')

# Exclude youth categories
youth_categories = ["BEN", "PRE", "MIN", "CAD"]
df = df[~df['category'].isin(youth_categories)]

# Filter for NG players only
df_ng = df[df['current_ranking'] == 'NG'].copy()
print(f"\nTotal NG players: {len(df_ng)}")

# Filter for NG players with valid ELO
df_ng_with_elo = df_ng[df_ng['elo'] > 0].copy()
print(f"NG players with valid ELO (>0): {len(df_ng_with_elo)}")
print(f"NG players without ELO: {len(df_ng) - len(df_ng_with_elo)}")

if len(df_ng_with_elo) == 0:
    print("\nNo NG players with valid ELO found!")
    exit()

# Load V3 model
print("\n" + "="*80)
print("LOADING MODELS")
print("="*80)

try:
    v3_model = joblib.load("model_v3_improved.pkl")
    v3_feature_cols = joblib.load("feature_cols_v3.pkl")
    v3_rank_to_int = joblib.load("rank_to_int_v3.pkl")
    v3_int_to_rank = joblib.load("int_to_rank_v3.pkl")
    print("[OK] V3 model loaded")
except Exception as e:
    print(f"[FAIL] V3 model failed to load: {e}")
    v3_model = None

try:
    v4_model = joblib.load("model_v4_with_elo.pkl")
    v4_feature_cols = joblib.load("feature_cols_v4.pkl")
    v4_rank_to_int = joblib.load("rank_to_int_v4.pkl")
    v4_int_to_rank = joblib.load("int_to_rank_v4.pkl")
    print("[OK] V4 ELO model loaded")
except Exception as e:
    print(f"[FAIL] V4 ELO model failed to load: {e}")
    v4_model = None

if v3_model is None or v4_model is None:
    print("\nCannot proceed without both models!")
    exit()

# Prepare features manually
from sklearn.preprocessing import LabelEncoder

ranking_order = list(v3_rank_to_int.keys())

def prepare_features_v3(df_subset):
    """Prepare V3 features"""
    features_list = []
    
    for idx, row in df_subset.iterrows():
        kaart = row['kaart']
        current_rank = row['current_ranking']
        category = row['category']
        
        current_rank_encoded = v3_rank_to_int.get(current_rank, -1)
        
        # Encode category
        category_encoder = LabelEncoder()
        category_encoder.fit(df['category'].unique())
        category_encoded = category_encoder.transform([category])[0]
        
        # Calculate stats
        total_wins = sum(kaart.get(rank, [0, 0])[0] for rank in ranking_order)
        total_losses = sum(kaart.get(rank, [0, 0])[1] for rank in ranking_order)
        total_matches = total_wins + total_losses
        win_rate = total_wins / total_matches if total_matches > 0 else 0.5
        
        # Nearby win rate
        nearby_wins = 0
        nearby_losses = 0
        for rank in ranking_order:
            rank_idx = v3_rank_to_int[rank]
            if abs(rank_idx - current_rank_encoded) <= 3:
                wins, losses = kaart.get(rank, [0, 0])
                nearby_wins += wins
                nearby_losses += losses
        nearby_total = nearby_wins + nearby_losses
        nearby_win_rate = nearby_wins / nearby_total if nearby_total > 0 else 0.5
        
        # Vs better
        better_wins = 0
        better_losses = 0
        for rank in ranking_order:
            rank_idx = v3_rank_to_int[rank]
            if rank_idx < current_rank_encoded:
                wins, losses = kaart.get(rank, [0, 0])
                better_wins += wins
                better_losses += losses
        better_total = better_wins + better_losses
        vs_better_win_rate = better_wins / better_total if better_total > 0 else 0
        
        # Vs worse
        worse_wins = 0
        worse_losses = 0
        for rank in ranking_order:
            rank_idx = v3_rank_to_int[rank]
            if rank_idx > current_rank_encoded:
                wins, losses = kaart.get(rank, [0, 0])
                worse_wins += wins
                worse_losses += losses
        worse_total = worse_wins + worse_losses
        vs_worse_win_rate = worse_wins / worse_total if worse_total > 0 else 0
        
        match_volume = np.log1p(total_matches)
        win_loss_ratio = total_wins / total_losses if total_losses > 0 else total_wins
        performance_consistency = nearby_win_rate * np.log1p(total_matches)
        level_dominance = (nearby_win_rate - 0.5) * 2
        
        win_rate_capped = np.clip(win_rate, 0.1, 0.9)
        nearby_win_rate_capped = np.clip(nearby_win_rate, 0.1, 0.9)
        
        improvement_signal = (
            max(0, nearby_win_rate - 0.6) * 2 +
            vs_better_win_rate * 1.5 +
            max(0, vs_worse_win_rate - 0.7)
        )
        
        decline_signal = (
            max(0, 0.4 - nearby_win_rate) * 2 +
            max(0, 0.5 - vs_worse_win_rate) * 1.5
        )
        
        is_junior = 1 if category in ['JUN', 'J19', 'J21'] else 0
        junior_volatility = is_junior * improvement_signal
        in_de_zone = 1 if current_rank_encoded in [9, 10, 11, 12, 13, 14, 15, 16] else 0
        match_reliability = np.tanh(total_matches / 30)
        reliable_performance = nearby_win_rate_capped * match_reliability
        is_low_rank = 1 if current_rank_encoded >= 13 else 0
        is_mid_rank = 1 if 9 <= current_rank_encoded < 13 else 0
        
        features_list.append({
            'current_rank_encoded': current_rank_encoded,
            'category_encoded': category_encoded,
            'win_rate': win_rate,
            'nearby_win_rate': nearby_win_rate,
            'vs_better_win_rate': vs_better_win_rate,
            'vs_worse_win_rate': vs_worse_win_rate,
            'total_matches': total_matches,
            'match_volume': match_volume,
            'win_loss_ratio': win_loss_ratio,
            'performance_consistency': performance_consistency,
            'level_dominance': level_dominance,
            'win_rate_capped': win_rate_capped,
            'nearby_win_rate_capped': nearby_win_rate_capped,
            'improvement_signal': improvement_signal,
            'decline_signal': decline_signal,
            'is_junior': is_junior,
            'junior_volatility': junior_volatility,
            'in_de_zone': in_de_zone,
            'match_reliability': match_reliability,
            'reliable_performance': reliable_performance,
            'is_low_rank': is_low_rank,
            'is_mid_rank': is_mid_rank
        })
    
    df_features = pd.DataFrame(features_list)
    return df_features[v3_feature_cols]

def prepare_features_v4(df_subset):
    """Prepare V4 features (with ELO)"""
    # Get all features first
    features_list = []
    
    for idx, row in df_subset.iterrows():
        kaart = row['kaart']
        current_rank = row['current_ranking']
        category = row['category']
        elo = row['elo']
        
        current_rank_encoded = v4_rank_to_int.get(current_rank, -1)
        
        # Encode category
        category_encoder = LabelEncoder()
        category_encoder.fit(df['category'].unique())
        category_encoded = category_encoder.transform([category])[0]
        
        # Calculate stats (same as V3)
        total_wins = sum(kaart.get(rank, [0, 0])[0] for rank in ranking_order)
        total_losses = sum(kaart.get(rank, [0, 0])[1] for rank in ranking_order)
        total_matches = total_wins + total_losses
        win_rate = total_wins / total_matches if total_matches > 0 else 0.5
        
        nearby_wins = 0
        nearby_losses = 0
        for rank in ranking_order:
            rank_idx = v4_rank_to_int[rank]
            if abs(rank_idx - current_rank_encoded) <= 3:
                wins, losses = kaart.get(rank, [0, 0])
                nearby_wins += wins
                nearby_losses += losses
        nearby_total = nearby_wins + nearby_losses
        nearby_win_rate = nearby_wins / nearby_total if nearby_total > 0 else 0.5
        
        better_wins = 0
        better_losses = 0
        for rank in ranking_order:
            rank_idx = v4_rank_to_int[rank]
            if rank_idx < current_rank_encoded:
                wins, losses = kaart.get(rank, [0, 0])
                better_wins += wins
                better_losses += losses
        better_total = better_wins + better_losses
        vs_better_win_rate = better_wins / better_total if better_total > 0 else 0
        
        worse_wins = 0
        worse_losses = 0
        for rank in ranking_order:
            rank_idx = v4_rank_to_int[rank]
            if rank_idx > current_rank_encoded:
                wins, losses = kaart.get(rank, [0, 0])
                worse_wins += wins
                worse_losses += losses
        worse_total = worse_wins + worse_losses
        vs_worse_win_rate = worse_wins / worse_total if worse_total > 0 else 0
        
        match_volume = np.log1p(total_matches)
        win_loss_ratio = total_wins / total_losses if total_losses > 0 else total_wins
        performance_consistency = nearby_win_rate * np.log1p(total_matches)
        level_dominance = (nearby_win_rate - 0.5) * 2
        
        win_rate_capped = np.clip(win_rate, 0.1, 0.9)
        nearby_win_rate_capped = np.clip(nearby_win_rate, 0.1, 0.9)
        
        improvement_signal = (
            max(0, nearby_win_rate - 0.6) * 2 +
            vs_better_win_rate * 1.5 +
            max(0, vs_worse_win_rate - 0.7)
        )
        
        decline_signal = (
            max(0, 0.4 - nearby_win_rate) * 2 +
            max(0, 0.5 - vs_worse_win_rate) * 1.5
        )
        
        is_junior = 1 if category in ['JUN', 'J19', 'J21'] else 0
        junior_volatility = is_junior * improvement_signal
        in_de_zone = 1 if current_rank_encoded in [9, 10, 11, 12, 13, 14, 15, 16] else 0
        match_reliability = np.tanh(total_matches / 30)
        reliable_performance = nearby_win_rate_capped * match_reliability
        is_low_rank = 1 if current_rank_encoded >= 13 else 0
        is_mid_rank = 1 if 9 <= current_rank_encoded < 13 else 0
        
        # ELO features
        elo_log = np.log1p(elo)
        rank_elo_map = {
            'A': 1800, 'B0': 1600, 'B2': 1500, 'B4': 1400, 'B6': 1300,
            'C0': 1200, 'C2': 1100, 'C4': 1000, 'C6': 900,
            'D0': 800, 'D2': 750, 'D4': 700, 'D6': 650,
            'E0': 600, 'E2': 550, 'E4': 500, 'E6': 450,
            'NG': 400
        }
        expected_elo = rank_elo_map.get(current_rank, 400)
        elo_advantage = np.clip((elo - expected_elo) / 100, -5, 5)
        
        features_list.append({
            'current_rank_encoded': current_rank_encoded,
            'category_encoded': category_encoded,
            'win_rate': win_rate,
            'nearby_win_rate': nearby_win_rate,
            'vs_better_win_rate': vs_better_win_rate,
            'vs_worse_win_rate': vs_worse_win_rate,
            'total_matches': total_matches,
            'match_volume': match_volume,
            'win_loss_ratio': win_loss_ratio,
            'performance_consistency': performance_consistency,
            'level_dominance': level_dominance,
            'win_rate_capped': win_rate_capped,
            'nearby_win_rate_capped': nearby_win_rate_capped,
            'improvement_signal': improvement_signal,
            'decline_signal': decline_signal,
            'is_junior': is_junior,
            'junior_volatility': junior_volatility,
            'in_de_zone': in_de_zone,
            'match_reliability': match_reliability,
            'reliable_performance': reliable_performance,
            'is_low_rank': is_low_rank,
            'is_mid_rank': is_mid_rank,
            'elo_log': elo_log,
            'elo_advantage': elo_advantage
        })
    
    df_features = pd.DataFrame(features_list)
    return df_features[v4_feature_cols]

# Test on NG players with ELO
print("\n" + "="*80)
print("TESTING ON NG PLAYERS WITH ELO")
print("="*80)

# Get actual next rankings
y_true = df_ng_with_elo['next_ranking'].map(v3_rank_to_int).values

# Prepare features
print("Preparing V3 features...")
X_v3 = prepare_features_v3(df_ng_with_elo)
print("Preparing V4 features...")
X_v4 = prepare_features_v4(df_ng_with_elo)

# Make predictions
print("Making predictions...")
y_pred_v3 = v3_model.predict(X_v3)
y_pred_v4 = v4_model.predict(X_v4)

# Calculate accuracies
acc_v3 = accuracy_score(y_true, y_pred_v3)
acc_v4 = accuracy_score(y_true, y_pred_v4)

print(f"\nV3 Model (no ELO) Accuracy: {acc_v3:.4f} ({acc_v3*100:.2f}%)")
print(f"V4 Model (with ELO) Accuracy: {acc_v4:.4f} ({acc_v4*100:.2f}%)")
print(f"Improvement: {(acc_v4 - acc_v3)*100:.2f}%")

# Show distribution of predictions
print("\n" + "="*80)
print("PREDICTION DISTRIBUTION")
print("="*80)

print("\nActual next rankings:")
actual_dist = pd.Series([v3_int_to_rank[i] for i in y_true]).value_counts().sort_index()
print(actual_dist)

print("\nV3 predictions:")
v3_pred_dist = pd.Series([v3_int_to_rank[i] for i in y_pred_v3]).value_counts().sort_index()
print(v3_pred_dist)

print("\nV4 predictions:")
v4_pred_dist = pd.Series([v4_int_to_rank[i] for i in y_pred_v4]).value_counts().sort_index()
print(v4_pred_dist)

# Show some examples
print("\n" + "="*80)
print("SAMPLE PREDICTIONS (First 10 NG players with ELO)")
print("="*80)

sample = df_ng_with_elo.head(10)
for idx, row in sample.iterrows():
    actual = row['next_ranking']
    v3_pred = v3_int_to_rank[v3_model.predict(df_ng_with_elo.loc[[idx]][v3_feature_cols])[0]]
    v4_pred = v4_int_to_rank[v4_model.predict(df_ng_with_elo.loc[[idx]][v4_feature_cols])[0]]
    elo = row['elo']
    
    total_wins = sum(row['kaart'].get(rank, [0, 0])[0] for rank in v3_rank_to_int.keys())
    total_losses = sum(row['kaart'].get(rank, [0, 0])[1] for rank in v3_rank_to_int.keys())
    total_matches = total_wins + total_losses
    win_rate = total_wins / total_matches if total_matches > 0 else 0
    
    v3_correct = "[OK]" if v3_pred == actual else "[X]"
    v4_correct = "[OK]" if v4_pred == actual else "[X]"
    
    print(f"\nELO: {elo:.0f} | Matches: {total_matches} | Win Rate: {win_rate:.2%}")
    print(f"  Actual: {actual}")
    print(f"  V3: {v3_pred} {v3_correct}")
    print(f"  V4: {v4_pred} {v4_correct}")

# Confusion matrix comparison
print("\n" + "="*80)
print("CONFUSION MATRIX - V3 vs V4")
print("="*80)

# Get unique ranks that appear in predictions
unique_ranks = sorted(set(y_true) | set(y_pred_v3) | set(y_pred_v4))
rank_labels = [v3_int_to_rank[i] for i in unique_ranks]

print("\nV3 Confusion Matrix:")
cm_v3 = confusion_matrix(y_true, y_pred_v3, labels=unique_ranks)
print("Predicted →")
print("Actual ↓")
print(pd.DataFrame(cm_v3, index=rank_labels, columns=rank_labels))

print("\nV4 Confusion Matrix:")
cm_v4 = confusion_matrix(y_true, y_pred_v4, labels=unique_ranks)
print("Predicted →")
print("Actual ↓")
print(pd.DataFrame(cm_v4, index=rank_labels, columns=rank_labels))

print("\n" + "="*80)
print("CONCLUSION")
print("="*80)

if acc_v4 > acc_v3:
    print(f"[WIN] V4 with ELO is BETTER for NG players (+{(acc_v4-acc_v3)*100:.2f}%)")
elif acc_v4 < acc_v3:
    print(f"[LOSE] V4 with ELO is WORSE for NG players ({(acc_v4-acc_v3)*100:.2f}%)")
else:
    print("[TIE] V4 with ELO performs the SAME as V3 for NG players")

print(f"\nRecommendation: {'Use V4 with ELO for NG players' if acc_v4 > acc_v3 else 'Keep using V3 for NG players'}")
