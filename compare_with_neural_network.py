"""
Compare Neural Network model with existing models (V3, V4, Filtered)
"""
import pandas as pd
import ast
import joblib
import numpy as np
import os
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')

print("=" * 80)
print("MODEL COMPARISON: Neural Network vs Random Forest Models")
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

# Test on regular categories (excluding youth)
youth_categories = ["BEN", "PRE", "MIN", "CAD"]
df_regular = df[~df['category'].isin(youth_categories)].copy()

# Test on youth categories
df_youth = df[df['category'].isin(youth_categories)].copy()

print(f"\nRegular categories: {len(df_regular)} players")
print(f"Youth categories: {len(df_youth)} players")

ranking_order = [
    "A", "B0", "B2", "B4", "B6",
    "C0", "C2", "C4", "C6",
    "D0", "D2", "D4", "D6",
    "E0", "E2", "E4", "E6",
    "NG"
]

rank_to_int = {r: i for i, r in enumerate(ranking_order)}
int_to_rank = {i: r for r, i in rank_to_int.items()}

def prepare_features(df_subset, feature_cols, category_encoder, rank_to_int, ranking_order):
    """Prepare features for prediction"""
    df_subset = df_subset.copy()
    
    df_subset["current_rank_encoded"] = df_subset["current_ranking"].map(rank_to_int)
    df_subset["next_rank_encoded"] = df_subset["next_ranking"].map(rank_to_int)
    
    # Basic stats
    total_wins = []
    total_losses = []
    for kaart in df_subset["kaart"]:
        wins = sum(w for w, l in kaart.values())
        losses = sum(l for w, l in kaart.values())
        total_wins.append(wins)
        total_losses.append(losses)
    
    df_subset['total_wins'] = total_wins
    df_subset['total_losses'] = total_losses
    df_subset['total_matches'] = df_subset['total_wins'] + df_subset['total_losses']
    df_subset['win_rate'] = df_subset['total_wins'] / df_subset['total_matches'].replace(0, 1)
    
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
    
    df_subset['nearby_win_rate'] = df_subset.apply(
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
    
    df_subset['vs_better_win_rate'] = df_subset.apply(
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
    
    df_subset['vs_worse_win_rate'] = df_subset.apply(
        lambda row: get_performance_vs_worse(row['kaart'], row['current_rank_encoded'], ranking_order, rank_to_int),
        axis=1
    )
    
    df_subset['match_volume'] = np.log1p(df_subset['total_matches'])
    df_subset['win_loss_ratio'] = np.where(df_subset['total_losses'] > 0, 
                                           df_subset['total_wins'] / df_subset['total_losses'], 
                                           df_subset['total_wins'])
    df_subset['performance_consistency'] = df_subset['nearby_win_rate'] * np.log1p(df_subset['total_matches'])
    df_subset['level_dominance'] = (df_subset['nearby_win_rate'] - 0.5) * 2
    df_subset['performance_score'] = (df_subset['win_rate'] * 0.4 + 
                                      df_subset['nearby_win_rate'] * 0.4 + 
                                      df_subset['vs_better_win_rate'] * 0.2)
    
    # V3 features
    df_subset['win_rate_capped'] = df_subset['win_rate'].clip(0.1, 0.9)
    df_subset['nearby_win_rate_capped'] = df_subset['nearby_win_rate'].clip(0.1, 0.9)
    
    df_subset['improvement_signal'] = (
        (df_subset['nearby_win_rate'] - 0.6).clip(0, 1) * 2 +
        df_subset['vs_better_win_rate'] * 1.5 +
        (df_subset['vs_worse_win_rate'] - 0.7).clip(0, 1)
    )
    
    df_subset['decline_signal'] = (
        (0.4 - df_subset['nearby_win_rate']).clip(0, 1) * 2 +
        (0.5 - df_subset['vs_worse_win_rate']).clip(0, 1) * 1.5
    )
    
    df_subset['is_junior'] = df_subset['category'].isin(['JUN', 'J19', 'J21']).astype(int)
    df_subset['junior_volatility'] = df_subset['is_junior'] * df_subset['improvement_signal']
    df_subset['in_de_zone'] = df_subset['current_rank_encoded'].isin([9, 10, 11, 12, 13, 14, 15, 16]).astype(int)
    df_subset['match_reliability'] = np.tanh(df_subset['total_matches'] / 30)
    df_subset['reliable_performance'] = df_subset['nearby_win_rate_capped'] * df_subset['match_reliability']
    df_subset['is_low_rank'] = (df_subset['current_rank_encoded'] >= 13).astype(int)
    df_subset['is_mid_rank'] = ((df_subset['current_rank_encoded'] >= 9) & (df_subset['current_rank_encoded'] < 13)).astype(int)
    
    # Youth-specific features (for filtered model)
    df_subset['is_unranked'] = (df_subset['current_ranking'] == 'NG').astype(int)
    df_subset['is_entry_rank'] = (df_subset['current_ranking'] == 'E6').astype(int)
    df_subset['breakthrough_signal'] = (
        (df_subset['win_rate'] - 0.6).clip(0, 1) * 2 +
        df_subset['vs_better_win_rate'] * 2 +
        (df_subset['nearby_win_rate'] - 0.7).clip(0, 1) * 1.5
    )
    df_subset['activity_volatility'] = np.tanh(df_subset['total_matches'] / 50) * df_subset['win_rate']
    df_subset['poor_performer_signal'] = (
        (0.4 - df_subset['win_rate']).clip(0, 1) * 2 +
        (0.3 - df_subset['nearby_win_rate']).clip(0, 1) * 1.5
    )
    df_subset['is_excellent'] = ((df_subset['win_rate'] > 0.7) & (df_subset['total_matches'] >= 20)).astype(int)
    df_subset['is_youngest'] = df_subset['category'].isin(['MIN', 'PRE']).astype(int)
    df_subset['is_oldest'] = df_subset['category'].isin(['BEN', 'CAD']).astype(int)
    
    # Ranked opponent ratio
    ranked_opponent_ratio = []
    ng_win_rate_list = []
    ng_has_many_matches_list = []
    
    for idx, row in df_subset.iterrows():
        kaart = row['kaart']
        ng_matches = kaart.get('NG', [0, 0])
        ng_total = ng_matches[0] + ng_matches[1]
        total = row['total_matches']
        ranked_matches = total - ng_total
        ranked_opponent_ratio.append(ranked_matches / total if total > 0 else 0)
        
        ng_win_rate_list.append(row['win_rate'] if row['current_ranking'] == 'NG' else 0)
        ng_has_many_matches_list.append(1 if (row['current_ranking'] == 'NG' and total >= 20) else 0)
    
    df_subset['ranked_opponent_ratio'] = ranked_opponent_ratio
    df_subset['ng_win_rate'] = ng_win_rate_list
    df_subset['ng_has_many_matches'] = ng_has_many_matches_list
    df_subset['e6_ready_to_advance'] = ((df_subset['current_ranking'] == 'E6') & 
                                        (df_subset['nearby_win_rate'] > 0.6) & 
                                        (df_subset['total_matches'] >= 15)).astype(int)
    df_subset['youth_match_reliability'] = np.tanh(df_subset['total_matches'] / 40)
    
    # Encode category
    df_subset['category_encoded'] = category_encoder.transform(df_subset['category'])
    
    # Ensure all feature columns exist
    for col in feature_cols:
        if col not in df_subset.columns:
            df_subset[col] = 0
    
    X = df_subset[feature_cols]
    y = df_subset['next_rank_encoded']
    
    return X, y, df_subset

def evaluate_model(model_name, model, X, y, int_to_rank, scaler=None):
    """Evaluate a model and return metrics"""
    if scaler is not None:
        X_scaled = scaler.transform(X)
    else:
        X_scaled = X
    
    y_pred = model.predict(X_scaled)
    
    accuracy = accuracy_score(y, y_pred)
    
    # Calculate off-by-one and off-by-two
    correct = (y_pred == y).sum()
    off_by_one = (np.abs(y_pred - y) == 1).sum()
    off_by_two = (np.abs(y_pred - y) == 2).sum()
    
    within_one = (correct + off_by_one) / len(y) * 100
    within_two = (correct + off_by_one + off_by_two) / len(y) * 100
    
    return {
        'model': model_name,
        'accuracy': accuracy * 100,
        'correct': correct,
        'total': len(y),
        'off_by_one': off_by_one,
        'off_by_two': off_by_two,
        'within_one': within_one,
        'within_two': within_two
    }

# ===== LOAD MODELS =====

print("\n" + "=" * 80)
print("LOADING MODELS")
print("=" * 80)

models_to_test = []

# Neural Network (sklearn MLPClassifier)
try:
    nn_model = joblib.load("model_neural_network.pkl")
    nn_scaler = joblib.load("scaler_neural_network.pkl")
    nn_category_encoder = joblib.load("category_encoder_neural_network.pkl")
    nn_feature_cols = joblib.load("feature_cols_neural_network.pkl")
    nn_int_to_rank = joblib.load("int_to_rank_neural_network.pkl")
    models_to_test.append(('Neural Network', nn_model, nn_scaler, nn_category_encoder, nn_feature_cols, nn_int_to_rank, False))
    print("✓ Loaded Neural Network model")
except Exception as e:
    print(f"✗ Could not load Neural Network model: {e}")

# V3 Improved (Regular)
try:
    v3_model = joblib.load("model_v3_improved.pkl")
    v3_category_encoder = joblib.load("category_encoder_v3.pkl")
    v3_feature_cols = joblib.load("feature_cols_v3.pkl")
    v3_int_to_rank = joblib.load("int_to_rank_v3.pkl")
    models_to_test.append(('V3 Improved', v3_model, None, v3_category_encoder, v3_feature_cols, v3_int_to_rank, False))
    print("✓ Loaded V3 Improved model")
except Exception as e:
    print(f"✗ Could not load V3 Improved model: {e}")

# V4 Final
try:
    v4_model = joblib.load("model_v4_final.pkl")
    v4_category_encoder = joblib.load("category_encoder_v4.pkl")
    v4_feature_cols = joblib.load("feature_cols_v4.pkl")
    v4_int_to_rank = joblib.load("int_to_rank_v4.pkl")
    models_to_test.append(('V4 Final', v4_model, None, v4_category_encoder, v4_feature_cols, v4_int_to_rank, False))
    print("✓ Loaded V4 Final model")
except Exception as e:
    print(f"✗ Could not load V4 Final model: {e}")

# Filtered V3 (Youth)
try:
    filtered_model = joblib.load("model_filtered_v3_improved.pkl")
    filtered_scaler = joblib.load("scaler_filtered.pkl") if os.path.exists("scaler_filtered.pkl") else None
    filtered_category_encoder = joblib.load("category_encoder_filtered_v3.pkl")
    filtered_feature_cols = joblib.load("feature_cols_filtered_v3.pkl")
    filtered_int_to_rank = joblib.load("int_to_rank_filtered_v3.pkl")
    models_to_test.append(('Filtered V3', filtered_model, filtered_scaler, filtered_category_encoder, filtered_feature_cols, filtered_int_to_rank, False))
    print("✓ Loaded Filtered V3 model")
except Exception as e:
    print(f"✗ Could not load Filtered V3 model: {e}")

if not models_to_test:
    print("\n❌ No models loaded! Please train models first.")
    exit(1)

# ===== TEST ON REGULAR CATEGORIES =====

print("\n" + "=" * 80)
print("TESTING ON REGULAR CATEGORIES (excluding youth)")
print("=" * 80)

results_regular = []

for model_name, model, scaler, category_encoder, feature_cols, int_to_rank_map, _ in models_to_test:
    if 'Filtered' in model_name:
        continue  # Skip filtered model for regular categories
    
    try:
        X, y, _ = prepare_features(df_regular, feature_cols, category_encoder, rank_to_int, ranking_order)
        result = evaluate_model(model_name, model, X, y, int_to_rank_map, scaler)
        results_regular.append(result)
        print(f"\n{model_name}:")
        print(f"  Accuracy: {result['accuracy']:.2f}%")
        print(f"  Within 1 rank: {result['within_one']:.2f}%")
        print(f"  Within 2 ranks: {result['within_two']:.2f}%")
    except Exception as e:
        print(f"\n{model_name}: Error - {e}")

# ===== TEST ON YOUTH CATEGORIES =====

print("\n" + "=" * 80)
print("TESTING ON YOUTH CATEGORIES")
print("=" * 80)

results_youth = []

for model_name, model, scaler, category_encoder, feature_cols, int_to_rank_map, _ in models_to_test:
    try:
        X, y, _ = prepare_features(df_youth, feature_cols, category_encoder, rank_to_int, ranking_order)
        result = evaluate_model(model_name, model, X, y, int_to_rank_map, scaler)
        results_youth.append(result)
        print(f"\n{model_name}:")
        print(f"  Accuracy: {result['accuracy']:.2f}%")
        print(f"  Within 1 rank: {result['within_one']:.2f}%")
        print(f"  Within 2 ranks: {result['within_two']:.2f}%")
    except Exception as e:
        print(f"\n{model_name}: Error - {e}")

# ===== SUMMARY =====

print("\n" + "=" * 80)
print("SUMMARY - REGULAR CATEGORIES")
print("=" * 80)

if results_regular:
    df_results_regular = pd.DataFrame(results_regular)
    df_results_regular = df_results_regular.sort_values('accuracy', ascending=False)
    print("\n" + df_results_regular.to_string(index=False))
    
    best_model = df_results_regular.iloc[0]
    print(f"\n🏆 Best Model: {best_model['model']} with {best_model['accuracy']:.2f}% accuracy")

print("\n" + "=" * 80)
print("SUMMARY - YOUTH CATEGORIES")
print("=" * 80)

if results_youth:
    df_results_youth = pd.DataFrame(results_youth)
    df_results_youth = df_results_youth.sort_values('accuracy', ascending=False)
    print("\n" + df_results_youth.to_string(index=False))
    
    best_model = df_results_youth.iloc[0]
    print(f"\n🏆 Best Model: {best_model['model']} with {best_model['accuracy']:.2f}% accuracy")

print("\n" + "=" * 80)
print("COMPARISON COMPLETE!")
print("=" * 80)
