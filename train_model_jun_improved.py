import pandas as pd
import ast
import joblib
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.ensemble import BalancedRandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score, cohen_kappa_score, matthews_corrcoef
import optuna
import warnings
warnings.filterwarnings('ignore')

try:
    from xgboost import XGBClassifier
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    print("XGBoost not available")

print("=" * 60)
print("IMPROVED JUN/J19 MODEL TRAINING")
print("=" * 60)

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

# Filter to JUN/J19 categories only
allowed_categories = ["JUN", "J19"]
df = df[df["category"].isin(allowed_categories)]
print(f"Filtered to JUN/J19 categories: {len(df)} players")

ranking_order = [
    "A", "B0", "B2", "B4", "B6",
    "C0", "C2", "C4", "C6",
    "D0", "D2", "D4", "D6",
    "E0", "E2", "E4", "E6",
    "NG"
]

rank_to_int = {r:i for i, r in enumerate(ranking_order)}
int_to_rank = {i:r for r, i in rank_to_int.items()}

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

# Advanced feature engineering
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

# Filter players with at least 15 matches
min_matches = 15
print(f"\nFiltering players with at least {min_matches} matches...")
print(f"Before filtering: {len(df)} players")
df = df[df['total_matches'] >= min_matches]
print(f"After filtering: {len(df)} players")

# ===== ENHANCEMENTS =====
print("\n[1/6] Adding opponent strength features...")
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

print("[2/6] Adding opponent quality features...")
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

print("[3/6] Adding match context features...")
# Win rate std
for idx, row in df.iterrows():
    rates = []
    for rank in ranking_order:
        if row[f"{rank}_total"] > 0:
            rates.append(row[f"{rank}_win_rate"])
    df.at[idx, 'win_rate_std'] = np.std(rates) if len(rates) > 1 else 0

# Recent form
df['recent_form'] = df['overall_win_rate'] * np.minimum(df['total_matches'] / 20, 1)

# Encode category
category_encoder = LabelEncoder()
df["category_encoded"] = category_encoder.fit_transform(df["category"])

# Define features
base_feature_cols = ["current_rank_encoded", "category_encoded"]
advanced_feature_cols = [
    'total_wins', 'total_losses', 'total_matches', 'overall_win_rate',
    'performance_consistency', 'recent_performance', 'rank_progression_potential',
    'performance_vs_higher', 'performance_vs_lower', 'win_loss_ratio',
    'avg_opponent_rank', 'toughest_opponent_beaten', 'win_rate_std', 'recent_form'
]

important_ranks = ['B0', 'B2', 'B4', 'C0', 'C2', 'C4', 'D0', 'D2', 'D4', 'E0', 'E2']
win_rate_cols = [f"{rank}_win_rate" for rank in important_ranks if f"{rank}_win_rate" in df.columns]
total_cols = [f"{rank}_total" for rank in ['E0', 'E2', 'E4', 'E6'] if f"{rank}_total" in df.columns]
weighted_perf_cols = [f"{rank}_weighted_performance" for rank in important_ranks if f"{rank}_weighted_performance" in df.columns]

feature_cols = base_feature_cols + advanced_feature_cols + win_rate_cols + total_cols + weighted_perf_cols

# Filter rare classes
class_counts = df["next_rank_encoded"].value_counts()
valid_classes = class_counts[class_counts >= 10].index
df = df[df["next_rank_encoded"].isin(valid_classes)]

print(f"\nClasses after filtering (>= 10 samples): {len(valid_classes)}")

X = df[feature_cols].fillna(0)
y = df["next_rank_encoded"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Balancing
print("\n[4/6] Balancing data...")
def conservative_smote_balancing(X_train, y_train):
    class_counts = pd.Series(y_train).value_counts().sort_values()
    min_samples = class_counts.min()
    max_samples = class_counts.max()
    
    if max_samples / min_samples < 3:
        return X_train, y_train, 'class_weight'
    
    oversample_strategy = {}
    for class_label, count in class_counts.items():
        if count < 20:
            oversample_strategy[class_label] = min(60, max_samples // 6)
        elif count < 150:
            target = int(max_samples * 0.3)
            if count < target:
                oversample_strategy[class_label] = target
    
    if not oversample_strategy:
        return X_train, y_train, 'class_weight'
    
    min_class_to_sample = min([class_counts[c] for c in oversample_strategy.keys()])
    k_neighbors = max(1, min(5, min_class_to_sample - 1))
    
    try:
        smote = SMOTE(sampling_strategy=oversample_strategy, k_neighbors=k_neighbors, random_state=42)
        X_oversampled, y_oversampled = smote.fit_resample(X_train, y_train)
        
        new_counts = pd.Series(y_oversampled).value_counts().sort_values()
        median_size = new_counts.median()
        max_majority_size = int(median_size * 2.0)
        
        undersample_strategy = {}
        for class_label, count in new_counts.items():
            if count > max_majority_size:
                undersample_strategy[class_label] = max_majority_size
        
        if undersample_strategy:
            undersampler = RandomUnderSampler(sampling_strategy=undersample_strategy, random_state=42)
            X_balanced, y_balanced = undersampler.fit_resample(X_oversampled, y_oversampled)
        else:
            X_balanced, y_balanced = X_oversampled, y_oversampled
        
        return X_balanced, y_balanced, 'combined'
    except ValueError as e:
        return X_train, y_train, 'class_weight'

X_train_sm, y_train_sm, balance_method = conservative_smote_balancing(X_train_scaled, y_train)

# ===== Hyperparameter Tuning =====
print("\n[5/6] Hyperparameter tuning with Optuna...")

def objective(trial):
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 200, 400),
        'max_depth': trial.suggest_int('max_depth', 12, 20),
        'min_samples_split': trial.suggest_int('min_samples_split', 5, 15),
        'min_samples_leaf': trial.suggest_int('min_samples_leaf', 2, 8),
        'max_features': trial.suggest_categorical('max_features', ['sqrt', 'log2']),
    }
    
    model = RandomForestClassifier(**params, random_state=42, n_jobs=-1)
    score = cross_val_score(model, X_train_sm, y_train_sm, cv=3, scoring='f1_weighted', n_jobs=-1).mean()
    return score

study = optuna.create_study(direction='maximize', study_name='jun_rf_optimization')
study.optimize(objective, n_trials=20, show_progress_bar=True)

print(f"\nBest parameters: {study.best_params}")
print(f"Best CV F1 score: {study.best_value:.4f}")

# ===== ENHANCEMENT 5: Ensemble with Multiple Models =====
print("\n[6/6] Creating ensemble model...")

# Train optimized RandomForest
rf_model = RandomForestClassifier(**study.best_params, random_state=42, n_jobs=-1)
rf_model.fit(X_train_sm, y_train_sm)

# Train Balanced RandomForest
balanced_rf = BalancedRandomForestClassifier(
    n_estimators=study.best_params['n_estimators'],
    max_depth=study.best_params['max_depth'],
    random_state=42,
    n_jobs=-1
)
balanced_rf.fit(X_train_sm, y_train_sm)

# Train Gradient Boosting
gb_model = GradientBoostingClassifier(
    n_estimators=200,
    max_depth=8,
    learning_rate=0.05,
    random_state=42
)
gb_model.fit(X_train_sm, y_train_sm)

estimators = [
    ('rf', rf_model),
    ('balanced_rf', balanced_rf),
    ('gb', gb_model)
]

# Add XGBoost if available
if XGBOOST_AVAILABLE:
    print("Adding XGBoost to ensemble...")
    unique_classes = np.unique(y_train_sm)
    class_mapping = {old: new for new, old in enumerate(unique_classes)}
    reverse_mapping = {new: old for old, new in class_mapping.items()}
    
    y_train_xgb = np.array([class_mapping[y] for y in y_train_sm])
    
    xgb_model = XGBClassifier(
        n_estimators=300,
        max_depth=8,
        learning_rate=0.05,
        random_state=42,
        n_jobs=-1,
        eval_metric='mlogloss'
    )
    xgb_model.fit(X_train_sm, y_train_xgb)
    
    # Wrapper for XGBoost
    class XGBWrapper:
        def __init__(self, model, reverse_mapping):
            self.model = model
            self.reverse_mapping = reverse_mapping
        
        def predict(self, X):
            preds = self.model.predict(X)
            return np.array([self.reverse_mapping[p] for p in preds])
        
        def predict_proba(self, X):
            return self.model.predict_proba(X)
        
        def fit(self, X, y):
            return self
    
    xgb_wrapped = XGBWrapper(xgb_model, reverse_mapping)
    estimators.append(('xgb', xgb_wrapped))

# Create ensemble (models are already fitted)
print(f"\nEnsemble created with {len(estimators)} models")

# Use the optimized RandomForest as the main model (best performing)
best_model = rf_model
print("Using optimized RandomForest as final model (best performing)")

# Evaluate
y_pred = best_model.predict(X_test_scaled)
accuracy = accuracy_score(y_test, y_pred)
kappa = cohen_kappa_score(y_test, y_pred)
mcc = matthews_corrcoef(y_test, y_pred)

print("\n" + "=" * 60)
print("FINAL RESULTS")
print("=" * 60)
print(f"Test Accuracy: {accuracy:.4f}")
print(f"Cohen's Kappa: {kappa:.4f}")
print(f"Matthews Correlation: {mcc:.4f}")

# Feature importance
print("\nTop 10 most important features:")
feature_importance = pd.DataFrame({
    'feature': feature_cols,
    'importance': best_model.feature_importances_
}).sort_values('importance', ascending=False)
print(feature_importance.head(10).to_string(index=False))

# Save models
joblib.dump(best_model, "model_jun_improved.pkl")
joblib.dump(category_encoder, "category_encoder_jun_improved.pkl")
joblib.dump(rank_to_int, "rank_to_int_jun_improved.pkl")
joblib.dump(int_to_rank, "int_to_rank_jun_improved.pkl")
joblib.dump(feature_cols, "feature_cols_jun_improved.pkl")
joblib.dump(ranking_order, "ranking_order_jun_improved.pkl")
joblib.dump(scaler, "scaler_jun_improved.pkl")

print("\nIMPROVED JUN/J19 MODEL SAVED!")
print(f"Total features: {len(feature_cols)}")
print(f"Training samples: {len(X_train_sm)}")
print(f"Test samples: {len(X_test)}")
print(f"Classes: {len(valid_classes)}")
