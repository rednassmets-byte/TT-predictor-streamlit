import pandas as pd
import ast
import joblib
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from sklearn.metrics import classification_report, accuracy_score, cohen_kappa_score, matthews_corrcoef
import warnings
warnings.filterwarnings('ignore')

# -------------------------
# Load data
# -------------------------
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

ranking_order = [
    "A",
    "B0", "B2", "B4", "B6",
    "C0", "C2", "C4", "C6",
    "D0", "D2", "D4", "D6",
    "E0", "E2", "E4", "E6",
    "NG"
]

rank_to_int = {r:i for i, r in enumerate(ranking_order)}
int_to_rank = {i:r for r, i in rank_to_int.items()}

df["current_rank_encoded"] = df["current_ranking"].map(rank_to_int)
df["next_rank_encoded"] = df["next_ranking"].map(rank_to_int)

# -------------------------
# Expand kaart and create advanced features
# -------------------------
def extract_kaart_features(kaart):
    f = {}
    for rank in ranking_order:
        wins, losses = kaart.get(rank, [0, 0])
        f[f"{rank}_wins"] = wins
        f[f"{rank}_losses"] = losses
    return f

df = pd.concat([df, df["kaart"].apply(extract_kaart_features).apply(pd.Series)], axis=1)

# Advanced feature engineering (same as filtered model)
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

# -------------------------
# Filter players with at least 15 matches
# -------------------------
min_matches = 15
print(f"\nFiltering players with at least {min_matches} matches...")
print(f"Before filtering: {len(df)} players")
df = df[df['total_matches'] >= min_matches]
print(f"After filtering: {len(df)} players")

# -------------------------
# Add opponent strength features
# -------------------------
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

print("Adding opponent strength features...")
df = add_opponent_strength_features(df, ranking_order, rank_to_int)

# -------------------------
# Add performance vs higher/lower ranks
# -------------------------
higher_ranks = ['A', 'B0', 'B2', 'B4']
lower_ranks = ['D0', 'D2', 'D4', 'E0', 'E2']

df['performance_vs_higher'] = df[[f"{r}_win_rate" for r in higher_ranks if f"{r}_win_rate" in df.columns]].mean(axis=1)
df['performance_vs_lower'] = df[[f"{r}_win_rate" for r in lower_ranks if f"{r}_win_rate" in df.columns]].mean(axis=1)

# Win/loss ratio
df['win_loss_ratio'] = np.divide(df['total_wins'], df['total_losses'], 
                                  out=np.zeros_like(df['total_wins'], dtype=float),
                                  where=df['total_losses'] > 0)

print("Added performance trend features")

category_encoder = LabelEncoder()
df["category_encoded"] = category_encoder.fit_transform(df["category"])

# Define feature columns - optimized selection
base_feature_cols = ["current_rank_encoded", "category_encoded"]
advanced_feature_cols = [
    'total_wins', 'total_losses', 'total_matches', 'overall_win_rate',
    'performance_consistency', 'recent_performance', 'rank_progression_potential',
    'performance_vs_higher', 'performance_vs_lower', 'win_loss_ratio'
]

# Add important win_rate features
important_ranks = ['B0', 'B2', 'B4', 'C0', 'C2', 'C4', 'D0', 'D2', 'D4', 'E0', 'E2']
win_rate_cols = [f"{rank}_win_rate" for rank in important_ranks if f"{rank}_win_rate" in df.columns]

# Add total games per rank for better context
total_cols = [f"{rank}_total" for rank in ['E0', 'E2', 'E4', 'E6'] if f"{rank}_total" in df.columns]

# Add weighted performance features for key ranks
weighted_perf_cols = [f"{rank}_weighted_performance" for rank in important_ranks if f"{rank}_weighted_performance" in df.columns]

feature_cols = base_feature_cols + advanced_feature_cols + win_rate_cols + total_cols + weighted_perf_cols

# -------------------------
# Filter out extremely rare classes
# -------------------------
class_counts = df["next_rank_encoded"].value_counts()
print(f"Original class distribution:\n{class_counts.sort_index()}")

# Keep classes with at least 10 samples
valid_classes = class_counts[class_counts >= 10].index
df = df[df["next_rank_encoded"].isin(valid_classes)]

print(f"\nClasses after filtering (>= 10 samples): {len(valid_classes)}")
print(f"Class distribution:\n{df['next_rank_encoded'].value_counts().sort_index()}")

X = df[feature_cols].fillna(0)
y = df["next_rank_encoded"]

# Split with larger test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

print(f"\nTraining set class distribution:\n{y_train.value_counts().sort_index()}")

# Feature scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# -------------------------
# Combined over/undersampling to reduce NG dominance
# -------------------------
def combined_sampling(X_train, y_train):
    """Apply combined over/under sampling"""
    class_counts = y_train.value_counts().sort_values()
    min_samples = class_counts.min()
    max_samples = class_counts.max()
    
    print(f"\nClass distribution before balancing:")
    print(f"  Min: {min_samples}, Max: {max_samples}, Ratio: {max_samples/min_samples:.1f}:1")
    print(f"  Class counts: {dict(class_counts)}")
    
    # Only balance if there's significant imbalance
    if max_samples / min_samples < 3:
        print("Classes are relatively balanced. Skipping resampling, using class weights.")
        return X_train, y_train, 'class_weight'
    
    # Step 1: Oversample minority classes
    oversample_strategy = {}
    
    for class_label, count in class_counts.items():
        if count < 20:
            oversample_strategy[class_label] = min(60, max_samples // 6)
        elif count < 150:
            target = int(max_samples * 0.3)
            if count < target:
                oversample_strategy[class_label] = target
    
    if not oversample_strategy:
        print("No resampling needed. Using class weights.")
        return X_train, y_train, 'class_weight'
    
    # Determine k_neighbors
    min_class_to_sample = min([class_counts[c] for c in oversample_strategy.keys()])
    k_neighbors = max(1, min(5, min_class_to_sample - 1))
    
    print(f"Step 1: Oversampling minorities with k_neighbors={k_neighbors}")
    print(f"  Oversample strategy: {oversample_strategy}")
    
    try:
        # Apply SMOTE
        smote = SMOTE(sampling_strategy=oversample_strategy, k_neighbors=k_neighbors, random_state=42)
        X_oversampled, y_oversampled = smote.fit_resample(X_train, y_train)
        
        # Step 2: Undersample majority class (NG)
        new_counts = pd.Series(y_oversampled).value_counts().sort_values()
        median_size = new_counts.median()
        max_majority_size = int(median_size * 2.0)  # NG can be 2x median
        
        undersample_strategy = {}
        for class_label, count in new_counts.items():
            if count > max_majority_size:
                undersample_strategy[class_label] = max_majority_size
        
        if undersample_strategy:
            print(f"Step 2: Undersampling majority class")
            print(f"  Undersample strategy: {undersample_strategy}")
            undersampler = RandomUnderSampler(sampling_strategy=undersample_strategy, random_state=42)
            X_balanced, y_balanced = undersampler.fit_resample(X_oversampled, y_oversampled)
        else:
            X_balanced, y_balanced = X_oversampled, y_oversampled
        
        print(f"Data resampled: {len(X_train)} -> {len(X_balanced)} samples")
        new_dist = pd.Series(y_balanced).value_counts().sort_index()
        print(f"Final class distribution:\n{new_dist}")
        return X_balanced, y_balanced, 'combined'
    except ValueError as e:
        print(f"Resampling failed: {e}. Using class weights instead.")
        return X_train, y_train, 'class_weight'

print("Balancing data...")
X_train_sm, y_train_sm, balance_method = combined_sampling(X_train_scaled, y_train)

# -------------------------
# Optimized Model Training
# -------------------------
def train_optimized_model(X_train, y_train, balance_method):
    """Train optimized RandomForest"""
    
    print("\nTraining optimized RandomForest...")
    
    # Use balanced weights only if we didn't do combined sampling
    use_class_weight = 'balanced' if balance_method == 'class_weight' else None
    
    model = RandomForestClassifier(
        n_estimators=300,
        max_depth=15,
        min_samples_split=8,
        min_samples_leaf=3,
        max_features='sqrt',
        bootstrap=True,
        oob_score=True,
        class_weight=use_class_weight,
        random_state=42,
        n_jobs=-1,
        verbose=0
    )
    
    # Cross-validation
    cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='accuracy')
    print(f"RandomForest CV Accuracy: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
    
    cv_f1_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='f1_weighted')
    print(f"RandomForest CV F1 (weighted): {cv_f1_scores.mean():.4f} (+/- {cv_f1_scores.std() * 2:.4f})")
    
    # Train final model
    model.fit(X_train, y_train)
    
    # Print feature importance
    if hasattr(model, 'feature_importances_'):
        print("\nTop 5 most important features:")
        feature_importance = pd.DataFrame({
            'feature': feature_cols,
            'importance': model.feature_importances_
        }).sort_values('importance', ascending=False)
        print(feature_importance.head(5).to_string(index=False))
    
    return model

# Train the model
best_model = train_optimized_model(X_train_sm, y_train_sm, balance_method)

# -------------------------
# Model Evaluation
# -------------------------
def evaluate_model(model, X_test, y_test, model_name):
    """Model evaluation with probability analysis"""
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)
    
    print(f"\n{model_name} Evaluation:")
    print("=" * 50)
    
    # Accuracy
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {accuracy:.4f}")
    
    # Get only the classes that are actually present
    present_classes = sorted(set(y_test) | set(y_pred))
    target_names = [int_to_rank[i] for i in present_classes]
    
    print(f"Classes in test set: {len(present_classes)}")
    
    # Analyze prediction confidence
    max_probas = y_pred_proba.max(axis=1)
    print(f"\nPrediction confidence:")
    print(f"  Mean: {max_probas.mean():.3f}")
    print(f"  Median: {np.median(max_probas):.3f}")
    print(f"  Low confidence (<0.5): {(max_probas < 0.5).sum()} samples")
    
    # Classification report
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, 
                              labels=present_classes,
                              target_names=target_names,
                              zero_division=0))
    
    # Better metrics for imbalanced data
    kappa = cohen_kappa_score(y_test, y_pred)
    mcc = matthews_corrcoef(y_test, y_pred)
    
    print(f"\nAdvanced Metrics:")
    print(f"  Cohen's Kappa: {kappa:.4f} (agreement beyond chance)")
    print(f"  Matthews Correlation: {mcc:.4f} (quality of binary classification)")
    
    # Per-class accuracy
    print(f"\nPer-Class Accuracy:")
    for class_label in present_classes:
        mask = y_test == class_label
        if mask.sum() > 0:
            class_acc = (y_pred[mask] == class_label).mean()
            rank_name = int_to_rank[class_label]
            print(f"  {rank_name}: {class_acc:.3f} ({mask.sum()} samples)")
    
    return accuracy

# Evaluate model
print("\nEvaluating model...")
final_accuracy = evaluate_model(best_model, X_test_scaled, y_test, "Optimized Model")

print(f"\nSelected Optimized Model with accuracy: {final_accuracy:.4f}")

# -------------------------
# Save assets
# -------------------------
joblib.dump(best_model, "model.pkl")
joblib.dump(category_encoder, "category_encoder.pkl")
joblib.dump(rank_to_int, "rank_to_int.pkl")
joblib.dump(int_to_rank, "int_to_rank.pkl")
joblib.dump(feature_cols, "feature_cols.pkl")
joblib.dump(ranking_order, "ranking_order.pkl")
joblib.dump(scaler, "scaler.pkl")

print("\nOPTIMIZED REGULAR MODEL TRAINED & SAVED")
print("Performance improvements:")
print("✓ Early class filtering (>= 10 samples)")
print("✓ Combined over/undersampling to reduce NG dominance")
print("✓ Advanced feature engineering (10 new features)")
print("✓ Opponent strength features (weighted performance)")
print("✓ Performance vs higher/lower ranks")
print("✓ Win/loss ratio")
print("✓ Feature scaling with StandardScaler")
print("✓ Better evaluation metrics (Kappa, MCC, per-class accuracy)")
print(f"✓ Final dataset shape: {df.shape}")
print(f"✓ Number of features: {len(feature_cols)}")
print(f"✓ Classes in training: {len(set(y_train_sm))}")
print(f"✓ Classes in test: {len(set(y_test))}")
print("✓ Optimized RandomForest parameters")
print("✓ Reduced feature dimensionality (focused selection)")
print(f"✓ Final dataset shape: {df.shape}")
print(f"✓ Number of features: {len(feature_cols)}")
print(f"✓ Classes in training: {len(set(y_train_sm))}")
print(f"✓ Classes in test: {len(set(y_test))}")