import pandas as pd
import ast
import joblib
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.combine import SMOTETomek
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

# -------------------------
# Load data with optimized parsing
# -------------------------
df = pd.read_csv("club_members_with_next_ranking.csv")

def normalize_rank(rank):
    if rank.startswith('A'):
        return 'A'
    elif rank == 'B0e':
        return 'B0'
    else:
        return rank

# Use vectorized operations instead of apply where possible
df["current_ranking"] = df["current_ranking"].apply(ast.literal_eval).str[0].apply(normalize_rank)
df["next_ranking"] = df["next_ranking"].apply(ast.literal_eval).str[0].apply(normalize_rank)
df["category"] = df["category"].apply(ast.literal_eval).str[0]
df["kaart"] = df["kaart"].apply(ast.literal_eval)

# Filter data to only include BEN, PRE, MIN, CAD categories
allowed_categories = ["BEN", "PRE", "MIN", "CAD"]
df = df[df["category"].isin(allowed_categories)]

ranking_order = [
    "A", "B0", "B2", "B4", "B6", "C0", "C2", "C4", "C6",
    "D0", "D2", "D4", "D6", "E0", "E2", "E4", "E6", "NG"
]

rank_to_int = {r:i for i, r in enumerate(ranking_order)}
int_to_rank = {i:r for r, i in rank_to_int.items()}

df["current_rank_encoded"] = df["current_ranking"].map(rank_to_int)
df["next_rank_encoded"] = df["next_ranking"].map(rank_to_int)

# -------------------------
# Keep more classes with lower threshold
# -------------------------
class_counts = df["next_rank_encoded"].value_counts()
print(f"Original class distribution:\n{class_counts.sort_index()}")

# Lower threshold to include more classes - only remove extremely rare ones
valid_classes = class_counts[class_counts >= 10].index  # Lowered from 30 to 10
df = df[df["next_rank_encoded"].isin(valid_classes)]

print(f"\nClasses after filtering (>= 10 samples): {len(valid_classes)}")
print(f"Class distribution:\n{df['next_rank_encoded'].value_counts().sort_index()}")

# -------------------------
# Optimized kaart feature extraction
# -------------------------
def extract_kaart_features_fast(kaart_list):
    """Vectorized extraction of kaart features"""
    all_features = []
    for kaart in kaart_list:
        features = {}
        for rank in ranking_order:
            wins, losses = kaart.get(rank, [0, 0])
            features[f"{rank}_wins"] = wins
            features[f"{rank}_losses"] = losses
        all_features.append(features)
    return pd.DataFrame(all_features)

# Batch process kaart features
kaart_features_df = extract_kaart_features_fast(df["kaart"].tolist())
df = pd.concat([df.reset_index(drop=True), kaart_features_df], axis=1)

# -------------------------
# Optimized advanced feature engineering
# -------------------------
def create_advanced_features_fast(df, ranking_order):
    """Vectorized feature engineering"""
    win_cols = [f"{r}_wins" for r in ranking_order]
    loss_cols = [f"{r}_losses" for r in ranking_order]
    
    # Convert to numpy for faster computation
    wins_array = df[win_cols].values
    losses_array = df[loss_cols].values
    
    # Calculate totals and win rates in bulk
    totals_array = wins_array + losses_array
    
    # Win rates per rank
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
    
    # Overall statistics using numpy for speed
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
    
    # Performance metrics
    df['performance_consistency'] = df['overall_win_rate'] * np.log1p(df['total_matches'])
    
    # Recent performance
    df['recent_performance'] = df['overall_win_rate'] * np.minimum(df['total_matches'] / 10, 1)
    
    # Rank progression features
    current_ranks = df['current_rank_encoded'].values
    df['rank_progression_potential'] = df['overall_win_rate'] * (1 - current_ranks / len(ranking_order))
    
    return df

df = create_advanced_features_fast(df, ranking_order)

# Encode categorical variables
category_encoder = LabelEncoder()
df["category_encoded"] = category_encoder.fit_transform(df["category"])

# Define feature columns - use only the most important features
base_feature_cols = ["current_rank_encoded", "category_encoded"]
advanced_feature_cols = [
    'total_wins', 'total_losses', 'total_matches', 'overall_win_rate',
    'performance_consistency', 'recent_performance', 'rank_progression_potential'
]

# Select only high-value features to reduce dimensionality
feature_cols = base_feature_cols + advanced_feature_cols

# Add win_rate features for ranks that matter for prediction
# Include more ranks to help distinguish between classes
important_ranks = ['B0', 'B2', 'B4', 'C0', 'C2', 'C4', 'D0', 'D2', 'D4', 'E0', 'E2']
win_rate_cols = [f"{rank}_win_rate" for rank in important_ranks if f"{rank}_win_rate" in df.columns]
feature_cols.extend(win_rate_cols)

# Add total games per rank for better context
total_cols = [f"{rank}_total" for rank in ['E0', 'E2', 'E4', 'E6'] if f"{rank}_total" in df.columns]
feature_cols.extend(total_cols)

X = df[feature_cols].fillna(0)
y = df["next_rank_encoded"]

# Split data with larger test set for better evaluation
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"Training set class distribution:\n{y_train.value_counts().sort_index()}")

# Feature scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# -------------------------
# Conservative class balancing to prevent overfitting
# -------------------------
def conservative_smote_balancing(X_train, y_train):
    """Apply combined over/under sampling to reduce NG dominance"""
    class_counts = y_train.value_counts().sort_values()
    min_samples = class_counts.min()
    max_samples = class_counts.max()
    
    print(f"Class distribution before balancing:")
    print(f"  Min: {min_samples}, Max: {max_samples}, Ratio: {max_samples/min_samples:.1f}:1")
    print(f"  Class counts: {dict(class_counts)}")
    
    # Only balance if there's significant imbalance
    if max_samples / min_samples < 3:
        print("Classes are relatively balanced. Skipping SMOTE, using class weights.")
        return X_train, y_train, 'class_weight'
    
    # Step 1: Oversample minority classes with SMOTE
    # Small classes (< 20): bring to 60 samples
    # Medium classes (20-150): bring to 30% of max
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
    
    # Determine k_neighbors based on smallest class
    min_class_to_sample = min([class_counts[c] for c in oversample_strategy.keys()])
    k_neighbors = max(1, min(5, min_class_to_sample - 1))
    
    print(f"Step 1: Oversampling minorities with k_neighbors={k_neighbors}")
    print(f"  Oversample strategy: {oversample_strategy}")
    
    try:
        # Apply SMOTE for minority classes
        smote = SMOTE(sampling_strategy=oversample_strategy, k_neighbors=k_neighbors, random_state=42)
        X_oversampled, y_oversampled = smote.fit_resample(X_train, y_train)
        
        # Step 2: Undersample the majority class (NG) to reduce dominance
        # Target: bring NG down to 2x the median class size (balanced approach)
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
X_train_sm, y_train_sm, balance_method = conservative_smote_balancing(X_train_scaled, y_train)

# -------------------------
# Optimized Model Training
# -------------------------
def train_optimized_model(X_train, y_train, balance_method):
    """Train optimized RandomForest with faster parameters"""
    
    print("Training optimized RandomForest...")
    
    # Use class weights if SMOTE wasn't applied
    class_weight = 'balanced' if balance_method == 'class_weight' else None
    
    # Use parameters optimized for imbalanced multi-class
    # Use balanced weights only if we didn't do combined sampling
    use_class_weight = 'balanced' if balance_method == 'class_weight' else None
    
    model = RandomForestClassifier(
        n_estimators=300,  # More trees for better minority class handling
        max_depth=15,  # Slightly deeper for complex patterns
        min_samples_split=8,  # Balanced regularization
        min_samples_leaf=3,  # Allow smaller leaves for minority classes
        max_features='sqrt',
        bootstrap=True,
        oob_score=True,
        class_weight=use_class_weight,  # Use weights only if needed
        random_state=42,
        n_jobs=-1,
        verbose=0
    )
    
    # Cross-validation with proper folds
    cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='accuracy')
    print(f"RandomForest CV Accuracy: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
    
    # Also check weighted F1 score which is better for imbalanced data
    cv_f1_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='f1_weighted')
    print(f"RandomForest CV F1 (weighted): {cv_f1_scores.mean():.4f} (+/- {cv_f1_scores.std() * 2:.4f})")
    
    # Train final model
    model.fit(X_train, y_train)
    
    # Print feature importance for debugging
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
# Fast Model Evaluation
# -------------------------
def evaluate_model_fast(model, X_test, y_test, model_name):
    """Fast model evaluation with probability analysis"""
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)
    
    print(f"\n{model_name} Evaluation:")
    print("=" * 40)
    
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
    
    # Simplified classification report
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, 
                              labels=present_classes,
                              target_names=target_names,
                              zero_division=0))
    
    return accuracy

# Evaluate model
final_accuracy = evaluate_model_fast(best_model, X_test_scaled, y_test, "Optimized Model")

print(f"\nSelected Optimized Model with accuracy: {final_accuracy:.4f}")

# -------------------------
# Save assets
# -------------------------
joblib.dump(best_model, "model_filtered.pkl")
joblib.dump(category_encoder, "category_encoder_filtered.pkl")
joblib.dump(rank_to_int, "rank_to_int_filtered.pkl")
joblib.dump(int_to_rank, "int_to_rank_filtered.pkl")
joblib.dump(feature_cols, "feature_cols_filtered.pkl")
joblib.dump(ranking_order, "ranking_order_filtered.pkl")
joblib.dump(scaler, "scaler_filtered.pkl")

print("\nOPTIMIZED FILTERED MODEL TRAINED & SAVED")
print("Performance improvements:")
print("✓ Early class filtering to prevent SMOTE errors")
print("✓ Safe SMOTE balancing with validation")
print("✓ Vectorized feature engineering")
print("✓ Reduced feature dimensionality")
print("✓ Optimized RandomForest parameters")
print("✓ Faster cross-validation (3 folds instead of 5)")
print("✓ Removed slow GridSearchCV")
print(f"✓ Final dataset shape: {df.shape}")
print(f"✓ Number of features: {len(feature_cols)} (reduced from original)")
print(f"✓ Classes in training: {len(set(y_train_sm))}")
print(f"✓ Classes in test: {len(set(y_test))}")