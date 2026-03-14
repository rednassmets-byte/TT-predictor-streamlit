"""
Neural Network Model for Ranking Prediction
Uses TensorFlow/Keras to create a deep learning model
"""
import pandas as pd
import ast
import joblib
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, callbacks
import warnings
warnings.filterwarnings('ignore')

print("=" * 80)
print("NEURAL NETWORK RANKING PREDICTION MODEL")
print("=" * 80)

# Set random seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

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

# Exclude youth categories (same as v3 model)
youth_categories = ["BEN", "PRE", "MIN", "CAD"]
df = df[~df['category'].isin(youth_categories)]

print(f"Training on {len(df)} players (excluding youth categories)")

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

# ===== FEATURE ENGINEERING (same as v3) =====

# Basic stats
total_wins = []
total_losses = []
for kaart in df["kaart"]:
    wins = sum(w for w, l in kaart.values())
    losses = sum(l for w, l in kaart.values())
    total_wins.append(wins)
    total_losses.append(losses)

df['total_wins'] = total_wins
df['total_losses'] = total_losses
df['total_matches'] = df['total_wins'] + df['total_losses']
df['win_rate'] = df['total_wins'] / df['total_matches'].replace(0, 1)

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

df['nearby_win_rate'] = df.apply(
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

df['vs_better_win_rate'] = df.apply(
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

df['vs_worse_win_rate'] = df.apply(
    lambda row: get_performance_vs_worse(row['kaart'], row['current_rank_encoded'], ranking_order, rank_to_int),
    axis=1
)

df['match_volume'] = np.log1p(df['total_matches'])
df['win_loss_ratio'] = np.where(df['total_losses'] > 0, df['total_wins'] / df['total_losses'], df['total_wins'])
df['performance_consistency'] = df['nearby_win_rate'] * np.log1p(df['total_matches'])
df['level_dominance'] = (df['nearby_win_rate'] - 0.5) * 2
df['performance_score'] = (df['win_rate'] * 0.4 + df['nearby_win_rate'] * 0.4 + df['vs_better_win_rate'] * 0.2)

# V3 features
df['win_rate_capped'] = df['win_rate'].clip(0.1, 0.9)
df['nearby_win_rate_capped'] = df['nearby_win_rate'].clip(0.1, 0.9)

df['improvement_signal'] = (
    (df['nearby_win_rate'] - 0.6).clip(0, 1) * 2 +
    df['vs_better_win_rate'] * 1.5 +
    (df['vs_worse_win_rate'] - 0.7).clip(0, 1)
)

df['decline_signal'] = (
    (0.4 - df['nearby_win_rate']).clip(0, 1) * 2 +
    (0.5 - df['vs_worse_win_rate']).clip(0, 1) * 1.5
)

df['is_junior'] = df['category'].isin(['JUN', 'J19', 'J21']).astype(int)
df['junior_volatility'] = df['is_junior'] * df['improvement_signal']
df['in_de_zone'] = df['current_rank_encoded'].isin([9, 10, 11, 12, 13, 14, 15, 16]).astype(int)
df['match_reliability'] = np.tanh(df['total_matches'] / 30)
df['reliable_performance'] = df['nearby_win_rate_capped'] * df['match_reliability']
df['is_low_rank'] = (df['current_rank_encoded'] >= 13).astype(int)
df['is_mid_rank'] = ((df['current_rank_encoded'] >= 9) & (df['current_rank_encoded'] < 13)).astype(int)

# Encode category
category_encoder = LabelEncoder()
df['category_encoded'] = category_encoder.fit_transform(df['category'])

# Select features
feature_cols = [
    'current_rank_encoded', 'category_encoded',
    'win_rate', 'nearby_win_rate', 'vs_better_win_rate', 'vs_worse_win_rate',
    'total_matches', 'match_volume', 'performance_score',
    'win_loss_ratio', 'performance_consistency', 'level_dominance',
    'win_rate_capped', 'nearby_win_rate_capped',
    'improvement_signal', 'decline_signal',
    'is_junior', 'junior_volatility', 'in_de_zone',
    'match_reliability', 'reliable_performance',
    'is_low_rank', 'is_mid_rank'
]

X = df[feature_cols]
y = df['next_rank_encoded']

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

print(f"\nTraining set: {len(X_train)} samples")
print(f"Test set: {len(X_test)} samples")
print(f"Number of features: {len(feature_cols)}")
print(f"Number of classes: {len(ranking_order)}")

# Scale features (important for neural networks)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ===== BUILD NEURAL NETWORK =====

print("\n" + "=" * 80)
print("BUILDING NEURAL NETWORK")
print("=" * 80)

num_classes = len(ranking_order)
input_dim = len(feature_cols)

# Create model with dropout and batch normalization for better generalization
model = keras.Sequential([
    layers.Input(shape=(input_dim,)),
    
    # First hidden layer
    layers.Dense(128, activation='relu', kernel_regularizer=keras.regularizers.l2(0.001)),
    layers.BatchNormalization(),
    layers.Dropout(0.3),
    
    # Second hidden layer
    layers.Dense(64, activation='relu', kernel_regularizer=keras.regularizers.l2(0.001)),
    layers.BatchNormalization(),
    layers.Dropout(0.3),
    
    # Third hidden layer
    layers.Dense(32, activation='relu', kernel_regularizer=keras.regularizers.l2(0.001)),
    layers.BatchNormalization(),
    layers.Dropout(0.2),
    
    # Output layer
    layers.Dense(num_classes, activation='softmax')
])

# Compile model
model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=0.001),
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

print("\nModel Architecture:")
model.summary()

# Callbacks
early_stopping = callbacks.EarlyStopping(
    monitor='val_loss',
    patience=15,
    restore_best_weights=True,
    verbose=1
)

reduce_lr = callbacks.ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.5,
    patience=5,
    min_lr=0.00001,
    verbose=1
)

# Train model
print("\n" + "=" * 80)
print("TRAINING NEURAL NETWORK")
print("=" * 80)

history = model.fit(
    X_train_scaled, y_train,
    validation_split=0.2,
    epochs=100,
    batch_size=32,
    callbacks=[early_stopping, reduce_lr],
    verbose=1
)

# ===== EVALUATE MODEL =====

print("\n" + "=" * 80)
print("EVALUATION")
print("=" * 80)

# Predictions
y_pred = model.predict(X_test_scaled)
y_pred_classes = np.argmax(y_pred, axis=1)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred_classes)
print(f"\nTest Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")

# Detailed classification report
print("\nClassification Report:")
print(classification_report(y_test, y_pred_classes, 
                          target_names=[int_to_rank[i] for i in range(num_classes)],
                          zero_division=0))

# Analyze predictions
print("\n" + "=" * 80)
print("PREDICTION ANALYSIS")
print("=" * 80)

correct = (y_pred_classes == y_test).sum()
off_by_one = (np.abs(y_pred_classes - y_test) == 1).sum()
off_by_two = (np.abs(y_pred_classes - y_test) == 2).sum()

print(f"Exact predictions: {correct}/{len(y_test)} ({correct/len(y_test)*100:.2f}%)")
print(f"Off by 1 rank: {off_by_one}/{len(y_test)} ({off_by_one/len(y_test)*100:.2f}%)")
print(f"Off by 2 ranks: {off_by_two}/{len(y_test)} ({off_by_two/len(y_test)*100:.2f}%)")
print(f"Within 1 rank: {(correct + off_by_one)/len(y_test)*100:.2f}%")
print(f"Within 2 ranks: {(correct + off_by_one + off_by_two)/len(y_test)*100:.2f}%")

# ===== SAVE MODEL =====

print("\n" + "=" * 80)
print("SAVING MODEL")
print("=" * 80)

# Save Keras model
model.save("model_neural_network.keras")
print("✓ Saved: model_neural_network.keras")

# Save supporting files
joblib.dump(scaler, "scaler_neural_network.pkl")
print("✓ Saved: scaler_neural_network.pkl")

joblib.dump(category_encoder, "category_encoder_neural_network.pkl")
print("✓ Saved: category_encoder_neural_network.pkl")

joblib.dump(feature_cols, "feature_cols_neural_network.pkl")
print("✓ Saved: feature_cols_neural_network.pkl")

joblib.dump(int_to_rank, "int_to_rank_neural_network.pkl")
print("✓ Saved: int_to_rank_neural_network.pkl")

joblib.dump(rank_to_int, "rank_to_int_neural_network.pkl")
print("✓ Saved: rank_to_int_neural_network.pkl")

joblib.dump(ranking_order, "ranking_order_neural_network.pkl")
print("✓ Saved: ranking_order_neural_network.pkl")

print("\n" + "=" * 80)
print("NEURAL NETWORK MODEL TRAINING COMPLETE!")
print("=" * 80)
