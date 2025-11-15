import pandas as pd
import ast
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from imblearn.over_sampling import SMOTE
import numpy as np

# ---------------------------------------------------------
# 1. Load data
# ---------------------------------------------------------

df = pd.read_csv("club_members_with_next_ranking.csv")

# Convert stringified dict/list columns back to Python objects
df["current_ranking"] = df["current_ranking"].apply(ast.literal_eval)
df["next_ranking"] = df["next_ranking"].apply(ast.literal_eval)
df["category"] = df["category"].apply(ast.literal_eval)
df["unique_index"] = df["unique_index"].apply(ast.literal_eval)
df["kaart"] = df["kaart"].apply(ast.literal_eval)

# Extract single-element lists into values
df["current_ranking"] = df["current_ranking"].apply(lambda x: x[0] if isinstance(x, list) else x)
df["next_ranking"] = df["next_ranking"].apply(lambda x: x[0] if isinstance(x, list) else x)
df["category"] = df["category"].apply(lambda x: x[0] if isinstance(x, list) else x)
df["unique_index"] = df["unique_index"].apply(lambda x: x[0] if isinstance(x, list) else x)

# ---------------------------------------------------------
# 2. Define ranking order (A highest → NG lowest)
# ---------------------------------------------------------

ranking_order = [
    "A", "A1", "A2", "A3", "A5", "A6", "A7", "A8", "A9", "A10", "A11", "A12", "A13", "A14", "A15", "A16", "A17", "A18", "A19", "A20", "A21", "A22", "A27",
    "Ae3", "Ae5", "Ae12", "Ae14",
    "As1", "As2", "As3", "As4", "As5", "As6", "As7", "As8", "As9", "As10", "As15", "As16", "As18", "As19", "As23",
    "B0", "B0e", "B2", "B4", "B6",
    "C0", "C2", "C4", "C6",
    "D0", "D2", "D4", "D6",
    "E0", "E2", "E4", "E6",
    "NG"
]

rank_to_int = {rank: i for i, rank in enumerate(ranking_order)}
int_to_rank = {i: rank for rank, i in rank_to_int.items()}

df["current_rank_encoded"] = df["current_ranking"].map(rank_to_int)
df["next_rank_encoded"] = df["next_ranking"].map(rank_to_int)

# ---------------------------------------------------------
# 3. Expand kaart into ML features
# ---------------------------------------------------------

def extract_kaart_features(kaart_dict):
    features = {}
    for rank in ranking_order:
        wins, losses = kaart_dict.get(rank, [0, 0])
        features[f"{rank}_wins"] = wins
        features[f"{rank}_losses"] = losses
    return features

kaart_expanded = df["kaart"].apply(extract_kaart_features).apply(pd.Series)
df = pd.concat([df, kaart_expanded], axis=1)

# ---------------------------------------------------------
# 4. Encode category
# ---------------------------------------------------------

category_encoder = LabelEncoder()
df["category_encoded"] = category_encoder.fit_transform(df["category"])

# ---------------------------------------------------------
# 5. Prepare ML features and target
# ---------------------------------------------------------

feature_columns = (
    ["current_rank_encoded", "category_encoded"] +
    [col for col in df.columns if col.endswith("_wins") or col.endswith("_losses")]
)

X = df[feature_columns]
y = df["next_rank_encoded"]

# Handle NaN values in X and y
X = X.fillna(0)  # Fill NaN with 0 for features
y = y.dropna()   # Drop rows where target is NaN
X = X.loc[y.index]  # Align X with y after dropping

# ---------------------------------------------------------
# 6. Train/test split
# ---------------------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.15, random_state=42
)

# ---------------------------------------------------------
# 7. Handle class imbalance with SMOTE
# ---------------------------------------------------------

# Filter out classes with very few samples to avoid SMOTE issues
class_counts = y_train.value_counts()
min_samples = 3  # Minimum samples per class for SMOTE
valid_classes = class_counts[class_counts >= min_samples].index
mask = y_train.isin(valid_classes)
X_train_filtered = X_train[mask]
y_train_filtered = y_train[mask]

smote = SMOTE(random_state=42, k_neighbors=2)
X_train_sm, y_train_sm = smote.fit_resample(X_train_filtered, y_train_filtered)

print(f"Original training set size: {len(X_train)}")
print(f"SMOTE training set size: {len(X_train_sm)}")

# ---------------------------------------------------------
# 8. Train Random Forest
# ---------------------------------------------------------

model = RandomForestClassifier(
    n_estimators=400,
    max_depth=12,
    class_weight="balanced",   # prevents NG bias
    random_state=42
)

model.fit(X_train_sm, y_train_sm)

# ---------------------------------------------------------
# 8. Evaluation
# ---------------------------------------------------------

preds = model.predict(X_test)

print("CLASSIFICATION REPORT:\n")
print(classification_report(
    y_test,
    preds,
    labels=sorted(int_to_rank.keys()),
    target_names=[int_to_rank[i] for i in sorted(int_to_rank)]
))

# ---------------------------------------------------------
# 9. Prediction function
# ---------------------------------------------------------

def predict_next_ranking(player_row):
    """
    player_row: a single row from df (df.loc[i])
    returns: (predicted_rank, probability_distribution)
    """
    X_input = player_row[feature_columns].values.reshape(1, -1)
    probas = model.predict_proba(X_input)[0]
    best_idx = np.argmax(probas)
    
    predicted_rank = int_to_rank[best_idx]
    probability_distribution = {
        int_to_rank[i]: float(probas[i]) for i in range(len(probas))
    }

    return predicted_rank, probability_distribution

# Example usage:


def prepare_new_player(current_ranking, category, kaart_dict):
    """
    Creates a dataframe row identical to training structure.
    
    current_ranking: string (e.g., "C2")
    category: string (e.g., "SEN")
    kaart_dict: dict, e.g. {"B4": [12,3], "B6": [5,7], "NG":[0,0], ...}
    """

    # Encode ranking
    current_rank_encoded = rank_to_int.get(current_ranking, None)
    if current_rank_encoded is None:
        raise ValueError(f"Unknown ranking: {current_ranking}")

    # Encode category
    category_encoded = category_encoder.transform([category])[0]

    # Expand kaart
    kaart_features = {}
    for rank in ranking_order:
        wins, losses = kaart_dict.get(rank, [0, 0])
        kaart_features[f"{rank}_wins"] = wins
        kaart_features[f"{rank}_losses"] = losses

    # Build row
    row = {
        "current_rank_encoded": current_rank_encoded,
        "category_encoded": category_encoded,
        **kaart_features
    }

    # Convert to DataFrame with correct order
    row_df = pd.DataFrame([row])[feature_columns]

    return row_df

new_player = prepare_new_player(
    current_ranking="E2",
    category="JUN",
    kaart_dict={ 'D6': [0, 5], 'E0': [5, 3], 'E2': [17, 6], 'E4': [23, 10], 'NG': [1, 1]
        
    }
)

# Predict
pred = model.predict(new_player)[0]
predicted_rank = int_to_rank[pred]

# Probability distribution
probas = model.predict_proba(new_player)[0]
probability_distribution = {
    int_to_rank[i]: float(probas[i]) for i in range(len(probas))
}

print("Predicted rank:", predicted_rank)
print("Probabilities:", probability_distribution)


