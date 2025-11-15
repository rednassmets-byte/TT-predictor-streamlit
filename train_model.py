import pandas as pd
import ast
import joblib
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE

# -------------------------
# Load data
# -------------------------
df = pd.read_csv("club_members_with_next_ranking.csv")

df["current_ranking"] = df["current_ranking"].apply(ast.literal_eval).str[0]
df["next_ranking"] = df["next_ranking"].apply(ast.literal_eval).str[0]
df["category"] = df["category"].apply(ast.literal_eval).str[0]
df["kaart"] = df["kaart"].apply(ast.literal_eval)

ranking_order = [
    "A", "A1", "A2", "A3", "A5", "A6", "A7", "A8", "A9", "A10", "A11", "A12", "A13", "A14", "A15", "A16", "A17", "A18",
    "A19", "A20", "A21", "A22", "A27",
    "Ae3", "Ae5", "Ae12", "Ae14",
    "As1", "As2", "As3", "As4", "As5", "As6", "As7", "As8", "As9", "As10", "As15", "As16", "As18", "As19", "As23",
    "B0", "B0e", "B2", "B4", "B6",
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
# Expand kaart
# -------------------------
def extract_kaart_features(kaart):
    f = {}
    for rank in ranking_order:
        wins, losses = kaart.get(rank, [0, 0])
        f[f"{rank}_wins"] = wins
        f[f"{rank}_losses"] = losses
    return f

df = pd.concat([df, df["kaart"].apply(extract_kaart_features).apply(pd.Series)], axis=1)

category_encoder = LabelEncoder()
df["category_encoded"] = category_encoder.fit_transform(df["category"])

feature_cols = ["current_rank_encoded", "category_encoded"] + \
               [c for c in df.columns if c.endswith("_wins") or c.endswith("_losses")]

X = df[feature_cols].fillna(0)
y = df["next_rank_encoded"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15)

# Filter low-frequency classes for SMOTE
valid_classes = y_train.value_counts()[lambda x: x >= 3].index
mask = y_train.isin(valid_classes)

sm = SMOTE(k_neighbors=2)
X_train_sm, y_train_sm = sm.fit_resample(X_train[mask], y_train[mask])

# -------------------------
# Train model
# -------------------------
model = RandomForestClassifier(
    n_estimators=400,
    max_depth=12,
    class_weight="balanced"
)

model.fit(X_train_sm, y_train_sm)

# -------------------------
# Save assets
# -------------------------
joblib.dump(model, "model.pkl")
joblib.dump(category_encoder, "category_encoder.pkl")
joblib.dump(rank_to_int, "rank_to_int.pkl")
joblib.dump(int_to_rank, "int_to_rank.pkl")
joblib.dump(feature_cols, "feature_cols.pkl")
joblib.dump(ranking_order, "ranking_order.pkl")

print("MODEL TRAINED & SAVED")
