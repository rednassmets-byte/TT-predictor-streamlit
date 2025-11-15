
import pandas as pd
import ast
import joblib
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE

# =========================================
# 1) LOAD DATA
# =========================================
df = pd.read_csv("club_members_with_next_ranking.csv")

df["current_ranking"] = df["current_ranking"].apply(ast.literal_eval).str[0]
df["next_ranking"] = df["next_ranking"].apply(ast.literal_eval).str[0]
df["category"] = df["category"].apply(ast.literal_eval).str[0]
df["kaart"] = df["kaart"].apply(ast.literal_eval)


# =========================================
# 2) NORMALIZE RANKS → All ranks starting with "A" become "A"
# =========================================
def normalize_rank(r):
    if str(r).startswith("A"):
        return "A"
    return r

df["current_ranking"] = df["current_ranking"].apply(normalize_rank)
df["next_ranking"] = df["next_ranking"].apply(normalize_rank)


# Also normalize kaart keys
def normalize_kaart(kaart_dict):
    new_dict = {}
    for k, v in kaart_dict.items():
        key = normalize_rank(k)   # A1, A9, As3 → "A"
        if key not in new_dict:
            new_dict[key] = [0, 0]
        new_dict[key][0] += v[0]  # wins
        new_dict[key][1] += v[1]  # losses
    return new_dict

df["kaart"] = df["kaart"].apply(normalize_kaart)


# =========================================
# 3) NEW SIMPLIFIED RANK LIST
# =========================================
ranking_order = [
    "A",   # all A ranks merged
    "B0", "B0e", "B2", "B4", "B6",
    "C0", "C2", "C4", "C6",
    "D0", "D2", "D4", "D6",
    "E0", "E2", "E4", "E6",
    "NG"
]

rank_to_int = {r: i for i, r in enumerate(ranking_order)}
int_to_rank = {i: r for r, i in rank_to_int.items()}

df["current_rank_encoded"] = df["current_ranking"].map(rank_to_int)
df["next_rank_encoded"] = df["next_ranking"].map(rank_to_int)


# =========================================
# 4) EXPAND KAART FEATURES
# =========================================
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


# =========================================
# 5) FEATURES + TRAIN SPLIT
# =========================================
feature_cols = ["current_rank_encoded", "category_encoded"] + \
               [c for c in df.columns if c.endswith("_wins") or c.endswith("_losses")]

X = df[feature_cols].fillna(0)
y = df["next_rank_encoded"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15)


# =========================================
# 6) SMOTE (oversampling rare classes)
# =========================================
valid_classes = y_train.value_counts()[lambda x: x >= 3].index
mask = y_train.isin(valid_classes)

sm = SMOTE(k_neighbors=2)
X_train_sm, y_train_sm = sm.fit_resample(X_train[mask], y_train[mask])


# =========================================
# 7) TRAIN MODEL
# =========================================
model = RandomForestClassifier(
    n_estimators=400,
    max_depth=12,
    class_weight="balanced"
)

model.fit(X_train_sm, y_train_sm)


# =========================================
# 8) SAVE ALL FILES
# =========================================
joblib.dump(model, "model.pkl")
joblib.dump(category_encoder, "category_encoder.pkl")
joblib.dump(rank_to_int, "rank_to_int.pkl")
joblib.dump(int_to_rank, "int_to_rank.pkl")
joblib.dump(feature_cols, "feature_cols.pkl")
joblib.dump(ranking_order, "ranking_order.pkl")

print("MODEL TRAINED WITH MERGED 'A' RANKS & SAVED")
