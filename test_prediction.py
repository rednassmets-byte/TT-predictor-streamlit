"""
Test if predictions work correctly
"""
import pandas as pd
import ast
import joblib
import numpy as np

print("=" * 80)
print("TESTING PREDICTION")
print("=" * 80)

# Load models
try:
    model = joblib.load("model_v3_improved.pkl")
    feature_cols = joblib.load("feature_cols_v3.pkl")
    rank_to_int = joblib.load("rank_to_int_v3.pkl")
    int_to_rank = joblib.load("int_to_rank_v3.pkl")
    print("✓ Loaded V3 model")
    print(f"✓ Feature columns: {len(feature_cols)}")
    print(f"✓ Features: {feature_cols}")
except Exception as e:
    print(f"✗ Error loading model: {e}")
    exit(1)

# Load test data
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

# Exclude youth
youth_categories = ["BEN", "PRE", "MIN", "CAD"]
df = df[~df['category'].isin(youth_categories)]

# Filter to players with 10+ matches
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

df = df[df['total_matches'] >= 10].head(5)

print(f"\n✓ Testing on {len(df)} players")

# Test prediction on first player
for idx, row in df.iterrows():
    print(f"\n{'='*80}")
    print(f"Player: {row['name']}")
    print(f"Current: {row['current_ranking']}")
    print(f"Actual next: {row['next_ranking']}")
    print(f"Matches: {int(row['total_matches'])}")
    
    # Create a simple feature vector with zeros
    test_features = pd.DataFrame([{col: 0 for col in feature_cols}])
    
    # Make prediction
    try:
        prediction = model.predict(test_features)[0]
        predicted_rank = int_to_rank.get(prediction, "Unknown")
        print(f"Predicted (with zeros): {predicted_rank}")
        
        if predicted_rank == "NG":
            print("⚠️  PROBLEM: Model predicts NG with zero features!")
            print("This means the model is not using the features correctly")
    except Exception as e:
        print(f"✗ Error predicting: {e}")
    
    break  # Just test first player

print("\n" + "=" * 80)
print("DIAGNOSIS")
print("=" * 80)

# Check what the model expects
print(f"\nModel expects {len(feature_cols)} features:")
for i, col in enumerate(feature_cols, 1):
    print(f"  {i}. {col}")

print("\nIf prediction is always NG, the issue is:")
print("  1. Features are not being calculated correctly in app.py")
print("  2. Feature names don't match between training and prediction")
print("  3. Features are all zeros (not being populated)")
