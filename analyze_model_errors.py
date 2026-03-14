import pandas as pd
import numpy as np
import joblib
import ast
from collections import Counter

# Load the data
df = pd.read_csv('club_members_with_next_ranking.csv')

# Define ranking order
ranking_order = ['A', 'B0', 'B2', 'B4', 'B6', 'C0', 'C2', 'C4', 'C6', 
                 'D0', 'D2', 'D4', 'D6', 'E0', 'E2', 'E4', 'E6', 'NG']

def extract_rank(rank_str):
    try:
        rank_list = ast.literal_eval(rank_str)
        return rank_list[0] if rank_list else None
    except:
        return None

def parse_kaart(kaart_str):
    try:
        kaart_dict = ast.literal_eval(kaart_str)
        features = {}
        for rank in ranking_order:
            if rank in kaart_dict:
                features[f'{rank}_wins'] = kaart_dict[rank][0]
                features[f'{rank}_losses'] = kaart_dict[rank][1]
            else:
                features[f'{rank}_wins'] = 0
                features[f'{rank}_losses'] = 0
        return features
    except:
        return {f'{rank}_wins': 0 for rank in ranking_order} | {f'{rank}_losses': 0 for rank in ranking_order}

# Extract ranks
df['current_rank'] = df['current_ranking'].apply(extract_rank)
df['next_rank'] = df['next_ranking'].apply(extract_rank)

# Filter valid data
df_valid = df[(df['current_rank'].notna()) & (df['next_rank'].notna())].copy()

# Parse kaart features
kaart_features = df_valid['kaart'].apply(parse_kaart)
kaart_df = pd.DataFrame(kaart_features.tolist())

# Prepare features
X = kaart_df
y = df_valid['next_rank']

# Load the trained models
try:
    model = joblib.load('model_v2.pkl')
    print("✓ Loaded main model (model_v2.pkl)")
except:
    try:
        model = joblib.load('model.pkl')
        print("✓ Loaded main model (model.pkl)")
    except:
        print("✗ Could not load any model")
        model = None

try:
    model_filtered = joblib.load('model_filtered_v2.pkl')
    print("✓ Loaded filtered model (model_filtered_v2.pkl)")
except:
    try:
        model_filtered = joblib.load('model_filtered.pkl')
        print("✓ Loaded filtered model (model_filtered.pkl)")
    except:
        print("✗ Could not load filtered model")
        model_filtered = None

print("\n" + "="*80)
print("ERROR ANALYSIS")
print("="*80)

# Analyze each model
for model_name, model_obj in [("Main Model", model), ("Filtered Model", model_filtered)]:
    if model_obj is None:
        continue
    
    print(f"\n{'='*80}")
    print(f"{model_name.upper()} - DETAILED ERROR ANALYSIS")
    print(f"{'='*80}")
    
    # Make predictions
    y_pred = model_obj.predict(X)
    
    # Calculate errors
    df_valid['predicted'] = y_pred
    df_valid['correct'] = df_valid['next_rank'] == df_valid['predicted']
    
    # Calculate rank distance for errors
    def rank_distance(actual, predicted):
        if actual not in ranking_order or predicted not in ranking_order:
            return None
        return abs(ranking_order.index(actual) - ranking_order.index(predicted))
    
    df_valid['error_distance'] = df_valid.apply(
        lambda row: rank_distance(row['next_rank'], row['predicted']), axis=1
    )
    
    # Overall accuracy
    accuracy = df_valid['correct'].mean()
    print(f"\nOverall Accuracy: {accuracy:.2%}")
    
    # Errors by current rank
    print(f"\n{'─'*80}")
    print("ERRORS BY CURRENT RANK")
    print(f"{'─'*80}")
    errors_by_rank = df_valid.groupby('current_rank').agg({
        'correct': ['count', 'sum', 'mean']
    }).round(3)
    errors_by_rank.columns = ['Total', 'Correct', 'Accuracy']
    errors_by_rank['Error_Rate'] = 1 - errors_by_rank['Accuracy']
    errors_by_rank = errors_by_rank.sort_values('Error_Rate', ascending=False)
    print(errors_by_rank.head(10))
    
    # Most common prediction errors
    print(f"\n{'─'*80}")
    print("TOP 10 MOST COMMON PREDICTION ERRORS")
    print(f"{'─'*80}")
    errors_df = df_valid[~df_valid['correct']].copy()
    error_patterns = errors_df.groupby(['current_rank', 'next_rank', 'predicted']).size().reset_index(name='count')
    error_patterns = error_patterns.sort_values('count', ascending=False).head(10)
    print(f"{'Current':<8} {'Actual':<8} {'Predicted':<10} {'Count':<8}")
    print("─"*40)
    for _, row in error_patterns.iterrows():
        print(f"{row['current_rank']:<8} {row['next_rank']:<8} {row['predicted']:<10} {int(row['count']):<8}")
    
    # Average error distance
    print(f"\n{'─'*80}")
    print("ERROR MAGNITUDE ANALYSIS")
    print(f"{'─'*80}")
    avg_error_distance = df_valid[~df_valid['correct']]['error_distance'].mean()
    print(f"Average error distance (ranks off): {avg_error_distance:.2f}")
    
    error_dist_counts = df_valid[~df_valid['correct']]['error_distance'].value_counts().sort_index()
    print("\nError distance distribution:")
    for dist, count in error_dist_counts.items():
        print(f"  {int(dist)} rank(s) off: {count} errors ({count/len(errors_df)*100:.1f}%)")
    
    # Category-specific errors
    print(f"\n{'─'*80}")
    print("ERRORS BY PLAYER CATEGORY")
    print(f"{'─'*80}")
    df_valid['category_clean'] = df_valid['category'].apply(lambda x: ast.literal_eval(x)[0] if isinstance(x, str) else x)
    category_errors = df_valid.groupby('category_clean').agg({
        'correct': ['count', 'mean']
    }).round(3)
    category_errors.columns = ['Total', 'Accuracy']
    category_errors['Error_Rate'] = 1 - category_errors['Accuracy']
    category_errors = category_errors.sort_values('Error_Rate', ascending=False)
    print(category_errors)
    
    # Specific problematic cases
    print(f"\n{'─'*80}")
    print("WORST PREDICTIONS (Largest Errors)")
    print(f"{'─'*80}")
    worst_errors = df_valid[~df_valid['correct']].nlargest(5, 'error_distance')
    for idx, row in worst_errors.iterrows():
        print(f"\nPlayer: {row['name']}")
        print(f"  Season: {row['season']}")
        print(f"  Current: {row['current_rank']} → Actual: {row['next_rank']} | Predicted: {row['predicted']}")
        print(f"  Error distance: {int(row['error_distance'])} ranks")
        print(f"  Category: {row['category_clean']}")

print("\n" + "="*80)
print("KEY INSIGHTS")
print("="*80)

# Compare models
if model and model_filtered:
    print("\nModel Comparison:")
    main_pred = model.predict(X)
    filtered_pred = model_filtered.predict(X)
    
    main_acc = (main_pred == y).mean()
    filtered_acc = (filtered_pred == y).mean()
    
    print(f"  Main Model: {main_acc:.2%}")
    print(f"  Filtered Model: {filtered_acc:.2%}")
