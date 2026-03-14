import pandas as pd
import ast

# Read the data
df = pd.read_csv('club_members_with_next_ranking.csv')

# Define ranking order (lower index = better rank)
ranking_order = ['A', 'B0', 'B2', 'B4', 'B6', 'C0', 'C2', 'C4', 'C6', 
                 'D0', 'D2', 'D4', 'D6', 'E0', 'E2', 'E4', 'E6', 'NG']

def extract_rank(rank_str):
    """Extract rank from string format like ['B0']"""
    try:
        rank_list = ast.literal_eval(rank_str)
        return rank_list[0] if rank_list else None
    except:
        return None

def calculate_jump(current_rank, next_rank):
    """Calculate jump between ranks (positive = improvement, negative = decline)"""
    if current_rank not in ranking_order or next_rank not in ranking_order:
        return None
    current_idx = ranking_order.index(current_rank)
    next_idx = ranking_order.index(next_rank)
    return current_idx - next_idx  # Positive means improvement (moving up)

# Extract ranks
df['current_rank'] = df['current_ranking'].apply(extract_rank)
df['next_rank'] = df['next_ranking'].apply(extract_rank)

# Calculate jumps
df['jump'] = df.apply(lambda row: calculate_jump(row['current_rank'], row['next_rank']), axis=1)

# Filter out None values
df_valid = df[df['jump'].notna()].copy()

# Find biggest positive jump (improvement)
biggest_improvement = df_valid.loc[df_valid['jump'].idxmax()]

# Find biggest negative jump (decline)
biggest_decline = df_valid.loc[df_valid['jump'].idxmin()]

print("=" * 80)
print("BIGGEST IMPROVEMENT (Positive Jump)")
print("=" * 80)
print(f"Name: {biggest_improvement['name']}")
print(f"Season: {biggest_improvement['season']}")
print(f"Current Rank: {biggest_improvement['current_rank']}")
print(f"Next Rank: {biggest_improvement['next_rank']}")
print(f"Jump: {int(biggest_improvement['jump'])} levels UP")
print(f"Category: {biggest_improvement['category']}")
print(f"Club: {biggest_improvement['club_name']}")
print()

print("=" * 80)
print("BIGGEST DECLINE (Negative Jump)")
print("=" * 80)
print(f"Name: {biggest_decline['name']}")
print(f"Season: {biggest_decline['season']}")
print(f"Current Rank: {biggest_decline['current_rank']}")
print(f"Next Rank: {biggest_decline['next_rank']}")
print(f"Jump: {abs(int(biggest_decline['jump']))} levels DOWN")
print(f"Category: {biggest_decline['category']}")
print(f"Club: {biggest_decline['club_name']}")
print()

# Show top 10 improvements
print("=" * 80)
print("TOP 10 BIGGEST IMPROVEMENTS")
print("=" * 80)
top_improvements = df_valid.nlargest(10, 'jump')[['name', 'season', 'current_rank', 'next_rank', 'jump', 'category']]
for idx, row in top_improvements.iterrows():
    print(f"{row['name']:30} | {row['season']:10} | {row['current_rank']:3} → {row['next_rank']:3} | +{int(row['jump'])} levels | {row['category']}")
