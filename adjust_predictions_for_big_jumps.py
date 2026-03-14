"""
Post-processing adjustment for V3 predictions
Only boost predictions for players with strong signals for big improvements
Keep everything else the same
"""
import numpy as np

def should_boost_prediction(player_data, current_rank_encoded, predicted_rank_encoded, ranking_order):
    """
    Determine if a player should get a boosted prediction for big improvement
    
    Criteria for boosting:
    1. Strong overall performance (>70% win rate)
    2. Beating better players (>35% win rate vs better)
    3. Dominating current level (>75% nearby win rate)
    4. Sufficient matches (20+)
    5. Model already predicts improvement (not stable or decline)
    """
    
    # Extract features
    win_rate = player_data.get('win_rate', 0)
    nearby_win_rate = player_data.get('nearby_win_rate', 0)
    vs_better_win_rate = player_data.get('vs_better_win_rate', 0)
    total_matches = player_data.get('total_matches', 0)
    
    # Model must already predict improvement
    predicted_change = current_rank_encoded - predicted_rank_encoded
    if predicted_change <= 0:
        return False  # Not predicting improvement, don't boost
    
    # Check for strong signals
    strong_overall = win_rate > 0.70
    beating_better = vs_better_win_rate > 0.35
    dominating_level = nearby_win_rate > 0.75
    enough_matches = total_matches >= 20
    
    # Need at least 3 of 4 strong signals
    signals = sum([strong_overall, beating_better, dominating_level, enough_matches])
    
    return signals >= 3


def adjust_prediction_for_big_jump(current_rank, predicted_rank, player_data, ranking_order):
    """
    Adjust prediction to be more optimistic for big improvements
    Only boosts if strong signals present
    
    Args:
        current_rank: Current rank (e.g., 'E4')
        predicted_rank: V3's prediction (e.g., 'E2')
        player_data: Dictionary with player features
        ranking_order: List of ranks in order
    
    Returns:
        adjusted_rank: Boosted prediction if criteria met, otherwise original
        was_boosted: Boolean indicating if prediction was adjusted
    """
    
    rank_to_int = {r: i for i, r in enumerate(ranking_order)}
    int_to_rank = {i: r for r, i in rank_to_int.items()}
    
    current_rank_encoded = rank_to_int.get(current_rank)
    predicted_rank_encoded = rank_to_int.get(predicted_rank)
    
    if current_rank_encoded is None or predicted_rank_encoded is None:
        return predicted_rank, False
    
    # Check if should boost
    if should_boost_prediction(player_data, current_rank_encoded, predicted_rank_encoded, ranking_order):
        # Boost by 1 additional rank
        boosted_rank_encoded = max(0, predicted_rank_encoded - 1)  # Lower index = better rank
        boosted_rank = int_to_rank[boosted_rank_encoded]
        return boosted_rank, True
    
    return predicted_rank, False


# Example usage
if __name__ == "__main__":
    ranking_order = [
        "A", "B0", "B2", "B4", "B6",
        "C0", "C2", "C4", "C6",
        "D0", "D2", "D4", "D6",
        "E0", "E2", "E4", "E6",
        "NG"
    ]
    
    # Example 1: Strong player - should boost
    player1 = {
        'win_rate': 0.75,
        'nearby_win_rate': 0.80,
        'vs_better_win_rate': 0.40,
        'total_matches': 30
    }
    
    current = 'E4'
    predicted = 'E2'  # V3 predicts 1 rank improvement
    adjusted, boosted = adjust_prediction_for_big_jump(current, predicted, player1, ranking_order)
    
    print("Example 1: Strong player")
    print(f"  Current: {current}")
    print(f"  V3 prediction: {predicted}")
    print(f"  Adjusted: {adjusted}")
    print(f"  Boosted: {boosted}")
    print()
    
    # Example 2: Average player - should NOT boost
    player2 = {
        'win_rate': 0.55,
        'nearby_win_rate': 0.60,
        'vs_better_win_rate': 0.20,
        'total_matches': 25
    }
    
    predicted2 = 'E2'
    adjusted2, boosted2 = adjust_prediction_for_big_jump(current, predicted2, player2, ranking_order)
    
    print("Example 2: Average player")
    print(f"  Current: {current}")
    print(f"  V3 prediction: {predicted2}")
    print(f"  Adjusted: {adjusted2}")
    print(f"  Boosted: {boosted2}")
    print()
    
    # Example 3: Stable prediction - should NOT boost
    player3 = {
        'win_rate': 0.75,
        'nearby_win_rate': 0.80,
        'vs_better_win_rate': 0.40,
        'total_matches': 30
    }
    
    predicted3 = 'E4'  # V3 predicts stable
    adjusted3, boosted3 = adjust_prediction_for_big_jump(current, predicted3, player3, ranking_order)
    
    print("Example 3: Strong player but V3 predicts stable")
    print(f"  Current: {current}")
    print(f"  V3 prediction: {predicted3}")
    print(f"  Adjusted: {adjusted3}")
    print(f"  Boosted: {boosted3}")
    print()
    
    print("=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print("\nThis adjustment:")
    print("  ✓ Only boosts predictions for big improvements")
    print("  ✓ Requires strong signals (3 of 4 criteria)")
    print("  ✓ Only boosts if V3 already predicts improvement")
    print("  ✓ Boosts by 1 additional rank")
    print("  ✓ Keeps all other predictions unchanged")
    print("\nCriteria for boosting:")
    print("  1. Win rate > 70%")
    print("  2. Beating better players > 35%")
    print("  3. Dominating current level > 75%")
    print("  4. Sufficient matches (20+)")
    print("\nNeed 3 of 4 criteria to boost")
