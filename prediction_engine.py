"""
TT Predictor - Prediction Engine
Handles all prediction logic separately from the frontend
"""

import pandas as pd
import numpy as np
import streamlit as st


def is_special_case(player_data, rank_to_int):
    """Check if player is a special case (NG or potential big jump)"""
    current_rank = player_data.get('ranking') or player_data.get('current_ranking')
    
    # NG players are always special cases
    if current_rank == 'NG':
        return True
    
    # Check for potential big jump (strong performance indicators)
    kaart = player_data.get('kaart', {})
    total_wins = sum(kaart.get(rank, [0, 0])[0] for rank in rank_to_int.keys())
    total_losses = sum(kaart.get(rank, [0, 0])[1] for rank in rank_to_int.keys())
    total_matches = total_wins + total_losses
    
    if total_matches < 10:
        return False
    
    win_rate = total_wins / total_matches
    
    # Calculate nearby win rate
    current_idx = rank_to_int.get(current_rank, 999)
    nearby_wins = 0
    nearby_losses = 0
    for rank, rank_idx in rank_to_int.items():
        if abs(rank_idx - current_idx) <= 3:
            wins, losses = kaart.get(rank, [0, 0])
            nearby_wins += wins
            nearby_losses += losses
    nearby_total = nearby_wins + nearby_losses
    nearby_win_rate = nearby_wins / nearby_total if nearby_total > 0 else 0
    
    # Strong performance = potential big jump
    # High overall win rate AND dominating at their level
    if win_rate > 0.70 and nearby_win_rate > 0.70 and total_matches >= 20:
        return True
    
    return False


def prepare_features(player_data, feature_cols, category_encoder, rank_to_int, ranking_order, scaler=None, include_elo=False):
    """Prepare features for model prediction (V3 or V4 with ELO)"""
    try:
        if not isinstance(player_data, dict):
            st.error("Player data is not a dictionary")
            return None

        # Extract current rank and category
        current_rank = player_data.get('ranking') or player_data.get('current_ranking')
        if current_rank is None:
            st.error("Current ranking not found in player data")
            return None

        category = player_data.get('category')
        kaart = player_data.get('kaart', {})
        elo = player_data.get('elo', 0)

        # Encode categorical variables
        current_rank_encoded = rank_to_int.get(current_rank, -1)
        
        # Handle category encoding with fallback for unseen categories
        try:
            category_encoded = category_encoder.transform([category])[0] if category else -1
        except ValueError:
            # Category not seen during training - use a default encoding
            # Map youth categories to similar ones that exist in the model
            category_mapping = {
                'PRE': 'MIN',  # Map PRE to MIN (similar age group)
                'BEN': 'CAD',  # Map BEN to CAD (similar age group)
            }
            fallback_category = category_mapping.get(category, 'SEN')  # Default to SEN
            try:
                category_encoded = category_encoder.transform([fallback_category])[0]
            except ValueError:
                category_encoded = 0  # Last resort fallback

        # Calculate overall performance
        total_wins = sum(kaart.get(rank, [0, 0])[0] for rank in ranking_order)
        total_losses = sum(kaart.get(rank, [0, 0])[1] for rank in ranking_order)
        total_matches = total_wins + total_losses
        win_rate = total_wins / total_matches if total_matches > 0 else 0.5
        
        # Calculate nearby_win_rate (performance at your level - within 3 ranks)
        nearby_wins = 0
        nearby_losses = 0
        for rank in ranking_order:
            rank_idx = rank_to_int.get(rank, 999)
            if abs(rank_idx - current_rank_encoded) <= 3:
                wins, losses = kaart.get(rank, [0, 0])
                nearby_wins += wins
                nearby_losses += losses
        nearby_total = nearby_wins + nearby_losses
        nearby_win_rate = nearby_wins / nearby_total if nearby_total > 0 else 0.5
        
        # Calculate vs_better_win_rate (performance against better players)
        better_wins = 0
        better_losses = 0
        for rank in ranking_order:
            rank_idx = rank_to_int.get(rank, 999)
            if rank_idx < current_rank_encoded:  # Better rank (lower index)
                wins, losses = kaart.get(rank, [0, 0])
                better_wins += wins
                better_losses += losses
        better_total = better_wins + better_losses
        vs_better_win_rate = better_wins / better_total if better_total > 0 else 0
        
        # Calculate vs_worse_win_rate (performance against worse players)
        worse_wins = 0
        worse_losses = 0
        for rank in ranking_order:
            rank_idx = rank_to_int.get(rank, 999)
            if rank_idx > current_rank_encoded:  # Worse rank (higher index)
                wins, losses = kaart.get(rank, [0, 0])
                worse_wins += wins
                worse_losses += losses
        worse_total = worse_wins + worse_losses
        vs_worse_win_rate = worse_wins / worse_total if worse_total > 0 else 0
        
        # Additional features for enhanced model
        win_loss_ratio = total_wins / total_losses if total_losses > 0 else total_wins
        performance_consistency = nearby_win_rate * np.log1p(total_matches)
        level_dominance = (nearby_win_rate - 0.5) * 2  # Scale -1 to 1
        
        # V3 NEW FEATURES - Improved prediction of changes
        win_rate_capped = np.clip(win_rate, 0.1, 0.9)
        nearby_win_rate_capped = np.clip(nearby_win_rate, 0.1, 0.9)
        
        improvement_signal = (
            max(0, nearby_win_rate - 0.6) * 2 +
            vs_better_win_rate * 1.5 +
            max(0, vs_worse_win_rate - 0.7)
        )
        
        decline_signal = (
            max(0, 0.4 - nearby_win_rate) * 2 +
            max(0, 0.5 - vs_worse_win_rate) * 1.5
        )
        
        is_junior = 1 if category in ['JUN', 'J19', 'J21'] else 0
        junior_volatility = is_junior * improvement_signal
        in_de_zone = 1 if current_rank_encoded in [9, 10, 11, 12, 13, 14, 15, 16] else 0
        match_reliability = np.tanh(total_matches / 30)
        reliable_performance = nearby_win_rate_capped * match_reliability
        is_low_rank = 1 if current_rank_encoded >= 13 else 0
        is_mid_rank = 1 if 9 <= current_rank_encoded < 13 else 0
        
        # Youth-specific features (for filtered model)
        is_unranked = 1 if current_rank == 'NG' else 0
        is_entry_rank = 1 if current_rank == 'E6' else 0
        breakthrough_signal = (
            max(0, win_rate - 0.6) * 2 +
            vs_better_win_rate * 2 +
            max(0, nearby_win_rate - 0.7) * 1.5
        )
        activity_volatility = np.tanh(total_matches / 50) * win_rate
        poor_performer_signal = (
            max(0, 0.4 - win_rate) * 2 +
            max(0, 0.3 - nearby_win_rate) * 1.5
        )
        is_excellent = 1 if (win_rate > 0.7 and total_matches >= 20) else 0
        is_youngest = 1 if category in ['MIN', 'PRE'] else 0
        is_oldest = 1 if category in ['BEN', 'CAD'] else 0
        
        # Ranked opponent ratio
        ng_matches = kaart.get('NG', [0, 0])
        ng_total = ng_matches[0] + ng_matches[1]
        ranked_matches = total_matches - ng_total
        ranked_opponent_ratio = ranked_matches / total_matches if total_matches > 0 else 0
        
        ng_win_rate = win_rate if current_rank == 'NG' else 0
        ng_has_many_matches = 1 if (current_rank == 'NG' and total_matches >= 20) else 0
        e6_ready_to_advance = 1 if (current_rank == 'E6' and nearby_win_rate > 0.6 and total_matches >= 15) else 0
        youth_match_reliability = np.tanh(total_matches / 40)
        
        # ELO features (only if include_elo=True for special cases)
        elo_log = 0
        elo_advantage = 0
        if include_elo:
            # Convert ELO to float if it's a string
            try:
                elo_val = float(elo) if elo else 0
            except (ValueError, TypeError):
                elo_val = 0
            
            if elo_val > 0:
                elo_log = np.log1p(elo_val)
                rank_elo_map = {
                    'A': 1800, 'B0': 1600, 'B2': 1500, 'B4': 1400, 'B6': 1300,
                    'C0': 1200, 'C2': 1100, 'C4': 1000, 'C6': 900,
                    'D0': 800, 'D2': 750, 'D4': 700, 'D6': 650,
                    'E0': 600, 'E2': 550, 'E4': 500, 'E6': 450,
                    'NG': 400
                }
                expected_elo = rank_elo_map.get(current_rank, 400)
                elo_advantage = np.clip((elo_val - expected_elo) / 100, -5, 5)
        
        # Create feature dictionary with all possible features
        features = {
            'current_rank_encoded': current_rank_encoded,
            'category_encoded': category_encoded,
            'win_rate': win_rate,
            'nearby_win_rate': nearby_win_rate,
            'vs_better_win_rate': vs_better_win_rate,
            'vs_worse_win_rate': vs_worse_win_rate,
            'total_matches': total_matches,
            'match_volume': np.log1p(total_matches),
            'performance_score': (win_rate * 0.4 + nearby_win_rate * 0.4 + vs_better_win_rate * 0.2),
            'win_loss_ratio': win_loss_ratio,
            'performance_consistency': performance_consistency,
            'level_dominance': level_dominance,
            # V3 features
            'win_rate_capped': win_rate_capped,
            'nearby_win_rate_capped': nearby_win_rate_capped,
            'improvement_signal': improvement_signal,
            'decline_signal': decline_signal,
            'is_junior': is_junior,
            'junior_volatility': junior_volatility,
            'in_de_zone': in_de_zone,
            'match_reliability': match_reliability,
            'reliable_performance': reliable_performance,
            'is_low_rank': is_low_rank,
            'is_mid_rank': is_mid_rank,
            # Youth-specific features (filtered model)
            'is_unranked': is_unranked,
            'is_entry_rank': is_entry_rank,
            'breakthrough_signal': breakthrough_signal,
            'activity_volatility': activity_volatility,
            'poor_performer_signal': poor_performer_signal,
            'is_excellent': is_excellent,
            'is_youngest': is_youngest,
            'is_oldest': is_oldest,
            'ranked_opponent_ratio': ranked_opponent_ratio,
            'ng_win_rate': ng_win_rate,
            'ng_has_many_matches': ng_has_many_matches,
            'e6_ready_to_advance': e6_ready_to_advance,
            'youth_match_reliability': youth_match_reliability,
            # ELO features (for special cases model)
            'elo_log': elo_log,
            'elo_advantage': elo_advantage
        }

        # Create DataFrame with all feature columns
        feature_df = pd.DataFrame([features])

        # Ensure all expected columns are present
        for col in feature_cols:
            if col not in feature_df.columns:
                feature_df[col] = 0

        # Select only the required features in the correct order
        feature_df = feature_df[feature_cols]
        
        # Apply scaling if scaler is provided (for filtered model)
        if scaler is not None:
            feature_df = pd.DataFrame(
                scaler.transform(feature_df),
                columns=feature_cols
            )

        return feature_df

    except Exception as e:
        st.error(f"Error preparing features: {e}")
        import traceback
        st.error(traceback.format_exc())
        return None


def should_boost_for_big_jump(player_data, current_rank, predicted_rank, rank_to_int, is_filtered_model=False):
    """
    Check if prediction should be boosted for big improvement
    More aggressive for filtered model (youth categories)
    """
    current_idx = rank_to_int.get(current_rank, 999)
    predicted_idx = rank_to_int.get(predicted_rank, 999)
    
    # Model must already predict improvement
    if predicted_idx >= current_idx:
        return False
    
    # Extract features
    kaart = player_data.get('kaart', {})
    
    # Calculate win rate
    total_wins = sum(kaart.get(rank, [0, 0])[0] for rank in rank_to_int.keys())
    total_losses = sum(kaart.get(rank, [0, 0])[1] for rank in rank_to_int.keys())
    total_matches = total_wins + total_losses
    win_rate = total_wins / total_matches if total_matches > 0 else 0
    
    # Calculate nearby win rate
    nearby_wins = 0
    nearby_losses = 0
    for rank, rank_idx in rank_to_int.items():
        if abs(rank_idx - current_idx) <= 3:
            wins, losses = kaart.get(rank, [0, 0])
            nearby_wins += wins
            nearby_losses += losses
    nearby_total = nearby_wins + nearby_losses
    nearby_win_rate = nearby_wins / nearby_total if nearby_total > 0 else 0
    
    # Calculate vs better win rate
    better_wins = 0
    better_losses = 0
    for rank, rank_idx in rank_to_int.items():
        if rank_idx < current_idx:
            wins, losses = kaart.get(rank, [0, 0])
            better_wins += wins
            better_losses += losses
    better_total = better_wins + better_losses
    vs_better_win_rate = better_wins / better_total if better_total > 0 else 0
    
    # More aggressive thresholds for filtered model (youth categories)
    if is_filtered_model:
        # Lower thresholds for youth - they improve faster
        strong_overall = win_rate > 0.65  # Was 0.70
        beating_better = vs_better_win_rate > 0.25  # Was 0.35
        dominating_level = nearby_win_rate > 0.70  # Was 0.75
        enough_matches = total_matches >= 15  # Was 20
        
        # Need at least 2 of 4 strong signals for youth
        signals = sum([strong_overall, beating_better, dominating_level, enough_matches])
        return signals >= 2
    else:
        # Conservative thresholds for regular model
        strong_overall = win_rate > 0.70
        beating_better = vs_better_win_rate > 0.35
        dominating_level = nearby_win_rate > 0.75
        enough_matches = total_matches >= 20
        
        # Need at least 3 of 4 strong signals
        signals = sum([strong_overall, beating_better, dominating_level, enough_matches])
        return signals >= 3


def predict_next_rank(player_data, model, feature_cols, category_encoder, rank_to_int, int_to_rank, ranking_order, scaler=None, special_model=None, special_feature_cols=None, is_filtered_model=False):
    """Predict the next rank for a player with optional boost for big improvements"""
    try:
        # Get category and current rank
        category = player_data.get('category')
        current_rank = player_data.get('ranking') or player_data.get('current_ranking')
        
        # Younger youth categories should NOT use ELO model
        younger_youth = ['BEN', 'PRE', 'MIN', 'CAD']
        is_younger_youth = category in younger_youth
        
        # Older youth (JUN, J19, J21) and adults CAN use ELO model
        # Check if this is a special case and we have the special model
        use_special_model = False
        if not is_younger_youth and special_model is not None and is_special_case(player_data, rank_to_int):
            elo = player_data.get('elo', 0)
            # Convert ELO to float if it's a string
            try:
                elo = float(elo) if elo else 0
            except (ValueError, TypeError):
                elo = 0
            
            if elo > 0:  # Only use special model if ELO is available
                use_special_model = True
                st.info("Using ELO-enhanced model for special case prediction")
        
        # Prepare features
        if use_special_model:
            features = prepare_features(player_data, special_feature_cols, category_encoder, rank_to_int, ranking_order, scaler, include_elo=True)
            prediction_model = special_model
        else:
            features = prepare_features(player_data, feature_cols, category_encoder, rank_to_int, ranking_order, scaler, include_elo=False)
            prediction_model = model

        if features is None:
            return None

        # Make prediction
        prediction = prediction_model.predict(features)[0]
        
        # Get prediction probabilities for confidence
        prediction_proba = prediction_model.predict_proba(features)[0]
        confidence = prediction_proba.max() * 100  # Convert to percentage

        # Convert prediction back to rank label
        predicted_rank = int_to_rank.get(prediction, "Unknown")
        
        # Get current rank and category
        current_rank = player_data.get('ranking') or player_data.get('current_ranking')
        category = player_data.get('category')
        
        # Check if should boost for big improvement (ONLY for filtered model - youth categories)
        was_boosted = False
        if is_filtered_model and current_rank and should_boost_for_big_jump(player_data, current_rank, predicted_rank, rank_to_int, is_filtered_model):
            # Boost by 1 additional rank
            boosted_idx = max(0, prediction - 1)
            predicted_rank = int_to_rank.get(boosted_idx, predicted_rank)
            was_boosted = True
            confidence = confidence * 0.9  # Slightly lower confidence for boosted predictions
        
        # JUNIOR-SPECIFIC ADJUSTMENTS (JUN/J19) - ONLY for filtered model
        if is_filtered_model and category in ['JUN', 'J19'] and current_rank:
            current_idx = rank_to_int.get(current_rank, 999)
            predicted_idx = rank_to_int.get(predicted_rank, 999)
            
            # Calculate performance metrics
            kaart = player_data.get('kaart', {})
            total_wins = sum(kaart.get(rank, [0, 0])[0] for rank in ranking_order)
            total_losses = sum(kaart.get(rank, [0, 0])[1] for rank in ranking_order)
            total_matches = total_wins + total_losses
            win_rate = total_wins / total_matches if total_matches > 0 else 0
            
            # Nearby win rate
            nearby_wins = 0
            nearby_losses = 0
            for rank in ranking_order:
                rank_idx = rank_to_int.get(rank, 999)
                if abs(rank_idx - current_idx) <= 3:
                    wins, losses = kaart.get(rank, [0, 0])
                    nearby_wins += wins
                    nearby_losses += losses
            nearby_total = nearby_wins + nearby_losses
            nearby_win_rate = nearby_wins / nearby_total if nearby_total > 0 else 0
            
            # Vs better win rate
            better_wins = 0
            better_losses = 0
            for rank in ranking_order:
                rank_idx = rank_to_int.get(rank, 999)
                if rank_idx < current_idx:
                    wins, losses = kaart.get(rank, [0, 0])
                    better_wins += wins
                    better_losses += losses
            better_total = better_wins + better_losses
            vs_better_win_rate = better_wins / better_total if better_total > 0 else 0
            
            # RULE 1: Boost high-performing juniors (especially JUN) - MORE AGGRESSIVE
            if category == 'JUN' and predicted_idx >= current_idx:  # Not predicting improvement
                if nearby_win_rate > 0.65 and vs_better_win_rate > 0.20 and total_matches >= 15:  # Lower thresholds
                    # Strong performer - boost by 1 rank
                    boosted_idx = max(0, predicted_idx - 1)
                    predicted_rank = int_to_rank.get(boosted_idx, predicted_rank)
                    was_boosted = True
                    confidence = confidence * 0.85
            
            # RULE 2: Be conservative with J19 declines (they're rare)
            if category == 'J19' and predicted_idx > current_idx:  # Predicting decline
                # Only predict decline if very strong signal
                if nearby_win_rate > 0.35:  # Not terrible performance
                    # Keep at same rank instead
                    predicted_rank = current_rank
                    confidence = confidence * 0.8
            
            # RULE 3: Active high-performers get extra boost - MORE AGGRESSIVE
            if total_matches >= 30 and win_rate > 0.60 and nearby_win_rate > 0.60:  # Lower thresholds
                if predicted_idx >= current_idx:  # Not already predicting improvement
                    boosted_idx = max(0, predicted_idx - 1)
                    predicted_rank = int_to_rank.get(boosted_idx, predicted_rank)
                    was_boosted = True
                    confidence = confidence * 0.85

        return predicted_rank, confidence, was_boosted

    except Exception as e:
        st.error(f"Error making prediction: {e}")
        import traceback
        st.error(traceback.format_exc())
        return None


def get_rank_comparison(current_rank, predicted_rank, rank_to_int):
    """Get color and arrow for rank comparison"""
    current_idx = rank_to_int.get(current_rank, 999)
    predicted_idx = rank_to_int.get(predicted_rank, 999)
    
    if predicted_idx < current_idx:
        return "+", "↑", "success", "Verbetering!"
    elif predicted_idx > current_idx:
        return "-", "↓", "error", "Achteruitgang"
    else:
        return "=", "→", "info", "Geen verandering"