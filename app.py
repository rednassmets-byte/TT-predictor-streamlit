import streamlit as st
import pandas as pd
import pickle
import joblib
import requests
import os
import sys
import numpy as np
from huggingface_hub import hf_hub_download
import plotly.graph_objects as go
import plotly.express as px
# Add current directory to path to ensure imports work
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    from database_maker import get_data, get_province_for_club, get_club_name_for_club, get_information
except ImportError as e:
    st.error(f"Could not import database_maker module: {e}")
    st.info("This may be due to missing dependencies. Installing pyvttl from GitHub...")
    st.code("pip install git+https://github.com/jacobstim/pyvttl.git")
    st.warning("Please ensure all dependencies from requirements.txt are installed.")
    st.stop()

# Load club data for province/club selection
try:
    df_clubs = pd.read_csv("club_data.csv", encoding='utf-8', header=1)
except FileNotFoundError:
    st.error("club_data.csv not found. Please ensure the club data file is available.")
    st.stop()

def get_clubs_for_province(province):
    """Get list of club codes for a given province."""
    return df_clubs[df_clubs['Provincie'] == province]['Clubnr.'].tolist()

def get_members_for_club_season(club, season):
    """Get list of member names for a club and season."""
    try:
        from database_maker import get_memberlist
        info = get_memberlist(club=club, season=season)
        members = []
        for member in info.MemberEntries:
            first_name = getattr(member, 'FirstName', '')
            last_name = getattr(member, 'LastName', '')
            if first_name and last_name:
                full_name = f"{first_name} {last_name}"
                members.append(full_name)
            elif first_name:
                members.append(first_name)
            elif last_name:
                members.append(last_name)
        return sorted(members)
    except Exception as e:
        st.sidebar.error(f"Error loading members: {e}")
        return []

# Load the pre-trained model and encoders
@st.cache_resource
def load_regular_model_and_encoders():
    try:
        # Load V3 model files (improved - better at predicting changes) - NO FALLBACKS
        category_encoder = joblib.load("category_encoder_v3.pkl")
        feature_cols = joblib.load("feature_cols_v3.pkl")
        int_to_rank = joblib.load("int_to_rank_v3.pkl")
        rank_to_int = joblib.load("rank_to_int_v3.pkl")
        ranking_order = joblib.load("ranking_order_v3.pkl")
        model = joblib.load("model_v3_improved.pkl")
        return model, category_encoder, feature_cols, int_to_rank, rank_to_int, ranking_order, None
    except Exception as e:
        st.error(f"❌ V3 Regular model not available: {str(e)}")
        st.error("Please ensure all model files are uploaded to the repository.")
        return None, None, None, None, None, None, None

@st.cache_resource
def load_filtered_model_and_encoders():
    """Load the V3 filtered model for youth categories (BEN/PRE/MIN/CAD)"""
    try:
        # Load V3 filtered model files (improved for youth) - NO FALLBACKS
        category_encoder = joblib.load("category_encoder_filtered_v3.pkl")
        feature_cols = joblib.load("feature_cols_filtered_v3.pkl")
        int_to_rank = joblib.load("int_to_rank_filtered_v3.pkl")
        rank_to_int = joblib.load("rank_to_int_filtered_v3.pkl")
        ranking_order = joblib.load("ranking_order_filtered_v3.pkl")
        model = joblib.load("model_filtered_v3_improved.pkl")
        return model, category_encoder, feature_cols, int_to_rank, rank_to_int, ranking_order, None
    except Exception as e:
        st.error(f"❌ V3 Filtered model not available: {str(e)}")
        st.error("Please ensure all model files are uploaded to the repository.")
        return None, None, None, None, None, None, None

@st.cache_resource
def load_special_cases_model():
    """Load the V4 hybrid special cases model (for NG + big jumps with ELO)"""
    try:
        model = joblib.load("model_v4_special_cases.pkl")
        feature_cols = joblib.load("feature_cols_v4_special.pkl")
        category_encoder = joblib.load("category_encoder_v4_hybrid.pkl")
        int_to_rank = joblib.load("int_to_rank_v4_hybrid.pkl")
        rank_to_int = joblib.load("rank_to_int_v4_hybrid.pkl")
        ranking_order = joblib.load("ranking_order_v4_hybrid.pkl")
        return model, category_encoder, feature_cols, int_to_rank, rank_to_int, ranking_order
    except Exception as e:
        # If special cases model not available, return None (will fall back to V3)
        return None, None, None, None, None, None

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
        # Check if this is a special case and we have the special model
        use_special_model = False
        if special_model is not None and is_special_case(player_data, rank_to_int):
            elo = player_data.get('elo', 0)
            # Convert ELO to float if it's a string
            try:
                elo = float(elo) if elo else 0
            except (ValueError, TypeError):
                elo = 0
            
            if elo > 0:  # Only use special model if ELO is available
                use_special_model = True
                st.info("🎯 Using ELO-enhanced model for special case prediction")
        
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
        return "🟢", "↑", "success", "Verbetering!"
    elif predicted_idx > current_idx:
        return "🔴", "↓", "error", "Achteruitgang"
    else:
        return "🟡", "→", "info", "Geen verandering"

def create_win_loss_chart(kaart_data, ranking_order):
    """Create a bar chart for win/loss data"""
    # Group ranks
    rank_groups = {}
    for rank in ranking_order:
        if rank.startswith('A'):
            group_key = 'A'
        elif rank in ['B0', 'B0e']:
            group_key = 'B0'
        else:
            group_key = rank
        
        if group_key not in rank_groups:
            rank_groups[group_key] = {'wins': 0, 'losses': 0}
        wins, losses = kaart_data.get(rank, [0, 0])
        rank_groups[group_key]['wins'] += wins
        rank_groups[group_key]['losses'] += losses
    
    # Create chart data - reverse order so best ranks appear first
    ranks = list(rank_groups.keys())
    ranks.reverse()
    
    # Calculate win percentages
    win_percentages = []
    for r in ranks:
        total = rank_groups[r]['wins'] + rank_groups[r]['losses']
        if total > 0:
            win_pct = (rank_groups[r]['wins'] / total) * 100
        else:
            win_pct = 0
        win_percentages.append(win_pct)
    
    # Color bars based on win percentage (green gradient)
    colors = [f'rgba(0, {int(204 * (p/100))}, {int(102 * (p/100))}, 0.8)' for p in win_percentages]
    
    fig = go.Figure(data=[
        go.Bar(x=ranks, y=win_percentages, marker_color=colors,
               text=[f'{p:.1f}%' for p in win_percentages], textposition='auto')
    ])
    
    fig.update_layout(
        title='Win Percentage per Rank',
        xaxis_title='Rank',
        yaxis_title='Win Rate (%)',
        yaxis_range=[0, 100],
        showlegend=False,
        height=300,
        margin=dict(l=20, r=20, t=40, b=20),
        xaxis={'categoryorder': 'array', 'categoryarray': ranks}
    )
    
    return fig

def create_rank_progression_chart(club_code, player_name, current_season, rank_to_int, predicted_rank=None):
    """Create a line chart showing rank progression over seasons"""
    seasons = []
    ranks = []
    rank_values = []
    
    # Fetch data for last 5 seasons (or available seasons)
    for season in range(max(15, current_season - 4), current_season + 1):
        try:
            player_data = get_data(club=club_code, name=player_name, season=season)
            if player_data and player_data.get('current_ranking'):
                rank = player_data['current_ranking']
                seasons.append(f"{season-1}-{season}")
                ranks.append(rank)
                rank_values.append(rank_to_int.get(rank, 999))
        except:
            continue
    
    if len(seasons) < 2:
        return None
    
    # Create line chart
    fig = go.Figure()
    
    # Historical data
    fig.add_trace(go.Scatter(
        x=seasons,
        y=rank_values,
        mode='lines+markers+text',
        text=ranks,
        textposition='top center',
        marker=dict(size=12, color='#00cc66'),
        line=dict(width=3, color='#00cc66'),
        name='Actual Rank'
    ))
    
    # Add predicted rank for next season if current season is 26
    if current_season == 26 and predicted_rank:
        predicted_season = f"26-27"
        predicted_value = rank_to_int.get(predicted_rank, 999)
        
        # Add dashed line from last actual to predicted
        fig.add_trace(go.Scatter(
            x=[seasons[-1], predicted_season],
            y=[rank_values[-1], predicted_value],
            mode='lines+markers+text',
            text=['', predicted_rank],
            textposition='top center',
            marker=dict(size=12, color='#ff9900', symbol='diamond'),
            line=dict(width=3, color='#ff9900', dash='dash'),
            name='Predicted'
        ))
    
    fig.update_layout(
        title='Rank Progression Over Seasons',
        xaxis_title='Season',
        yaxis_title='Rank',
        yaxis=dict(
            autorange='reversed',  # Better ranks (lower values) at top
            tickmode='array',
            tickvals=list(range(len(rank_to_int))),
            ticktext=[k for k, v in sorted(rank_to_int.items(), key=lambda x: x[1])]
        ),
        showlegend=True if (current_season == 26 and predicted_rank) else False,
        height=400,
        margin=dict(l=20, r=20, t=40, b=20),
        hovermode='x unified'
    )
    
    return fig

def main():
    st.set_page_config(page_title="TT KlassementPredictor", page_icon="🏓", layout="wide")
    
    # Custom CSS for responsive design
    st.markdown("""
        <style>
        /* Responsive adjustments */
        @media (max-width: 768px) {
            .stColumn {
                width: 100% !important;
            }
            h1 {
                font-size: 1.5rem !important;
            }
        }
        
        /* Better spacing */
        .stProgress > div > div {
            background-color: #00cc66;
        }
        
        /* Card-like containers */
        .stMetric {
            background-color: #f0f2f6;
            padding: 10px;
            border-radius: 5px;
        }
        </style>
    """, unsafe_allow_html=True)

    st.title("TT KlassementPredictor")
    st.markdown("Voorspel je klassement op basis van je kaart")
    
    # Model accuracy note
    st.info(" Model Accuracy: ~85% (trained on Antwerpen data, seasons 15-26)")

    # Load models and encoders (ORIGINAL SETUP - BEST PERFORMANCE)
    regular_model, regular_category_encoder, regular_feature_cols, regular_int_to_rank, regular_rank_to_int, regular_ranking_order, regular_scaler = load_regular_model_and_encoders()
    filtered_model, filtered_category_encoder, filtered_feature_cols, filtered_int_to_rank, filtered_rank_to_int, filtered_ranking_order, filtered_scaler = load_filtered_model_and_encoders()
    
    # Load special cases model (V4 hybrid with ELO)
    special_model, special_category_encoder, special_feature_cols, special_int_to_rank, special_rank_to_int, special_ranking_order = load_special_cases_model()

    if None in [regular_model, regular_category_encoder, regular_feature_cols, regular_int_to_rank, regular_rank_to_int, regular_ranking_order]:
        st.error("Failed to load regular model components. Please check the model files.")
        return

    if None in [filtered_model, filtered_category_encoder, filtered_feature_cols, filtered_int_to_rank, filtered_rank_to_int, filtered_ranking_order]:
        st.error("Failed to load filtered model components. Please check the filtered model files.")
        return
    
    # Sidebar for input
    st.sidebar.header("Player Information")

    # Fixed to Antwerpen province only
    selected_province = 'Antwerpen'
    st.sidebar.info("Province: Antwerpen")

    # Club selection based on province
    clubs_in_province = get_clubs_for_province(selected_province)
    club_names = [f"{code} - {get_club_name_for_club(code)}" for code in clubs_in_province]
    selected_club_display = st.sidebar.selectbox("Select Club", club_names)
    club_code = selected_club_display.split(' - ')[0] if selected_club_display else ""

    # Season selection
    season = st.sidebar.number_input("Seizoen", min_value=15, max_value=26, value=26)

    # Initialize session state for selected player
    if 'selected_player' not in st.session_state:
        st.session_state.selected_player = None
    if 'previous_season' not in st.session_state:
        st.session_state.previous_season = season
    if 'auto_predict' not in st.session_state:
        st.session_state.auto_predict = False

    # Member selection based on club and season
    if club_code and season:
        members = get_members_for_club_season(club_code, season)
        if members:
            # Check if season changed
            season_changed = st.session_state.previous_season != season
            
            # Try to keep the same player if season changed
            if season_changed and st.session_state.selected_player:
                if st.session_state.selected_player in members:
                    # Player exists in new season - keep them selected and auto-predict
                    default_index = members.index(st.session_state.selected_player)
                    st.session_state.auto_predict = True
                else:
                    # Player doesn't exist - reset to first player, don't auto-predict
                    default_index = 0
                    st.session_state.auto_predict = False
                    st.session_state.selected_player = members[0]
            else:
                # No season change or no previous player
                default_index = members.index(st.session_state.selected_player) if st.session_state.selected_player in members else 0
            
            player_name = st.sidebar.selectbox("Select Player", members, index=default_index)
            st.session_state.selected_player = player_name
            st.session_state.previous_season = season
        else:
            player_name = st.sidebar.text_input("Player Name (manual entry)", value="")
            st.sidebar.warning("Could not load members automatically. Please enter name manually.")
    else:
        player_name = st.sidebar.text_input("Player Name", value="")
    
    # Predict button in sidebar
    st.sidebar.markdown("---")
    predict_button = st.sidebar.button(" Voorspel Klassement", type="primary", use_container_width=True)
    
    # Auto-predict if season changed and player exists
    if st.session_state.auto_predict:
        predict_button = True
        st.session_state.auto_predict = False
    
    # Main content area
    st.header("Kaart & Prediction")
    
    if not predict_button:
        # Empty state
        st.markdown("---")
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            st.markdown("#### Selecteer een speler en klik op 'Voorspel Klassement'")
            st.markdown("De AI zal het toekomstige klassement voorspellen op basis van de huidige kaart.")
    
    if predict_button:
        with st.spinner("Fetching player data and making prediction..."):
            try:
                # Get player data from API
                player_data = get_data(club=club_code, name=player_name, season=season)
                
                if player_data:
                    # Display player information
                    st.subheader("Kaart")
                    
                    # Display player name in large text
                    st.markdown(f"# {player_name}")
                    st.markdown("---")
                    
                    info_col1, info_col2, info_col3 = st.columns(3)
                    
                    with info_col1:
                        st.metric("Klassementt", player_data.get('current_ranking', 'Unknown'))
                        st.metric("Categorie", player_data.get('category', 'Unknown'))
                    
                    with info_col2:
                        st.metric("Club", player_data.get('club_name', 'Unknown'))
                        st.metric("Provincie", player_data.get('province', 'Unknown'))
                    
                    with info_col3:
                        st.metric("Seizoen", player_data.get('season', 'Unknown'))
                        st.metric("Unique Index", player_data.get('unique_index', 'Unknown'))
                    
                    # Choose model based on category AND rank
                    category = player_data.get('category')
                    current_rank = player_data.get('ranking') or player_data.get('current_ranking')
                    
                    # Model selection logic:
                    # 1. BEN/PRE/MIN/CAD categories with rank C6 or lower (worse) -> use filtered model
                    # 2. BEN/PRE/MIN/CAD categories with rank above C6 (better) -> use regular model
                    # 3. All other categories -> use regular model
                    
                    # Define rank order (lower index = better rank)
                    rank_order = ['A', 'B0', 'B2', 'B4', 'B6', 'C0', 'C2', 'C4', 'C6', 
                                  'D0', 'D2', 'D4', 'D6', 'E0', 'E2', 'E4', 'E6', 'NG']
                    
                    use_filtered = False
                    if category in ["BEN", "PRE", "MIN", "CAD"]:
                        # Check if rank is C6 or lower (worse)
                        try:
                            current_rank_idx = rank_order.index(current_rank)
                            c6_idx = rank_order.index('C6')
                            # Use filtered model only if rank is C6 or worse (higher index)
                            if current_rank_idx >= c6_idx:
                                use_filtered = True
                            else:
                                # For youth at high ranks (above C6), still use filtered model
                                # but we'll disable the aggressive boost
                                use_filtered = True
                        except (ValueError, AttributeError):
                            # If rank not found, default to filtered for youth
                            use_filtered = True
                    
                    if use_filtered:
                        # Use filtered model for youth categories
                        model = filtered_model
                        category_encoder = filtered_category_encoder
                        feature_cols = filtered_feature_cols
                        rank_to_int = filtered_rank_to_int
                        int_to_rank = filtered_int_to_rank
                        ranking_order = filtered_ranking_order
                        scaler = filtered_scaler
                        
                        # Check if rank is above C6 (better rank)
                        try:
                            current_rank_idx = rank_order.index(current_rank)
                            c6_idx = rank_order.index('C6')
                            if current_rank_idx < c6_idx:
                                model_type = "Filtered (youth high rank - no boost)"
                            else:
                                model_type = "Filtered (youth C6+)"
                        except (ValueError, AttributeError):
                            model_type = "Filtered (youth)"
                    else:
                        # Use regular model for all other cases
                        model = regular_model
                        category_encoder = regular_category_encoder
                        feature_cols = regular_feature_cols
                        rank_to_int = regular_rank_to_int
                        int_to_rank = regular_int_to_rank
                        ranking_order = regular_ranking_order
                        scaler = regular_scaler
                        model_type = "Regular"

                    # Display performance data
                    st.subheader("KAART")

                    # Group specific ranks: A together, B0 and B0e together, others separate
                    rank_groups = {}
                    for rank in ranking_order:
                        if rank.startswith('A'):
                            group_key = 'A'
                        elif rank in ['B0', 'B0e']:
                            group_key = 'B0'
                        else:
                            group_key = rank  # Keep others separate

                        if group_key not in rank_groups:
                            rank_groups[group_key] = {'wins': 0, 'losses': 0}
                        wins, losses = player_data.get('kaart', {}).get(rank, [0, 0])
                        rank_groups[group_key]['wins'] += wins
                        rank_groups[group_key]['losses'] += losses

                    # Create performance DataFrame
                    performance_data = []
                    for group_key, stats in rank_groups.items():
                        wins = stats['wins']
                        losses = stats['losses']
                        total_matches = wins + losses
                        performance_data.append({
                            'Rank': group_key,
                            'Wins': wins,
                            'Losses': losses,
                            'Win Rate': f"{(wins/total_matches)*100:.1f}%" if total_matches > 0 else "0%"
                        })

                    performance_df = pd.DataFrame(performance_data)
                    
                    # Display table and chart side by side
                    col_table, col_chart = st.columns([1, 1])
                    with col_table:
                        st.dataframe(performance_df, use_container_width=True)
                    with col_chart:
                        fig = create_win_loss_chart(player_data.get('kaart', {}), ranking_order)
                        st.plotly_chart(fig, use_container_width=True)

                    # Make prediction first (needed for progression chart)
                    # Only apply boost for youth at C6 or lower
                    apply_boost = "Filtered (youth C6+)" in model_type
                    result = predict_next_rank(
                        player_data, model, feature_cols, category_encoder, 
                        rank_to_int, int_to_rank, ranking_order, scaler,
                        special_model=special_model, special_feature_cols=special_feature_cols,
                        is_filtered_model=apply_boost
                    )
                    
                    predicted_rank = None
                    if result:
                        predicted_rank, confidence, was_boosted = result
                    
                    # Display rank progression chart with prediction
                    st.subheader("Rank Progression")
                    progression_fig = create_rank_progression_chart(club_code, player_name, season, rank_to_int, predicted_rank)
                    if progression_fig:
                        st.plotly_chart(progression_fig, use_container_width=True)
                    else:
                        st.info("Not enough historical data to show rank progression (need at least 2 seasons)")

                    # Display prediction
                    st.subheader("Voorspelling  Klassement")
                    
                    if result:
                        current_rank = player_data.get('ranking') or player_data.get('current_ranking')
                        
                        # Get comparison indicators
                        emoji, arrow, color_type, message = get_rank_comparison(current_rank, predicted_rank, rank_to_int)
                        
                        # Display prediction
                        st.markdown(f"# {predicted_rank}")
                        
                        # Show if prediction was boosted
                       
                        
                        # Show confidence with interpretation
                        if confidence >= 70:
                            confidence_label = "Zeer zeker"
                            confidence_color = "🟢"
                        elif confidence >= 50:
                            confidence_label = "Redelijk zeker"
                            confidence_color = "🟡"
                        else:
                            confidence_label = "Onzeker"
                            confidence_color = "🟠"
                        
                        st.caption(f"{confidence_color} Confidence: {confidence:.1f}% ({confidence_label}) | {message}")
                        
                        # Warning for low confidence
                        if confidence < 30:
                            st.info("Lage zekerheid - voorspelling kan variëren. Meer wedstrijden spelen verhoogt de nauwkeurigheid.")
                    else:
                        st.error("Unable to make prediction. Please check the input data.")
                        predicted_rank = None
                    
                    # Calculate performance score (1-100)
                    st.subheader("Performance Score")
                    total_wins = sum(wins for wins, _ in player_data.get('kaart', {}).values())
                    total_losses = sum(losses for _, losses in player_data.get('kaart', {}).values())
                    total_matches = total_wins + total_losses

                    if total_matches > 0:
                        # Calculate performance score based on predicted vs current ranking difference and win rate
                        # Higher predicted rank (lower number) gives positive boost, modulated by win rate
                        current_rank = player_data.get('ranking') or player_data.get('current_ranking', '')
                        current_rank_num = rank_to_int.get(current_rank, 0)
                        predicted_rank_num = rank_to_int.get(predicted_rank, current_rank_num)
                        rank_difference = current_rank_num - predicted_rank_num  # Positive if predicted is higher rank

                        win_rate = total_wins / total_matches if total_matches > 0 else 0

                        # Scale to 0-100: base from rank difference, modulated by win rate for more dynamic scoring
                        performance_score = int(50 + rank_difference * 10 + (win_rate - 0.5) * 20)  # Win rate modulation adds/subtracts up to 10 points
                        performance_score = max(0, min(100, performance_score))

                        # Color gradient from dark red (0) to dark green (100)
                        win_rate = total_wins / total_matches if total_matches > 0 else 0
                        rank_improvement = rank_difference
                        
                        # Calculate color gradient from dark red to dark green
                        # At 0: dark red (139, 0, 0)
                        # At 100: dark green (0, 100, 0)
                        ratio = performance_score / 100
                        red = int(139 * (1 - ratio))
                        green = int(100 * ratio)
                        color = f"rgb({red}, {green}, 0)"
                        
                        # Display with color gradient
                        st.markdown(f"""
                            <div style="background-color: {color}; padding: 20px; border-radius: 10px; text-align: center;">
                                <h2 style="color: white; margin: 0;">Performance Score</h2>
                                <h1 style="color: white; margin: 10px 0; font-size: 3rem;">{performance_score}</h1>
                                <p style="color: white; margin: 0;">out of 100</p>
                            </div>
                        """, unsafe_allow_html=True)
                        
                        st.caption(f"Predicted ranking improvement: {rank_improvement} levels. {total_wins} wins out of {total_matches} matches ({win_rate:.1%} win rate)")
                    else:
                        st.markdown(f"""
                            <div style="background-color: rgb(139, 0, 0); padding: 20px; border-radius: 10px; text-align: center;">
                                <h2 style="color: white; margin: 0;">Performance Score</h2>
                                <h1 style="color: white; margin: 10px 0; font-size: 3rem;">0</h1>
                                <p style="color: white; margin: 0;">out of 100</p>
                            </div>
                        """, unsafe_allow_html=True)
                        st.info("No matches played this season")
                    
                    # Display info at bottom
                    st.markdown("---")
                    st.caption(f"Model: {model_type} | Categorie: {category}")
                    st.caption("Machine learning op data van Antwerpen seizoen 15-26")
                    st.caption("Gemaakt door Smets Sander | Credits: Smets Steven, Tim Jacobs, vttl api")
                    
                else:
                    st.error("Could not fetch player data. Please check the club code and player name.")
                    
            except Exception as e:
                st.error(f"Error: {e}")

if __name__ == "__main__":
    main()