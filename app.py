import streamlit as st
import pandas as pd
import pickle
import joblib
import requests
import os
import sys
import numpy as np
from huggingface_hub import hf_hub_download
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
        # Try loading local files first (for improved model)
        try:
            category_encoder = joblib.load("category_encoder.pkl")
            feature_cols = joblib.load("feature_cols.pkl")
            int_to_rank = joblib.load("int_to_rank.pkl")
            rank_to_int = joblib.load("rank_to_int.pkl")
            ranking_order = joblib.load("ranking_order.pkl")
            model = joblib.load("model.pkl")
            scaler = joblib.load("scaler.pkl")
            return model, category_encoder, feature_cols, int_to_rank, rank_to_int, ranking_order, scaler
        except:
            # Fallback to Hugging Face
            repo_id = "zouteboom4/ai_prediction"
            category_encoder = joblib.load(hf_hub_download(repo_id=repo_id, filename="category_encoder.pkl"))
            feature_cols = joblib.load(hf_hub_download(repo_id=repo_id, filename="feature_cols.pkl"))
            int_to_rank = joblib.load(hf_hub_download(repo_id=repo_id, filename="int_to_rank.pkl"))
            rank_to_int = joblib.load(hf_hub_download(repo_id=repo_id, filename="rank_to_int.pkl"))
            ranking_order = joblib.load(hf_hub_download(repo_id=repo_id, filename="ranking_order.pkl"))
            model = joblib.load(hf_hub_download(repo_id=repo_id, filename="model.pkl"))
            return model, category_encoder, feature_cols, int_to_rank, rank_to_int, ranking_order, None
    except Exception as e:
        st.error(f"Error loading regular model files: {e}")
        return None, None, None, None, None, None, None

@st.cache_resource
def load_filtered_model_and_encoders():
    try:
        # Load local filtered model files
        category_encoder = joblib.load("category_encoder_filtered.pkl")
        feature_cols = joblib.load("feature_cols_filtered.pkl")
        int_to_rank = joblib.load("int_to_rank_filtered.pkl")
        rank_to_int = joblib.load("rank_to_int_filtered.pkl")
        ranking_order = joblib.load("ranking_order_filtered.pkl")
        model = joblib.load("model_filtered.pkl")
        scaler = joblib.load("scaler_filtered.pkl")

        return model, category_encoder, feature_cols, int_to_rank, rank_to_int, ranking_order, scaler
    except Exception as e:
        st.error(f"Error loading filtered model files: {e}")
        return None, None, None, None, None, None, None

def prepare_features(player_data, feature_cols, category_encoder, rank_to_int, ranking_order, scaler=None):
    """Prepare features for model prediction with advanced feature engineering"""
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

        # Encode categorical variables
        current_rank_encoded = rank_to_int.get(current_rank, -1)
        category_encoded = category_encoder.transform([category])[0] if category else -1

        # Create feature dictionary
        features = {
            'current_rank_encoded': current_rank_encoded,
            'category_encoded': category_encoded
        }

        # Add win/loss data from kaart and calculate totals/win rates
        total_wins = 0
        total_losses = 0
        
        for rank in ranking_order:
            wins = kaart.get(rank, [0, 0])[0]
            losses = kaart.get(rank, [0, 0])[1]
            total = wins + losses
            
            features[f"{rank}_wins"] = wins
            features[f"{rank}_losses"] = losses
            features[f"{rank}_total"] = total
            features[f"{rank}_win_rate"] = wins / total if total > 0 else 0
            
            total_wins += wins
            total_losses += losses

        # Add advanced features
        total_matches = total_wins + total_losses
        overall_win_rate = total_wins / total_matches if total_matches > 0 else 0
        
        features['total_wins'] = total_wins
        features['total_losses'] = total_losses
        features['total_matches'] = total_matches
        features['overall_win_rate'] = overall_win_rate
        features['performance_consistency'] = overall_win_rate * np.log1p(total_matches)
        features['recent_performance'] = overall_win_rate * min(total_matches / 10, 1)
        features['rank_progression_potential'] = overall_win_rate * (1 - current_rank_encoded / len(ranking_order))

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

def predict_next_rank(player_data, model, feature_cols, category_encoder, rank_to_int, int_to_rank, ranking_order, scaler=None):
    """Predict the next rank for a player with automatic junior leniency"""
    try:
        # Prepare features
        features = prepare_features(player_data, feature_cols, category_encoder, rank_to_int, ranking_order, scaler)

        if features is None:
            return None

        # Make prediction
        prediction = model.predict(features)[0]
        original_prediction = prediction
        
        # Get prediction probabilities
        prediction_proba = model.predict_proba(features)[0]
        confidence = prediction_proba.max()
        
        # Apply automatic leniency for JUN and J19 categories
        category = player_data.get('category', '')
        if category in ['JUN', 'J19']:
            # Calculate win rate for leniency determination
            kaart = player_data.get('kaart', {})
            total_wins = sum(wins for wins, _ in kaart.values())
            total_losses = sum(losses for _, losses in kaart.values())
            total_matches = total_wins + total_losses
            win_rate = total_wins / total_matches if total_matches > 0 else 0
            
            # Determine leniency based on performance and confidence
            if win_rate >= 0.6 and confidence >= 0.5:
                leniency = 2  # Strong performance + confident = 2 ranks better
            elif win_rate >= 0.5 or confidence >= 0.6:
                leniency = 1  # Good performance OR confident = 1 rank better
            else:
                leniency = 1  # Default junior leniency = 1 rank better
            
            # Apply leniency (lower number = better rank)
            adjusted_prediction = max(0, prediction - leniency)
            
            # Don't predict worse than current rank for juniors
            current_rank = player_data.get('ranking') or player_data.get('current_ranking')
            current_rank_encoded = rank_to_int.get(current_rank, prediction)
            if adjusted_prediction > current_rank_encoded:
                adjusted_prediction = current_rank_encoded
            
            if adjusted_prediction != original_prediction:
                original_rank_name = int_to_rank.get(original_prediction, "Unknown")
                adjusted_rank_name = int_to_rank.get(adjusted_prediction, "Unknown")
                st.info(f"🎯 Junior leniency: {original_rank_name} → {adjusted_rank_name} (win rate: {win_rate:.1%}, confidence: {confidence:.1%})")
            
            prediction = adjusted_prediction

        # Convert prediction back to rank label
        predicted_rank = int_to_rank.get(prediction, "Unknown")

        return predicted_rank

    except Exception as e:
        st.error(f"Error making prediction: {e}")
        import traceback
        st.error(traceback.format_exc())
        return None

def main():
    st.set_page_config(page_title="TT KlassementPredictor", page_icon="🏓", layout="wide")

    st.title("TT KlassementPredictor")
    st.markdown("Voorspel je klassment op basis van je kaart")

    # Load models and encoders
    regular_model, regular_category_encoder, regular_feature_cols, regular_int_to_rank, regular_rank_to_int, regular_ranking_order, regular_scaler = load_regular_model_and_encoders()
    filtered_model, filtered_category_encoder, filtered_feature_cols, filtered_int_to_rank, filtered_rank_to_int, filtered_ranking_order, filtered_scaler = load_filtered_model_and_encoders()

    if None in [regular_model, regular_category_encoder, regular_feature_cols, regular_int_to_rank, regular_rank_to_int, regular_ranking_order]:
        st.error("Failed to load regular model components. Please check the model files.")
        return

    if None in [filtered_model, filtered_category_encoder, filtered_feature_cols, filtered_int_to_rank, filtered_rank_to_int, filtered_ranking_order, filtered_scaler]:
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

    # Member selection based on club and season
    if club_code and season:
        members = get_members_for_club_season(club_code, season)
        if members:
            player_name = st.sidebar.selectbox("Select Player", members)
        else:
            player_name = st.sidebar.text_input("Player Name (manual entry)", value="")
            st.sidebar.warning("Could not load members automatically. Please enter name manually.")
    else:
        player_name = st.sidebar.text_input("Player Name", value="")
    
    # Main content area
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.header("Kaart & Prediction")
        
        if st.button("Get Player Data and Predict", type="primary"):
            with st.spinner("Fetching player data and making prediction..."):
                try:
                    # Get player data from API
                    player_data = get_data(club=club_code, name=player_name, season=season)
                    
                    if player_data:
                        # Display player information
                        st.subheader("Kaart")
                        
                        info_col1, info_col2, info_col3 = st.columns(3)
                        
                        with info_col1:
                            st.metric("Current Ranking", player_data.get('current_ranking', 'Unknown'))
                            st.metric("Category", player_data.get('category', 'Unknown'))
                        
                        with info_col2:
                            st.metric("Club", player_data.get('club_name', 'Unknown'))
                            st.metric("Province", player_data.get('province', 'Unknown'))
                        
                        with info_col3:
                            st.metric("Season", player_data.get('season', 'Unknown'))
                            st.metric("Unique Index", player_data.get('unique_index', 'Unknown'))
                        
                        # Choose model based on category
                        category = player_data.get('category')
                        if category in ["BEN", "PRE", "MIN", "CAD"]:
                            model = filtered_model
                            category_encoder = filtered_category_encoder
                            feature_cols = filtered_feature_cols
                            rank_to_int = filtered_rank_to_int
                            int_to_rank = filtered_int_to_rank
                            ranking_order = filtered_ranking_order
                            scaler = filtered_scaler
                            model_type = "filtered (youth categories)"
                        else:
                            model = regular_model
                            category_encoder = regular_category_encoder
                            feature_cols = regular_feature_cols
                            rank_to_int = regular_rank_to_int
                            int_to_rank = regular_int_to_rank
                            ranking_order = regular_ranking_order
                            scaler = regular_scaler  # Regular model now uses scaler too
                            model_type = "regular (all categories)"

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
                                'Rank/Category': group_key,
                                'Wins': wins,
                                'Losses': losses,
                                'Total Matches': total_matches,
                                'Win Rate': f"{(wins/total_matches)*100:.1f}%" if total_matches > 0 else "0%"
                            })

                        performance_df = pd.DataFrame(performance_data)
                        st.dataframe(performance_df, width='stretch')

                        # Make prediction
                        st.subheader("Voorspelling  Klassement")

                        predicted_rank = predict_next_rank(
                            player_data, model, feature_cols, category_encoder, 
                            rank_to_int, int_to_rank, ranking_order, scaler
                        )

                        if predicted_rank:
                            # Show prediction
                            current_rank = player_data.get('ranking') or player_data.get('current_ranking')
                            
                            # Display prediction with comparison
                            col1, col2, col3 = st.columns([1, 1, 1])
                            
                            with col1:
                                st.metric("Huidig Klassement", current_rank)
                            
                            with col2:
                                st.metric("Voorspeld Klassement", predicted_rank)
                            
                            with col3:
                                # Calculate rank change
                                if current_rank != predicted_rank:
                                    current_idx = rank_to_int.get(current_rank, 999)
                                    predicted_idx = rank_to_int.get(predicted_rank, 999)
                                    rank_diff = current_idx - predicted_idx
                                    
                                    if rank_diff > 0:
                                        st.metric("Verandering", f"↑ {rank_diff} rank{'s' if rank_diff > 1 else ''}", delta=f"+{rank_diff}", delta_color="normal")
                                    elif rank_diff < 0:
                                        st.metric("Verandering", f"↓ {abs(rank_diff)} rank{'s' if abs(rank_diff) > 1 else ''}", delta=f"{rank_diff}", delta_color="inverse")
                                    else:
                                        st.metric("Verandering", "Geen verandering", delta="0", delta_color="off")
                                else:
                                    st.metric("Verandering", "Geen verandering", delta="0", delta_color="off")
                            
                            st.caption(f"Model: {model_type} | Categorie: {category}")
                        else:
                            st.error("Unable to make prediction. Please check the input data.")
                        
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

                            # Color coding: green for good (>=60), yellow for average (40-59), red for poor (<40)
                            win_rate = total_wins / total_matches if total_matches > 0 else 0
                            rank_improvement = rank_difference
                            if performance_score >= 60:
                                st.metric("Performance Score (1-100)", performance_score, delta="Excellent")
                                st.success(f"Predicted ranking improvement: {rank_improvement} levels. {total_wins} wins out of {total_matches} matches ({win_rate:.1%} win rate)")
                            elif performance_score >= 40:
                                st.metric("Performance Score (1-100)", performance_score, delta="Good")
                                st.warning(f"Predicted ranking improvement: {rank_improvement} levels. {total_wins} wins out of {total_matches} matches ({win_rate:.1%} win rate)")
                            else:
                                st.metric("Performance Score (1-100)", performance_score, delta="Needs Improvement")
                                st.error(f"Predicted ranking improvement: {rank_improvement} levels. {total_wins} wins out of {total_matches} matches ({win_rate:.1%} win rate)")
                        else:
                            st.metric("Performance Score (1-100)", 0, delta="No Data")
                            st.info("No matches played this season")
                        
                    else:
                        st.error("Could not fetch player data. Please check the club code and player name.")
                        
                except Exception as e:
                    st.error(f"Error: {e}")
    
    with col2:
        st.header("DATA")
        st.markdown("""
        machine learning op data van Antwerpen seizoen 15-26
        """)
        
        st.header("Over Ons")
        st.markdown("""
        Gemaakt door Smets Sander
        credits:
        Smets Steven,
        Tim Jacobs,
        vttl api

        
        *Uses machine learning model trained on historical performance data.*
        """)
        
       

if __name__ == "__main__":
    main()