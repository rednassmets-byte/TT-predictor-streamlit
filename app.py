import streamlit as st
import pandas as pd
import pickle
import joblib
import requests
import os
import sys

# Add current directory to path to ensure imports work
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    from database_maker import get_data, get_province_for_club, get_club_name_for_club, get_information
except ImportError:
    st.error("Could not import database_maker module. Please ensure database_maker.py is in the same directory.")
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
def load_model_and_encoders():
    try:
        # Load all files using joblib (consistent with how they were saved)
        category_encoder = joblib.load('ai/category_encoder.pkl')
        feature_cols = joblib.load('ai/feature_cols.pkl')
        int_to_rank = joblib.load('ai/int_to_rank.pkl')
        rank_to_int = joblib.load('ai/rank_to_int.pkl')
        ranking_order = joblib.load('ai/ranking_order.pkl')
        model = joblib.load('ai/model.pkl')

        return model, category_encoder, feature_cols, int_to_rank, rank_to_int, ranking_order
    except Exception as e:
        st.error(f"Error loading model files: {e}")
        return None, None, None, None, None, None

def prepare_features(player_data, feature_cols, category_encoder, rank_to_int, ranking_order):
    """Prepare features for model prediction"""
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

        # Add win/loss data from kaart
        for rank in ranking_order:
            wins_key = f"{rank}_wins"
            losses_key = f"{rank}_losses"

            wins = kaart.get(rank, [0, 0])[0]
            losses = kaart.get(rank, [0, 0])[1]

            features[wins_key] = wins
            features[losses_key] = losses

        # Create DataFrame with all feature columns
        feature_df = pd.DataFrame([features])

        # Ensure all expected columns are present
        for col in feature_cols:
            if col not in feature_df.columns:
                feature_df[col] = 0

        return feature_df[feature_cols]

    except Exception as e:
        st.error(f"Error preparing features: {e}")
        return None

def predict_next_rank(player_data, model, feature_cols, category_encoder, rank_to_int, int_to_rank, ranking_order):
    """Predict the next rank for a player"""
    try:
        # Prepare features
        features = prepare_features(player_data, feature_cols, category_encoder, rank_to_int, ranking_order)

        if features is None:
            return None

        # Make prediction
        prediction = model.predict(features)[0]

        # Convert prediction back to rank label
        predicted_rank = int_to_rank.get(prediction, "Unknown")

        return predicted_rank

    except Exception as e:
        st.error(f"Error making prediction: {e}")
        return None

def main():
    st.set_page_config(page_title="TT KlassementPredictor", page_icon="🏓", layout="wide")
    
    st.title("TT KlassementPredictor")
    st.markdown("Voorspel je klassment op basis van je kaart")
    
    # Load model and encoders
    model, category_encoder, feature_cols, int_to_rank, rank_to_int, ranking_order = load_model_and_encoders()

    if None in [model, category_encoder, feature_cols, int_to_rank, rank_to_int, ranking_order]:
        st.error("Failed to load model components. Please check the model files.")
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

                        predicted_rank = predict_next_rank(player_data, model, feature_cols, category_encoder, rank_to_int, int_to_rank, ranking_order)

                        if predicted_rank:
                            st.success(f"**Voorspelling volgend klassement:** {predicted_rank}")
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