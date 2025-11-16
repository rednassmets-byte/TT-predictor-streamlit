import streamlit as st
import pandas as pd
import joblib
import os
import sys
import requests
from io import BytesIO

# Add path to TT predictor directory for database_maker import
sys.path.append("../../OneDrive/Documenten/TT predictor")

from database_maker import get_data, get_province_for_club, get_club_name_for_club

# Load saved model + encoders from ai folder
# Load model from Hugging Face
model_url = "https://huggingface.co/zouteboom4/ai_prediction/resolve/main/model.pkl"
response = requests.get(model_url)
response.raise_for_status()
model = joblib.load(BytesIO(response.content))
category_encoder = joblib.load("ai/category_encoder.pkl")
rank_to_int = joblib.load("ai/rank_to_int.pkl")
int_to_rank = joblib.load("ai/int_to_rank.pkl")
feature_cols = joblib.load("ai/feature_cols.pkl")
ranking_order = joblib.load("ai/ranking_order.pkl")

st.title("🏓 VTTL Ranking Predictor")

# -------------------------
# Define available categories
# -------------------------
available_categories = ["BEN", "PRE", "MIN", "CAD", "JUN", "J19", "SEN", "V40", "V50", "V60", "V70", "V80"]

# -------------------------
# Input Methods
# -------------------------
input_method = st.radio("Choose input method:", 
                       ["Manual Input", "Auto-fill from VTTL Database"])

# Initialize session state for auto-filled data
if 'auto_filled_data' not in st.session_state:
    st.session_state.auto_filled_data = None

if input_method == "Auto-fill from VTTL Database":
    st.subheader("Auto-fill from VTTL Database")
    
    col1, col2 = st.columns(2)
    with col1:
        club_code = st.text_input("Club Code (e.g., A182)", "A182").strip().upper()
        player_name = st.text_input("Player Name", "").strip()
    
    with col2:
        season = st.number_input("Season (e.g., 25 for 2024-2025)", 
                               min_value=20, max_value=30, value=25)
    
    if st.button("Fetch Player Data"):
        if club_code and player_name:
            try:
                with st.spinner("Fetching data from VTTL database..."):
                    player_data = get_data(club=club_code, name=player_name, season=season)

                    st.session_state.auto_filled_data = player_data
                    st.success("✅ Data fetched successfully!")

                    # Display player info
                    st.info(f"""
                    **Player Information:**
                    - **Club:** {player_data['club_naam']} ({club_code})
                    - **Province:** {player_data['provincie']}
                    - **Current Ranking:** {player_data['huidig_klassement']}
                    - **Next Season Ranking:** {player_data['volgend_klassement']}
                    - **Category:** {player_data['categorie']}
                    - **Season:** {player_data['seizoen']}
                    """)

            except Exception as e:
                st.error(f"❌ Error: {str(e)}")
        else:
            st.warning("⚠️ Please enter both club code and player name")
    
    st.divider()

# -------------------------
# Player Information Section
# -------------------------
st.subheader("Player Information")

# Set default values based on auto-filled data
default_ranking = st.session_state.auto_filled_data['huidig_klassement'][0] if st.session_state.auto_filled_data else ranking_order[0]
default_category = st.session_state.auto_filled_data['categorie'][0] if st.session_state.auto_filled_data else available_categories[0]

current_ranking = st.selectbox("Current Ranking", ranking_order, 
                              index=ranking_order.index(default_ranking) if default_ranking in ranking_order else 0)
category = st.selectbox("Category", available_categories,
                       index=available_categories.index(default_category) if default_category in available_categories else 0)

# -------------------------
# Match Results Section
# -------------------------
st.subheader("Match Results (Wins/Losses per Ranking)")
kaart_input = {}

# Pre-fill kaart if auto-filled data exists
auto_kaart = st.session_state.auto_filled_data['kaart'] if st.session_state.auto_filled_data else {}

for rank in ranking_order:
    with st.expander(f"Results vs {rank}"):
        default_wins = auto_kaart.get(rank, [0, 0])[0] if st.session_state.auto_filled_data else 0
        default_losses = auto_kaart.get(rank, [0, 0])[1] if st.session_state.auto_filled_data else 0
        
        wins = st.number_input(f"{rank} wins", min_value=0, value=default_wins, key=f"{rank}_wins")
        losses = st.number_input(f"{rank} losses", min_value=0, value=default_losses, key=f"{rank}_losses")
        
        if wins > 0 or losses > 0:
            kaart_input[rank] = [wins, losses]

# -------------------------
# Prepare row for prediction
# -------------------------
def prepare(new_rank, cat, kaart):
    data = {
        "current_rank_encoded": rank_to_int[new_rank],
        "category_encoded": category_encoder.transform([cat])[0]
    }

    for r in ranking_order:
        w, l = kaart.get(r, [0, 0])
        data[f"{r}_wins"] = w
        data[f"{r}_losses"] = l

    return pd.DataFrame([data])[feature_cols]

# -------------------------
# Predict
# -------------------------
if st.button("Predict Ranking"):
    try:
        row = prepare(current_ranking, category, kaart_input)
        pred = model.predict(row)[0]
        predicted_rank = int_to_rank[pred]
        
        st.success(f"**Predicted Next Ranking:** {predicted_rank}")
        
        # Show some additional info
        current_rank_num = rank_to_int[current_ranking]
        pred_rank_num = pred
        
        if pred_rank_num < current_rank_num:
            st.info("🎉 Prediction: **Promotion** expected!")
        elif pred_rank_num > current_rank_num:
            st.info("📉 Prediction: **Relegation** possible")
        else:
            st.info("➡️ Prediction: **Same ranking** likely")
            
    except ValueError as e:
        st.error(f"Error: Category '{category}' might not be recognized by the model. Please try another category.")
    except Exception as e:
        st.error(f"An error occurred: {str(e)}")

# Clear auto-filled data button
if st.session_state.auto_filled_data:
    if st.button("Clear Auto-filled Data"):
        st.session_state.auto_filled_data = None
        st.rerun()