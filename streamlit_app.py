import streamlit as st
import pandas as pd
import joblib
import os
from database_maker import get_data, get_province_for_club, get_club_name_for_club

# Load saved model + encoders from ai folder
model = joblib.load("ai/model.pkl")
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
        season = st.number_input("Season (e.g., 26 for 2025-2026)", 
                               min_value=20, max_value=26, value=26)
    
    if st.button("Fetch Player Data"):
        if club_code and player_name:
            try:
                with st.spinner("Fetching data from VTTL database..."):
                    # Call the function and store the result
                    player_data = get_data(club=club_code, name=player_name, season=season)
                    
                    if player_data:
                        # Store in session state
                        st.session_state.auto_filled_data = player_data
                        
                        st.success("✅ Data fetched successfully!")
                        
                        # Display player info
                        st.info(f"""
                        **Player Information:**
                        - **Club:** {player_data['club_name']} ({club_code})
                        - **Province:** {player_data['province']}
                        - **Current Ranking:** {player_data['current_ranking']}
                        - **Category:** {player_data['category']}
                        - **Season:** {player_data['season']}
                        """)
                    else:
                        st.error("❌ No data found for this player and club combination.")
                        
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
if st.session_state.auto_filled_data:
    default_ranking = st.session_state.auto_filled_data.get('current_ranking', ranking_order[0])
    default_category = st.session_state.auto_filled_data.get('category', available_categories[0])
else:
    default_ranking = ranking_order[0]
    default_category = available_categories[0]

# Ensure the default values are in the respective lists
if default_ranking not in ranking_order:
    default_ranking = ranking_order[0]
if default_category not in available_categories:
    default_category = available_categories[0]

current_ranking = st.selectbox("Current Ranking", ranking_order, 
                              index=ranking_order.index(default_ranking))

category = st.selectbox("Category", available_categories,
                       index=available_categories.index(default_category))

# -------------------------
# Match Results Section
# -------------------------
st.subheader("Match Results")

# Initialize kaart_input
kaart_input = {}

if input_method == "Auto-fill from VTTL Database" and st.session_state.auto_filled_data:
    # For auto-fill mode, use the auto-filled kaart data
    st.info("📊 Match results automatically loaded from VTTL database")
    
    kaart_input = st.session_state.auto_filled_data.get('kaart', {})
    
    # Show summary of loaded data
    total_wins = sum(wins for wins, losses in kaart_input.values())
    total_losses = sum(losses for wins, losses in kaart_input.values())
    
    st.write(f"**Loaded:** {total_wins} wins and {total_losses} losses from current season")
    
    # Show detailed results in a table
    results_data = []
    for rank, (wins, losses) in kaart_input.items():
        if wins > 0 or losses > 0:
            results_data.append({"Rank": rank, "Wins": wins, "Losses": losses})
    
    if results_data:
        st.write("**Match Results Summary:**")
        results_df = pd.DataFrame(results_data)
        st.dataframe(results_df, hide_index=True, use_container_width=True)
    else:
        st.info("No match results found for this player in the current season.")
        
else:
    # Manual input mode
    st.write("Enter wins/losses per ranking:")
    
    for rank in ranking_order:
        with st.expander(f"Results vs {rank}"):
            wins = st.number_input(f"{rank} wins", min_value=0, value=0, key=f"{rank}_wins")
            losses = st.number_input(f"{rank} losses", min_value=0, value=0, key=f"{rank}_losses")
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
        # For auto-fill mode, use the auto-filled kaart data
        if input_method == "Auto-fill from VTTL Database" and st.session_state.auto_filled_data:
            kaart_to_use = st.session_state.auto_filled_data.get('kaart', {})
        else:
            kaart_to_use = kaart_input
        
        row = prepare(current_ranking, category, kaart_to_use)
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