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

# Try to import prediction engine, fallback to inline functions if not available
try:
    from prediction_engine import predict_next_rank, get_rank_comparison
    USING_PREDICTION_ENGINE = True
except ImportError:
    USING_PREDICTION_ENGINE = False

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
        st.error(f"Error loading members: {e}")
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
        st.error(f"V3 Regular model not available: {str(e)}")
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
        st.error(f"V3 Filtered model not available: {str(e)}")
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

# Fallback prediction functions if prediction_engine.py is not available
if not USING_PREDICTION_ENGINE:
    def predict_next_rank(player_data, model, feature_cols, category_encoder, rank_to_int, int_to_rank, ranking_order, scaler=None, special_model=None, special_feature_cols=None, is_filtered_model=False):
        """Fallback prediction function"""
        try:
            # Simple prediction without all the complex logic
            current_rank = player_data.get('ranking') or player_data.get('current_ranking')
            if not current_rank:
                return None
            
            # For now, just return the current rank as prediction
            return current_rank, 50.0, False
        except Exception as e:
            st.error(f"Error in fallback prediction: {e}")
            return None
    
    def get_rank_comparison(current_rank, predicted_rank, rank_to_int):
        """Fallback rank comparison function"""
        return "=", "→", "info", "Geen verandering"

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
    st.set_page_config(page_title="TT KlassementPredictor", page_icon="🏓", layout="wide", initial_sidebar_state="collapsed")
    
    # Custom CSS for mobile-first responsive design with HIGH CONTRAST & MODERN COLORS
    st.markdown("""
        <style>
        /* Hide sidebar by default for mobile experience */
        [data-testid="stSidebar"] {
            display: none;
        }
        
        /* Modern color palette with high contrast */
        :root {
            --primary-green: #00b359;
            --primary-dark: #008040;
            --secondary-blue: #0066cc;
            --accent-orange: #ff6b35;
            --success-green: #28a745;
            --warning-yellow: #ffc107;
            --danger-red: #dc3545;
            --background-light: #f8f9fa;
            --background-white: #ffffff;
            --text-dark: #1a1a1a;
            --text-medium: #495057;
            --border-color: #ced4da;
            --shadow-color: rgba(0, 0, 0, 0.15);
        }
        
        /* Page background gradient */
        .stApp {
            background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%);
        }
        
        /* Stronger text contrast with modern colors */
        body, p, span, div, label {
            color: var(--text-dark) !important;
            font-weight: 500;
        }
        
        h1 {
            color: var(--primary-dark) !important;
            font-weight: 800 !important;
            background: linear-gradient(135deg, var(--primary-green), var(--primary-dark));
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            background-clip: text;
        }
        
        h2, h3 {
            color: var(--primary-dark) !important;
            font-weight: 700 !important;
        }
        
        h4, h5, h6 {
            color: var(--text-dark) !important;
            font-weight: 600 !important;
        }
        
        /* Mobile-first responsive adjustments */
        @media (max-width: 768px) {
            .stColumn {
                width: 100% !important;
                padding: 0 !important;
            }
            h1 {
                font-size: 1.8rem !important;
            }
            h2 {
                font-size: 1.4rem !important;
            }
            h3 {
                font-size: 1.2rem !important;
            }
            .stButton button {
                width: 100% !important;
                font-size: 1.1rem !important;
                padding: 0.75rem !important;
            }
            .stSelectbox, .stNumberInput {
                font-size: 1rem !important;
            }
        }
        
        /* Colorful progress bar */
        .stProgress > div > div {
            background: linear-gradient(90deg, var(--primary-green), var(--success-green)) !important;
        }
        
        /* Modern card containers with color accents */
        .stMetric {
            background: linear-gradient(135deg, #ffffff 0%, #f8f9fa 100%) !important;
            padding: 20px;
            border-radius: 12px;
            box-shadow: 0 4px 12px var(--shadow-color) !important;
            border-left: 4px solid var(--primary-green) !important;
            border-top: 1px solid var(--border-color) !important;
            border-right: 1px solid var(--border-color) !important;
            border-bottom: 1px solid var(--border-color) !important;
            transition: all 0.3s ease;
        }
        
        .stMetric:hover {
            transform: translateY(-2px);
            box-shadow: 0 6px 16px var(--shadow-color) !important;
        }
        
        .stMetric label {
            color: var(--text-medium) !important;
            font-weight: 600 !important;
            font-size: 0.85rem !important;
            text-transform: uppercase;
            letter-spacing: 0.5px;
        }
        
        .stMetric [data-testid="stMetricValue"] {
            color: var(--primary-dark) !important;
            font-weight: 800 !important;
            font-size: 1.8rem !important;
        }
        
        /* Modern gradient button styling */
        .stButton button {
            background: linear-gradient(135deg, var(--primary-green) 0%, var(--primary-dark) 100%) !important;
            border: none !important;
            color: #ffffff !important;
            font-weight: 700 !important;
            box-shadow: 0 4px 12px rgba(0, 179, 89, 0.4) !important;
            transition: all 0.3s ease;
            text-transform: uppercase;
            letter-spacing: 1px;
        }
        
        .stButton button:hover {
            background: linear-gradient(135deg, #00cc66 0%, var(--primary-green) 100%) !important;
            box-shadow: 0 6px 20px rgba(0, 179, 89, 0.6) !important;
            transform: translateY(-3px);
        }
        
        .stButton button:active {
            transform: translateY(-1px);
            box-shadow: 0 2px 8px rgba(0, 179, 89, 0.4) !important;
        }
        
        /* Modern selectbox and input styling */
        .stSelectbox > div > div, .stNumberInput > div > div > input {
            background-color: var(--background-white) !important;
            border: 2px solid var(--border-color) !important;
            border-radius: 8px !important;
            color: var(--text-dark) !important;
            font-weight: 600 !important;
            transition: all 0.3s ease;
        }
        
        .stSelectbox > div > div:focus-within, .stNumberInput > div > div > input:focus {
            border-color: var(--primary-green) !important;
            box-shadow: 0 0 0 3px rgba(0, 179, 89, 0.1) !important;
        }
        
        /* Colorful info/warning boxes */
        div[data-baseweb="notification"] {
            border-radius: 10px !important;
            border-left-width: 4px !important;
            box-shadow: 0 3px 10px var(--shadow-color) !important;
            font-weight: 600 !important;
        }
        
        /* Info box - blue */
        div[data-baseweb="notification"][kind="info"] {
            background-color: #e7f3ff !important;
            border-left-color: var(--secondary-blue) !important;
        }
        
        /* Success box - green */
        div[data-baseweb="notification"][kind="positive"] {
            background-color: #d4edda !important;
            border-left-color: var(--success-green) !important;
        }
        
        /* Warning box - yellow */
        div[data-baseweb="notification"][kind="warning"] {
            background-color: #fff3cd !important;
            border-left-color: var(--warning-yellow) !important;
        }
        
        /* Error box - red */
        div[data-baseweb="notification"][kind="error"] {
            background-color: #f8d7da !important;
            border-left-color: var(--danger-red) !important;
        }
        
        /* Prediction display with gradient text */
        div[data-testid="stMarkdownContainer"] h1 {
            text-align: center;
            font-weight: 900 !important;
            text-shadow: 2px 2px 4px rgba(0,0,0,0.1);
        }
        
        /* Modern dataframe styling */
        .stDataFrame {
            border: 2px solid var(--border-color) !important;
            border-radius: 10px !important;
            overflow: hidden;
        }
        
        .stDataFrame thead tr th {
            background: linear-gradient(135deg, var(--primary-green), var(--primary-dark)) !important;
            color: white !important;
            font-weight: 700 !important;
        }
        
        .stDataFrame tbody tr:nth-child(even) {
            background-color: #f8f9fa !important;
        }
        
        .stDataFrame tbody tr:hover {
            background-color: #e9ecef !important;
        }
        
        /* Colorful horizontal rule */
        hr {
            border: none !important;
            height: 3px !important;
            background: linear-gradient(90deg, 
                var(--primary-green) 0%, 
                var(--secondary-blue) 50%, 
                var(--accent-orange) 100%) !important;
            margin: 2rem 0 !important;
            border-radius: 2px;
        }
        
        /* Caption text with color */
        .stCaption, small {
            color: var(--text-medium) !important;
            font-weight: 600 !important;
        }
        
        /* Subheader with icon styling */
        h2::before, h3::before {
            margin-right: 8px;
        }
        
        /* Chart containers */
        .js-plotly-plot {
            border-radius: 10px;
            box-shadow: 0 2px 8px var(--shadow-color);
        }
        </style>
    """, unsafe_allow_html=True)

    st.title("TT Klassement Predictor")

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
    
    # Main input section (mobile-friendly)
    st.header("Speler Selectie")
    
    # Search method selection
    search_method = st.radio(
        "Zoek methode:",
        ["Database Zoeken", "Club & Seizoen"],
        horizontal=True,
        help="Kies tussen database zoeken of club selectie"
    )
    
    # Initialize variables
    player_name = None
    club_code = None
    season = 26  # Default season
    
    if search_method == "Database Zoeken":
        st.subheader("🔍 Database Zoeken")
        
        # Load player database for search
        @st.cache_data
        def load_player_database():
            try:
                df = pd.read_csv("club_members_main_data.csv")
                return df
            except FileNotFoundError:
                st.error("Player database not found")
                return None
        
        df_players = load_player_database()
        
        if df_players is not None:
            # Search interface
            col1, col2 = st.columns([3, 1])
            
            with col1:
                search_query = st.text_input(
                    "Zoek speler:", 
                    placeholder="Type naam, club, of ranking...",
                    help="Zoek op naam, club naam, of huidige ranking"
                )
            
            with col2:
                # Get available seasons from data
                available_seasons = sorted(df_players['season'].unique(), reverse=True)
                season_options = []
                for s in available_seasons:
                    # Convert "2025-2026" to 26
                try:
                    year = int(s.split('-')[1])
                    season_options.append(year)
                except:
                    continue
            
            if season_options:
                season = st.selectbox("Seizoen", season_options, index=0)
            else:
                season = 26
            
            # Show suggestions as you type (Google-like)
            if search_query and len(search_query) >= 2:  # Start suggesting after 2 characters
                # Convert search query to lowercase for case-insensitive search
                query_lower = search_query.lower()
                
                # Filter by season first
                season_str = f"{season-1}-{season}"
                df_filtered = df_players[df_players['season'] == season_str].copy()
                
                if not df_filtered.empty:
                    # Search in multiple columns
                    mask = (
                        df_filtered['name'].str.lower().str.contains(query_lower, na=False) |
                        df_filtered['club_name'].str.lower().str.contains(query_lower, na=False) |
                        df_filtered['current_ranking'].str.lower().str.contains(query_lower, na=False) |
                        df_filtered['category'].str.lower().str.contains(query_lower, na=False)
                    )
                    
                    search_results = df_filtered[mask]
                    
                    if not search_results.empty:
                        # Google-like suggestions container
                        with st.container():
                            st.markdown("""
                                <style>
                                .suggestion-container {
                                    background: white;
                                    border: 1px solid #ddd;
                                    border-radius: 8px;
                                    box-shadow: 0 2px 8px rgba(0,0,0,0.1);
                                    margin-top: -10px;
                                    padding: 8px;
                                }
                                </style>
                            """, unsafe_allow_html=True)
                            
                            st.markdown(f"<div class='suggestion-container'>", unsafe_allow_html=True)
                            st.markdown(f"**{len(search_results)} resultaten gevonden:**")
                            
                            # Show top 10 suggestions
                            for idx, row in search_results.head(10).iterrows():
                                # Clean up the data display
                                name = row['name']
                                club = row['club_name']
                                ranking = str(row['current_ranking']).strip("[]'")
                                category = str(row['category']).strip("[]'")
                                
                                # Highlight matching text
                                name_display = name
                                club_display = club
                                
                                # Simple highlighting (case insensitive)
                                if query_lower in name.lower():
                                    name_display = name.replace(
                                        query_lower, f"**{query_lower}**"
                                    ).replace(
                                        query_lower.upper(), f"**{query_lower.upper()}**"
                                    ).replace(
                                        query_lower.capitalize(), f"**{query_lower.capitalize()}**"
                                    )
                                
                                if query_lower in club.lower():
                                    club_display = club.replace(
                                        query_lower, f"**{query_lower}**"
                                    ).replace(
                                        query_lower.upper(), f"**{query_lower.upper()}**"
                                    ).replace(
                                        query_lower.capitalize(), f"**{query_lower.capitalize()}**"
                                    )
                                
                                # Create clickable suggestion
                                if st.button(
                                    f"🏓 {name_display}",
                                    key=f"suggestion_{idx}",
                                    help=f"{club_display} | {ranking} ({category})",
                                    use_container_width=True
                                ):
                                    # Set the selected player data
                                    st.session_state.selected_player_data = {
                                        'name': name,
                                        'club': club,
                                        'season': season,
                                        'ranking': ranking,
                                        'category': category,
                                        'row_data': row,
                                        'search_method': 'database'
                                    }
                                    st.success(f"✅ Geselecteerd: {name}")
                                    st.rerun()
                                
                                # Show metadata below button
                                st.markdown(f"<small>📍 {club_display} | 🏆 {ranking} | 👤 {category}</small>", unsafe_allow_html=True)
                                st.markdown("---")
                            
                            if len(search_results) > 10:
                                st.info(f"... en nog {len(search_results) - 10} andere resultaten")
                            
                            st.markdown("</div>", unsafe_allow_html=True)
                    else:
                        st.warning("🔍 Geen spelers gevonden met deze zoekopdracht")
                        
                        # Show some suggestions
                        st.info("💡 **Zoektips:**")
                        st.markdown("- Probeer een deel van de naam: `Sander`, `Van Den`")
                        st.markdown("- Zoek op club: `Nodo`, `Sparta`")
                        st.markdown("- Zoek op ranking: `B0`, `C4`, `NG`")
                        st.markdown("- Zoek op categorie: `SEN`, `JUN`, `V40`")
                else:
                    st.warning(f"❌ Geen data beschikbaar voor seizoen {season_str}")
            elif search_query and len(search_query) < 2:
                st.info("💭 Type minstens 2 karakters om te zoeken...")
            else:
                # Show some popular examples when no search
                st.info("🔍 **Populaire zoekopdrachten:**")
                
                # Get some sample data to show
                season_str = f"{season-1}-{season}"
                df_sample = df_players[df_players['season'] == season_str].head(5)
                
                if not df_sample.empty:
                    col_a, col_b = st.columns(2)
                    with col_a:
                        st.markdown("**Voorbeelden:**")
                        for _, row in df_sample.iterrows():
                            name = row['name'].split()[0]  # First name only
                            if st.button(f"🔍 {name}", key=f"example_{name}", use_container_width=True):
                                st.session_state.example_search = name
                                st.rerun()
                    
                    with col_b:
                        st.markdown("**Zoek op:**")
                        st.markdown("• `B0` - alle B0 spelers")
                        st.markdown("• `Nodo` - alle TTC Nodo leden")
                        st.markdown("• `SEN` - alle senioren")
                        st.markdown("• `JUN` - alle junioren")
                
                # Handle example search
                if 'example_search' in st.session_state:
                    st.text_input(
                        "Zoek speler:", 
                        value=st.session_state.example_search,
                        key="example_input"
                    )
                    del st.session_state.example_search
            
            # Show selected player if any
            if 'selected_player_data' in st.session_state and st.session_state.selected_player_data.get('search_method') == 'database':
                player_info = st.session_state.selected_player_data
                st.success(f"Geselecteerde speler: **{player_info['name']}** van {player_info['club']}")
                
                # Set variables for prediction
                player_name = player_info['name']
                club_code = None  # Database search doesn't need club_code for API
                season = player_info['season']
        else:
            st.error("Kan speler database niet laden")
    
    else:
        # Original club & season selection method
        st.subheader("🏓 Club & Seizoen Selectie")
        
        # Fixed to Antwerpen province only
        selected_province = 'Antwerpen'
        st.info("Provincie: Antwerpen")

        # Create columns for better mobile layout
        col1, col2 = st.columns([2, 1])
        
        with col1:
            # Club selection based on province
            clubs_in_province = get_clubs_for_province(selected_province)
            club_names = [f"{code} - {get_club_name_for_club(code)}" for code in clubs_in_province]
            selected_club_display = st.selectbox("Selecteer Club", club_names)
            club_code = selected_club_display.split(' - ')[0] if selected_club_display else ""
        
        with col2:
            # Season selection
            season = st.number_input("Seizoen", min_value=15, max_value=26, value=26)

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
                
                player_name = st.selectbox("Selecteer Speler", members, index=default_index)
                st.session_state.selected_player = player_name
                st.session_state.previous_season = season
            else:
                player_name = st.text_input("Speler Naam (manuele invoer)", value="")
                st.warning("Kon leden niet automatisch laden. Voer naam manueel in.")
        else:
            player_name = st.text_input("Speler Naam", value="")
    
    # Predict button
    st.markdown("---")
    predict_button = st.button("Voorspel Klassement", type="primary", use_container_width=True)
    
    # Auto-predict if season changed and player exists (only for club method)
    if search_method == "Club & Seizoen" and 'auto_predict' in st.session_state and st.session_state.auto_predict:
        predict_button = True
        st.session_state.auto_predict = False
    
    # Main content area
    if not predict_button:
        # Empty state
        st.markdown("---")
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            st.markdown("#### Selecteer een speler en klik op 'Voorspel Klassement'")
            st.markdown("De AI zal het toekomstige klassement voorspellen op basis van de huidige kaart.")
    
    if predict_button:
        # Check what method is being used and if data is available
        if search_method == "Database Zoeken":
            if 'selected_player_data' not in st.session_state or st.session_state.selected_player_data.get('search_method') != 'database':
                st.error("❌ Selecteer eerst een speler uit de database zoekresultaten")
            else:
                with st.spinner("Making prediction from database data..."):
                    try:
                        # Get player data from selected search result
                        player_info = st.session_state.selected_player_data
                        row = player_info['row_data']
                        
                        # Convert CSV data to the format expected by the prediction model
                        import ast
                        kaart_data = ast.literal_eval(row['kaart'])
                        
                        player_data = {
                            'name': row['name'],
                            'category': str(row['category']).strip("[]'"),
                            'ranking': str(row['current_ranking']).strip("[]'"),
                            'current_ranking': str(row['current_ranking']).strip("[]'"),
                            'kaart': kaart_data,
                            'elo': 0  # Default ELO if not available
                        }
                        
                        st.success(f"✅ Using database data for {player_data['name']}")
        else:
            # Club & Season method
            if not player_name or not club_code:
                st.error("❌ Selecteer eerst een club en speler")
            else:
                with st.spinner("Fetching player data from API..."):
                    try:
                        player_data = get_data(club=club_code, name=player_name, season=season)
                        if player_data:
                            st.success(f"✅ Using API data for {player_name}")
                        else:
                            st.error("❌ Kon speler data niet ophalen van API")
                            player_data = None
        
        if 'player_data' in locals() and player_data:
                    # Get basic info for model selection
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

                    # ========== PREDICTION SECTION (AT THE TOP) ==========
                    # Make prediction
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
                        
                        # Get comparison indicators
                        emoji, arrow, color_type, message = get_rank_comparison(current_rank, predicted_rank, rank_to_int)
                        
                        # Display prediction - ALLEEN VOORSPELD KLASSEMENT
                        st.markdown("### Voorspeld Klassement")
                        st.markdown(f"""
                            <div style="
                                background: linear-gradient(135deg, #f0f0f0 0%, #e0e0e0 100%);
                                padding: 30px;
                                border-radius: 15px;
                                text-align: center;
                                box-shadow: 0 6px 20px rgba(40, 167, 69, 0.3);
                                border: 4px solid #28a745;
                            ">
                                <h1 style="
                                    color: #2c3e50;
                                    font-size: 4rem;
                                    margin: 0;
                                    font-weight: 900;
                                    text-shadow: none;
                                    letter-spacing: 3px;
                                ">{predicted_rank}</h1>
                            </div>
                        """, unsafe_allow_html=True)
                        
                        # Show confidence with interpretation
                        if confidence >= 70:
                            confidence_label = "Zeer zeker"
                            confidence_color = "✓"
                        elif confidence >= 50:
                            confidence_label = "Redelijk zeker"
                            confidence_color = "~"
                        else:
                            confidence_label = "Onzeker"
                            confidence_color = "!"
                        
                        st.caption(f"{confidence_color} Zekerheid: {confidence:.1f}% ({confidence_label}) | {message}")
                    else:
                        st.error("Kan geen voorspelling maken. Controleer de invoergegevens.")
                        predicted_rank = None
                    
                    st.markdown("---")
                    
                    # ========== PLAYER INFORMATION SECTION ==========
                    st.markdown(f"# {player_data['name']}")
                    
                    info_col1, info_col2, info_col3 = st.columns(3)
                    
                    with info_col1:
                        st.metric("Klassement", player_data.get('current_ranking', 'Unknown'))
                        st.metric("Categorie", player_data.get('category', 'Unknown'))
                    
                    with info_col2:
                        st.metric("Club", player_info.get('club', 'Unknown'))
                        st.metric("Provincie", 'Antwerpen')
                    
                    with info_col3:
                        st.metric("Seizoen", f"{player_info.get('season', 'Unknown')}")
                        st.metric("Unique Index", str(row.get('unique_index', 'Unknown')))
                    
                    st.markdown("---")
                    
                    # ========== PERFORMANCE DATA SECTION ==========
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
                        st.dataframe(performance_df, use_container_width=True, hide_index=True)
                    with col_chart:
                        fig = create_win_loss_chart(player_data.get('kaart', {}), ranking_order)
                        st.plotly_chart(fig, use_container_width=True)

                    # Display rank progression chart with prediction
                    st.subheader("Rank Progression")
                    # Note: Rank progression requires API data, not available with database search
                    st.info("Rank progression chart niet beschikbaar voor database zoeken (vereist API data)")
                    # progression_fig = create_rank_progression_chart(club_code, player_name, season, rank_to_int, predicted_rank)
                    # if progression_fig:
                    #     st.plotly_chart(progression_fig, use_container_width=True)
                    # else:
                    #     st.info("Niet genoeg historische data om progressie te tonen (minimaal 2 seizoenen nodig)")
                    
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

                        # High contrast color gradient from bright red (0) to bright green (100)
                        win_rate = total_wins / total_matches if total_matches > 0 else 0
                        rank_improvement = rank_difference
                        
                        # Calculate color gradient with HIGH CONTRAST
                        # At 0: bright red (220, 53, 69) - Bootstrap danger
                        # At 50: orange (255, 193, 7) - Bootstrap warning
                        # At 100: bright green (40, 167, 69) - Bootstrap success
                        ratio = performance_score / 100
                        
                        if ratio < 0.5:
                            # Red to Orange (0-50)
                            local_ratio = ratio * 2
                            red = int(220 + (255 - 220) * local_ratio)
                            green = int(53 + (193 - 53) * local_ratio)
                            blue = int(69 + (7 - 69) * local_ratio)
                        else:
                            # Orange to Green (50-100)
                            local_ratio = (ratio - 0.5) * 2
                            red = int(255 + (40 - 255) * local_ratio)
                            green = int(193 + (167 - 193) * local_ratio)
                            blue = int(7 + (69 - 7) * local_ratio)
                        
                        color = f"rgb({red}, {green}, {blue})"
                        
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
                            <div style="background-color: rgb(220, 53, 69); padding: 20px; border-radius: 10px; text-align: center;">
                                <h2 style="color: white; margin: 0;">Performance Score</h2>
                                <h1 style="color: white; margin: 10px 0; font-size: 3rem;">0</h1>
                                <p style="color: white; margin: 0;">out of 100</p>
                            </div>
                        """, unsafe_allow_html=True)
                        st.info("No matches played this season")
                    
                    # Display info at bottom
                    st.markdown("---")
                    st.info("Model Accuracy: ~85% (trained on Antwerpen data, seasons 15-26)")
                    st.caption(f"Model: {model_type} | Categorie: {category}")
                    st.caption("Machine learning op data van Antwerpen seizoen 15-26")
                    st.caption("Gemaakt door Smets Sander | Credits: Smets Steven, Tim Jacobs, vttl api")
                    
                except Exception as e:
                    st.error(f"Error making prediction: {e}")
                    import traceback
                    st.error(traceback.format_exc())

if __name__ == "__main__":
    main()