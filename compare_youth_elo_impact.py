"""
Compare predictions for youth players with and without ELO model
"""
import pandas as pd
import joblib
import numpy as np
from database_maker import get_data

# Load models
def load_models():
    # V3 Filtered model (youth)
    category_encoder_v3 = joblib.load("category_encoder_filtered_v3.pkl")
    feature_cols_v3 = joblib.load("feature_cols_filtered_v3.pkl")
    int_to_rank_v3 = joblib.load("int_to_rank_filtered_v3.pkl")
    rank_to_int_v3 = joblib.load("rank_to_int_filtered_v3.pkl")
    ranking_order_v3 = joblib.load("ranking_order_filtered_v3.pkl")
    model_v3 = joblib.load("model_filtered_v3_improved.pkl")
    
    # V4 Special cases model (with ELO)
    try:
        model_v4 = joblib.load("model_v4_special_cases.pkl")
        feature_cols_v4 = joblib.load("feature_cols_v4_special.pkl")
        category_encoder_v4 = joblib.load("category_encoder_v4_hybrid.pkl")
        int_to_rank_v4 = joblib.load("int_to_rank_v4_hybrid.pkl")
        rank_to_int_v4 = joblib.load("rank_to_int_v4_hybrid.pkl")
        ranking_order_v4 = joblib.load("ranking_order_v4_hybrid.pkl")
    except:
        model_v4 = None
        feature_cols_v4 = None
        category_encoder_v4 = None
        int_to_rank_v4 = None
        rank_to_int_v4 = None
        ranking_order_v4 = None
    
    return {
        'v3': (model_v3, category_encoder_v3, feature_cols_v3, int_to_rank_v3, rank_to_int_v3, ranking_order_v3),
        'v4': (model_v4, category_encoder_v4, feature_cols_v4, int_to_rank_v4, rank_to_int_v4, ranking_order_v4)
    }

def prepare_features(player_data, feature_cols, category_encoder, rank_to_int, ranking_order, include_elo=False):
    """Prepare features for prediction"""
    current_rank = player_data.get('ranking') or player_data.get('current_ranking')
    category = player_data.get('category')
    kaart = player_data.get('kaart', {})
    elo = player_data.get('elo', 0)
    
    current_rank_encoded = rank_to_int.get(current_rank, -1)
    
    try:
        category_encoded = category_encoder.transform([category])[0] if category else -1
    except ValueError:
        category_mapping = {'PRE': 'MIN', 'BEN': 'CAD'}
        fallback_category = category_mapping.get(category, 'SEN')
        try:
            category_encoded = category_encoder.transform([fallback_category])[0]
        except ValueError:
            category_encoded = 0
    
    # Calculate performance metrics
    total_wins = sum(kaart.get(rank, [0, 0])[0] for rank in ranking_order)
    total_losses = sum(kaart.get(rank, [0, 0])[1] for rank in ranking_order)
    total_matches = total_wins + total_losses
    win_rate = total_wins / total_matches if total_matches > 0 else 0.5
    
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
    
    better_wins = 0
    better_losses = 0
    for rank in ranking_order:
        rank_idx = rank_to_int.get(rank, 999)
        if rank_idx < current_rank_encoded:
            wins, losses = kaart.get(rank, [0, 0])
            better_wins += wins
            better_losses += losses
    better_total = better_wins + better_losses
    vs_better_win_rate = better_wins / better_total if better_total > 0 else 0
    
    worse_wins = 0
    worse_losses = 0
    for rank in ranking_order:
        rank_idx = rank_to_int.get(rank, 999)
        if rank_idx > current_rank_encoded:
            wins, losses = kaart.get(rank, [0, 0])
            worse_wins += wins
            worse_losses += losses
    worse_total = worse_wins + worse_losses
    vs_worse_win_rate = worse_wins / worse_total if worse_total > 0 else 0
    
    win_loss_ratio = total_wins / total_losses if total_losses > 0 else total_wins
    performance_consistency = nearby_win_rate * np.log1p(total_matches)
    level_dominance = (nearby_win_rate - 0.5) * 2
    
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
    
    ng_matches = kaart.get('NG', [0, 0])
    ng_total = ng_matches[0] + ng_matches[1]
    ranked_matches = total_matches - ng_total
    ranked_opponent_ratio = ranked_matches / total_matches if total_matches > 0 else 0
    
    ng_win_rate = win_rate if current_rank == 'NG' else 0
    ng_has_many_matches = 1 if (current_rank == 'NG' and total_matches >= 20) else 0
    e6_ready_to_advance = 1 if (current_rank == 'E6' and nearby_win_rate > 0.6 and total_matches >= 15) else 0
    youth_match_reliability = np.tanh(total_matches / 40)
    
    # ELO features
    elo_log = 0
    elo_advantage = 0
    if include_elo:
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
        'elo_log': elo_log,
        'elo_advantage': elo_advantage
    }
    
    feature_df = pd.DataFrame([features])
    
    for col in feature_cols:
        if col not in feature_df.columns:
            feature_df[col] = 0
    
    feature_df = feature_df[feature_cols]
    
    return feature_df

def test_youth_players():
    """Test predictions for youth players"""
    models = load_models()
    
    # Load club members data to find youth players
    try:
        df = pd.read_csv("club_members_with_next_ranking.csv")
        # Filter for youth categories
        youth_df = df[df['category'].isin(['JUN', 'J19', 'CAD', 'MIN', 'PRE', 'BEN'])]
        # Take a sample of 10 youth players
        sample_df = youth_df.sample(min(10, len(youth_df)))
        
        test_cases = []
        for _, row in sample_df.iterrows():
            test_cases.append({
                'club': str(row['club']),
                'name': row['name'],
                'category': row['category']
            })
    except Exception as e:
        print(f"Could not load from CSV: {e}")
        print("Using manual test cases...")
        # Fallback to manual test cases
        test_cases = [
            {'club': '362', 'name': 'Smets Sander', 'category': 'JUN'},
        ]
    
    print("=" * 80)
    print("VERGELIJKING: V3 Model (zonder ELO) vs V4 Model (met ELO) voor JEUGD")
    print("=" * 80)
    print()
    
    results = []
    
    for test_case in test_cases:
        try:
            player_data = get_data(club=test_case['club'], name=test_case['name'], season=26)
            
            if not player_data:
                print(f"⚠️  Geen data gevonden voor {test_case['name']}")
                continue
            
            current_rank = player_data.get('ranking') or player_data.get('current_ranking')
            category = player_data.get('category')
            elo = player_data.get('elo', 0)
            
            # V3 prediction (without ELO)
            model_v3, cat_enc_v3, feat_v3, int_rank_v3, rank_int_v3, rank_order_v3 = models['v3']
            features_v3 = prepare_features(player_data, feat_v3, cat_enc_v3, rank_int_v3, rank_order_v3, include_elo=False)
            pred_v3 = model_v3.predict(features_v3)[0]
            pred_rank_v3 = int_rank_v3.get(pred_v3, "Unknown")
            conf_v3 = model_v3.predict_proba(features_v3)[0].max() * 100
            
            # V4 prediction (with ELO) - if available
            pred_rank_v4 = "N/A"
            conf_v4 = 0
            if models['v4'][0] is not None:
                model_v4, cat_enc_v4, feat_v4, int_rank_v4, rank_int_v4, rank_order_v4 = models['v4']
                features_v4 = prepare_features(player_data, feat_v4, cat_enc_v4, rank_int_v4, rank_order_v4, include_elo=True)
                pred_v4 = model_v4.predict(features_v4)[0]
                pred_rank_v4 = int_rank_v4.get(pred_v4, "Unknown")
                conf_v4 = model_v4.predict_proba(features_v4)[0].max() * 100
            
            # Calculate stats
            kaart = player_data.get('kaart', {})
            total_wins = sum(kaart.get(rank, [0, 0])[0] for rank in rank_order_v3)
            total_losses = sum(kaart.get(rank, [0, 0])[1] for rank in rank_order_v3)
            total_matches = total_wins + total_losses
            win_rate = total_wins / total_matches if total_matches > 0 else 0
            
            print(f"Speler: {test_case['name']}")
            print(f"Categorie: {category}")
            print(f"Huidig klassement: {current_rank}")
            print(f"ELO: {elo}")
            print(f"Wedstrijden: {total_matches} (Win rate: {win_rate:.1%})")
            print(f"")
            print(f"  V3 Model (ZONDER ELO): {pred_rank_v3} (zekerheid: {conf_v3:.1f}%)")
            print(f"  V4 Model (MET ELO):    {pred_rank_v4} (zekerheid: {conf_v4:.1f}%)")
            
            if pred_rank_v3 != pred_rank_v4:
                print(f"  ⚠️  VERSCHIL GEDETECTEERD!")
            else:
                print(f"  ✅ Zelfde voorspelling")
            
            print("-" * 80)
            print()
            
            results.append({
                'name': test_case['name'],
                'category': category,
                'current': current_rank,
                'elo': elo,
                'matches': total_matches,
                'win_rate': win_rate,
                'v3_pred': pred_rank_v3,
                'v3_conf': conf_v3,
                'v4_pred': pred_rank_v4,
                'v4_conf': conf_v4,
                'different': pred_rank_v3 != pred_rank_v4
            })
            
        except Exception as e:
            print(f"❌ Error voor {test_case['name']}: {e}")
            print()
    
    # Summary
    if results:
        print("=" * 80)
        print("SAMENVATTING")
        print("=" * 80)
        different_count = sum(1 for r in results if r['different'])
        print(f"Totaal getest: {len(results)} jeugdspelers")
        print(f"Verschillende voorspellingen: {different_count}")
        print(f"Zelfde voorspellingen: {len(results) - different_count}")
        print()
        
        if different_count > 0:
            print("CONCLUSIE: ELO model geeft ANDERE voorspellingen voor jeugd")
            print("           → Het is CORRECT om ELO model NIET te gebruiken voor jeugd")
        else:
            print("CONCLUSIE: ELO model geeft ZELFDE voorspellingen voor jeugd")
            print("           → Geen impact, maar toch beter om V3 te gebruiken")

if __name__ == "__main__":
    test_youth_players()
