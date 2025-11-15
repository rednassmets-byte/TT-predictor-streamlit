import pandas as pd
import numpy as np
import ast
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import classification_report, accuracy_score
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import joblib

# Data laden en voorbereiden
def load_and_preprocess_data(file_path):
    df = pd.read_csv(file_path)
    
    # Kaart data parsen
    def parse_kaart(kaart_str):
        try:
            kaart_dict = ast.literal_eval(kaart_str)
            return kaart_dict
        except:
            return {}
    
    df['kaart_dict'] = df['kaart'].apply(parse_kaart)
    
    return df

def create_features(df):
    features = []
    target = []
    
    # Ranking volgorde voor numerieke representatie
    ranking_order = ['NG', 'E6', 'E4', 'E2', 'E0', 'D6', 'D4', 'D2', 'D0', 
                   'C6', 'C4', 'C2', 'C0', 'B6', 'B4', 'B2', 'B0', 'A']
    ranking_mapping = {rank: i for i, rank in enumerate(ranking_order)}
    
    for _, row in df.iterrows():
        if not row['kaart_dict']:
            continue
            
        # Basis features
        current_rank = row['current_ranking'] if row['current_ranking'] else 'NG'
        category = row['category'] if row['category'] else 'SEN'
        next_rank = row['next_ranking'] if row['next_ranking'] else 'NG'
        
        # Huidige ranking als numerieke waarde
        current_rank_num = ranking_mapping.get(current_rank, 0)
        
        # Category encoding
        category_encoded = hash(category) % 100  # Simpele encoding
        
        # Statistieken uit kaart data
        kaart = row['kaart_dict']
        total_wins = 0
        total_losses = 0
        win_rate_per_rank = []
        
        for rank in ranking_order:
            if rank in kaart:
                wins, losses = kaart[rank]
                total_wins += wins
                total_losses += losses
                if wins + losses > 0:
                    win_rate = wins / (wins + losses)
                    win_rate_per_rank.append(win_rate)
        
        # Algemene statistieken
        total_matches = total_wins + total_losses
        overall_win_rate = total_wins / total_matches if total_matches > 0 else 0
        avg_win_rate = np.mean(win_rate_per_rank) if win_rate_per_rank else 0
        
        # Prestatie tegen hogere/ lagere rankings
        current_rank_idx = ranking_mapping.get(current_rank, 0)
        performance_against_higher = 0
        performance_against_lower = 0
        matches_against_higher = 0
        matches_against_lower = 0
        
        for rank, (wins, losses) in kaart.items():
            rank_idx = ranking_mapping.get(rank, 0)
            if rank_idx > current_rank_idx:  # Hogere ranking
                performance_against_higher += wins
                matches_against_higher += wins + losses
            elif rank_idx < current_rank_idx:  # Lagere ranking
                performance_against_lower += wins
                matches_against_lower += wins + losses
        
        perf_vs_higher = performance_against_higher / matches_against_higher if matches_against_higher > 0 else 0
        perf_vs_lower = performance_against_lower / matches_against_lower if matches_against_lower > 0 else 0
        
        # Feature vector
        feature_vector = [
            current_rank_num,
            category_encoded,
            total_wins,
            total_losses,
            total_matches,
            overall_win_rate,
            avg_win_rate,
            perf_vs_higher,
            perf_vs_lower,
            len(win_rate_per_rank)  # Aantal rankings waartegen gespeeld
        ]
        
        features.append(feature_vector)
        target.append(ranking_mapping.get(next_rank, 0))
    
    return np.array(features), np.array(target), ranking_mapping

# Feature namen voor debugging
feature_names = [
    'current_rank_numeric', 'category_encoded', 'total_wins', 'total_losses',
    'total_matches', 'overall_win_rate', 'avg_win_rate_per_rank',
    'performance_vs_higher', 'performance_vs_lower', 'num_opponent_ranks'
]


class DeepLearningModel(nn.Module):
    def __init__(self, input_dim, num_classes):
        super(DeepLearningModel, self).__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.bn1 = nn.BatchNorm1d(128)
        self.dropout1 = nn.Dropout(0.3)
        
        self.fc2 = nn.Linear(128, 64)
        self.bn2 = nn.BatchNorm1d(64)
        self.dropout2 = nn.Dropout(0.3)
        
        self.fc3 = nn.Linear(64, 32)
        self.dropout3 = nn.Dropout(0.2)
        
        self.fc4 = nn.Linear(32, num_classes)
        
    def forward(self, x):
        x = torch.relu(self.bn1(self.fc1(x)))
        x = self.dropout1(x)
        x = torch.relu(self.bn2(self.fc2(x)))
        x = self.dropout2(x)
        x = torch.relu(self.fc3(x))
        x = self.dropout3(x)
        x = self.fc4(x)
        return x

def create_deep_learning_model(input_dim, num_classes):
    model = DeepLearningModel(input_dim, num_classes)
    return model

def train_models(features, target, ranking_mapping):
    # Filter out classes with too few samples
    unique_classes, counts = np.unique(target, return_counts=True)
    valid_classes = unique_classes[counts >= 10]  # Require at least 10 samples per class
    mask = np.isin(target, valid_classes)

    features_filtered = features[mask]
    target_filtered = target[mask]

    # Check if we have enough classes for stratification
    unique_filtered, counts_filtered = np.unique(target_filtered, return_counts=True)
    if len(unique_filtered) >= 2 and np.min(counts_filtered) >= 2:
        # Use stratification
        X_train, X_test, y_train, y_test = train_test_split(
            features_filtered, target_filtered, test_size=0.2, random_state=42, stratify=target_filtered
        )
    else:
        # Fall back to regular split
        X_train, X_test, y_train, y_test = train_test_split(
            features_filtered, target_filtered, test_size=0.2, random_state=42
        )

    # Normaliseren
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    num_classes = len(ranking_mapping)

    # Deep Learning Model with PyTorch
    dl_model = create_deep_learning_model(X_train.shape[1], num_classes)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(dl_model.parameters(), lr=0.001)

    # Convert to tensors
    X_train_tensor = torch.FloatTensor(X_train_scaled)
    y_train_tensor = torch.LongTensor(y_train)
    X_test_tensor = torch.FloatTensor(X_test_scaled)
    y_test_tensor = torch.LongTensor(y_test)

    # DataLoader
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

    # Training loop with early stopping
    best_loss = float('inf')
    patience = 10
    patience_counter = 0
    best_model_state = None

    for epoch in range(100):
        dl_model.train()
        epoch_loss = 0
        for batch_X, batch_y in train_loader:
            optimizer.zero_grad()
            outputs = dl_model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        epoch_loss /= len(train_loader)

        # Validation
        dl_model.eval()
        with torch.no_grad():
            val_outputs = dl_model(X_test_tensor)
            val_loss = criterion(val_outputs, y_test_tensor).item()

        if val_loss < best_loss:
            best_loss = val_loss
            patience_counter = 0
            best_model_state = dl_model.state_dict()
        else:
            patience_counter += 1

        if patience_counter >= patience:
            print(f"Early stopping at epoch {epoch}")
            break

    # Load best model
    if best_model_state:
        dl_model.load_state_dict(best_model_state)

    # Traditionele ML modellen voor vergelijking
    rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
    gb_model = GradientBoostingClassifier(n_estimators=100, random_state=42)

    rf_model.fit(X_train, y_train)
    try:
        gb_model.fit(X_train, y_train)
    except ValueError as e:
        print(f"Gradient Boosting failed: {e}")
        print("Skipping Gradient Boosting due to insufficient class diversity")
        gb_model = None

    # Evaluatie
    models = {
        'Deep Learning': dl_model,
        'Random Forest': rf_model,
        'Gradient Boosting': gb_model if gb_model is not None else None
    }

    results = {}
    for name, model in models.items():
        if model is None:
            continue
        if name == 'Deep Learning':
            dl_model.eval()
            with torch.no_grad():
                y_pred_tensor = dl_model(X_test_tensor)
                y_pred = torch.argmax(y_pred_tensor, dim=1).numpy()
        else:
            y_pred = model.predict(X_test)

        accuracy = accuracy_score(y_test, y_pred)
        results[name] = {
            'model': model,
            'accuracy': accuracy,
            'predictions': y_pred
        }

        print(f"\n{name} Results:")
        print(f"Accuracy: {accuracy:.4f}")
        print(classification_report(y_test, y_pred))

    return results, scaler, ranking_mapping

def predict_next_ranking(player_data, models, scaler, ranking_mapping):
    """
    Voorspel next_ranking voor nieuwe speler data
    """
    # Maak ranking mapping reverse voor terugconversie
    reverse_mapping = {v: k for k, v in ranking_mapping.items()}

    # Voorbereiden features (zelfde als tijdens training)
    features = create_single_player_features(player_data)

    # Schalen voor DL model
    features_scaled = scaler.transform([features])

    # Voorspellingen van alle modellen
    predictions = {}

    # Deep Learning
    dl_model = models['Deep Learning']['model']
    dl_model.eval()
    with torch.no_grad():
        features_tensor = torch.FloatTensor(features_scaled)
        dl_outputs = dl_model(features_tensor)
        dl_probs = torch.softmax(dl_outputs, dim=1).numpy()[0]
        dl_pred = np.argmax(dl_probs)
    predictions['Deep Learning'] = {
        'ranking': reverse_mapping[dl_pred],
        'confidence': dl_probs[dl_pred]
    }

    # Random Forest
    rf_pred = models['Random Forest']['model'].predict([features])[0]
    rf_probs = models['Random Forest']['model'].predict_proba([features])[0]
    predictions['Random Forest'] = {
        'ranking': reverse_mapping[rf_pred],
        'confidence': rf_probs[rf_pred]
    }

    # Gradient Boosting
    gb_pred = models['Gradient Boosting']['model'].predict([features])[0]
    gb_probs = models['Gradient Boosting']['model'].predict_proba([features])[0]
    predictions['Gradient Boosting'] = {
        'ranking': reverse_mapping[gb_pred],
        'confidence': gb_probs[gb_pred]
    }

    return predictions, features

def create_single_player_features(player_data):
    """
    Creëer features voor een individuele speler
    player_data moet bevatten: current_ranking, category, kaart_dict
    """
    ranking_order = ['NG', 'E6', 'E4', 'E2', 'E0', 'D6', 'D4', 'D2', 'D0', 
                   'C6', 'C4', 'C2', 'C0', 'B6', 'B4', 'B2', 'B0', 'A']
    ranking_mapping = {rank: i for i, rank in enumerate(ranking_order)}
    
    current_rank = player_data['current_ranking']
    category = player_data['category']
    kaart = player_data['kaart_dict']
    
    # Zelfde feature engineering als tijdens training
    current_rank_num = ranking_mapping.get(current_rank, 0)
    category_encoded = hash(category) % 100
    
    total_wins = 0
    total_losses = 0
    win_rate_per_rank = []
    
    for rank in ranking_order:
        if rank in kaart:
            wins, losses = kaart[rank]
            total_wins += wins
            total_losses += losses
            if wins + losses > 0:
                win_rate = wins / (wins + losses)
                win_rate_per_rank.append(win_rate)
    
    total_matches = total_wins + total_losses
    overall_win_rate = total_wins / total_matches if total_matches > 0 else 0
    avg_win_rate = np.mean(win_rate_per_rank) if win_rate_per_rank else 0
    
    # Performance tegen hogere/lagere rankings
    current_rank_idx = ranking_mapping.get(current_rank, 0)
    performance_against_higher = 0
    performance_against_lower = 0
    matches_against_higher = 0
    matches_against_lower = 0
    
    for rank, (wins, losses) in kaart.items():
        rank_idx = ranking_mapping.get(rank, 0)
        if rank_idx > current_rank_idx:
            performance_against_higher += wins
            matches_against_higher += wins + losses
        elif rank_idx < current_rank_idx:
            performance_against_lower += wins
            matches_against_lower += wins + losses
    
    perf_vs_higher = performance_against_higher / matches_against_higher if matches_against_higher > 0 else 0
    perf_vs_lower = performance_against_lower / matches_against_lower if matches_against_lower > 0 else 0
    
    return [
        current_rank_num, category_encoded, total_wins, total_losses,
        total_matches, overall_win_rate, avg_win_rate,
        perf_vs_higher, perf_vs_lower, len(win_rate_per_rank)
    ]

def main():
    # Data laden
    df = load_and_preprocess_data('club_members_with_next_ranking.csv')
    
    # Features maken
    features, target, ranking_mapping = create_features(df)
    
    print(f"Dataset grootte: {len(features)} samples")
    print(f"Aantal features: {len(feature_names)}")
    print(f"Feature namen: {feature_names}")
    
    # Modellen trainen
    results, scaler, ranking_mapping = train_models(features, target, ranking_mapping)
    
    # Voorbeeld voorspelling
    example_player = {
        'current_ranking': 'C2',
        'category': 'SEN',
        'kaart_dict': {
            'C0': [5, 3],
            'C2': [8, 4],
            'C4': [3, 2],
            'B6': [1, 2]
        }
    }
    
    predictions, features_used = predict_next_ranking(example_player, results, scaler, ranking_mapping)
    
    print("\n" + "="*50)
    print("VOORSPELLING VOOR EXAMPLE SPELER:")
    print("="*50)
    print(f"Current ranking: {example_player['current_ranking']}")
    print(f"Category: {example_player['category']}")
    print(f"Kaart data: {example_player['kaart_dict']}")
    print("\nVoorspellingen:")
    for model_name, pred in predictions.items():
        print(f"{model_name}: {pred['ranking']} (confidence: {pred['confidence']:.3f})")
    
    # Feature importance voor Random Forest
    feature_importance = results['Random Forest']['model'].feature_importances_
    importance_df = pd.DataFrame({
        'feature': feature_names,
        'importance': feature_importance
    }).sort_values('importance', ascending=False)
    
    print("\nFeature Importance (Random Forest):")
    print(importance_df)
    
    return results, scaler, ranking_mapping

if __name__ == "__main__":
    trained_models, scaler, ranking_mapping = main()
    
    
def save_models(models, scaler, ranking_mapping, filepath='ranking_model'):
    """Sla getrainde modellen op"""
    # Deep Learning model
    torch.save(models['Deep Learning']['model'].state_dict(), f'{filepath}_dl.pth')

    # Traditional models
    joblib.dump(models['Random Forest']['model'], f'{filepath}_rf.pkl')
    joblib.dump(models['Gradient Boosting']['model'], f'{filepath}_gb.pkl')

    # Scaler en mapping
    joblib.dump(scaler, f'{filepath}_scaler.pkl')
    joblib.dump(ranking_mapping, f'{filepath}_mapping.pkl')

    print("Modellen succesvol opgeslagen")

def load_models(filepath='ranking_model'):
    """Laad getrainde modellen"""
    # Load mapping first to get num_classes
    ranking_mapping = joblib.load(f'{filepath}_mapping.pkl')
    num_classes = len(ranking_mapping)

    # Deep Learning model
    dl_model = create_deep_learning_model(10, num_classes)  # Assuming 10 features
    dl_model.load_state_dict(torch.load(f'{filepath}_dl.pth'))
    dl_model.eval()

    # Traditional models
    rf_model = joblib.load(f'{filepath}_rf.pkl')
    gb_model = joblib.load(f'{filepath}_gb.pkl')

    # Scaler
    scaler = joblib.load(f'{filepath}_scaler.pkl')

    models = {
        'Deep Learning': {'model': dl_model},
        'Random Forest': {'model': rf_model},
        'Gradient Boosting': {'model': gb_model}
    }

    return models, scaler, ranking_mapping
