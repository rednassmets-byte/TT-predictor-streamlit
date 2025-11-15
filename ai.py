import pandas as pd
import numpy as np
import ast
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import classification_report, accuracy_score
from imblearn.over_sampling import SMOTE
import joblib
import matplotlib.pyplot as plt
from collections import Counter

# Custom Dataset class voor PyTorch
class TableTennisDataset(Dataset):
    def __init__(self, features, targets):
        self.features = torch.FloatTensor(features)
        self.targets = torch.LongTensor(targets)
    
    def __len__(self):
        return len(self.features)
    
    def __getitem__(self, idx):
        return self.features[idx], self.targets[idx]

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
    targets = []

    # Ranking volgorde
    ranking_order = ['NG', 'E6', 'E4', 'E2', 'E0', 'D6', 'D4', 'D2', 'D0',
                   'C6', 'C4', 'C2', 'C0', 'B6', 'B4', 'B2', 'B0', 'A']
    ranking_mapping = {rank: i for i, rank in enumerate(ranking_order)}

    # Category encoding
    categories = df['category'].unique()
    category_mapping = {cat: i for i, cat in enumerate(categories)}

    for _, row in df.iterrows():
        if not row['kaart_dict']:
            continue

        # Basis features
        current_rank = row['current_ranking'] if row['current_ranking'] else 'NG'
        category = row['category'] if row['category'] else 'SEN'
        next_rank = row['next_ranking'] if row['next_ranking'] else 'NG'
        
        current_rank_num = ranking_mapping.get(current_rank, 0)
        category_num = category_mapping.get(category, 0)
        
        # Statistieken uit kaart data
        kaart = row['kaart_dict']
        total_wins = 0
        total_losses = 0
        win_rates = []
        matches_per_rank = []
        
        current_rank_idx = ranking_mapping.get(current_rank, 0)
        
        for rank in ranking_order:
            if rank in kaart:
                wins, losses = kaart[rank]
                total_wins += wins
                total_losses += losses
                
                if wins + losses > 0:
                    win_rate = wins / (wins + losses)
                    win_rates.append(win_rate)
                    
                    rank_idx = ranking_mapping.get(rank, 0)
                    # Relatieve ranking positie
                    rel_position = rank_idx - current_rank_idx
                    matches_per_rank.append((rel_position, wins + losses, win_rate))
        
        total_matches = total_wins + total_losses
        overall_win_rate = total_wins / total_matches if total_matches > 0 else 0
        
        # Geavanceerde features
        if matches_per_rank:
            matches_per_rank = np.array(matches_per_rank)
            # Performance tegen verschillende niveau's
            higher_rank_matches = matches_per_rank[matches_per_rank[:, 0] > 0]
            lower_rank_matches = matches_per_rank[matches_per_rank[:, 0] < 0]
            same_rank_matches = matches_per_rank[matches_per_rank[:, 0] == 0]
            
            perf_vs_higher = higher_rank_matches[:, 2].mean() if len(higher_rank_matches) > 0 else 0
            perf_vs_lower = lower_rank_matches[:, 2].mean() if len(lower_rank_matches) > 0 else 0
            perf_vs_same = same_rank_matches[:, 2].mean() if len(same_rank_matches) > 0 else 0
            
            # Gewogen performance
            higher_weight = higher_rank_matches[:, 1].sum() if len(higher_rank_matches) > 0 else 0
            lower_weight = lower_rank_matches[:, 1].sum() if len(lower_rank_matches) > 0 else 0
        else:
            perf_vs_higher = perf_vs_lower = perf_vs_same = 0
            higher_weight = lower_weight = 0
        
        # Form metrics
        recent_matches = min(15, total_matches)
        recent_performance = min(1.0, total_wins / recent_matches) if recent_matches > 0 else 0
        
        # Consistency metric
        consistency = 1 - np.std(win_rates) if win_rates else 0
        
        # Feature vector
        feature_vector = [
            current_rank_num,
            category_num,
            total_wins,
            total_losses,
            total_matches,
            overall_win_rate,
            perf_vs_higher,
            perf_vs_lower,
            perf_vs_same,
            higher_weight / total_matches if total_matches > 0 else 0,
            lower_weight / total_matches if total_matches > 0 else 0,
            recent_performance,
            consistency,
            len(win_rates)  # Variety of opponents
        ]
        
        features.append(feature_vector)
        targets.append(ranking_mapping.get(next_rank, 0))
    
    return np.array(features), np.array(targets), ranking_mapping, category_mapping

feature_names = [
    'current_rank', 'category', 'total_wins', 'total_losses', 'total_matches',
    'overall_win_rate', 'perf_vs_higher', 'perf_vs_lower', 'perf_vs_same',
    'higher_opponent_ratio', 'lower_opponent_ratio', 'recent_performance',
    'consistency', 'opponent_variety'
]


class RankingPredictor(nn.Module):
    def __init__(self, input_size, num_classes, hidden_layers=[256, 128, 64, 32], dropout_rate=0.3):
        super(RankingPredictor, self).__init__()
        
        layers = []
        prev_size = input_size
        
        for hidden_size in hidden_layers:
            layers.extend([
                nn.Linear(prev_size, hidden_size),
                nn.BatchNorm1d(hidden_size),
                nn.ReLU(),
                nn.Dropout(dropout_rate)
            ])
            prev_size = hidden_size
        
        self.feature_layers = nn.Sequential(*layers)
        self.output_layer = nn.Linear(prev_size, num_classes)
        
    def forward(self, x):
        x = self.feature_layers(x)
        return self.output_layer(x)

class AdvancedRankingPredictor(nn.Module):
    def __init__(self, input_size, num_classes):
        super(AdvancedRankingPredictor, self).__init__()
        
        # Main feature processing
        self.main_layers = nn.Sequential(
            nn.Linear(input_size, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.4),
            
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.3),
            
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.2),
            
            nn.Linear(128, 64),
            nn.ReLU(),
        )
        
        # Attention mechanism voor belangrijke features
        self.attention = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 64),
            nn.Sigmoid()
        )
        
        self.output_layer = nn.Linear(64, num_classes)
        
    def forward(self, x):
        features = self.main_layers(x)
        attention_weights = self.attention(features)
        weighted_features = features * attention_weights
        return self.output_layer(weighted_features)

def train_pytorch_model(model, train_loader, val_loader, num_epochs=100, learning_rate=0.001, class_weights=None):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    if class_weights is not None:
        class_weights = torch.FloatTensor(class_weights).to(device)
        criterion = nn.CrossEntropyLoss(weight=class_weights)
    else:
        criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=10, factor=0.5)
    
    train_losses = []
    val_accuracies = []
    
    best_val_accuracy = 0
    patience_counter = 0
    patience = 20
    
    for epoch in range(num_epochs):
        # Training
        model.train()
        train_loss = 0
        for batch_features, batch_targets in train_loader:
            batch_features, batch_targets = batch_features.to(device), batch_targets.to(device)
            
            optimizer.zero_grad()
            outputs = model(batch_features)
            loss = criterion(outputs, batch_targets)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
        
        # Validation
        model.eval()
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for batch_features, batch_targets in val_loader:
                batch_features, batch_targets = batch_features.to(device), batch_targets.to(device)
                
                outputs = model(batch_features)
                _, predicted = torch.max(outputs.data, 1)
                val_total += batch_targets.size(0)
                val_correct += (predicted == batch_targets).sum().item()
        
        val_accuracy = val_correct / val_total
        avg_train_loss = train_loss / len(train_loader)
        
        train_losses.append(avg_train_loss)
        val_accuracies.append(val_accuracy)
        
        scheduler.step(avg_train_loss)
        
        if (epoch + 1) % 10 == 0:
            print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {avg_train_loss:.4f}, Val Accuracy: {val_accuracy:.4f}')
        
        # Early stopping
        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            patience_counter = 0
            # Save best model
            torch.save(model.state_dict(), 'best_model.pth')
        else:
            patience_counter += 1
            
        if patience_counter >= patience:
            print(f'Early stopping at epoch {epoch+1}')
            break
    
    # Load best model
    model.load_state_dict(torch.load('best_model.pth'))
    
    return model, train_losses, val_accuracies

def plot_training_history(train_losses, val_accuracies):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    ax1.plot(train_losses)
    ax1.set_title('Training Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    
    ax2.plot(val_accuracies)
    ax2.set_title('Validation Accuracy')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy')
    
    plt.tight_layout()
    plt.show()
    
def train_all_models(features, targets, ranking_mapping):
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        features, targets, test_size=0.4, random_state=42, stratify=targets
    )

    # Check class distribution before SMOTE
    print(f"Original training set class distribution: {Counter(y_train)}")
    print(f"Test set class distribution: {Counter(y_test)}")

    # Apply SMOTE only if there are multiple classes
    if len(np.unique(y_train)) > 1:
        smote = SMOTE(random_state=42, k_neighbors=1)  # k_neighbors=1 to handle small classes
        X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)
        print(f"After SMOTE training set class distribution: {Counter(y_train_resampled)}")
    else:
        print("Only one class in training data. Skipping SMOTE.")
        X_train_resampled, y_train_resampled = X_train, y_train

    # Calculate class weights for imbalanced dataset (after SMOTE, weights should be more balanced)
    class_counts = Counter(y_train_resampled)
    total_samples = len(y_train_resampled)
    num_classes = len(ranking_mapping)
    class_weights = []

    for i in range(num_classes):
        count = class_counts.get(i, 0)
        if count > 0:
            weight = total_samples / (num_classes * count)
        else:
            weight = 1.0
        class_weights.append(weight)

    # Normalize weights to prevent extreme values
    max_weight = max(class_weights)
    class_weights = [w / max_weight for w in class_weights]

    print(f"Class weights: {class_weights}")

    # Normalize for PyTorch
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_resampled)
    X_test_scaled = scaler.transform(X_test)

    # PyTorch DataLoaders
    train_dataset = TableTennisDataset(X_train_scaled, y_train_resampled)
    test_dataset = TableTennisDataset(X_test_scaled, y_test)

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    # Create validation split
    train_size = int(0.8 * len(train_dataset))
    val_size = len(train_dataset) - train_size
    train_subset, val_subset = torch.utils.data.random_split(train_dataset, [train_size, val_size])

    train_loader = DataLoader(train_subset, batch_size=32, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_subset, batch_size=32, shuffle=False, drop_last=True)

    # PyTorch models
    input_size = features.shape[1]
    num_classes = len(ranking_mapping)

    print("Training PyTorch models...")

    # Simple model
    simple_model = RankingPredictor(input_size, num_classes, [128, 64, 32])
    simple_model, simple_losses, simple_accs = train_pytorch_model(
        simple_model, train_loader, val_loader, num_epochs=150, class_weights=class_weights
    )

    # Advanced model
    advanced_model = AdvancedRankingPredictor(input_size, num_classes)
    advanced_model, advanced_losses, advanced_accs = train_pytorch_model(
        advanced_model, train_loader, val_loader, num_epochs=200, class_weights=class_weights
    )

    # Traditional ML models for comparison
    print("Training traditional ML models...")
    rf_model = RandomForestClassifier(n_estimators=200, random_state=42)
    gb_model = GradientBoostingClassifier(n_estimators=200, random_state=42)

    rf_model.fit(X_train, y_train)

    # Check if we have enough classes for Gradient Boosting
    if len(np.unique(y_train)) < 2:
        print("Warning: Only one class in training data for Gradient Boosting. Skipping...")
        gb_model = None
    else:
        gb_model.fit(X_train, y_train)
    
    # Evaluate all models
    results = {}
    
    # PyTorch models evaluation
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    for model_name, model in [('PyTorch Simple', simple_model), ('PyTorch Advanced', advanced_model)]:
        model.eval()
        correct = 0
        total = 0
        all_predictions = []
        
        with torch.no_grad():
            for batch_features, batch_targets in test_loader:
                batch_features, batch_targets = batch_features.to(device), batch_targets.to(device)
                outputs = model(batch_features)
                _, predicted = torch.max(outputs.data, 1)
                total += batch_targets.size(0)
                correct += (predicted == batch_targets).sum().item()
                all_predictions.extend(predicted.cpu().numpy())
        
        accuracy = correct / total
        results[model_name] = {
            'model': model,
            'accuracy': accuracy,
            'predictions': np.array(all_predictions),
            'type': 'pytorch'
        }
        print(f"{model_name} Test Accuracy: {accuracy:.4f}")
    
    # Traditional models evaluation
    for model_name, model in [('Random Forest', rf_model), ('Gradient Boosting', gb_model)]:
        if model is not None:
            y_pred = model.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)
            results[model_name] = {
                'model': model,
                'accuracy': accuracy,
                'predictions': y_pred,
                'type': 'sklearn'
            }
            print(f"{model_name} Test Accuracy: {accuracy:.4f}")
        else:
            print(f"{model_name} skipped due to insufficient class diversity")
    
    # Plot training history
    plot_training_history(simple_losses, simple_accs)
    
    return results, scaler, ranking_mapping

def predict_single_player(player_data, models, scaler, ranking_mapping):
    """Make prediction for a single player"""
    reverse_mapping = {v: k for k, v in ranking_mapping.items()}
    
    # Create features
    features = create_single_player_features(player_data, ranking_mapping)
    features_scaled = scaler.transform([features])
    
    predictions = {}
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    for model_name, model_info in models.items():
        if model_info['type'] == 'pytorch':
            # PyTorch model prediction
            model = model_info['model']
            model.eval()
            
            with torch.no_grad():
                features_tensor = torch.FloatTensor(features_scaled).to(device)
                output = model(features_tensor)
                probabilities = torch.softmax(output, dim=1)
                confidence, prediction = torch.max(probabilities, 1)
                
                predictions[model_name] = {
                    'ranking': reverse_mapping[prediction.item()],
                    'confidence': confidence.item(),
                    'all_probabilities': {
                        reverse_mapping[i]: prob.item() for i, prob in enumerate(probabilities[0])
                    }
                }
        else:
            # Scikit-learn model prediction
            model = model_info['model']
            prediction = model.predict([features])[0]
            probabilities = model.predict_proba([features])[0]
            
            predictions[model_name] = {
                'ranking': reverse_mapping[prediction],
                'confidence': probabilities[prediction],
                'all_probabilities': {
                    reverse_mapping[i]: prob for i, prob in enumerate(probabilities)
                }
            }
    
    return predictions, features

def create_single_player_features(player_data, ranking_mapping):
    """Create features for single player prediction"""
    ranking_order = list(ranking_mapping.keys())
    
    current_rank = player_data['current_ranking']
    category = player_data['category']
    kaart = player_data['kaart_dict']
    
    current_rank_num = ranking_mapping.get(current_rank, 0)
    category_num = hash(category) % 100  # Simple encoding
    
    # Calculate all features (same as in create_features)
    total_wins = 0
    total_losses = 0
    win_rates = []
    matches_per_rank = []
    
    current_rank_idx = ranking_mapping.get(current_rank, 0)
    
    for rank in ranking_order:
        if rank in kaart:
            wins, losses = kaart[rank]
            total_wins += wins
            total_losses += losses
            
            if wins + losses > 0:
                win_rate = wins / (wins + losses)
                win_rates.append(win_rate)
                
                rank_idx = ranking_mapping.get(rank, 0)
                rel_position = rank_idx - current_rank_idx
                matches_per_rank.append((rel_position, wins + losses, win_rate))
    
    total_matches = total_wins + total_losses
    overall_win_rate = total_wins / total_matches if total_matches > 0 else 0
    
    # Calculate advanced features
    if matches_per_rank:
        matches_per_rank = np.array(matches_per_rank)
        higher_rank_matches = matches_per_rank[matches_per_rank[:, 0] > 0]
        lower_rank_matches = matches_per_rank[matches_per_rank[:, 0] < 0]
        same_rank_matches = matches_per_rank[matches_per_rank[:, 0] == 0]
        
        perf_vs_higher = higher_rank_matches[:, 2].mean() if len(higher_rank_matches) > 0 else 0
        perf_vs_lower = lower_rank_matches[:, 2].mean() if len(lower_rank_matches) > 0 else 0
        perf_vs_same = same_rank_matches[:, 2].mean() if len(same_rank_matches) > 0 else 0
        
        higher_weight = higher_rank_matches[:, 1].sum() if len(higher_rank_matches) > 0 else 0
        lower_weight = lower_rank_matches[:, 1].sum() if len(lower_rank_matches) > 0 else 0
    else:
        perf_vs_higher = perf_vs_lower = perf_vs_same = 0
        higher_weight = lower_weight = 0
    
    recent_matches = min(15, total_matches)
    recent_performance = min(1.0, total_wins / recent_matches) if recent_matches > 0 else 0
    consistency = 1 - np.std(win_rates) if win_rates else 0
    
    return [
        current_rank_num, category_num, total_wins, total_losses, total_matches,
        overall_win_rate, perf_vs_higher, perf_vs_lower, perf_vs_same,
        higher_weight / total_matches if total_matches > 0 else 0,
        lower_weight / total_matches if total_matches > 0 else 0,
        recent_performance, consistency, len(win_rates)
    ]
    
def main():
    print("Loading and preprocessing data...")
    df = load_and_preprocess_data('club_members_with_next_ranking.csv')
    
    print("Creating features...")
    features, targets, ranking_mapping, category_mapping = create_features(df)
    
    print(f"Dataset size: {len(features)} samples")
    print(f"Number of features: {features.shape[1]}")
    print(f"Number of classes: {len(ranking_mapping)}")
    print(f"Ranking order: {list(ranking_mapping.keys())}")
    
    print("\nTraining models...")
    results, scaler, ranking_mapping = train_all_models(features, targets, ranking_mapping)
    
    # Example prediction
    example_player = {
        'current_ranking': 'C2',
        'category': 'SEN',
        'kaart_dict': {
            'C0': [12, 3],   # Excellent against lower
            'C2': [8, 6],    # Good against same
            'C4': [5, 8],    # Decent against higher
            'B6': [2, 5]     # Some experience vs much higher
        }
    }

    print("\n" + "="*70)
    print("PREDICTION FOR EXAMPLE PLAYER:")
    print("="*70)
    print(f"Current ranking: {example_player['current_ranking']}")
    print(f"Category: {example_player['category']}")
    print(f"Match history: {example_player['kaart_dict']}")

    predictions, features_used = predict_single_player(example_player, results, scaler, ranking_mapping)

    print("\nPredictions from all models:")
    for model_name, pred in predictions.items():
        print(f"  {model_name:20}: {pred['ranking']} (confidence: {pred['confidence']:.3f})")

    # Show detailed probabilities from best model
    best_model = max(predictions.items(), key=lambda x: x[1]['confidence'])
    print(f"\nDetailed probabilities from {best_model[0]}:")
    for rank, prob in sorted(best_model[1]['all_probabilities'].items(),
                           key=lambda x: x[1], reverse=True)[:5]:
        print(f"  {rank}: {prob:.3f}")

    # User player prediction
    user_player = {
        'current_ranking': 'E2',  # Inferred from match history (strong performance at E ranks)
        'category': 'CAD',
        'kaart_dict': {
            'A': [0, 0],
            'B0': [0, 0],
            'B2': [0, 0],
            'B4': [0, 0],
            'B6': [0, 0],
            'C0': [0, 0],
            'C2': [0, 0],
            'C4': [0, 3],
            'C6': [0, 0],
            'D0': [0, 1],
            'D2': [0, 0],
            'D4': [0, 3],
            'D6': [1, 14],
            'E0': [18, 22],
            'E2': [20, 3],
            'E4': [11, 2],
            'E6': [10, 0],
            'NG': [7, 0]
        }
    }

    print("\n" + "="*70)
    print("PREDICTION FOR USER PLAYER:")
    print("="*70)
    print(f"Current ranking: {user_player['current_ranking']}")
    print(f"Category: {user_player['category']}")
    print(f"Match history: {user_player['kaart_dict']}")

    predictions, features_used = predict_single_player(user_player, results, scaler, ranking_mapping)

    print("\nPredictions from all models:")
    for model_name, pred in predictions.items():
        print(f"  {model_name:20}: {pred['ranking']} (confidence: {pred['confidence']:.3f})")

    # Show detailed probabilities from best model
    best_model = max(predictions.items(), key=lambda x: x[1]['confidence'])
    print(f"\nDetailed probabilities from {best_model[0]}:")
    for rank, prob in sorted(best_model[1]['all_probabilities'].items(),
                           key=lambda x: x[1], reverse=True)[:5]:
        print(f"  {rank}: {prob:.3f}")
    
    return results, scaler, ranking_mapping

if __name__ == "__main__":
    trained_models, scaler, ranking_mapping = main()
def save_pytorch_models(results, scaler, ranking_mapping, filepath='pytorch_model'):
    """Save trained models"""
    for name, result in results.items():
        if result['type'] == 'pytorch':
            # Save PyTorch model state dict
            model = result['model']
            safe_name = name.lower().replace(' ', '_')
            torch.save(model.state_dict(), f'{filepath}_{safe_name}.pth')
        else:
            # Save scikit-learn model
            model = result['model']
            safe_name = name.lower().replace(' ', '_')
            joblib.dump(model, f'{filepath}_{safe_name}.pkl')
    
    joblib.dump(scaler, f'{filepath}_scaler.pkl')
    joblib.dump(ranking_mapping, f'{filepath}_mapping.pkl')
    joblib.dump(feature_names, f'{filepath}_features.pkl')
    
    print("All models saved successfully")

def load_pytorch_models(filepath='pytorch_model'):
    """Load trained models"""
    import glob
    import os
    
    ranking_mapping = joblib.load(f'{filepath}_mapping.pkl')
    scaler = joblib.load(f'{filepath}_scaler.pkl')
    feature_names = joblib.load(f'{filepath}_features.pkl')
    
    input_size = len(feature_names)
    num_classes = len(ranking_mapping)
    
    models = {}
    
    # Load PyTorch models
    pytorch_files = glob.glob(f'{filepath}_pytorch_*.pth')
    for model_file in pytorch_files:
        model_name = os.path.basename(model_file).replace(f'{filepath}_', '').replace('.pth', '')
        model_name = model_name.replace('_', ' ').title()
        
        if 'simple' in model_file:
            model = RankingPredictor(input_size, num_classes)
        else:
            model = AdvancedRankingPredictor(input_size, num_classes)
        
        model.load_state_dict(torch.load(model_file))
        model.eval()
        
        models[model_name] = {
            'model': model,
            'type': 'pytorch'
        }
    
    # Load scikit-learn models
    sklearn_files = glob.glob(f'{filepath}_*.pkl')
    sklearn_files = [f for f in sklearn_files if 'scaler' not in f and 'mapping' not in f and 'features' not in f]
    
    for model_file in sklearn_files:
        model_name = os.path.basename(model_file).replace(f'{filepath}_', '').replace('.pkl', '')
        model_name = model_name.replace('_', ' ').title()
        
        model = joblib.load(model_file)
        models[model_name] = {
            'model': model,
            'type': 'sklearn'
        }
    
    return models, scaler, ranking_mapping, feature_names