# 🏓 TT KlassementPredictor

A Streamlit web application for predicting table tennis player rankings in the VTTL (Vlaamse Tafeltennisliga) using advanced machine learning.

## ✨ Features

- **Smart Club Selection**: Choose from all clubs in Antwerpen province
- **Member Selection**: Select players from the chosen club and season
- **Dual Model System**: 
  - Regular model for all categories (72% accuracy)
  - Filtered model for youth categories BEN/PRE/MIN/CAD (67% accuracy)
- **Junior Leniency**: Automatic adjustment for JUN/J19 categories (more lenient predictions)
- **Performance Prediction**: AI-powered ranking prediction with confidence scores
- **Performance Score**: Dynamic scoring system (1-100) with color coding
- **Detailed Statistics**: View win/loss records across different ranking categories
- **Visual Feedback**: Shows rank improvements/declines with metrics

## 🚀 Quick Start

### Streamlit Cloud Deployment (Recommended)

1. **Push to GitHub**:
   ```bash
   git add .
   git commit -m "Deploy to Streamlit"
   git push origin main
   ```

2. **Deploy on Streamlit Cloud**:
   - Go to [share.streamlit.io](https://share.streamlit.io)
   - Click "New app"
   - Connect your GitHub repository
   - Select branch: `main`
   - Main file path: `app.py`
   - Click "Deploy"

### Local Development

```bash
# Install dependencies
pip install -r requirements.txt

# Run the app
streamlit run app.py
```

## 📊 Model Details

### Regular Model (All Categories)
- **Accuracy**: 72.04% on seasons 24-25
- **Training Data**: 18,453 players from seasons 15-26
- **Features**: 24 optimized features
- **Classes**: 18 ranking classes (A to NG)
- **Techniques**: 
  - Combined over/undersampling (SMOTE + RandomUnderSampler)
  - StandardScaler normalization
  - Advanced feature engineering
  - Optimized RandomForest (300 trees, depth 15)

### Filtered Model (Youth Categories)
- **Accuracy**: 66.84% on seasons 24-25
- **Training Data**: 3,112 youth players (BEN/PRE/MIN/CAD)
- **Features**: 24 optimized features
- **Classes**: 13 ranking classes
- **Special**: Trained specifically for youth performance patterns

### Key Features Used
- Current rank and category
- Total wins/losses/matches
- Overall win rate
- Performance consistency
- Recent performance
- Rank progression potential
- Win rates per rank (B0-E2)
- Total matches per rank (E0-E6)

## 🎯 Junior Leniency System

For JUN and J19 categories, the system automatically applies leniency:
- **Strong performance** (≥60% win rate + high confidence): +2 ranks
- **Good performance** (≥50% win rate OR high confidence): +1 rank
- **Default**: +1 rank
- Never predicts worse than current rank

## 📁 Required Files for Deployment

Make sure these files are in your repository:
- `app.py` - Main Streamlit application
- `database_maker.py` - VTTL API integration
- `club_data.csv` - Club information
- `requirements.txt` - Python dependencies
- `.streamlit/config.toml` - Streamlit configuration

**Model Files** (must be included):
- `model.pkl` & `model_filtered.pkl` - Trained models
- `scaler.pkl` & `scaler_filtered.pkl` - Feature scalers
- `category_encoder.pkl` & `category_encoder_filtered.pkl` - Category encoders
- `rank_to_int.pkl`, `int_to_rank.pkl` - Rank mappings
- `feature_cols.pkl` - Feature column names
- `ranking_order.pkl` - Ranking order

## 🛠️ Tech Stack

- **Frontend**: Streamlit
- **ML Framework**: Scikit-learn
- **Data Processing**: Pandas, NumPy
- **Class Balancing**: Imbalanced-learn
- **API Integration**: Zeep (SOAP), Requests
- **Model Storage**: Joblib

## 📈 Performance Improvements

From initial model to current:
- Regular model: 69.04% → **72.04%** (+3%)
- Filtered model: 40.15% → **66.84%** (+26.7%)

Key improvements:
- ✅ Combined over/undersampling to reduce NG dominance
- ✅ Advanced feature engineering (7 new features)
- ✅ Feature scaling with StandardScaler
- ✅ Optimized RandomForest parameters
- ✅ Focused feature selection (24 features)
- ✅ Junior category leniency system

## 👥 Credits

- **Sander Smets** - Development
- **Steven Smets** - Support
- **Tim Jacobs** - Support
- **VTTL API** - Data source

## 📝 License

This project is for educational and personal use.
