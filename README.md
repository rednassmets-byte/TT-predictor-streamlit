# TT KlassementPredictor

A Streamlit web application for predicting table tennis player rankings in the VTTL (Vlaamse Tafeltennisliga) using machine learning.

## Features

- **Club Selection**: Choose from all clubs in Antwerpen province
- **Member Selection**: Select players from the chosen club and season
- **Performance Prediction**: AI-powered ranking prediction based on historical data
- **Performance Score**: Dynamic scoring system (1-100) with color coding
- **Detailed Statistics**: View win/loss records across different ranking categories

## Deployment

This app can be deployed on various platforms:

### Heroku Deployment

1. Create a Heroku account
2. Install Heroku CLI
3. Initialize git repository:
   ```bash
   git init
   git add .
   git commit -m "Initial commit"
   ```

4. Create Heroku app:
   ```bash
   heroku create your-app-name
   ```

5. Deploy:
   ```bash
   git push heroku main
   ```

### Streamlit Cloud

1. Go to [share.streamlit.io](https://share.streamlit.io)
2. Connect your GitHub repository
3. Deploy the app

### Local Development

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Requirements

- Python 3.11
- Streamlit
- Pandas
- Scikit-learn
- Joblib
- PyVTTL API access

## Model

The prediction model is trained on historical VTTL data from Antwerpen province (seasons 15-26) using a RandomForest classifier with 38 features including win/loss ratios across different ranking categories.

## Data Sources

- VTTL API for real-time player data
- Historical performance data for model training
- Club information from VTTL club database

## Credits

- Sander Smets
- Steven Smets
- Tim Jacobs
- VTTL API
