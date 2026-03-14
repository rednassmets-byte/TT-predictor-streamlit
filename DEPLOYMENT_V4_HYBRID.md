# V4 Hybrid Model Deployment Guide

## Quick Deploy to Streamlit Cloud

### 1. Push to GitHub
Run the deployment script:
```bash
push_v4_hybrid.bat
```

Or manually:
```bash
git add app.py database_maker.py club_members_with_next_ranking.csv
git add add_elo_to_csv.py train_model_v4_hybrid.py test_v4_elo_ng.py
git add model_v4_special_cases.pkl feature_cols_v4_special.pkl
git add category_encoder_v4_hybrid.pkl int_to_rank_v4_hybrid.pkl
git add rank_to_int_v4_hybrid.pkl ranking_order_v4_hybrid.pkl
git add CHANGELOG_V4_HYBRID.md
git commit -m "Add V4 Hybrid Model with ELO"
git push origin master
```

### 2. Deploy on Streamlit Cloud

1. Go to https://share.streamlit.io/
2. Click "New app"
3. Select your repository: `rednassmets-byte/TT-predictor-streamlit`
4. Branch: `master`
5. Main file: `app.py`
6. Click "Deploy"

### 3. Verify Deployment

The app should:
- ✅ Load V3 models successfully
- ✅ Load V4 special cases model (optional, falls back to V3)
- ✅ Show "🎯 Using ELO-enhanced model" for NG players with ELO
- ✅ Display correct model type in predictions
- ✅ Handle youth categories properly (boost only for C6+)

## What's New in V4 Hybrid

### Model Performance
- **V3 Regular**: 85.55% accuracy (normal cases)
- **V4 Special Cases**: 96.08% accuracy (NG players with ELO)
- **Improvement**: +1.91% for NG players

### Features
1. **Automatic Model Selection**
   - NG players with ELO → V4 special cases model
   - Strong performers (potential big jumps) → V4 special cases model
   - Everyone else → V3 model

2. **Smart Boost Logic**
   - Youth at C6 or lower → Aggressive boost
   - Youth above C6 → Conservative (no boost)
   - Adults → No boost

3. **ELO Integration**
   - 16,440 players with valid ELO data
   - ELO helps predict initial rank for NG players
   - Falls back gracefully if ELO missing

## Files Included

### Core Application
- `app.py` - Main Streamlit app with hybrid model
- `database_maker.py` - Data fetching with ELO extraction
- `requirements.txt` - All dependencies

### Data
- `club_members_with_next_ranking.csv` - Training data with ELO
- `club_data.csv` - Club information

### V3 Models (Primary)
- `model_v3_improved.pkl`
- `model_filtered_v3_improved.pkl`
- Supporting files: category_encoder, feature_cols, rank mappings

### V4 Models (Special Cases)
- `model_v4_special_cases.pkl` - For NG + big jumps with ELO
- `feature_cols_v4_special.pkl`
- `category_encoder_v4_hybrid.pkl`
- `int_to_rank_v4_hybrid.pkl`
- `rank_to_int_v4_hybrid.pkl`
- `ranking_order_v4_hybrid.pkl`

### Training Scripts
- `train_model_v4_hybrid.py` - Train hybrid models
- `train_model_v4_with_elo.py` - Train ELO model
- `add_elo_to_csv.py` - Fetch ELO data

### Testing
- `test_v4_elo_ng.py` - Verify NG player improvements

## Troubleshooting

### Model Not Loading
- Check all .pkl files are committed
- Verify file sizes (GitHub has 100MB limit)
- Check Streamlit logs for errors

### ELO Not Working
- App will fall back to V3 automatically
- Check if `elo` column exists in CSV
- Verify ELO values are numeric

### Wrong Model Being Used
- Check model_type display in app
- Verify category and rank detection
- Look for "🎯 Using ELO-enhanced model" message

## Monitoring

After deployment, monitor:
1. Model accuracy in production
2. ELO-enhanced predictions vs regular
3. User feedback on predictions
4. Error rates by category/rank

## Rollback Plan

If issues occur:
```bash
git revert HEAD
git push origin master
```

App will automatically redeploy with previous version.

## Support

- GitHub: https://github.com/rednassmets-byte/TT-predictor-streamlit
- Issues: Report on GitHub Issues tab
- Logs: Check Streamlit Cloud dashboard
