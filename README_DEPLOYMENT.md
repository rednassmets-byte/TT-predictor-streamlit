# Table Tennis Ranking Prediction App - Deployment Ready

## 🎯 What's Included

### Models
- **V3 Regular Model**: 86% accuracy (non-youth players)
- **V3 Filtered Model**: 88% accuracy (youth players)
- **V2 Fallback Models**: Automatic fallback if V3 not available
- **Boost Logic**: Selective boost for exceptional performers

### Features
✓ Rank prediction with confidence scores
✓ Boost for big improvements (strong performers)
✓ Youth-specific predictions
✓ Performance analysis
✓ Win/loss visualization
✓ Professional Dutch UI

## 🚀 Deploy to Streamlit Cloud

### Quick Start (Windows)
```bash
deploy_streamlit.bat
```

### Quick Start (Mac/Linux)
```bash
chmod +x deploy_streamlit.sh
./deploy_streamlit.sh
```

### Manual Deployment
See `QUICK_DEPLOY.md` for step-by-step instructions

## 📊 Model Performance

| Model | Accuracy | Dataset | Best For |
|-------|----------|---------|----------|
| V3 Regular | 86% | 10,753 players | Non-youth |
| V3 Filtered | 88% | 1,630 players | Youth (BEN/PRE/MIN/CAD) |

### Boost Logic
- **40%** of big improvements get boosted
- **Criteria**: 3 of 4 strong signals required
- **Impact**: More optimistic for exceptional performers
- **Safety**: Only boosts existing improvements

## 📁 Required Files

### Essential (Must Deploy)
- `app.py` - Main application
- `requirements.txt` - Dependencies
- `club_data.csv` - Club data
- `.gitattributes` - Git LFS configuration

### Model Files (V3 - Primary)
- `model_v3_improved.pkl` (155MB)
- `model_filtered_v3_improved.pkl` (18MB)
- `category_encoder_v3.pkl`
- `feature_cols_v3.pkl`
- `int_to_rank_v3.pkl`
- `rank_to_int_v3.pkl`
- `ranking_order_v3.pkl`
- (+ filtered versions)

### Model Files (V2 - Fallback)
- `model_v2.pkl` (58MB)
- `model_filtered_v2.pkl` (12MB)
- (+ supporting files)

## 🔧 Technical Details

### Dependencies
- Python 3.11
- Streamlit 1.28+
- scikit-learn 1.3+
- pandas 2.0+
- plotly 5.18+
- pyvttl (from GitHub)

### File Sizes
- Total: ~250MB
- Uses Git LFS for large files
- Streamlit caches models after first load

## 🌐 After Deployment

Your app will be available at:
```
https://YOUR-APP-NAME.streamlit.app
```

### First Load
- Takes 2-5 minutes (downloading models)
- Shows loading spinner

### Subsequent Loads
- Fast (models cached)
- Instant predictions

## 📖 Documentation

- `QUICK_DEPLOY.md` - Step-by-step deployment
- `DEPLOY_TO_STREAMLIT.md` - Detailed guide
- `BOOST_LOGIC_SUMMARY.md` - How boost works
- `MODEL_IMPROVEMENTS.md` - V3 improvements
- `FILTERED_MODEL_V3_IMPROVEMENTS.md` - Youth model

## 🎨 Features in App

### Prediction Tab
- Select player by club/season/name
- Shows current rank
- Predicts next rank
- Confidence score
- Boost indicator (if applied)

### Performance Analysis
- Win/loss breakdown
- Performance score (1-100)
- Rank comparison
- Visual charts

### Model Selection
- Regular model (non-youth)
- Filtered model (youth)
- Automatic selection based on category

## 🔒 Security & Privacy

- No personal data stored
- All processing client-side
- Models are static (no training on user data)
- HTTPS by default on Streamlit Cloud

## 📞 Support

For issues or questions:
1. Check `QUICK_DEPLOY.md`
2. Check Streamlit logs
3. Verify all .pkl files are committed

## 🎉 Ready to Deploy!

Run the deployment script and your app will be live in minutes!
