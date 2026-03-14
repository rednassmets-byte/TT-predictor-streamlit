# Deploy to Streamlit Cloud - Step by Step

## Prerequisites
- GitHub account
- Streamlit Cloud account (free at share.streamlit.io)

## Step 1: Prepare Your Repository

### Required Files (Already Present ✓)
- `app.py` - Main application
- `requirements.txt` - Dependencies
- `club_data.csv` - Club data
- Model files (V3):
  - `model_v3_improved.pkl`
  - `model_filtered_v3_improved.pkl`
  - `category_encoder_v3.pkl`
  - `category_encoder_filtered_v3.pkl`
  - `feature_cols_v3.pkl`
  - `feature_cols_filtered_v3.pkl`
  - `int_to_rank_v3.pkl`
  - `int_to_rank_filtered_v3.pkl`
  - `rank_to_int_v3.pkl`
  - `rank_to_int_filtered_v3.pkl`
  - `ranking_order_v3.pkl`
  - `ranking_order_filtered_v3.pkl`

### Optional: Create .streamlit/config.toml
```toml
[theme]
primaryColor = "#FF4B4B"
backgroundColor = "#FFFFFF"
secondaryBackgroundColor = "#F0F2F6"
textColor = "#262730"
font = "sans serif"

[server]
maxUploadSize = 200
enableCORS = false
enableXsrfProtection = true
```

## Step 2: Push to GitHub

```bash
# Initialize git (if not already)
git init

# Add files
git add app.py requirements.txt club_data.csv *.pkl

# Commit
git commit -m "Deploy V3 models with boost logic"

# Add remote (replace with your repo)
git remote add origin https://github.com/YOUR_USERNAME/YOUR_REPO.git

# Push
git push -u origin main
```

## Step 3: Deploy on Streamlit Cloud

1. Go to https://share.streamlit.io
2. Click "New app"
3. Select your GitHub repository
4. Set:
   - **Main file path**: `app.py`
   - **Python version**: 3.11
5. Click "Deploy"

## Step 4: Wait for Deployment

Streamlit will:
- Install dependencies from requirements.txt
- Load model files
- Start the app

This takes 2-5 minutes.

## Important Notes

### Model Files Size
The .pkl files are large (~100MB total). GitHub has limits:
- Single file: 100MB max
- Repository: 1GB recommended max

If files are too large, use Git LFS:
```bash
git lfs install
git lfs track "*.pkl"
git add .gitattributes
git commit -m "Track model files with LFS"
```

### Alternative: Host Models Separately
If model files are too large for GitHub:
1. Upload to Hugging Face Hub
2. Update app.py to download from HF (already has fallback code)

## Troubleshooting

### Error: "File too large"
Use Git LFS or host models on Hugging Face

### Error: "Module not found"
Check requirements.txt has all dependencies

### Error: "Model file not found"
Ensure all .pkl files are committed to repo

### App is slow
Model files are large - first load takes time

## Your App URL

After deployment, you'll get a URL like:
`https://YOUR_APP_NAME.streamlit.app`

Share this URL with users!

## Features Included

✓ V3 models (86% accuracy)
✓ Boost logic for big improvements
✓ Youth model (88% accuracy)
✓ Automatic fallback to V2
✓ All features working
