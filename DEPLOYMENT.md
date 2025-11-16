# Deployment Guide for TT KlassementPredictor

## Prerequisites

1. GitHub account
2. Streamlit Cloud account (free at share.streamlit.io)
3. Git installed on your computer

## Step-by-Step Deployment

### 1. Prepare Your Repository

First, make sure all files are committed to git:

```bash
# Check git status
git status

# Add all files (including .pkl model files)
git add .

# Commit changes
git commit -m "Prepare for Streamlit deployment"
```

### 2. Push to GitHub

If you haven't already connected to GitHub:

```bash
# Add remote (replace with your GitHub repo URL)
git remote add origin https://github.com/YOUR_USERNAME/YOUR_REPO_NAME.git

# Push to GitHub
git push -u origin main
```

If you already have a remote:

```bash
# Just push
git push origin main
```

### 3. Deploy on Streamlit Cloud

1. Go to https://share.streamlit.io
2. Sign in with your GitHub account
3. Click "New app"
4. Fill in the form:
   - **Repository**: Select your repository
   - **Branch**: `main`
   - **Main file path**: `app.py`
   - **App URL**: Choose a custom URL (optional)
5. Click "Deploy!"

### 4. Wait for Deployment

Streamlit will:
- Install dependencies from `requirements.txt`
- Load your model files
- Start the app

This usually takes 2-5 minutes.

### 5. Access Your App

Once deployed, you'll get a URL like:
```
https://YOUR_APP_NAME.streamlit.app
```

## Important Files Checklist

Make sure these files are in your repository:

### Application Files
- ✅ `app.py` - Main application
- ✅ `database_maker.py` - API integration
- ✅ `club_data.csv` - Club data
- ✅ `requirements.txt` - Dependencies
- ✅ `.streamlit/config.toml` - Configuration

### Model Files (CRITICAL - Must be included!)
- ✅ `model.pkl` (Regular model)
- ✅ `model_filtered.pkl` (Youth model)
- ✅ `scaler.pkl` (Regular scaler)
- ✅ `scaler_filtered.pkl` (Youth scaler)
- ✅ `category_encoder.pkl` (Regular encoder)
- ✅ `category_encoder_filtered.pkl` (Youth encoder)
- ✅ `rank_to_int.pkl` (Rank mappings)
- ✅ `int_to_rank.pkl` (Rank mappings)
- ✅ `rank_to_int_filtered.pkl` (Youth rank mappings)
- ✅ `int_to_rank_filtered.pkl` (Youth rank mappings)
- ✅ `feature_cols.pkl` (Feature columns)
- ✅ `feature_cols_filtered.pkl` (Youth feature columns)
- ✅ `ranking_order.pkl` (Ranking order)
- ✅ `ranking_order_filtered.pkl` (Youth ranking order)

## Troubleshooting

### "Module not found" error
- Check `requirements.txt` has all dependencies
- Make sure versions are compatible

### "File not found" error
- Verify all .pkl files are committed to git
- Check .gitignore is not excluding .pkl files

### App crashes on startup
- Check Streamlit Cloud logs
- Verify model files are not corrupted
- Test locally first: `streamlit run app.py`

### Large file size warning
If GitHub complains about large files:
1. Model files should be < 100MB each
2. If larger, consider using Git LFS or hosting models elsewhere

## Updating Your Deployment

To update your deployed app:

```bash
# Make changes to your code
# Commit changes
git add .
git commit -m "Update app"

# Push to GitHub
git push origin main
```

Streamlit Cloud will automatically detect changes and redeploy!

## Environment Variables (Optional)

If you need to add secrets (API keys, etc.):

1. Go to your app settings on Streamlit Cloud
2. Click "Secrets"
3. Add your secrets in TOML format:
   ```toml
   [api]
   key = "your-api-key"
   ```

Access in code:
```python
import streamlit as st
api_key = st.secrets["api"]["key"]
```

## Support

If you encounter issues:
- Check Streamlit Cloud logs
- Review Streamlit documentation: https://docs.streamlit.io
- Test locally before deploying
