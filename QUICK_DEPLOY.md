# Quick Deploy to Streamlit Cloud

## Option 1: Automatic (Recommended)

### Windows:
```bash
deploy_streamlit.bat
```

### Mac/Linux:
```bash
chmod +x deploy_streamlit.sh
./deploy_streamlit.sh
```

## Option 2: Manual Steps

### 1. Install Git LFS
```bash
# Windows: Download from https://git-lfs.github.com/
# Mac: brew install git-lfs
# Linux: sudo apt-get install git-lfs

git lfs install
```

### 2. Initialize Repository
```bash
git init
git lfs track "*.pkl"
git lfs track "*.csv"
```

### 3. Add Files
```bash
git add .gitattributes
git add app.py requirements.txt club_data.csv
git add *_v3*.pkl *_v2*.pkl
```

### 4. Commit and Push
```bash
git commit -m "Deploy to Streamlit"
git remote add origin https://github.com/YOUR_USERNAME/YOUR_REPO.git
git push -u origin main
```

### 5. Deploy on Streamlit Cloud
1. Go to https://share.streamlit.io
2. Sign in with GitHub
3. Click "New app"
4. Select your repository
5. Main file: `app.py`
6. Click "Deploy"

## What Gets Deployed

✓ V3 models (155MB + 18MB)
✓ V2 fallback models (58MB + 12MB)
✓ Boost logic for big improvements
✓ All features working

## Your App Will Have

- 86% accuracy predictions
- Boost for exceptional performers
- Youth-specific model (88% accuracy)
- Automatic fallback to V2
- Professional UI

## Troubleshooting

### "File too large" error
- Make sure Git LFS is installed and initialized
- Check .gitattributes file exists

### "Module not found" error
- Streamlit will install from requirements.txt
- Wait 2-5 minutes for first deployment

### App is slow
- First load downloads large model files
- Subsequent loads are cached and fast

## After Deployment

Your app URL: `https://YOUR-APP-NAME.streamlit.app`

Share this with users!
