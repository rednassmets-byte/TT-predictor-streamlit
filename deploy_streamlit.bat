@echo off
REM Streamlit Deployment Script for Windows

echo ===================================
echo Deploying to Streamlit Cloud
echo ===================================

REM Step 1: Check Git LFS
echo Step 1: Checking Git LFS...
git lfs version >nul 2>&1
if errorlevel 1 (
    echo Git LFS not found. Please install it first:
    echo   Download from https://git-lfs.github.com/
    exit /b 1
)

REM Step 2: Initialize Git LFS
echo Step 2: Initializing Git LFS...
git lfs install

REM Step 3: Track large files
echo Step 3: Tracking large files...
git lfs track "*.pkl"
git lfs track "*.csv"

REM Step 4: Add .gitattributes
echo Step 4: Adding .gitattributes...
git add .gitattributes

REM Step 5: Add essential files only
echo Step 5: Adding essential files...
git add app.py
git add requirements.txt
git add club_data.csv

REM Add V3 model files
git add model_v3_improved.pkl
git add model_filtered_v3_improved.pkl
git add category_encoder_v3.pkl
git add category_encoder_filtered_v3.pkl
git add feature_cols_v3.pkl
git add feature_cols_filtered_v3.pkl
git add int_to_rank_v3.pkl
git add int_to_rank_filtered_v3.pkl
git add rank_to_int_v3.pkl
git add rank_to_int_filtered_v3.pkl
git add ranking_order_v3.pkl
git add ranking_order_filtered_v3.pkl

REM Add V2 fallback files
git add model_v2.pkl
git add model_filtered_v2.pkl
git add category_encoder_v2.pkl
git add category_encoder_filtered_v2.pkl
git add feature_cols_v2.pkl
git add feature_cols_filtered_v2.pkl
git add int_to_rank_v2.pkl
git add int_to_rank_filtered_v2.pkl
git add rank_to_int_v2.pkl
git add rank_to_int_filtered_v2.pkl
git add ranking_order_v2.pkl
git add ranking_order_filtered_v2.pkl

REM Step 6: Commit
echo Step 6: Committing...
git commit -m "Deploy V3 models with boost logic to Streamlit"

REM Step 7: Push
echo Step 7: Ready to push to GitHub...
echo Make sure you've set up your remote:
echo   git remote add origin https://github.com/YOUR_USERNAME/YOUR_REPO.git
echo.
pause
git push -u origin main

echo.
echo ===================================
echo Deployment Complete!
echo ===================================
echo Next steps:
echo 1. Go to https://share.streamlit.io
echo 2. Click 'New app'
echo 3. Select your repository
echo 4. Set main file: app.py
echo 5. Click 'Deploy'
echo ===================================
pause
