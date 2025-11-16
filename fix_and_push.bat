@echo off
echo ========================================
echo Fixing Large Files and Pushing to GitHub
echo ========================================
echo.

echo Step 1: Removing large files from tracking...
git rm -r --cached "ai/" "TT predictor/" "*.exe" 2>nul
echo Done!
echo.

echo Step 2: Adding .gitignore changes...
git add .gitignore
echo Done!
echo.

echo Step 3: Setting up Git LFS for model.pkl...
git lfs track "model.pkl"
git add .gitattributes
echo Done!
echo.

echo Step 4: Adding remaining files...
git add model.pkl model_filtered.pkl *.pkl app.py database_maker.py club_data.csv requirements.txt README.md .streamlit/ *.md *.bat
echo Done!
echo.

echo Step 5: Committing changes...
git commit -m "Fix: Remove large files, use Git LFS for model.pkl"
echo Done!
echo.

echo Step 6: Pushing to GitHub...
git push origin blackboxai/update-app
echo.

if %ERRORLEVEL% EQU 0 (
    echo ========================================
    echo SUCCESS! Pushed to GitHub
    echo ========================================
    echo.
    echo Repository: https://github.com/rednassmets-byte/TT-predictor-streamlit
    echo Branch: blackboxai/update-app
    echo.
    echo Large files removed:
    echo - ai/ folder (231 MB model.pkl)
    echo - TT predictor/ folder (duplicates)
    echo - *.exe files (183 MB)
    echo.
    echo Git LFS enabled for:
    echo - model.pkl (118 MB)
    echo.
    echo Next: Deploy on Streamlit Cloud
    echo 1. Go to https://share.streamlit.io
    echo 2. New app
    echo 3. Repository: rednassmets-byte/TT-predictor-streamlit
    echo 4. Branch: blackboxai/update-app
    echo 5. Main file: app.py
    echo 6. Deploy!
    echo.
) else (
    echo ========================================
    echo ERROR: Push failed
    echo ========================================
    echo.
    echo If Git LFS is not installed:
    echo 1. Download from: https://git-lfs.github.com/
    echo 2. Install Git LFS
    echo 3. Run: git lfs install
    echo 4. Run this script again
    echo.
)

pause
