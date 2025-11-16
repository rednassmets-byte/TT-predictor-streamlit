@echo off
echo ========================================
echo Pushing to GitHub: TT-predictor-streamlit
echo ========================================
echo.

echo Current branch:
git branch
echo.

echo Step 1: Adding all files...
git add .
echo Done!
echo.

echo Step 2: Committing changes...
git commit -m "Deploy improved models to Streamlit - 72%% accuracy regular, 67%% filtered, junior leniency"
echo Done!
echo.

echo Step 3: Switching to master branch...
git checkout master
if %ERRORLEVEL% NEQ 0 (
    echo Creating master branch...
    git checkout -b master
)
echo Done!
echo.

echo Step 4: Merging changes from blackboxai/update-app...
git merge blackboxai/update-app -m "Merge improved models and features"
echo Done!
echo.

echo Step 5: Pushing to GitHub (TT-predictor-streamlit)...
git push -u origin master
echo.

if %ERRORLEVEL% EQU 0 (
    echo ========================================
    echo SUCCESS! Pushed to GitHub
    echo ========================================
    echo.
    echo Repository: https://github.com/rednassmets-byte/TT-predictor-streamlit
    echo.
    echo Next steps:
    echo 1. Go to https://share.streamlit.io
    echo 2. Sign in with GitHub
    echo 3. Click "New app"
    echo 4. Repository: rednassmets-byte/TT-predictor-streamlit
    echo 5. Branch: master
    echo 6. Main file: app.py
    echo 7. Click "Deploy!"
    echo.
    echo Your app will be live at:
    echo https://tt-predictor-streamlit.streamlit.app
    echo.
) else (
    echo ========================================
    echo ERROR: Push failed
    echo ========================================
    echo.
    echo You may need to authenticate with GitHub.
    echo Try running: git push -u origin master
    echo.
)

pause
