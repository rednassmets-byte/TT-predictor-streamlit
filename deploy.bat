@echo off
echo ========================================
echo TT KlassementPredictor - GitHub Deploy
echo ========================================
echo.

echo Step 1: Checking git status...
git status
echo.

echo Step 2: Adding all files...
git add .
echo.

echo Step 3: Committing changes...
set /p commit_msg="Enter commit message (or press Enter for default): "
if "%commit_msg%"=="" set commit_msg=Update app for Streamlit deployment

git commit -m "%commit_msg%"
echo.

echo Step 4: Pushing to GitHub...
git push origin main
echo.

if %ERRORLEVEL% EQU 0 (
    echo ========================================
    echo SUCCESS! Files pushed to GitHub
    echo ========================================
    echo.
    echo Next steps:
    echo 1. Go to https://share.streamlit.io
    echo 2. Sign in with GitHub
    echo 3. Click "New app"
    echo 4. Select your repository
    echo 5. Set main file to: app.py
    echo 6. Click Deploy!
    echo.
) else (
    echo ========================================
    echo ERROR: Push failed
    echo ========================================
    echo.
    echo Possible issues:
    echo - Not connected to GitHub remote
    echo - Authentication required
    echo - Network issues
    echo.
    echo To set up GitHub remote:
    echo git remote add origin https://github.com/YOUR_USERNAME/YOUR_REPO.git
    echo.
)

pause
