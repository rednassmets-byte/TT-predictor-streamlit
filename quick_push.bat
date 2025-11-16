@echo off
echo ========================================
echo Quick Push to GitHub
echo ========================================
echo.

echo Adding all files...
git add .

echo.
echo Committing...
git commit -m "Deploy to Streamlit - Improved models with junior leniency"

echo.
echo Pushing to GitHub...
git push origin blackboxai/update-app

echo.
if %ERRORLEVEL% EQU 0 (
    echo ========================================
    echo SUCCESS!
    echo ========================================
    echo.
    echo Pushed to: https://github.com/rednassmets-byte/TT-predictor-streamlit
    echo Branch: blackboxai/update-app
    echo.
    echo IMPORTANT: You may want to merge this to master branch
    echo.
    echo To deploy on Streamlit:
    echo 1. Go to https://share.streamlit.io
    echo 2. New app
    echo 3. Repository: rednassmets-byte/TT-predictor-streamlit
    echo 4. Branch: blackboxai/update-app (or master after merge)
    echo 5. Main file: app.py
    echo 6. Deploy!
    echo.
) else (
    echo ========================================
    echo FAILED - You may need to authenticate
    echo ========================================
)

pause
