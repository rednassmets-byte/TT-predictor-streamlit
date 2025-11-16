@echo off
echo ========================================
echo Update Streamlit App
echo ========================================
echo.

echo What did you change?
set /p update_msg="Enter update description: "

if "%update_msg%"=="" set update_msg=Update app

echo.
echo Adding files...
git add .

echo.
echo Committing changes...
git commit -m "%update_msg%"

echo.
echo Pushing to GitHub...
git push origin master

echo.
if %ERRORLEVEL% EQU 0 (
    echo ========================================
    echo SUCCESS! App Updated
    echo ========================================
    echo.
    echo Streamlit Cloud will automatically redeploy in 1-2 minutes.
    echo.
    echo Check your app at:
    echo https://tt-predictor-streamlit.streamlit.app
    echo.
) else (
    echo ========================================
    echo ERROR: Update failed
    echo ========================================
)

pause
