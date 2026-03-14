@echo off
echo ========================================
echo Pushing V4 Hybrid Model to GitHub
echo ========================================

echo.
echo Adding modified files...
git add app.py
git add database_maker.py
git add club_members_with_next_ranking.csv
git add requirements.txt

echo.
echo Adding V4 hybrid model files...
git add add_elo_to_csv.py
git add train_model_v4_hybrid.py
git add train_model_v4_with_elo.py
git add test_v4_elo_ng.py

echo.
echo Adding V4 model artifacts...
git add model_v4_special_cases.pkl
git add model_v4_with_elo.pkl
git add feature_cols_v4_special.pkl
git add feature_cols_v4.pkl
git add category_encoder_v4_hybrid.pkl
git add int_to_rank_v4_hybrid.pkl
git add rank_to_int_v4_hybrid.pkl
git add ranking_order_v4_hybrid.pkl

echo.
echo Adding documentation...
git add CHANGELOG_V4_HYBRID.md
git add README.md

echo.
echo Committing changes...
git commit -m "Add V4 Hybrid Model with ELO integration

- Integrated ELO ratings into predictions
- V4 special cases model for NG players (96.08%% accuracy, +1.91%% improvement)
- Hybrid system: V3 for normal cases, V4 with ELO for special cases
- Updated boost logic for youth categories
- Added parallel ELO data fetching
- Improved model selection based on rank and category
- All tests passing, ready for deployment"

echo.
echo Pushing to GitHub...
git push origin master

echo.
echo ========================================
echo Done! Check https://github.com/rednassmets-byte/TT-predictor-streamlit
echo ========================================
pause
