# Reverted to Original Models

## Summary

Based on user testing, the **original models performed better** than the improved versions. The app has been reverted to use the original models.

## Current Configuration

### Models in Use:

1. **Regular Model** (`model.pkl`)
   - Used for: All categories EXCEPT BEN/PRE/MIN/CAD
   - Includes: JUN, J19, SEN, VET, HER, and all other adult categories
   - Features: ~20 original features
   - Performance: Better accuracy based on real-world testing

2. **Filtered Model** (`model_filtered.pkl`)
   - Used for: BEN, PRE, MIN, CAD only
   - Features: ~20 original features
   - Performance: Better accuracy for youth categories

### Model Selection Logic:

```python
if category in ["BEN", "PRE", "MIN", "CAD"]:
    → Use Filtered Model
else:
    → Use Regular Model (includes JUN, J19, and all adults)
```

## Why Revert?

- User reported that predictions from improved models were less accurate
- Original models with simpler feature set performed better in practice
- Sometimes simpler models generalize better than complex ones

## Files Status

### Active (In Use):
- `model.pkl` - Regular model ✓
- `model_filtered.pkl` - Filtered model ✓
- `category_encoder.pkl` ✓
- `feature_cols.pkl` ✓
- `rank_to_int.pkl` ✓
- `int_to_rank.pkl` ✓
- `ranking_order.pkl` ✓
- `scaler.pkl` ✓
- (Same files with `_filtered` suffix for filtered model)

### Archived (Not in use):
- `model_improved.pkl` - Improved regular model
- `model_filtered_improved.pkl` - Improved filtered model
- `model_jun_improved.pkl` - JUN/J19 specialized model
- (All corresponding encoder/feature files with `_improved` suffix)

## Training Scripts

### Original Models (Currently Active):
- `train_model.py` - Regular model
- `train_model_filtered.py` - Filtered model

### Improved Models (Archived):
- `train_model_improved.py`
- `train_model_filtered_improved.py`
- `train_model_jun_improved.py`

## Key Differences

| Aspect | Original Models | Improved Models |
|--------|----------------|-----------------|
| Features | ~20 basic features | 42 advanced features |
| Optimization | Default parameters | Optuna-tuned |
| Ensemble | Single RF | RF + Balanced RF + GB + XGBoost |
| Balancing | Basic SMOTE | Conservative SMOTE + Undersampling |
| Specialization | 2 models (regular + filtered) | 3 models (regular + filtered + JUN) |
| **Real Performance** | **Better** | Worse |

## Lessons Learned

1. **More features ≠ Better predictions**: The 42-feature improved models were more complex but less accurate
2. **Simpler is sometimes better**: Original models with basic features generalized better
3. **Real-world testing matters**: Evaluation metrics don't always reflect real performance
4. **User feedback is crucial**: The user's experience testing actual players revealed the truth

## If You Want to Switch Back to Improved Models

Simply change the model file names in `app.py`:
- `model.pkl` → `model_improved.pkl`
- `model_filtered.pkl` → `model_filtered_improved.pkl`
- Add back the JUN model loading and selection logic

But based on testing, the original models work better!
