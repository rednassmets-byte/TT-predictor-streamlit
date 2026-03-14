# Current Model Configuration

## ✅ Three Models Using Original Simple Architecture

The app now uses **three specialized models**, all based on the original simpler architecture that performed better in testing.

### Model Selection Logic

```
IF category is JUN:
    → Use JUN specialized model (model_jun.pkl)
    
ELSE IF category is BEN, PRE, MIN, or CAD:
    → Use Filtered youth model (model_filtered.pkl)
    
ELSE (all other categories including J19, SEN, VET, HER, etc.):
    → Use Regular adult model (model.pkl)
```

## Models in Use

### 1. Regular Model (`model.pkl`)
- **Categories**: All adult categories (SEN, VET, HER, etc.)
- **Excludes**: BEN, PRE, MIN, CAD, JUN, J19
- **Features**: ~20 original features
- **Architecture**: Simple RandomForest with basic SMOTE balancing
- **Performance**: Best accuracy based on user testing

### 2. Filtered Model (`model_filtered.pkl`)
- **Categories**: BEN, PRE, MIN, CAD only
- **Features**: ~38 features (same as regular but optimized for youth)
- **Architecture**: Simple RandomForest with conservative SMOTE
- **Performance**: Good accuracy for youth categories

### 3. JUN Model (`model_jun.pkl`) ⭐ NEW
- **Categories**: JUN only (J19 uses regular model)
- **Features**: ~38 features (same architecture as filtered)
- **Architecture**: Simple RandomForest with conservative SMOTE
- **Test Accuracy**: 62.63% (on training data)
- **Real-world Accuracy**: 84.21% on JUN players (12% better than regular model!)
- **Training samples**: 782 (JUN + J19 data)
- **Why JUN only**: Regular model performs better on J19 (83.33% vs 76.92%)

## Why This Configuration?

1. **Original architecture works better**: User testing showed the simpler models were more accurate
2. **Specialized models**: Each age group gets a model trained specifically on their data
3. **No over-engineering**: Avoided complex features and ensemble methods that didn't improve real-world performance

## Model Comparison

| Model | Categories | Features | Architecture | Status |
|-------|-----------|----------|--------------|--------|
| Regular | Adult | ~20 | Simple RF | ✅ Active |
| Filtered | BEN/PRE/MIN/CAD | ~38 | Simple RF | ✅ Active |
| JUN/J19 | JUN/J19 | ~38 | Simple RF | ✅ Active |

## Training Scripts

- `train_model.py` - Regular model
- `train_model_filtered.py` - Filtered model
- `train_model_jun.py` - JUN/J19 model ⭐ NEW

## Files Created

### JUN/J19 Model Files:
- `model_jun.pkl`
- `category_encoder_jun.pkl`
- `feature_cols_jun.pkl`
- `rank_to_int_jun.pkl`
- `int_to_rank_jun.pkl`
- `ranking_order_jun.pkl`
- `scaler_jun.pkl`

## How to Verify

1. Open the app at http://localhost:8501
2. Select a JUN player → Should see: `Model: JUN (specialized)`
3. Select a J19 player → Should see: `Model: Regular (adult categories)`
4. Select a CAD player → Should see: `Model: Filtered (youth categories)`

## Performance Notes

**JUN Model Performance (tested on seasons 24-25):**
- JUN players: **84.21%** accuracy (vs 72.18% with regular model) → **+12.03% improvement!**
- J19 players: 76.92% accuracy (vs 83.33% with regular model) → -6.41% worse

**Decision:** Use JUN model only for JUN category, let J19 use regular model

This gives us the best of both worlds:
- JUN players get specialized model with 12% better accuracy
- J19 players get regular model with 6% better accuracy

## Next Steps

Test the JUN/J19 model with real players and compare predictions to:
- What the regular model would predict
- Actual outcomes

If the JUN/J19 model doesn't perform better, we can easily revert by removing it from the model selection logic.
