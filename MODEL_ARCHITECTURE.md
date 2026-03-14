# Table Tennis Ranking Prediction - Model Architecture

## Overview
Three specialized models using identical improved architecture for different player categories.

## Model Architecture (All Models)

### Core Components
- **Base Algorithm**: Optimized RandomForest (via Optuna hyperparameter tuning)
- **Ensemble**: RandomForest + Balanced RandomForest + Gradient Boosting + XGBoost
- **Features**: 42 advanced features per model
- **Balancing**: Conservative SMOTE + Random Undersampling
- **Scaling**: StandardScaler for feature normalization

### Feature Engineering (42 features)
1. **Base Features** (2)
   - Current rank encoded
   - Category encoded

2. **Performance Metrics** (14)
   - Total wins, losses, matches
   - Overall win rate
   - Performance consistency
   - Recent performance
   - Rank progression potential
   - Performance vs higher ranks
   - Performance vs lower ranks
   - Win/loss ratio
   - Average opponent rank
   - Toughest opponent beaten
   - Win rate standard deviation
   - Recent form

3. **Rank-Specific Features** (26)
   - Win rates for important ranks (B0-E2)
   - Total matches for E-ranks
   - Weighted performance by opponent strength

### Hyperparameter Optimization
- **Method**: Optuna (20 trials)
- **Optimized Parameters**:
  - n_estimators: 200-400
  - max_depth: 12-20
  - min_samples_split: 5-15
  - min_samples_leaf: 2-8
  - max_features: sqrt or log2

## Three Specialized Models

### 1. Regular Model (Adult Categories)
**File**: `model_improved.pkl`

**Training Data**:
- Excludes: BEN, PRE, MIN, CAD, JUN, J19
- Includes: All adult categories (SEN, VET, etc.)
- Min matches: 15
- Training samples: ~6,000+

**Performance**:
- Test accuracy: ~70-75%
- Zero outliers (>3 ranks difference)

**Use Case**: All adult category players

---

### 2. Filtered Model (Youth Categories)
**File**: `model_filtered_improved.pkl`

**Training Data**:
- Categories: BEN, PRE, MIN, CAD only
- Min matches: 15
- Training samples: 1,805
- Test samples: 325

**Performance**:
- Test accuracy: 73.85%
- Evaluation accuracy: 71.21%
- Zero outliers
- Best on BEN (80%), PRE (77.78%)

**Use Case**: Youth categories (BEN/PRE/MIN/CAD)

---

### 3. JUN/J19 Model
**File**: `model_jun_improved.pkl`

**Training Data**:
- Categories: JUN, J19 only
- Min matches: 15
- Training samples: 782
- Test samples: 198

**Performance**:
- Test accuracy: 67.68%
- Evaluation accuracy: 70.40%
- Only 1 outlier (0.47%)
- Balanced between JUN (70.12%) and J19 (70.93%)

**Use Case**: Junior categories (JUN/J19)

---

## Model Selection Logic (app.py)

```python
if category in ["JUN", "J19"]:
    # Use JUN/J19 specialized model
    model = jun_model
    model_type = "JUN/J19 (specialized)"
    
elif category in ["BEN", "PRE", "MIN", "CAD"]:
    # Use filtered model for youth categories
    model = filtered_model
    model_type = "Filtered (youth categories)"
    
else:
    # Use regular model for all other categories
    model = regular_model
    model_type = "Regular (adult categories)"
```

## Training Scripts

1. **Regular Model**: `train_model_improved.py`
2. **Filtered Model**: `train_model_filtered_improved.py`
3. **JUN/J19 Model**: `train_model_jun_improved.py`

## Evaluation Scripts

1. **Regular Model**: `evaluate_improved.py`
2. **Filtered Model**: `evaluate_filtered_improved.py`
3. **JUN/J19 Model**: `evaluate_jun_improved.py`

## Key Improvements Over Original Models

1. **Advanced Feature Engineering**: 42 features vs ~20 original
2. **Opponent Strength Weighting**: Considers quality of opponents
3. **Hyperparameter Optimization**: Optuna-tuned parameters
4. **Ensemble Methods**: Multiple models for robustness
5. **Conservative Balancing**: SMOTE + undersampling for class imbalance
6. **Category Specialization**: Separate models for different age groups

## Performance Summary

| Model | Categories | Accuracy | Outliers | Training Samples |
|-------|-----------|----------|----------|------------------|
| Regular | Adult | ~70-75% | 0 | 6,000+ |
| Filtered | BEN/PRE/MIN/CAD | 71.21% | 0 | 1,805 |
| JUN/J19 | JUN/J19 | 70.40% | 1 (0.47%) | 782 |

All models achieve consistent ~70% accuracy with minimal outliers, providing reliable predictions across all player categories.
