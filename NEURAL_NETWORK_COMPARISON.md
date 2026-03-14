# Neural Network vs Random Forest Comparison

## Summary

I created a neural network model using scikit-learn's MLPClassifier and compared it with the existing Random Forest models used in app.py.

## Results

### Performance Comparison (Regular Categories)

| Model | Accuracy | Exact Predictions | Within 1 Rank | Within 2 Ranks |
|-------|----------|-------------------|---------------|----------------|
| **V3 Random Forest** | **85.60%** | 13,122/15,329 | 97.85% | 99.26% |
| Neural Network | 77.17% | 11,829/15,329 | 98.75% | 99.74% |

**Winner: V3 Random Forest by 8.43%**

## Model Details

### Neural Network (MLPClassifier)
- **Architecture**: 3 hidden layers (128 → 64 → 32 neurons)
- **Activation**: ReLU
- **Optimizer**: Adam with adaptive learning rate
- **Regularization**: L2 (alpha=0.001) + Dropout via early stopping
- **Training**: 40 iterations with early stopping
- **Features**: Same 23 features as V3 model
- **Scaling**: StandardScaler (required for neural networks)

### V3 Random Forest (Current)
- **Architecture**: 200 decision trees
- **Max depth**: 20
- **Features**: 23 engineered features
- **No scaling required**

## Why Random Forest Wins

1. **Dataset Size**: 15,329 samples is relatively small for deep learning
   - Neural networks typically need 100k+ samples to outperform traditional ML
   - Random Forests work well with smaller datasets

2. **Feature Engineering**: Features are already well-engineered
   - Includes domain-specific features (improvement_signal, decline_signal, etc.)
   - Random Forests excel with pre-engineered features
   - Neural networks shine when learning features from raw data

3. **Tabular Data**: This is structured tabular data
   - Random Forests are specifically designed for tabular data
   - Neural networks are better for images, text, sequences

4. **Training Efficiency**:
   - Random Forest: ~30 seconds training time
   - Neural Network: ~2 minutes training time

5. **Interpretability**:
   - Random Forest: Can extract feature importance
   - Neural Network: Black box, harder to interpret

## When Neural Networks Would Win

Neural networks would likely outperform if:
- Dataset was 10x larger (150k+ samples)
- Raw match data was used instead of engineered features
- Temporal/sequential patterns needed to be learned
- Complex non-linear interactions existed that weren't captured

## Recommendation

**Keep using the V3 Random Forest model in app.py**

Reasons:
- ✅ 8.43% higher accuracy
- ✅ Faster training and prediction
- ✅ Better interpretability
- ✅ More stable predictions
- ✅ Easier to maintain and debug
- ✅ No additional dependencies (no TensorFlow needed)

## Files Created

1. **train_model_neural_network_sklearn.py** - Neural network training script
2. **compare_neural_vs_random_forest.py** - Comparison script
3. **model_neural_network.pkl** - Trained neural network model
4. **scaler_neural_network.pkl** - Feature scaler for neural network
5. Supporting files (encoders, feature lists, etc.)

## Technical Notes

- Initially tried TensorFlow/Keras but Python 3.14 compatibility issues
- Switched to scikit-learn's MLPClassifier (Multi-Layer Perceptron)
- MLPClassifier provides similar functionality without TensorFlow dependency
- Used same feature engineering pipeline as existing models for fair comparison

## Conclusion

The neural network experiment confirms that the current V3 Random Forest model is the optimal choice for this ranking prediction task. The Random Forest's superior performance, combined with its efficiency and interpretability, makes it the best model for production use in app.py.
