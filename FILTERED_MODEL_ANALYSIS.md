# Filtered Model (Youth Categories) - Error Analysis

## Model Purpose
The filtered model is specifically trained for **youth categories only**:
- **BEN** (Benjamins)
- **PRE** (Pre-cadets)  
- **MIN** (Minimes)
- **CAD** (Cadets)

## Overall Performance

### Accuracy: 84.05%
- **Perfect predictions**: 84.0%
- **Within 1 rank**: 98.0%
- **Within 2 ranks**: 99.3%
- **Mean error**: 0.19 ranks

This is **excellent performance** for youth players, who are typically more volatile and harder to predict.

## What the Filtered Model Gets Most Wrong

### 1. Worst Performing Ranks
- **D2**: 37.5% error rate (but only 8 players)
- **C4**: 33.3% error rate (but only 12 players)
- **E6**: 28.0% error rate (239 players - significant)

The low-sample ranks (D2, C4) have high error rates but aren't statistically significant. **E6 is the real problem** with 239 players and 28% error rate.

### 2. Worst Performing Categories
- **MIN (Minimes)**: 16.8% error rate (535 players)
- **PRE (Pre-cadets)**: 16.4% error rate (293 players)
- **CAD (Cadets)**: 16.0% error rate (717 players)
- **BEN (Benjamins)**: 8.2% error rate (85 players) ✓ Best!

Younger players (MIN, PRE) are slightly harder to predict than older youth (CAD, BEN).

### 3. Prediction Tendencies

#### By Rank Change Type:
- **Improvements** (moving up): 81.1% accuracy (771 players)
- **Stable** (same rank): 86.9% accuracy (850 players)
- **Declines** (moving down): 66.7% accuracy (9 players)

The model struggles with declines, but there are very few declines in youth categories (only 9 players), which makes sense - youth players typically improve or stay stable.

#### By Activity Level:
- **Low activity**: 89.1% accuracy ✓ Best
- **Medium activity**: 86.7% accuracy
- **High activity**: 79.7% accuracy
- **Very high activity**: 80.6% accuracy

Interestingly, the model is **more accurate for low-activity players**. This might be because high-activity youth players are more volatile.

#### By Performance Level:
- **Poor performers**: 96.4% accuracy ✓ Best
- **Below average**: 84.9% accuracy
- **Above average**: 79.4% accuracy
- **Excellent performers**: 75.4% accuracy

The model is **best at predicting poor performers** (likely to stay at same rank or decline) and **worst at predicting excellent performers** (who might make big jumps).

### 4. Most Common Mistakes

1. **NG → NG predicted as E6** (62 times)
   - Model is too optimistic about unranked youth players

2. **E6 → E6 predicted as E4** (36 times)
   - Model predicts improvement when players stay stable

3. **NG → E6 predicted as NG** (28 times)
   - Model is too pessimistic about unranked players getting ranked

### 5. Prediction Bias

**Too Optimistic** (predicts better than actual):
- D2: -0.38
- C2: -0.30
- E6: -0.26
- D0: -0.25

**Too Pessimistic** (predicts worse than actual):
- D4: +0.54
- D6: +0.23

Most ranks are well-balanced, but the model tends to be slightly optimistic for mid-level youth players.

## Key Insights

### Strengths:
1. ✓ **Overall excellent accuracy** (84.05%)
2. ✓ **Very good at predicting stability** (86.9%)
3. ✓ **Excellent for poor performers** (96.4%)
4. ✓ **Best for Benjamins** (91.8% accuracy)
5. ✓ **98% within 1 rank** - very reliable

### Weaknesses:
1. ✗ **Struggles with E6 rank** (28% error rate, 239 players)
2. ✗ **Too optimistic about NG players** (predicts E6 when they stay NG)
3. ✗ **Underestimates excellent performers** (75.4% accuracy)
4. ✗ **Struggles with high-activity players** (79.7% accuracy)
5. ✗ **Poor at predicting declines** (66.7%, but only 9 cases)

## Comparison with Regular Model

### Filtered Model (Youth):
- **Accuracy**: 84.05%
- **Dataset**: 1,630 youth players (≥15 matches)
- **Categories**: BEN, PRE, MIN, CAD only

### Regular Model V3 (Non-Youth):
- **Accuracy**: 85.82%
- **Dataset**: 10,753 non-youth players (≥10 matches)
- **Categories**: SEN, V40, V50, V60, V65, V70, V75, V80, V85, JUN, J19, J21

The filtered model performs **slightly worse** than the regular model, which makes sense because:
1. Youth players are more volatile (rapid development)
2. Smaller dataset (1,630 vs 10,753)
3. More NG (unranked) players who are harder to predict

## Recommendations

### To Improve Filtered Model:

1. **Add youth-specific features**:
   - Age within category (younger MIN vs older MIN)
   - Months in current rank
   - Growth velocity (rapid improvers)

2. **Special handling for NG players**:
   - Different model for NG → first rank
   - Consider match quality, not just quantity

3. **Special handling for E6**:
   - This is the entry rank for many youth
   - Add features for "ready to advance" signals

4. **Handle excellent performers better**:
   - Add "breakthrough" signal for high performers
   - Consider wins against much better players

5. **More training data**:
   - Only 1,630 players is relatively small
   - Consider lowering match threshold to 10 (like regular model)

## Conclusion

The filtered model performs **very well** for youth categories (84.05% accuracy), especially considering youth players are inherently more volatile. The main issues are:
- E6 rank predictions (entry-level youth)
- NG (unranked) player predictions
- Excellent performers who might make big jumps

These are all related to the **high volatility of youth development**, which is difficult to predict from match statistics alone.
