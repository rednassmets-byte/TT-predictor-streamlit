"""
Visualize Neural Network vs Random Forest comparison
Creates charts showing performance differences
"""
import matplotlib.pyplot as plt
import numpy as np

# Results from comparison
models = ['V3 Random\nForest', 'Neural\nNetwork']
accuracy = [85.60, 77.17]
within_1 = [97.85, 98.75]
within_2 = [99.26, 99.74]
training_time = [30, 120]  # seconds

# Create figure with subplots
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle('Neural Network vs Random Forest Comparison', fontsize=16, fontweight='bold')

# Colors
colors = ['#2ecc71', '#3498db']  # Green for RF, Blue for NN

# 1. Accuracy Comparison
ax1 = axes[0, 0]
bars1 = ax1.bar(models, accuracy, color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)
ax1.set_ylabel('Accuracy (%)', fontsize=12, fontweight='bold')
ax1.set_title('Exact Prediction Accuracy', fontsize=13, fontweight='bold')
ax1.set_ylim([70, 90])
ax1.grid(axis='y', alpha=0.3, linestyle='--')
ax1.axhline(y=80, color='red', linestyle='--', alpha=0.5, label='80% threshold')

# Add value labels on bars
for bar, val in zip(bars1, accuracy):
    height = bar.get_height()
    ax1.text(bar.get_x() + bar.get_width()/2., height + 0.5,
             f'{val:.2f}%', ha='center', va='bottom', fontsize=11, fontweight='bold')

# Add winner annotation
ax1.annotate('WINNER!', xy=(0, 85.60), xytext=(0, 88),
            arrowprops=dict(arrowstyle='->', color='green', lw=2),
            fontsize=12, fontweight='bold', color='green', ha='center')

# 2. Within N Ranks Comparison
ax2 = axes[0, 1]
x = np.arange(len(models))
width = 0.35

bars2a = ax2.bar(x - width/2, within_1, width, label='Within 1 rank', 
                 color='#3498db', alpha=0.8, edgecolor='black', linewidth=1.5)
bars2b = ax2.bar(x + width/2, within_2, width, label='Within 2 ranks',
                 color='#9b59b6', alpha=0.8, edgecolor='black', linewidth=1.5)

ax2.set_ylabel('Accuracy (%)', fontsize=12, fontweight='bold')
ax2.set_title('Prediction Tolerance', fontsize=13, fontweight='bold')
ax2.set_xticks(x)
ax2.set_xticklabels(models)
ax2.set_ylim([95, 100])
ax2.legend(fontsize=10)
ax2.grid(axis='y', alpha=0.3, linestyle='--')

# Add value labels
for bars in [bars2a, bars2b]:
    for bar in bars:
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                f'{height:.1f}%', ha='center', va='bottom', fontsize=9)

# 3. Training Time Comparison
ax3 = axes[1, 0]
bars3 = ax3.bar(models, training_time, color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)
ax3.set_ylabel('Training Time (seconds)', fontsize=12, fontweight='bold')
ax3.set_title('Training Efficiency', fontsize=13, fontweight='bold')
ax3.grid(axis='y', alpha=0.3, linestyle='--')

# Add value labels
for bar, val in zip(bars3, training_time):
    height = bar.get_height()
    ax3.text(bar.get_x() + bar.get_width()/2., height + 2,
             f'{val}s', ha='center', va='bottom', fontsize=11, fontweight='bold')

# Add efficiency annotation
ax3.annotate('4x Faster!', xy=(0, 30), xytext=(0.5, 60),
            arrowprops=dict(arrowstyle='->', color='green', lw=2),
            fontsize=11, fontweight='bold', color='green', ha='center')

# 4. Feature Comparison Table
ax4 = axes[1, 1]
ax4.axis('off')

# Create comparison table
features = [
    ['Feature', 'Random Forest', 'Neural Network'],
    ['Accuracy', '85.60%', '77.17%'],
    ['Training Time', '30s', '120s'],
    ['Interpretability', 'High', 'Low'],
    ['Data Efficiency', 'Excellent', 'Needs More'],
    ['Maintenance', 'Easy', 'Complex'],
    ['Dependencies', 'Minimal', 'More'],
]

table = ax4.table(cellText=features, cellLoc='center', loc='center',
                 colWidths=[0.35, 0.325, 0.325])
table.auto_set_font_size(False)
table.set_fontsize(10)
table.scale(1, 2.5)

# Style header row
for i in range(3):
    cell = table[(0, i)]
    cell.set_facecolor('#34495e')
    cell.set_text_props(weight='bold', color='white')

# Style data rows - highlight winner
for i in range(1, len(features)):
    for j in range(3):
        cell = table[(i, j)]
        if j == 0:
            cell.set_facecolor('#ecf0f1')
            cell.set_text_props(weight='bold')
        elif j == 1:  # Random Forest column
            cell.set_facecolor('#d5f4e6')  # Light green
        else:  # Neural Network column
            cell.set_facecolor('#ebf5fb')  # Light blue

ax4.set_title('Feature Comparison', fontsize=13, fontweight='bold', pad=20)

plt.tight_layout()
plt.savefig('model_comparison_chart.png', dpi=300, bbox_inches='tight')
print("✓ Saved: model_comparison_chart.png")
plt.show()

print("\n" + "=" * 80)
print("VISUALIZATION COMPLETE")
print("=" * 80)
print("\nKey Findings:")
print("  • Random Forest is 8.43% more accurate")
print("  • Random Forest trains 4x faster")
print("  • Both models have excellent tolerance (>97% within 1 rank)")
print("  • Random Forest is better suited for this tabular data problem")
print("\nRecommendation: Continue using V3 Random Forest in production")
