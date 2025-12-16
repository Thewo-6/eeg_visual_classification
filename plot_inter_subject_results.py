import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Set style for better-looking plots
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (10, 6)
plt.rcParams['font.size'] = 10

# Read the data
df = pd.read_csv('inter_subject_results.csv')

# Create a figure with multiple subplots
fig = plt.figure(figsize=(16, 10))

# 1. Best Test Accuracy by Model and Frequency Band
ax1 = plt.subplot(2, 3, 1)
pivot_data = df.pivot(index='freq_band', columns='model', values='best_test_accuracy')
pivot_data.plot(kind='bar', ax=ax1, color=['#3498db', '#e74c3c'])
ax1.set_title('Best Test Accuracy by Model and Frequency Band', fontsize=14, fontweight='bold')
ax1.set_xlabel('Frequency Band', fontsize=12)
ax1.set_ylabel('Test Accuracy', fontsize=12)
ax1.legend(title='Model', fontsize=10)
ax1.set_xticklabels(ax1.get_xticklabels(), rotation=45)
ax1.grid(axis='y', alpha=0.3)

# 2. Comparison: Best Val Accuracy vs Best Test Accuracy
ax2 = plt.subplot(2, 3, 2)
x = np.arange(len(df))
width = 0.35
ax2.bar(x - width/2, df['best_val_accuracy'], width, label='Validation', color='#2ecc71', alpha=0.8)
ax2.bar(x + width/2, df['best_test_accuracy'], width, label='Test', color='#e67e22', alpha=0.8)
ax2.set_xlabel('Experiment', fontsize=12)
ax2.set_ylabel('Accuracy', fontsize=12)
ax2.set_title('Validation vs Test Accuracy', fontsize=14, fontweight='bold')
ax2.set_xticks(x)
ax2.set_xticklabels([f"{df.iloc[i]['model'][:4]}\n{df.iloc[i]['freq_band']}" for i in range(len(df))], fontsize=9)
ax2.legend(fontsize=10)
ax2.grid(axis='y', alpha=0.3)

# 3. Training Progression: Final Accuracies
ax3 = plt.subplot(2, 3, 3)
accuracies = df[['final_train_accuracy', 'final_val_accuracy', 'final_test_accuracy']].values
experiments = [f"{df.iloc[i]['model'][:4]}-{df.iloc[i]['freq_band']}" for i in range(len(df))]
x_pos = np.arange(len(experiments))
ax3.plot(x_pos, accuracies[:, 0], 'o-', label='Train', linewidth=2, markersize=8, color='#3498db')
ax3.plot(x_pos, accuracies[:, 1], 's-', label='Validation', linewidth=2, markersize=8, color='#2ecc71')
ax3.plot(x_pos, accuracies[:, 2], '^-', label='Test', linewidth=2, markersize=8, color='#e74c3c')
ax3.set_xlabel('Experiment', fontsize=12)
ax3.set_ylabel('Final Accuracy', fontsize=12)
ax3.set_title('Final Train/Val/Test Accuracies', fontsize=14, fontweight='bold')
ax3.set_xticks(x_pos)
ax3.set_xticklabels(experiments, rotation=45, ha='right', fontsize=9)
ax3.legend(fontsize=10)
ax3.grid(alpha=0.3)

# 4. Average Accuracy by Model
ax4 = plt.subplot(2, 3, 4)
model_avg = df.groupby('model')['best_test_accuracy'].agg(['mean', 'std'])
ax4.bar(model_avg.index, model_avg['mean'], yerr=model_avg['std'], 
        capsize=10, color=['#3498db', '#e74c3c'], alpha=0.8, edgecolor='black', linewidth=1.5)
ax4.set_title('Average Test Accuracy by Model', fontsize=14, fontweight='bold')
ax4.set_ylabel('Test Accuracy', fontsize=12)
ax4.set_xlabel('Model', fontsize=12)
ax4.grid(axis='y', alpha=0.3)
for i, (idx, row) in enumerate(model_avg.iterrows()):
    ax4.text(i, row['mean'] + row['std'] + 0.01, f"{row['mean']:.3f}", 
             ha='center', va='bottom', fontweight='bold', fontsize=11)

# 5. Average Accuracy by Frequency Band
ax5 = plt.subplot(2, 3, 5)
freq_avg = df.groupby('freq_band')['best_test_accuracy'].agg(['mean', 'std'])
colors = ['#9b59b6', '#1abc9c', '#f39c12']
ax5.bar(freq_avg.index, freq_avg['mean'], yerr=freq_avg['std'], 
        capsize=10, color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)
ax5.set_title('Average Test Accuracy by Frequency Band', fontsize=14, fontweight='bold')
ax5.set_ylabel('Test Accuracy', fontsize=12)
ax5.set_xlabel('Frequency Band (Hz)', fontsize=12)
ax5.set_xticklabels(ax5.get_xticklabels(), rotation=45)
ax5.grid(axis='y', alpha=0.3)
for i, (idx, row) in enumerate(freq_avg.iterrows()):
    ax5.text(i, row['mean'] + row['std'] + 0.01, f"{row['mean']:.3f}", 
             ha='center', va='bottom', fontweight='bold', fontsize=11)

# 6. Overfitting Analysis (Train - Test Gap)
ax6 = plt.subplot(2, 3, 6)
df['overfitting_gap'] = df['final_train_accuracy'] - df['final_test_accuracy']
experiments = [f"{df.iloc[i]['model'][:4]}-{df.iloc[i]['freq_band']}" for i in range(len(df))]
colors_overfit = ['#e74c3c' if gap > 0.5 else '#f39c12' if gap > 0.3 else '#2ecc71' 
                   for gap in df['overfitting_gap']]
ax6.barh(experiments, df['overfitting_gap'], color=colors_overfit, alpha=0.8, edgecolor='black')
ax6.set_xlabel('Overfitting Gap (Train - Test Accuracy)', fontsize=12)
ax6.set_title('Overfitting Analysis', fontsize=14, fontweight='bold')
ax6.axvline(x=0.3, color='orange', linestyle='--', linewidth=2, alpha=0.5, label='Moderate')
ax6.axvline(x=0.5, color='red', linestyle='--', linewidth=2, alpha=0.5, label='Severe')
ax6.legend(fontsize=9)
ax6.grid(axis='x', alpha=0.3)

plt.tight_layout()
plt.savefig('inter_subject_results_analysis.png', dpi=300, bbox_inches='tight')
print("Saved: inter_subject_results_analysis.png")

# Create a second figure with detailed comparison
fig2, axes = plt.subplots(1, 2, figsize=(14, 6))

# Grouped bar chart: Model performance across frequency bands
ax_left = axes[0]
freq_bands = df['freq_band'].unique()
x = np.arange(len(freq_bands))
width = 0.35

lstm_data = df[df['model'] == 'lstm'].set_index('freq_band').loc[freq_bands, 'best_test_accuracy'].values
eeg_data = df[df['model'] == 'EEGChannelNet'].set_index('freq_band').loc[freq_bands, 'best_test_accuracy'].values

ax_left.bar(x - width/2, lstm_data, width, label='LSTM', color='#3498db', alpha=0.8, edgecolor='black')
ax_left.bar(x + width/2, eeg_data, width, label='EEGChannelNet', color='#e74c3c', alpha=0.8, edgecolor='black')
ax_left.set_xlabel('Frequency Band', fontsize=13)
ax_left.set_ylabel('Best Test Accuracy', fontsize=13)
ax_left.set_title('Model Performance Across Frequency Bands', fontsize=15, fontweight='bold')
ax_left.set_xticks(x)
ax_left.set_xticklabels(freq_bands)
ax_left.legend(fontsize=11)
ax_left.grid(axis='y', alpha=0.3)

# Add value labels on bars
for i, (l, e) in enumerate(zip(lstm_data, eeg_data)):
    ax_left.text(i - width/2, l + 0.01, f'{l:.3f}', ha='center', va='bottom', fontsize=10, fontweight='bold')
    ax_left.text(i + width/2, e + 0.01, f'{e:.3f}', ha='center', va='bottom', fontsize=10, fontweight='bold')

# Summary table
ax_right = axes[1]
ax_right.axis('off')

# Create summary statistics table
summary_data = []
summary_data.append(['Metric', 'LSTM', 'EEGChannelNet'])
summary_data.append(['Avg Test Acc', 
                     f"{df[df['model']=='lstm']['best_test_accuracy'].mean():.4f}",
                     f"{df[df['model']=='EEGChannelNet']['best_test_accuracy'].mean():.4f}"])
summary_data.append(['Best Test Acc', 
                     f"{df[df['model']=='lstm']['best_test_accuracy'].max():.4f}",
                     f"{df[df['model']=='EEGChannelNet']['best_test_accuracy'].max():.4f}"])
summary_data.append(['Avg Val Acc', 
                     f"{df[df['model']=='lstm']['best_val_accuracy'].mean():.4f}",
                     f"{df[df['model']=='EEGChannelNet']['best_val_accuracy'].mean():.4f}"])
summary_data.append(['Avg Overfitting Gap', 
                     f"{df[df['model']=='lstm']['overfitting_gap'].mean():.4f}",
                     f"{df[df['model']=='EEGChannelNet']['overfitting_gap'].mean():.4f}"])
summary_data.append(['', '', ''])
summary_data.append(['Best Frequency', '', ''])
for model in ['lstm', 'EEGChannelNet']:
    best_row = df[df['model']==model].loc[df[df['model']==model]['best_test_accuracy'].idxmax()]
    summary_data.append([f"{model}", 
                         f"{best_row['freq_band']}", 
                         f"{best_row['best_test_accuracy']:.4f}"])

table = ax_right.table(cellText=summary_data, cellLoc='center', loc='center',
                       colWidths=[0.4, 0.3, 0.3])
table.auto_set_font_size(False)
table.set_fontsize(11)
table.scale(1, 2.5)

# Style header row
for i in range(3):
    table[(0, i)].set_facecolor('#34495e')
    table[(0, i)].set_text_props(weight='bold', color='white')

# Style alternating rows
for i in range(1, len(summary_data)):
    for j in range(3):
        if i % 2 == 0:
            table[(i, j)].set_facecolor('#ecf0f1')
        if i == 6:  # Section header
            table[(i, j)].set_facecolor('#95a5a6')
            table[(i, j)].set_text_props(weight='bold')

ax_right.set_title('Performance Summary', fontsize=15, fontweight='bold', pad=20)

plt.tight_layout()
plt.savefig('inter_subject_comparison.png', dpi=300, bbox_inches='tight')
print("Saved: inter_subject_comparison.png")

# Print summary to console
print("\n" + "="*60)
print("INTER-SUBJECT EXPERIMENT RESULTS SUMMARY")
print("="*60)
print(df.to_string(index=False))
print("\n" + "="*60)
print("AVERAGE PERFORMANCE BY MODEL")
print("="*60)
print(df.groupby('model')['best_test_accuracy'].agg(['mean', 'std', 'min', 'max']))
print("\n" + "="*60)
print("AVERAGE PERFORMANCE BY FREQUENCY BAND")
print("="*60)
print(df.groupby('freq_band')['best_test_accuracy'].agg(['mean', 'std', 'min', 'max']))
print("\n" + "="*60)
print(f"BEST OVERALL RESULT:")
best_idx = df['best_test_accuracy'].idxmax()
print(f"Model: {df.loc[best_idx, 'model']}")
print(f"Frequency Band: {df.loc[best_idx, 'freq_band']}")
print(f"Test Accuracy: {df.loc[best_idx, 'best_test_accuracy']:.4f}")
print(f"Epoch: {df.loc[best_idx, 'best_epoch']}")
print("="*60)

plt.show()
