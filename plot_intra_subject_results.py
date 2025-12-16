import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Set style for better-looking plots
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (10, 6)
plt.rcParams['font.size'] = 11

# Read the data
df = pd.read_csv('intra_subject_exp.csv')

# Create main figure with multiple subplots
fig = plt.figure(figsize=(18, 18))

# 1. Average Test Accuracy by Subject and Model
ax1 = plt.subplot(3, 3, 1)
subject_model = df.groupby(['subject', 'model'])['best_test_accuracy'].mean().unstack()
subject_model.plot(kind='bar', ax=ax1, color=['#3498db', '#e74c3c'], width=0.8)
ax1.set_title('Average Test Accuracy by Subject and Model', fontsize=13, fontweight='bold')
ax1.set_xlabel('Subject', fontsize=11)
ax1.set_ylabel('Test Accuracy', fontsize=11)
ax1.legend(title='Model', fontsize=9)
ax1.set_xticklabels(ax1.get_xticklabels(), rotation=0)
ax1.grid(axis='y', alpha=0.3)

# 2. Average Test Accuracy by Frequency Band
ax2 = plt.subplot(3, 3, 2)
freq_avg = df.groupby('freq_band')['best_test_accuracy'].agg(['mean', 'std'])
colors = ['#9b59b6', '#1abc9c', '#f39c12']
ax2.bar(freq_avg.index, freq_avg['mean'], yerr=freq_avg['std'], 
        capsize=8, color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)
ax2.set_title('Average Accuracy by Frequency Band', fontsize=13, fontweight='bold')
ax2.set_ylabel('Test Accuracy', fontsize=11)
ax2.set_xlabel('Frequency Band (Hz)', fontsize=11)
ax2.grid(axis='y', alpha=0.3)
for i, (idx, row) in enumerate(freq_avg.iterrows()):
    ax2.text(i, row['mean'] + row['std'] + 0.01, f"{row['mean']:.3f}", 
             ha='center', va='bottom', fontweight='bold', fontsize=10)

# 3. Model Performance Comparison
ax3 = plt.subplot(3, 3, 3)
model_avg = df.groupby('model')['best_test_accuracy'].agg(['mean', 'std'])
ax3.bar(model_avg.index, model_avg['mean'], yerr=model_avg['std'], 
        capsize=10, color=['#3498db', '#e74c3c'], alpha=0.8, edgecolor='black', linewidth=1.5)
ax3.set_title('Overall Model Performance', fontsize=13, fontweight='bold')
ax3.set_ylabel('Test Accuracy', fontsize=11)
ax3.grid(axis='y', alpha=0.3)
for i, (idx, row) in enumerate(model_avg.iterrows()):
    ax3.text(i, row['mean'] + row['std'] + 0.01, f"{row['mean']:.3f}", 
             ha='center', va='bottom', fontweight='bold', fontsize=11)

# 4. Heatmap: Subject x Frequency Band Performance
ax4 = plt.subplot(3, 3, 4)
heatmap_data = df.groupby(['subject', 'freq_band'])['best_test_accuracy'].mean().unstack()
sns.heatmap(heatmap_data, annot=True, fmt='.3f', cmap='RdYlGn', ax=ax4, 
            cbar_kws={'label': 'Test Accuracy'}, vmin=0, vmax=0.5)
ax4.set_title('Performance Heatmap: Subject × Frequency', fontsize=13, fontweight='bold')
ax4.set_xlabel('Frequency Band', fontsize=11)
ax4.set_ylabel('Subject', fontsize=11)

# 5. Box plot: Distribution of accuracies by model
ax5 = plt.subplot(3, 3, 5)
df.boxplot(column='best_test_accuracy', by='model', ax=ax5, 
           patch_artist=True, grid=True)
ax5.set_title('Test Accuracy Distribution by Model', fontsize=13, fontweight='bold')
ax5.set_xlabel('Model', fontsize=11)
ax5.set_ylabel('Test Accuracy', fontsize=11)
plt.suptitle('')  # Remove auto title

# 6. Overfitting Analysis by Model
ax6 = plt.subplot(3, 3, 6)
df['overfitting_gap'] = df['final_train_accuracy'] - df['final_test_accuracy']
overfit_by_model = df.groupby('model')['overfitting_gap'].agg(['mean', 'std'])
ax6.bar(overfit_by_model.index, overfit_by_model['mean'], yerr=overfit_by_model['std'],
        capsize=10, color=['#3498db', '#e74c3c'], alpha=0.8, edgecolor='black', linewidth=1.5)
ax6.set_title('Overfitting Gap by Model', fontsize=13, fontweight='bold')
ax6.set_ylabel('Train - Test Accuracy Gap', fontsize=11)
ax6.axhline(y=0.5, color='orange', linestyle='--', linewidth=2, alpha=0.5)
ax6.grid(axis='y', alpha=0.3)

# 7. Subject Performance Ranking
ax7 = plt.subplot(3, 3, 7)
subject_ranking = df.groupby('subject')['best_test_accuracy'].mean().sort_values(ascending=True)
colors_rank = plt.cm.viridis(np.linspace(0, 1, len(subject_ranking)))
ax7.barh(range(len(subject_ranking)), subject_ranking.values, color=colors_rank, 
         alpha=0.8, edgecolor='black', linewidth=1.5)
ax7.set_yticks(range(len(subject_ranking)))
ax7.set_yticklabels([f'Subject {int(s)}' for s in subject_ranking.index])
ax7.set_xlabel('Average Test Accuracy', fontsize=11)
ax7.set_title('Subject Performance Ranking', fontsize=13, fontweight='bold')
ax7.grid(axis='x', alpha=0.3)
for i, v in enumerate(subject_ranking.values):
    ax7.text(v + 0.01, i, f'{v:.3f}', va='center', fontweight='bold', fontsize=10)

# 8. Best Configuration per Subject
ax8 = plt.subplot(3, 3, 8)
best_per_subject = df.loc[df.groupby('subject')['best_test_accuracy'].idxmax()]
x_pos = np.arange(len(best_per_subject))
colors_best = ['#e74c3c' if m == 'lstm' else '#3498db' for m in best_per_subject['model']]
ax8.bar(x_pos, best_per_subject['best_test_accuracy'], color=colors_best, 
        alpha=0.8, edgecolor='black', linewidth=1.5)
ax8.set_xticks(x_pos)
ax8.set_xticklabels([f"S{int(s)}" for s in best_per_subject['subject']])
ax8.set_xlabel('Subject', fontsize=11)
ax8.set_ylabel('Best Test Accuracy', fontsize=11)
ax8.set_title('Best Result per Subject', fontsize=13, fontweight='bold')
ax8.grid(axis='y', alpha=0.3)
# Add labels showing model and freq band
for i, (idx, row) in enumerate(best_per_subject.iterrows()):
    ax8.text(i, row['best_test_accuracy'] + 0.01, 
             f"{row['model'][:4]}\n{row['freq_band']}", 
             ha='center', va='bottom', fontsize=8, fontweight='bold')

# 9. Model x Frequency Band Performance
ax9 = plt.subplot(3, 3, 9)
model_freq = df.groupby(['model', 'freq_band'])['best_test_accuracy'].mean().unstack()
x = np.arange(len(model_freq.columns))
width = 0.35
for i, (model, values) in enumerate(model_freq.iterrows()):
    offset = (i - 0.5) * width
    color = '#3498db' if model == 'lstm' else '#e74c3c'
    ax9.bar(x + offset, values, width, label=model, color=color, alpha=0.8, edgecolor='black')
ax9.set_xlabel('Frequency Band', fontsize=11)
ax9.set_ylabel('Average Test Accuracy', fontsize=11)
ax9.set_title('Model Performance Across Frequency Bands', fontsize=13, fontweight='bold')
ax9.set_xticks(x)
ax9.set_xticklabels(model_freq.columns)
ax9.legend(fontsize=10)
ax9.grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig('intra_subject_results_analysis.png', dpi=300, bbox_inches='tight')
print("Saved: intra_subject_results_analysis.png")

# Create second figure: Detailed subject-by-subject comparison
fig2, axes = plt.subplots(2, 3, figsize=(18, 10))
axes = axes.flatten()

for idx, subject in enumerate(sorted(df['subject'].unique())):
    ax = axes[idx]
    subject_data = df[df['subject'] == subject]
    
    # Group by model and freq_band
    pivot = subject_data.pivot(index='freq_band', columns='model', values='best_test_accuracy')
    pivot.plot(kind='bar', ax=ax, color=['#3498db', '#e74c3c'], width=0.7)
    
    ax.set_title(f'Subject {int(subject)} Performance', fontsize=12, fontweight='bold')
    ax.set_xlabel('Frequency Band', fontsize=10)
    ax.set_ylabel('Test Accuracy', fontsize=10)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
    ax.legend(title='Model', fontsize=9)
    ax.grid(axis='y', alpha=0.3)
    
    # Add value labels
    for container in ax.containers:
        ax.bar_label(container, fmt='%.3f', fontsize=8, padding=3)

plt.tight_layout()
plt.savefig('intra_subject_by_subject_analysis.png', dpi=300, bbox_inches='tight')
print("Saved: intra_subject_by_subject_analysis.png")

# Create third figure: Summary statistics
fig3 = plt.figure(figsize=(16, 12))

# Add title at top of figure
fig3.text(0.5, 0.98, 'Intra-Subject Experiment Summary', 
          ha='center', va='top', fontsize=18, fontweight='bold')

# Summary table
ax_table = plt.subplot(2, 1, 1)
ax_table.axis('off')

summary_data = []
summary_data.append(['Metric', 'Value'])
summary_data.append(['Total Experiments', str(len(df))])
summary_data.append(['Number of Subjects', str(len(df['subject'].unique()))])
summary_data.append(['Models Tested', ', '.join(df['model'].unique())])
summary_data.append(['Frequency Bands', ', '.join(df['freq_band'].unique())])
summary_data.append(['', ''])
summary_data.append(['Overall Statistics', ''])
summary_data.append(['Mean Test Accuracy', f"{df['best_test_accuracy'].mean():.4f}"])
summary_data.append(['Std Test Accuracy', f"{df['best_test_accuracy'].std():.4f}"])
summary_data.append(['Min Test Accuracy', f"{df['best_test_accuracy'].min():.4f}"])
summary_data.append(['Max Test Accuracy', f"{df['best_test_accuracy'].max():.4f}"])
summary_data.append(['', ''])
summary_data.append(['Best Result', ''])
best_idx = df['best_test_accuracy'].idxmax()
summary_data.append(['Best Subject', f"Subject {int(df.loc[best_idx, 'subject'])}"])
summary_data.append(['Best Model', df.loc[best_idx, 'model']])
summary_data.append(['Best Frequency', df.loc[best_idx, 'freq_band']])
summary_data.append(['Best Accuracy', f"{df.loc[best_idx, 'best_test_accuracy']:.4f}"])

table = ax_table.table(cellText=summary_data, cellLoc='left', loc='center',
                       colWidths=[0.3, 0.4])
table.auto_set_font_size(False)
table.set_fontsize(12)
table.scale(1, 3)

# Style table
for i in range(len(summary_data)):
    if i in [0, 6, 12]:  # Headers
        table[(i, 0)].set_facecolor('#34495e')
        table[(i, 1)].set_facecolor('#34495e')
        table[(i, 0)].set_text_props(weight='bold', color='white')
        table[(i, 1)].set_text_props(weight='bold', color='white')
    elif i in [5, 11]:  # Separators
        table[(i, 0)].set_facecolor('#95a5a6')
        table[(i, 1)].set_facecolor('#95a5a6')
    elif i % 2 == 0:
        table[(i, 0)].set_facecolor('#ecf0f1')
        table[(i, 1)].set_facecolor('#ecf0f1')

# Comparison table
ax_comp = plt.subplot(2, 1, 2)
ax_comp.axis('off')

comp_data = []
comp_data.append(['Subject', 'Best Model', 'Best Freq Band', 'Best Accuracy', 'Worst Accuracy', 'Improvement'])

for subject in sorted(df['subject'].unique()):
    subj_data = df[df['subject'] == subject]
    best_idx = subj_data['best_test_accuracy'].idxmax()
    best_acc = subj_data.loc[best_idx, 'best_test_accuracy']
    worst_acc = subj_data['best_test_accuracy'].min()
    improvement = best_acc - worst_acc
    
    comp_data.append([
        f'Subject {int(subject)}',
        subj_data.loc[best_idx, 'model'],
        subj_data.loc[best_idx, 'freq_band'],
        f'{best_acc:.4f}',
        f'{worst_acc:.4f}',
        f'{improvement:.4f}'
    ])

comp_table = ax_comp.table(cellText=comp_data, cellLoc='center', loc='center',
                            colWidths=[0.12, 0.15, 0.15, 0.15, 0.15, 0.15])
comp_table.auto_set_font_size(False)
comp_table.set_fontsize(11)
comp_table.scale(1, 2.5)

# Style comparison table
for i in range(len(comp_data[0])):
    comp_table[(0, i)].set_facecolor('#34495e')
    comp_table[(0, i)].set_text_props(weight='bold', color='white')

for i in range(1, len(comp_data)):
    for j in range(len(comp_data[0])):
        if i % 2 == 0:
            comp_table[(i, j)].set_facecolor('#ecf0f1')

plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.savefig('intra_subject_summary.png', dpi=300, bbox_inches='tight')
print("Saved: intra_subject_summary.png")

# Print console summary
print("\n" + "="*80)
print("INTRA-SUBJECT EXPERIMENT RESULTS SUMMARY")
print("="*80)
print(f"\nTotal experiments: {len(df)}")
print(f"Subjects: {sorted(df['subject'].unique())}")
print(f"Models: {list(df['model'].unique())}")
print(f"Frequency bands: {list(df['freq_band'].unique())}")

print("\n" + "="*80)
print("OVERALL STATISTICS")
print("="*80)
print(f"Mean test accuracy: {df['best_test_accuracy'].mean():.4f} ± {df['best_test_accuracy'].std():.4f}")
print(f"Range: [{df['best_test_accuracy'].min():.4f}, {df['best_test_accuracy'].max():.4f}]")

print("\n" + "="*80)
print("PERFORMANCE BY MODEL")
print("="*80)
print(df.groupby('model')['best_test_accuracy'].agg(['mean', 'std', 'min', 'max']))

print("\n" + "="*80)
print("PERFORMANCE BY SUBJECT")
print("="*80)
print(df.groupby('subject')['best_test_accuracy'].agg(['mean', 'std', 'min', 'max']))

print("\n" + "="*80)
print("PERFORMANCE BY FREQUENCY BAND")
print("="*80)
print(df.groupby('freq_band')['best_test_accuracy'].agg(['mean', 'std', 'min', 'max']))

print("\n" + "="*80)
print("BEST RESULT")
print("="*80)
best_idx = df['best_test_accuracy'].idxmax()
print(f"Subject: {int(df.loc[best_idx, 'subject'])}")
print(f"Model: {df.loc[best_idx, 'model']}")
print(f"Frequency band: {df.loc[best_idx, 'freq_band']}")
print(f"Test accuracy: {df.loc[best_idx, 'best_test_accuracy']:.4f}")
print(f"Epoch: {int(df.loc[best_idx, 'best_epoch'])}")
print("="*80)

plt.show()
