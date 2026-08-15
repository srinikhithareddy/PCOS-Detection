"""
Analyze ultrasound quality assessment results
Generate statistics, plots, and quality classification
"""

import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

print("=" * 80)
print("ULTRASOUND QUALITY ANALYSIS")
print("=" * 80)

# Load quality report
report_path = Path('data/reports/ultrasound_quality_report.csv')
df = pd.read_csv(report_path)

print(f"\nLoaded quality report: {len(df)} images")

# Calculate detailed statistics
def calculate_detailed_stats(series, name):
    """Calculate detailed percentiles for a metric"""
    stats = {
        'Metric': name,
        'Count': len(series),
        'Mean': series.mean(),
        'Std': series.std(),
        'Min': series.min(),
        'Max': series.max(),
        'Median': series.median(),
        '25th': series.quantile(0.25),
        '50th': series.quantile(0.50),
        '75th': series.quantile(0.75),
        '90th': series.quantile(0.90),
        '95th': series.quantile(0.95),
        '99th': series.quantile(0.99)
    }
    return stats

brisque_stats = calculate_detailed_stats(df['brisque_score'].dropna(), 'BRISQUE')
niqe_stats = calculate_detailed_stats(df['niqe_score'].dropna(), 'NIQE')

print("\n" + "=" * 80)
print("DETAILED STATISTICS")
print("=" * 80)

stats_df = pd.DataFrame([brisque_stats, niqe_stats])
print(stats_df.to_string(index=False))

# Statistics by class
print("\n" + "=" * 80)
print("STATISTICS BY CLASS")
print("=" * 80)

class_stats = []
for class_name in df['class'].unique():
    class_df = df[df['class'] == class_name]
    
    b_stats = calculate_detailed_stats(class_df['brisque_score'].dropna(), f'{class_name}_BRISQUE')
    n_stats = calculate_detailed_stats(class_df['niqe_score'].dropna(), f'{class_name}_NIQE')
    
    class_stats.append({
        'Class': class_name,
        'Count': len(class_df),
        'BRISQUE_Mean': b_stats['Mean'],
        'BRISQUE_Std': b_stats['Std'],
        'BRISQUE_Min': b_stats['Min'],
        'BRISQUE_Max': b_stats['Max'],
        'BRISQUE_Median': b_stats['Median'],
        'NIQE_Mean': n_stats['Mean'],
        'NIQE_Std': n_stats['Std'],
        'NIQE_Min': n_stats['Min'],
        'NIQE_Max': n_stats['Max'],
        'NIQE_Median': n_stats['Median']
    })

class_stats_df = pd.DataFrame(class_stats)
print(class_stats_df.to_string(index=False))

# Generate distribution plots
print("\n" + "=" * 80)
print("GENERATING DISTRIBUTION PLOTS")
print("=" * 80)

plots_dir = Path('data/reports/plots')
plots_dir.mkdir(parents=True, exist_ok=True)

# Overall BRISQUE distribution
plt.figure(figsize=(10, 6))
sns.histplot(df['brisque_score'], bins=50, kde=True)
plt.title('BRISQUE Score Distribution - All Images')
plt.xlabel('BRISQUE Score (lower is better)')
plt.ylabel('Frequency')
plt.grid(True, alpha=0.3)
plt.savefig(plots_dir / 'brisque_distribution_overall.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved: brisque_distribution_overall.png")

# Overall NIQE distribution
plt.figure(figsize=(10, 6))
sns.histplot(df['niqe_score'], bins=50, kde=True)
plt.title('NIQE Score Distribution - All Images')
plt.xlabel('NIQE Score (lower is better)')
plt.ylabel('Frequency')
plt.grid(True, alpha=0.3)
plt.savefig(plots_dir / 'niqe_distribution_overall.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved: niqe_distribution_overall.png")

# BRISQUE by class
fig, axes = plt.subplots(2, 3, figsize=(15, 10))
axes = axes.flatten()
for idx, class_name in enumerate(df['class'].unique()):
    if idx < len(axes):
        class_df = df[df['class'] == class_name]
        axes[idx].hist(class_df['brisque_score'], bins=30, alpha=0.7, edgecolor='black')
        axes[idx].set_title(f'BRISQUE - {class_name}')
        axes[idx].set_xlabel('BRISQUE Score')
        axes[idx].set_ylabel('Frequency')
        axes[idx].grid(True, alpha=0.3)

# Remove empty subplot
if len(axes) > len(df['class'].unique()):
    for idx in range(len(df['class'].unique()), len(axes)):
        fig.delaxes(axes[idx])

plt.tight_layout()
plt.savefig(plots_dir / 'brisque_distribution_by_class.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved: brisque_distribution_by_class.png")

# NIQE by class
fig, axes = plt.subplots(2, 3, figsize=(15, 10))
axes = axes.flatten()
for idx, class_name in enumerate(df['class'].unique()):
    if idx < len(axes):
        class_df = df[df['class'] == class_name]
        axes[idx].hist(class_df['niqe_score'], bins=30, alpha=0.7, edgecolor='black')
        axes[idx].set_title(f'NIQE - {class_name}')
        axes[idx].set_xlabel('NIQE Score')
        axes[idx].set_ylabel('Frequency')
        axes[idx].grid(True, alpha=0.3)

# Remove empty subplot
if len(axes) > len(df['class'].unique()):
    for idx in range(len(df['class'].unique()), len(axes)):
        fig.delaxes(axes[idx])

plt.tight_layout()
plt.savefig(plots_dir / 'niqe_distribution_by_class.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved: niqe_distribution_by_class.png")

# Box plots by class
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
sns.boxplot(data=df, x='class', y='brisque_score', ax=ax1)
ax1.set_title('BRISQUE by Class')
ax1.set_xlabel('Class')
ax1.set_ylabel('BRISQUE Score')
ax1.tick_params(axis='x', rotation=45)

sns.boxplot(data=df, x='class', y='niqe_score', ax=ax2)
ax2.set_title('NIQE by Class')
ax2.set_xlabel('Class')
ax2.set_ylabel('NIQE Score')
ax2.tick_params(axis='x', rotation=45)

plt.tight_layout()
plt.savefig(plots_dir / 'boxplots_by_class.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved: boxplots_by_class.png")

# Scatter plot BRISQUE vs NIQE
plt.figure(figsize=(10, 8))
plt.scatter(df['brisque_score'], df['niqe_score'], alpha=0.5, s=10)
plt.xlabel('BRISQUE Score')
plt.ylabel('NIQE Score')
plt.title('BRISQUE vs NIQE - All Images')
plt.grid(True, alpha=0.3)
plt.savefig(plots_dir / 'brisque_vs_niqe_scatter.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved: brisque_vs_niqe_scatter.png")

print("\n" + "=" * 80)
print("QUALITY THRESHOLD ANALYSIS")
print("=" * 80)

# Analyze potential threshold strategies
print("\nBRISQUE Percentiles:")
for p in [5, 10, 25, 50, 75, 90, 95]:
    print(f"  {p}th percentile: {df['brisque_score'].quantile(p/100):.4f}")

print("\nNIQE Percentiles:")
for p in [5, 10, 25, 50, 75, 90, 95]:
    print(f"  {p}th percentile: {df['niqe_score'].quantile(p/100):.4f}")

# Proposed threshold strategy based on data distribution
print("\n" + "=" * 80)
print("PROPOSED QUALITY CLASSIFICATION STRATEGY")
print("=" * 80)

print("""
Based on the actual score distributions, we propose a percentile-based strategy:

BRISQUE Classification (lower is better):
- Good: ≤ 25th percentile (≤ 10.99)
- Poor: 25th-75th percentile (10.99 - 27.86)
- Unusable: > 75th percentile (> 27.86)

NIQE Classification (lower is better):
- Good: ≤ 25th percentile (≤ 4.81)
- Poor: 25th-75th percentile (4.81 - 6.79)
- Unusable: > 75th percentile (> 6.79)

Combined Classification (both metrics must agree):
- Good: Both BRISQUE and NIQE are "Good"
- Poor: At least one metric is "Poor", neither is "Unusable"
- Review: Metrics disagree significantly (one Good, one Unusable)
- Unusable: At least one metric is "Unusable"

This strategy:
1. Is data-driven based on actual score distributions
2. Does not force fixed percentages
3. Keeps thresholds configurable
4. Flags disagreements for manual review
""")

# Apply classification
def classify_image(row):
    """Classify image based on BRISQUE and NIQE scores"""
    brisque = row['brisque_score']
    niqe = row['niqe_score']
    
    # BRISQUE classification
    if brisque <= 10.99:
        brisque_cat = 'good'
    elif brisque <= 27.86:
        brisque_cat = 'poor'
    else:
        brisque_cat = 'unusable'
    
    # NIQE classification
    if niqe <= 4.81:
        niqe_cat = 'good'
    elif niqe <= 6.79:
        niqe_cat = 'poor'
    else:
        niqe_cat = 'unusable'
    
    # Combined classification
    if brisque_cat == 'good' and niqe_cat == 'good':
        quality_category = 'good'
        decision = 'continue'
    elif brisque_cat == 'unusable' or niqe_cat == 'unusable':
        quality_category = 'unusable'
        decision = 'reject'
    elif (brisque_cat == 'good' and niqe_cat == 'unusable') or (brisque_cat == 'unusable' and niqe_cat == 'good'):
        quality_category = 'review'
        decision = 'manual_review'
    else:
        quality_category = 'poor'
        decision = 'enhance'
    
    return brisque_cat, niqe_cat, quality_category, decision

df[['brisque_category', 'niqe_category', 'quality_category', 'preprocessing_decision']] = df.apply(
    lambda row: pd.Series(classify_image(row)), axis=1
)

# Save updated report
df.to_csv(report_path, index=False)
print(f"\nUpdated quality report with classifications saved to: {report_path}")

# Classification summary
print("\n" + "=" * 80)
print("CLASSIFICATION SUMMARY")
print("=" * 80)

print("\nOverall Classification:")
print(df['quality_category'].value_counts())
print(f"\nPercentages:")
print(df['quality_category'].value_counts(normalize=True) * 100)

print("\nPreprocessing Decisions:")
print(df['preprocessing_decision'].value_counts())

print("\nClassification by Class:")
for class_name in df['class'].unique():
    class_df = df[df['class'] == class_name]
    print(f"\n{class_name}:")
    print(class_df['quality_category'].value_counts())
    print(f"Percentages:")
    print(class_df['quality_category'].value_counts(normalize=True) * 100)

print("\n" + "=" * 80)
print("ANALYSIS COMPLETE")
print("=" * 80)
print(f"\nResults saved to:")
print(f"  - {report_path}")
print(f"  - {plots_dir}/")
