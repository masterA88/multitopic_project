"""
================================================================================
STEP 1: Exploratory Data Analysis & Preprocessing Verification
================================================================================
Research: Pendekatan Multitopic Stacking Ensemble untuk Klasifikasi Ulasan
          Emosi pada Sosial Media Berbasis LSA

References:
  [1] Saputri, M. S., Mahendra, R., & Adriani, M. (2018). Emotion Classification
      on Indonesian Twitter Dataset. IALP 2018.
  [2] Deerwester, S., Dumais, S. T., Furnas, G. W., Landauer, T. K., & Harshman, R.
      (1990). Indexing by Latent Semantic Analysis. JASIS, 41(6), 391-407.
  [3] Mienye, I. D., & Sun, Y. (2022). A Survey of Ensemble Learning: Concepts,
      Algorithms, Applications, and Prospects. IEEE Access, 10, 99129-99149.
  [4] Junianto, E. et al. (2024). Klasifikasi Emosi pada Teks Berbahasa Inggris
      Menggunakan Pendekatan Ensemble Bagging. JNTETI, 13(4), 272-281.
  [5] Glenn, A., LaCasse, P., & Cox, B. (2023). Emotion classification of
      Indonesian Tweets using Bidirectional LSTM. Neural Computing and
      Applications, 35, 345-360.
  [6] Wolpert, D. H. (1992). Stacked Generalization. Neural Networks, 5(2), 241-259.
  [7] Chawla, N. V. et al. (2002). SMOTE: Synthetic Minority Over-sampling
      Technique. JAIR, 16, 321-357.
================================================================================
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import warnings
warnings.filterwarnings('ignore')

# ============================================================
# Configuration
# ============================================================
DATA_PATH = "preprocessing_instagram_xlsx_-_JASMINE.csv"
OUTPUT_DIR = "outputs"
FIGURES_DIR = os.path.join(OUTPUT_DIR, "figures")
RANDOM_STATE = 42

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(FIGURES_DIR, exist_ok=True)

np.random.seed(RANDOM_STATE)

# ============================================================
# 1. Load Dataset
# ============================================================
print("=" * 70)
print("STEP 1: EXPLORATORY DATA ANALYSIS & PREPROCESSING VERIFICATION")
print("=" * 70)

df = pd.read_csv(DATA_PATH)
print(f"\n[INFO] Dataset loaded: {df.shape[0]} rows, {df.shape[1]} columns")
print(f"[INFO] Columns: {df.columns.tolist()}")

# ============================================================
# 2. Data Cleaning
# ============================================================
print("\n--- Data Cleaning ---")
print(f"Missing values in 'final_text': {df['final_text'].isna().sum()}")
print(f"Missing values in 'label_emosi': {df['label_emosi'].isna().sum()}")

# Drop rows with missing final_text or label_emosi
df_clean = df.dropna(subset=['final_text', 'label_emosi']).copy()
df_clean = df_clean[df_clean['final_text'].str.strip() != ''].copy()
df_clean.reset_index(drop=True, inplace=True)

print(f"[INFO] After cleaning: {df_clean.shape[0]} rows")

# ============================================================
# 3. Label Distribution Analysis
# ============================================================
print("\n--- Label Distribution ---")
label_counts = df_clean['label_emosi'].value_counts()
label_pct = df_clean['label_emosi'].value_counts(normalize=True) * 100

dist_df = pd.DataFrame({
    'Count': label_counts,
    'Percentage (%)': label_pct.round(2)
})
print(dist_df)
print(f"\nTotal samples: {df_clean.shape[0]}")

# Imbalance ratio
majority = label_counts.max()
minority = label_counts.min()
print(f"Imbalance ratio (majority/minority): {majority/minority:.2f}")

# ============================================================
# 4. Visualization: Label Distribution
# ============================================================
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Bar chart
colors = ['#2ecc71', '#e74c3c', '#3498db', '#f39c12', '#9b59b6']
ax1 = axes[0]
bars = ax1.bar(label_counts.index, label_counts.values, color=colors[:len(label_counts)])
ax1.set_title('Distribusi Label Emosi', fontsize=14, fontweight='bold')
ax1.set_xlabel('Emosi')
ax1.set_ylabel('Jumlah')
for bar, val in zip(bars, label_counts.values):
    ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 10,
             str(val), ha='center', va='bottom', fontweight='bold')

# Pie chart
ax2 = axes[1]
ax2.pie(label_counts.values, labels=label_counts.index, autopct='%1.1f%%',
        colors=colors[:len(label_counts)], startangle=90)
ax2.set_title('Proporsi Label Emosi', fontsize=14, fontweight='bold')

plt.tight_layout()
plt.savefig(os.path.join(FIGURES_DIR, 'fig1_label_distribution.png'), dpi=150, bbox_inches='tight')
plt.close()
print("\n[SAVED] fig1_label_distribution.png")

# ============================================================
# 5. Text Length Analysis
# ============================================================
df_clean['text_length'] = df_clean['final_text'].apply(lambda x: len(str(x).split()))

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Histogram
ax1 = axes[0]
ax1.hist(df_clean['text_length'], bins=30, color='#3498db', edgecolor='black', alpha=0.7)
ax1.set_title('Distribusi Panjang Teks (Jumlah Kata)', fontsize=13, fontweight='bold')
ax1.set_xlabel('Jumlah Kata')
ax1.set_ylabel('Frekuensi')
ax1.axvline(df_clean['text_length'].mean(), color='red', linestyle='--',
            label=f"Mean: {df_clean['text_length'].mean():.1f}")
ax1.legend()

# Boxplot per emotion
ax2 = axes[1]
sns.boxplot(data=df_clean, x='label_emosi', y='text_length', ax=ax2, palette=colors)
ax2.set_title('Panjang Teks per Label Emosi', fontsize=13, fontweight='bold')
ax2.set_xlabel('Emosi')
ax2.set_ylabel('Jumlah Kata')

plt.tight_layout()
plt.savefig(os.path.join(FIGURES_DIR, 'fig2_text_length_analysis.png'), dpi=150, bbox_inches='tight')
plt.close()
print("[SAVED] fig2_text_length_analysis.png")

# Text length statistics per emotion
print("\n--- Text Length Statistics per Emotion ---")
print(df_clean.groupby('label_emosi')['text_length'].describe().round(2))

# ============================================================
# 6. Word Frequency Analysis (Top Words per Emotion)
# ============================================================
from collections import Counter

fig, axes = plt.subplots(2, 3, figsize=(18, 10))
axes = axes.flatten()

emotions = df_clean['label_emosi'].unique()
for idx, emotion in enumerate(sorted(emotions)):
    if idx >= 5:
        break
    subset = df_clean[df_clean['label_emosi'] == emotion]
    all_words = ' '.join(subset['final_text'].astype(str)).split()
    word_freq = Counter(all_words).most_common(15)

    words, counts = zip(*word_freq) if word_freq else ([], [])
    ax = axes[idx]
    ax.barh(range(len(words)), counts, color=colors[idx % len(colors)])
    ax.set_yticks(range(len(words)))
    ax.set_yticklabels(words)
    ax.invert_yaxis()
    ax.set_title(f'Top 15 Words - {emotion.upper()}', fontweight='bold')
    ax.set_xlabel('Frekuensi')

# Hide the unused subplot
if len(emotions) < 6:
    axes[5].axis('off')

plt.tight_layout()
plt.savefig(os.path.join(FIGURES_DIR, 'fig3_word_frequency_per_emotion.png'), dpi=150, bbox_inches='tight')
plt.close()
print("[SAVED] fig3_word_frequency_per_emotion.png")

# ============================================================
# 7. Save Cleaned Data
# ============================================================
cleaned_path = os.path.join(OUTPUT_DIR, "data_cleaned.csv")
df_clean.to_csv(cleaned_path, index=False)
print(f"\n[SAVED] Cleaned dataset: {cleaned_path}")
print(f"[INFO] Final dataset shape: {df_clean.shape}")

# ============================================================
# 8. Summary Statistics
# ============================================================
summary = {
    'Total Samples': df_clean.shape[0],
    'Unique Labels': df_clean['label_emosi'].nunique(),
    'Labels': ', '.join(sorted(df_clean['label_emosi'].unique())),
    'Avg Text Length (words)': df_clean['text_length'].mean().round(2),
    'Median Text Length (words)': df_clean['text_length'].median(),
    'Vocabulary Size (approx)': len(set(' '.join(df_clean['final_text'].astype(str)).split())),
    'Imbalance Ratio': round(majority/minority, 2)
}

print("\n--- Dataset Summary ---")
for k, v in summary.items():
    print(f"  {k}: {v}")

print("\n[DONE] Step 1 completed successfully.")
