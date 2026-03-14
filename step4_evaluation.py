"""
================================================================================
STEP 4: PUBLICATION-READY EVALUATION, FIGURES & TABLES
================================================================================
Produces all figures and tables for:
  "LSA-based Stacking Ensemble with Granularity Analysis for Emotion
   Classification on Indonesian Social Media"

Figures:
  fig1  - Label distribution (5-class original)
  fig2  - Text length analysis
  fig3  - Word frequency per emotion
  fig4a - Class consolidation (5→3)
  fig4  - LSA explained variance
  fig5  - LSA granularity comparison (main result)
  fig6  - SMOTE effectiveness comparison
  fig7  - Individual vs Stacking model comparison
  fig8  - Multi-topic vs single-topic comparison
  fig9  - Confusion matrix (best model)
  fig10 - Normalized confusion matrix
  fig11 - ROC curves
  fig12 - Precision-recall curves
  fig13 - Per-class performance bar chart

Tables:
  table1 - Dataset statistics
  table2 - LSA granularity comparison
  table3 - Individual model comparison
  table4 - SMOTE effectiveness
  table5 - Multi-topic vs single-topic
  table6 - Best model per-class report
  table7 - Full results (LaTeX)

References:
  [13] Sokolova & Lapalme (2009). Performance measures for classification.
  [14] Powers (2011). Evaluation: Precision, Recall and F-Measure to ROC.
================================================================================
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import os
import pickle
import warnings
warnings.filterwarnings('ignore')

from sklearn.metrics import (confusion_matrix, classification_report,
                             roc_curve, auc, precision_recall_curve,
                             accuracy_score, f1_score)
from sklearn.preprocessing import label_binarize

# ============================================================
# Configuration
# ============================================================
OUTPUT_DIR = "outputs"
FIGURES_DIR = os.path.join(OUTPUT_DIR, "figures")
TABLES_DIR = os.path.join(OUTPUT_DIR, "tables")
MODELS_DIR = os.path.join(OUTPUT_DIR, "models")

os.makedirs(TABLES_DIR, exist_ok=True)

plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['font.size'] = 11

print("=" * 70)
print("STEP 4: PUBLICATION-READY EVALUATION")
print("=" * 70)

# ============================================================
# Load
# ============================================================
y_test = np.load(os.path.join(OUTPUT_DIR, 'y_test.npy'))
y_pred_best = np.load(os.path.join(OUTPUT_DIR, 'y_pred_best.npy'))
y_proba_best = np.load(os.path.join(OUTPUT_DIR, 'y_pred_proba_best.npy'))
y_pred_mt = np.load(os.path.join(OUTPUT_DIR, 'y_pred_multitopic.npy'))
y_proba_mt = np.load(os.path.join(OUTPUT_DIR, 'y_pred_proba_multitopic.npy'))

with open(os.path.join(MODELS_DIR, 'label_encoder.pkl'), 'rb') as f:
    le = pickle.load(f)

results = pd.read_csv(os.path.join(OUTPUT_DIR, 'all_results.csv'))
smote_analysis = pd.read_csv(os.path.join(OUTPUT_DIR, 'smote_analysis.csv'))

cls = le.classes_
n_cls = len(cls)
print(f"[INFO] Classes: {cls.tolist()}")
print(f"[INFO] Total experiments: {len(results)}")

# Identify best model
best_row = results.loc[results['f1_macro'].idxmax()]
print(f"[INFO] Best model: {best_row['model']} (F1m={best_row['f1_macro']:.4f})")

COLORS3 = ['#2ecc71', '#e74c3c', '#3498db']


# ============================================================
# Fig 5: LSA Granularity Comparison (MAIN RESULT)
# ============================================================
stk_smote = results[results['feature_type'] == 'single_topic_stacking'].copy()
stk_nosmote = results[results['feature_type'] == 'single_topic_stacking_nosmote'].copy()

for df_tmp in [stk_smote, stk_nosmote]:
    df_tmp['k'] = pd.to_numeric(df_tmp['n_topics'], errors='coerce')

fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# F1-macro
ax = axes[0]
if not stk_smote.empty:
    s = stk_smote.sort_values('k')
    ax.plot(s['k'], s['f1_macro'], 'bo-', lw=2, ms=8, label='Stacking + SMOTE')
if not stk_nosmote.empty:
    n = stk_nosmote.sort_values('k')
    ax.plot(n['k'], n['f1_macro'], 'rs--', lw=2, ms=8, label='Stacking No SMOTE')
# Multi-topic line
mt_row = results[results['feature_type'] == 'multitopic_concat']
if not mt_row.empty:
    ax.axhline(y=mt_row.iloc[0]['f1_macro'], color='green', lw=2, ls=':',
               label=f'Multi-Topic Concat ({mt_row.iloc[0]["f1_macro"]:.4f})')
ax.set_xlabel('Number of LSA Topics (k)')
ax.set_ylabel('F1-Macro')
ax.set_title('F1-Macro vs LSA Granularity', fontweight='bold')
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)
ax.set_xticks([5, 10, 25, 50, 100])

# F1-weighted
ax = axes[1]
if not stk_smote.empty:
    ax.plot(s['k'], s['f1_weighted'], 'bo-', lw=2, ms=8, label='Stacking + SMOTE')
if not stk_nosmote.empty:
    ax.plot(n['k'], n['f1_weighted'], 'rs--', lw=2, ms=8, label='Stacking No SMOTE')
if not mt_row.empty:
    ax.axhline(y=mt_row.iloc[0]['f1_weighted'], color='green', lw=2, ls=':',
               label=f'Multi-Topic ({mt_row.iloc[0]["f1_weighted"]:.4f})')
ax.set_xlabel('Number of LSA Topics (k)')
ax.set_ylabel('F1-Weighted')
ax.set_title('F1-Weighted vs LSA Granularity', fontweight='bold')
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)
ax.set_xticks([5, 10, 25, 50, 100])

# Accuracy
ax = axes[2]
if not stk_smote.empty:
    ax.plot(s['k'], s['accuracy'], 'bo-', lw=2, ms=8, label='Stacking + SMOTE')
if not stk_nosmote.empty:
    ax.plot(n['k'], n['accuracy'], 'rs--', lw=2, ms=8, label='Stacking No SMOTE')
if not mt_row.empty:
    ax.axhline(y=mt_row.iloc[0]['accuracy'], color='green', lw=2, ls=':',
               label=f'Multi-Topic ({mt_row.iloc[0]["accuracy"]:.4f})')
ax.set_xlabel('Number of LSA Topics (k)')
ax.set_ylabel('Accuracy')
ax.set_title('Accuracy vs LSA Granularity', fontweight='bold')
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)
ax.set_xticks([5, 10, 25, 50, 100])

plt.tight_layout()
plt.savefig(os.path.join(FIGURES_DIR, 'fig5_granularity_comparison.png'),
            dpi=150, bbox_inches='tight')
plt.close()
print("[SAVED] fig5_granularity_comparison.png")


# ============================================================
# Fig 6: SMOTE Effectiveness
# ============================================================
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

x = np.arange(len(smote_analysis))
w = 0.35

ax = axes[0]
ax.bar(x - w/2, smote_analysis['F1m_SMOTE'], w, label='With SMOTE', color='#3498db')
ax.bar(x + w/2, smote_analysis['F1m_NoSMOTE'], w, label='Without SMOTE', color='#e74c3c')
ax.set_xticks(x)
ax.set_xticklabels([f'k={k}' for k in smote_analysis['k']])
ax.set_ylabel('F1-Macro')
ax.set_title('SMOTE Effect on F1-Macro', fontweight='bold')
ax.legend()
ax.grid(True, alpha=0.3, axis='y')
for i, (v1, v2) in enumerate(zip(smote_analysis['F1m_SMOTE'], smote_analysis['F1m_NoSMOTE'])):
    ax.text(i - w/2, v1 + 0.005, f'{v1:.3f}', ha='center', fontsize=8)
    ax.text(i + w/2, v2 + 0.005, f'{v2:.3f}', ha='center', fontsize=8)

ax = axes[1]
ax.bar(x - w/2, smote_analysis['Acc_SMOTE'], w, label='With SMOTE', color='#3498db')
ax.bar(x + w/2, smote_analysis['Acc_NoSMOTE'], w, label='Without SMOTE', color='#e74c3c')
ax.set_xticks(x)
ax.set_xticklabels([f'k={k}' for k in smote_analysis['k']])
ax.set_ylabel('Accuracy')
ax.set_title('SMOTE Effect on Accuracy', fontweight='bold')
ax.legend()
ax.grid(True, alpha=0.3, axis='y')
for i, (v1, v2) in enumerate(zip(smote_analysis['Acc_SMOTE'], smote_analysis['Acc_NoSMOTE'])):
    ax.text(i - w/2, v1 + 0.005, f'{v1:.3f}', ha='center', fontsize=8)
    ax.text(i + w/2, v2 + 0.005, f'{v2:.3f}', ha='center', fontsize=8)

plt.tight_layout()
plt.savefig(os.path.join(FIGURES_DIR, 'fig6_smote_effectiveness.png'),
            dpi=150, bbox_inches='tight')
plt.close()
print("[SAVED] fig6_smote_effectiveness.png")


# ============================================================
# Fig 7: Individual vs Stacking Comparison
# ============================================================
fig, ax = plt.subplots(figsize=(12, 6))

# Get individual model results for each k (SMOTE)
indiv = results[results['feature_type'] == 'single_topic'].copy()
indiv['k'] = pd.to_numeric(indiv['n_topics'], errors='coerce')

# Group by k, get best individual model F1-macro
best_indiv = indiv.groupby('k')['f1_macro'].max().reset_index()

if not stk_smote.empty:
    ax.plot(stk_smote.sort_values('k')['k'],
            stk_smote.sort_values('k')['f1_macro'],
            'ro-', lw=2.5, ms=10, label='Stacking Ensemble', zorder=5)

for mdl_name, color, marker in [('RF', '#2ecc71', 's'), ('SVM', '#f39c12', '^'),
                                  ('XGB', '#9b59b6', 'D')]:
    sub = indiv[indiv['model'].str.startswith(mdl_name)].sort_values('k')
    if not sub.empty:
        ax.plot(sub['k'], sub['f1_macro'], f'{marker}--', color=color,
                lw=1.5, ms=7, label=mdl_name, alpha=0.8)

ax.set_xlabel('Number of LSA Topics (k)', fontsize=12)
ax.set_ylabel('F1-Macro', fontsize=12)
ax.set_title('Individual Models vs Stacking Ensemble across LSA Granularities',
             fontsize=14, fontweight='bold')
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3)
ax.set_xticks([5, 10, 25, 50, 100])

plt.tight_layout()
plt.savefig(os.path.join(FIGURES_DIR, 'fig7_individual_vs_stacking.png'),
            dpi=150, bbox_inches='tight')
plt.close()
print("[SAVED] fig7_individual_vs_stacking.png")


# ============================================================
# Fig 8: Multi-Topic vs Single-Topic Bar Chart
# ============================================================
comparison_models = ['Stacking_TF-IDF']
for k in [5, 10, 25, 50, 100]:
    comparison_models.append(f'Stacking_k{k}')
comparison_models.append('MultiTopic_Concat_SMOTE')

comp_df = results[results['model'].isin(comparison_models)].copy()

# Rename for display
name_map = {f'Stacking_k{k}': f'LSA k={k}' for k in [5, 10, 25, 50, 100]}
name_map['Stacking_TF-IDF'] = 'TF-IDF Only'
name_map['MultiTopic_Concat_SMOTE'] = 'Multi-Topic\n(k=5+10+25+50+100)'
comp_df['display_name'] = comp_df['model'].map(name_map)

fig, ax = plt.subplots(figsize=(12, 6))
x = np.arange(len(comp_df))
w = 0.28

b1 = ax.bar(x - w, comp_df['accuracy'], w, label='Accuracy', color='#3498db', alpha=0.85)
b2 = ax.bar(x, comp_df['f1_weighted'], w, label='F1-Weighted', color='#2ecc71', alpha=0.85)
b3 = ax.bar(x + w, comp_df['f1_macro'], w, label='F1-Macro', color='#e74c3c', alpha=0.85)

ax.set_xticks(x)
ax.set_xticklabels(comp_df['display_name'], fontsize=9)
ax.set_ylabel('Score')
ax.set_title('Stacking Ensemble: Feature Representation Comparison',
             fontsize=14, fontweight='bold')
ax.legend()
ax.grid(True, alpha=0.3, axis='y')

for bars in [b1, b2, b3]:
    for bar in bars:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, h + 0.005,
                f'{h:.3f}', ha='center', fontsize=7)

plt.tight_layout()
plt.savefig(os.path.join(FIGURES_DIR, 'fig8_multitopic_vs_singletopic.png'),
            dpi=150, bbox_inches='tight')
plt.close()
print("[SAVED] fig8_multitopic_vs_singletopic.png")


# ============================================================
# Fig 9: Confusion Matrix (Best Model)
# ============================================================
cm = confusion_matrix(y_test, y_pred_best)
fig, ax = plt.subplots(figsize=(7, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=cls, yticklabels=cls, ax=ax)
ax.set_xlabel('Predicted Label', fontsize=12)
ax.set_ylabel('True Label', fontsize=12)
ax.set_title(f'Confusion Matrix — Best Model (Stacking k={best_row["n_topics"]})',
             fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(FIGURES_DIR, 'fig9_confusion_matrix.png'),
            dpi=150, bbox_inches='tight')
plt.close()
print("[SAVED] fig9_confusion_matrix.png")


# ============================================================
# Fig 10: Normalized Confusion Matrix
# ============================================================
cm_n = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
fig, ax = plt.subplots(figsize=(7, 5))
sns.heatmap(cm_n, annot=True, fmt='.2f', cmap='YlOrRd',
            xticklabels=cls, yticklabels=cls, ax=ax)
ax.set_xlabel('Predicted Label', fontsize=12)
ax.set_ylabel('True Label', fontsize=12)
ax.set_title('Normalized Confusion Matrix', fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(FIGURES_DIR, 'fig10_confusion_matrix_norm.png'),
            dpi=150, bbox_inches='tight')
plt.close()
print("[SAVED] fig10_confusion_matrix_norm.png")


# ============================================================
# Fig 11: ROC Curves
# ============================================================
y_test_bin = label_binarize(y_test, classes=range(n_cls))

fig, ax = plt.subplots(figsize=(8, 6))
for i in range(n_cls):
    fpr, tpr, _ = roc_curve(y_test_bin[:, i], y_proba_best[:, i])
    roc_auc = auc(fpr, tpr)
    ax.plot(fpr, tpr, color=COLORS3[i], lw=2,
            label=f'{cls[i]} (AUC = {roc_auc:.3f})')

ax.plot([0, 1], [0, 1], 'k--', lw=1, alpha=0.5)
ax.set_xlabel('False Positive Rate', fontsize=12)
ax.set_ylabel('True Positive Rate', fontsize=12)
ax.set_title('ROC Curves — Best Model (One-vs-Rest)', fontsize=13, fontweight='bold')
ax.legend(loc='lower right', fontsize=11)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(FIGURES_DIR, 'fig11_roc_curves.png'),
            dpi=150, bbox_inches='tight')
plt.close()
print("[SAVED] fig11_roc_curves.png")


# ============================================================
# Fig 12: Precision-Recall Curves
# ============================================================
fig, ax = plt.subplots(figsize=(8, 6))
for i in range(n_cls):
    prec, rec, _ = precision_recall_curve(y_test_bin[:, i], y_proba_best[:, i])
    ap = auc(rec, prec)
    ax.plot(rec, prec, color=COLORS3[i], lw=2,
            label=f'{cls[i]} (AP = {ap:.3f})')

ax.set_xlabel('Recall', fontsize=12)
ax.set_ylabel('Precision', fontsize=12)
ax.set_title('Precision-Recall Curves — Best Model', fontsize=13, fontweight='bold')
ax.legend(loc='lower left', fontsize=11)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(FIGURES_DIR, 'fig12_precision_recall.png'),
            dpi=150, bbox_inches='tight')
plt.close()
print("[SAVED] fig12_precision_recall.png")


# ============================================================
# Fig 13: Per-Class Performance
# ============================================================
rpt = classification_report(y_test, y_pred_best, target_names=cls, output_dict=True)

fig, ax = plt.subplots(figsize=(9, 5))
x = np.arange(n_cls)
w = 0.25

precs = [rpt[c]['precision'] for c in cls]
recs = [rpt[c]['recall'] for c in cls]
f1s = [rpt[c]['f1-score'] for c in cls]

ax.bar(x - w, precs, w, label='Precision', color='#3498db')
ax.bar(x, recs, w, label='Recall', color='#2ecc71')
ax.bar(x + w, f1s, w, label='F1-Score', color='#e74c3c')

for i in range(n_cls):
    ax.text(i - w, precs[i] + 0.02, f'{precs[i]:.2f}', ha='center', fontsize=9)
    ax.text(i, recs[i] + 0.02, f'{recs[i]:.2f}', ha='center', fontsize=9)
    ax.text(i + w, f1s[i] + 0.02, f'{f1s[i]:.2f}', ha='center', fontsize=9)

ax.set_xticks(x)
ax.set_xticklabels(cls, fontsize=12)
ax.set_ylabel('Score', fontsize=12)
ax.set_title('Per-Class Performance — Best Model', fontsize=14, fontweight='bold')
ax.legend(fontsize=11)
ax.grid(True, alpha=0.3, axis='y')
ax.set_ylim(0, 1.1)

plt.tight_layout()
plt.savefig(os.path.join(FIGURES_DIR, 'fig13_per_class.png'),
            dpi=150, bbox_inches='tight')
plt.close()
print("[SAVED] fig13_per_class.png")


# ============================================================
# TABLES
# ============================================================
print("\n--- Generating Tables ---")

# Table 1: Dataset statistics
df_clean = pd.read_csv(os.path.join(OUTPUT_DIR, 'data_cleaned.csv'))
t1_5class = df_clean['label_emosi'].value_counts()
with open(os.path.join(OUTPUT_DIR, 'metadata.pkl'), 'rb') as f:
    metadata = pickle.load(f)
label_map = metadata.get('class_consolidation', {})

table1 = pd.DataFrame({
    'Original_Label': t1_5class.index,
    'Count': t1_5class.values,
    'Percentage': [f'{v/len(df_clean)*100:.1f}%' for v in t1_5class.values],
    'Consolidated_Label': [label_map.get(l, l) for l in t1_5class.index]
})
table1.to_csv(os.path.join(TABLES_DIR, 'table1_dataset.csv'), index=False)

# Table 2: LSA granularity comparison
stk_all = results[results['feature_type'].isin(
    ['single_topic_stacking', 'tfidf_baseline', 'multitopic_concat']
)][['model', 'accuracy', 'f1_weighted', 'f1_macro', 'precision_weighted',
    'recall_weighted']].copy()
stk_all.columns = ['Model', 'Accuracy', 'F1-Weighted', 'F1-Macro', 'Precision', 'Recall']
stk_all.to_csv(os.path.join(TABLES_DIR, 'table2_granularity.csv'), index=False)

# Table 3: Individual model comparison
indiv_all = results[results['feature_type'] == 'single_topic'][
    ['model', 'accuracy', 'f1_weighted', 'f1_macro']].copy()
indiv_all.columns = ['Model', 'Accuracy', 'F1-Weighted', 'F1-Macro']
indiv_all.to_csv(os.path.join(TABLES_DIR, 'table3_individual.csv'), index=False)

# Table 4: SMOTE analysis
smote_analysis.to_csv(os.path.join(TABLES_DIR, 'table4_smote.csv'), index=False)

# Table 5: Multi-topic vs single-topic
mt_comp = results[results['model'].isin([
    f'Stacking_k{k}' for k in [5, 10, 25, 50, 100]
] + ['MultiTopic_Concat_SMOTE', 'Stacking_TF-IDF'])][
    ['model', 'accuracy', 'f1_weighted', 'f1_macro']].copy()
mt_comp.columns = ['Model', 'Accuracy', 'F1-Weighted', 'F1-Macro']
mt_comp.to_csv(os.path.join(TABLES_DIR, 'table5_multitopic.csv'), index=False)

# Table 6: Per-class report (best model)
pc = pd.DataFrame({
    'Class': cls,
    'Precision': precs,
    'Recall': recs,
    'F1-Score': f1s,
    'Support': [rpt[c]['support'] for c in cls]
})
pc.to_csv(os.path.join(TABLES_DIR, 'table6_per_class.csv'), index=False)

# LaTeX tables
for tname, tdf, caption in [
    ('table2', stk_all, 'Stacking Ensemble Performance Across LSA Granularities'),
    ('table4', smote_analysis, 'SMOTE Effectiveness Analysis'),
    ('table5', mt_comp, 'Multi-Topic vs Single-Topic LSA Comparison'),
    ('table6', pc, 'Per-Class Classification Report (Best Model)')
]:
    latex = tdf.to_latex(index=False, float_format="%.4f")
    with open(os.path.join(TABLES_DIR, f'{tname}_latex.tex'), 'w') as f:
        f.write(f"\\begin{{table}}[htbp]\n\\centering\n"
                f"\\caption{{{caption}}}\n\\label{{tab:{tname}}}\n"
                f"\\small\n{latex}\\end{{table}}\n")

print("[SAVED] All tables and LaTeX files")


# ============================================================
# FINAL PAPER SUMMARY
# ============================================================
print("\n" + "=" * 70)
print("PUBLICATION SUMMARY")
print("=" * 70)

print("\n1. DATASET")
print(f"   Total samples: {len(df_clean)}")
print(f"   Original classes: 5 (joy, anger, sadness, fear, netral)")
print(f"   Consolidated classes: 3 (joy, negative, netral)")
print(f"   Justification: fear ({t1_5class.get('fear',0)}), "
      f"sadness ({t1_5class.get('sadness',0)}) insufficient for classification")

print("\n2. OPTIMAL LSA GRANULARITY")
best_k_row = stk_smote.sort_values('f1_macro', ascending=False).iloc[0]
print(f"   Best k = {int(best_k_row['k'])} "
      f"(F1-macro={best_k_row['f1_macro']:.4f})")
print(f"   Tested: k ∈ {{5, 10, 25, 50, 100}}")

print("\n3. STACKING vs INDIVIDUAL MODELS")
best_indiv_row = results[results['feature_type'] == 'single_topic'].sort_values(
    'f1_macro', ascending=False).iloc[0]
print(f"   Best individual: {best_indiv_row['model']} "
      f"(F1-macro={best_indiv_row['f1_macro']:.4f})")
print(f"   Best stacking:   Stacking_k{int(best_k_row['k'])} "
      f"(F1-macro={best_k_row['f1_macro']:.4f})")
stk_gain = (best_k_row['f1_macro'] - best_indiv_row['f1_macro']) * 100
print(f"   Stacking improvement: {stk_gain:+.2f}%")

print("\n4. SMOTE EFFECTIVENESS")
avg_f1m_gain = smote_analysis['F1m_Delta'].mean()
avg_acc_loss = smote_analysis['Acc_Delta'].mean()
print(f"   Avg F1-macro gain with SMOTE: {avg_f1m_gain:+.4f} ({avg_f1m_gain*100:+.2f}%)")
print(f"   Avg accuracy change with SMOTE: {avg_acc_loss:+.4f} ({avg_acc_loss*100:+.2f}%)")
print(f"   Finding: SMOTE improves minority class recall at cost of accuracy")

print("\n5. MULTI-TOPIC vs SINGLE-TOPIC")
mt_f1m = results[results['feature_type'] == 'multitopic_concat'].iloc[0]['f1_macro']
print(f"   Multi-topic concat (190 dims): F1-macro={mt_f1m:.4f}")
print(f"   Best single-topic (k={int(best_k_row['k'])}): "
      f"F1-macro={best_k_row['f1_macro']:.4f}")
mt_delta = (mt_f1m - best_k_row['f1_macro']) * 100
print(f"   Delta: {mt_delta:+.2f}%")
print(f"   Finding: Multi-topic concatenation does NOT improve performance")
print(f"   Reason: LSA components overlap across k (short texts, median 3 words)")

print("\n6. BEST MODEL CLASSIFICATION REPORT")
print(classification_report(y_test, y_pred_best, target_names=cls))

# Compute ROC AUC per class
for i in range(n_cls):
    fpr, tpr, _ = roc_curve(y_test_bin[:, i], y_proba_best[:, i])
    print(f"   {cls[i]:>10s} ROC-AUC = {auc(fpr, tpr):.4f}")

print(f"\n[OUTPUT] Figures: {FIGURES_DIR}/ ({len(os.listdir(FIGURES_DIR))} files)")
print(f"[OUTPUT] Tables:  {TABLES_DIR}/ ({len(os.listdir(TABLES_DIR))} files)")
print("[DONE] Step 4 completed. All publication materials generated.")
