"""
================================================================================
STEP 3: COMPLETE EXPERIMENT SUITE
================================================================================
Research: LSA-based Stacking Ensemble with Granularity Analysis
          for Emotion Classification on Indonesian Social Media

Experiments:
  3A. Individual classifiers (RF, SVM, XGB) × 5 LSA granularities + SMOTE
  3B. Stacking ensemble × 5 granularities + SMOTE
  3C. TF-IDF baseline (no LSA)
  3D. Stacking × 5 granularities WITHOUT SMOTE (SMOTE analysis)
  3E. Multi-topic feature concatenation + Stacking + SMOTE
  3F. Multi-topic feature concatenation WITHOUT SMOTE

References:
  [1] Saputri et al. (2018). Emotion Classification on Indonesian Twitter.
  [2] Deerwester et al. (1990). Indexing by Latent Semantic Analysis.
  [3] Mienye & Sun (2022). A Survey of Ensemble Learning. IEEE Access.
  [6] Wolpert (1992). Stacked Generalization. Neural Networks.
  [7] Chawla et al. (2002). SMOTE. JAIR.
  [10] Breiman (2001). Random Forests. Machine Learning.
  [11] Cortes & Vapnik (1995). Support-Vector Networks.
  [12] Chen & Guestrin (2016). XGBoost. KDD.
================================================================================
"""

import numpy as np
import pandas as pd
import os
import pickle
import time
import warnings
warnings.filterwarnings('ignore')

from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from xgboost import XGBClassifier
from sklearn.metrics import (accuracy_score, f1_score, precision_score,
                             recall_score, classification_report,
                             confusion_matrix)
from imblearn.over_sampling import SMOTE
from scipy import sparse

# ============================================================
# Configuration
# ============================================================
OUTPUT_DIR = "outputs"
MODELS_DIR = os.path.join(OUTPUT_DIR, "models")
RANDOM_STATE = 42
TOPIC_GRANULARITIES = [5, 10, 25, 50, 100]

print("=" * 70)
print("STEP 3: COMPLETE EXPERIMENT SUITE")
print("=" * 70)

# ============================================================
# Load data
# ============================================================
y_train = np.load(os.path.join(OUTPUT_DIR, 'y_train.npy'))
y_test = np.load(os.path.join(OUTPUT_DIR, 'y_test.npy'))

with open(os.path.join(MODELS_DIR, 'label_encoder.pkl'), 'rb') as f:
    le = pickle.load(f)

n_classes = len(le.classes_)
print(f"[INFO] Train: {len(y_train)}, Test: {len(y_test)}")
print(f"[INFO] Classes: {le.classes_.tolist()}")
print(f"[INFO] Train distribution: {dict(zip(le.classes_, np.bincount(y_train)))}")
print(f"[INFO] Test  distribution: {dict(zip(le.classes_, np.bincount(y_test)))}")


# ============================================================
# Shared model constructors
# ============================================================
def make_rf(rs=42):
    return RandomForestClassifier(
        n_estimators=200, max_depth=20, min_samples_split=5,
        class_weight='balanced', random_state=rs, n_jobs=-1)

def make_svm(rs=42):
    return make_pipeline(
        StandardScaler(),
        SVC(kernel='rbf', C=10, gamma='scale', probability=True,
            class_weight='balanced', random_state=rs))

def make_xgb(rs=42):
    return XGBClassifier(
        n_estimators=200, max_depth=6, learning_rate=0.1,
        eval_metric='mlogloss', random_state=rs, n_jobs=-1)

def make_svm_sparse(rs=42):
    """SVM for sparse TF-IDF (no mean centering)."""
    return make_pipeline(
        StandardScaler(with_mean=False),
        SVC(kernel='rbf', C=10, gamma='scale', probability=True,
            class_weight='balanced', random_state=rs))

def make_stacking(rs=42, cv=5):
    return StackingClassifier(
        estimators=[('rf', make_rf(rs)), ('svm', make_svm(rs)), ('xgb', make_xgb(rs))],
        final_estimator=LogisticRegression(max_iter=1000, C=1.0, solver='lbfgs',
                                           random_state=rs),
        cv=cv, stack_method='predict_proba', n_jobs=-1)

def make_stacking_sparse(rs=42, cv=5):
    """Stacking with sparse-safe SVM for TF-IDF input."""
    return StackingClassifier(
        estimators=[('rf', make_rf(rs)), ('svm', make_svm_sparse(rs)), ('xgb', make_xgb(rs))],
        final_estimator=LogisticRegression(max_iter=1000, C=1.0, solver='lbfgs',
                                           random_state=rs),
        cv=cv, stack_method='predict_proba', n_jobs=-1)

def metrics_dict(y_true, y_pred, name, train_time=0):
    return {
        'model': name,
        'accuracy': round(accuracy_score(y_true, y_pred), 4),
        'f1_weighted': round(f1_score(y_true, y_pred, average='weighted'), 4),
        'f1_macro': round(f1_score(y_true, y_pred, average='macro'), 4),
        'precision_weighted': round(precision_score(y_true, y_pred, average='weighted'), 4),
        'recall_weighted': round(recall_score(y_true, y_pred, average='weighted'), 4),
        'train_time_sec': round(train_time, 2)
    }


all_results = []

# ============================================================
# 3A. Individual Models × 5 Granularities + SMOTE
# ============================================================
print("\n" + "=" * 60)
print("3A. Individual Models per LSA Granularity (+ SMOTE)")
print("=" * 60)

for k in TOPIC_GRANULARITIES:
    Xtr = np.load(os.path.join(OUTPUT_DIR, f'X_train_lsa_{k}.npy'))
    Xte = np.load(os.path.join(OUTPUT_DIR, f'X_test_lsa_{k}.npy'))

    smote = SMOTE(random_state=RANDOM_STATE, k_neighbors=3)
    Xtr_s, ytr_s = smote.fit_resample(Xtr, y_train)

    for name, mdl_fn in [('RF', make_rf), ('SVM', make_svm), ('XGB', make_xgb)]:
        mdl = mdl_fn(RANDOM_STATE)
        t0 = time.time()
        mdl.fit(Xtr_s, ytr_s)
        yp = mdl.predict(Xte)
        m = metrics_dict(y_test, yp, f'{name}_k{k}', time.time()-t0)
        m['n_topics'] = k
        m['feature_type'] = 'single_topic'
        m['smote'] = 'yes'
        all_results.append(m)

    print(f"  k={k:>3d}: RF={all_results[-3]['f1_macro']:.4f}, "
          f"SVM={all_results[-2]['f1_macro']:.4f}, "
          f"XGB={all_results[-1]['f1_macro']:.4f} (F1-macro)")


# ============================================================
# 3B. Stacking Ensemble × 5 Granularities + SMOTE
# ============================================================
print("\n" + "=" * 60)
print("3B. Stacking Ensemble per Granularity (+ SMOTE)")
print("=" * 60)

best_stacking_f1m = 0
best_stacking_k = None
best_stacking_ypred = None
best_stacking_yproba = None

for k in TOPIC_GRANULARITIES:
    Xtr = np.load(os.path.join(OUTPUT_DIR, f'X_train_lsa_{k}.npy'))
    Xte = np.load(os.path.join(OUTPUT_DIR, f'X_test_lsa_{k}.npy'))

    smote = SMOTE(random_state=RANDOM_STATE, k_neighbors=3)
    Xtr_s, ytr_s = smote.fit_resample(Xtr, y_train)

    stk = make_stacking(RANDOM_STATE)
    t0 = time.time()
    stk.fit(Xtr_s, ytr_s)
    yp = stk.predict(Xte)
    yproba = stk.predict_proba(Xte)
    elapsed = time.time() - t0

    m = metrics_dict(y_test, yp, f'Stacking_k{k}', elapsed)
    m['n_topics'] = k
    m['feature_type'] = 'single_topic_stacking'
    m['smote'] = 'yes'
    all_results.append(m)

    # Track best
    if m['f1_macro'] > best_stacking_f1m:
        best_stacking_f1m = m['f1_macro']
        best_stacking_k = k
        best_stacking_ypred = yp
        best_stacking_yproba = yproba

    print(f"  k={k:>3d}: acc={m['accuracy']:.4f}, f1w={m['f1_weighted']:.4f}, "
          f"f1m={m['f1_macro']:.4f}  ({elapsed:.0f}s)")

print(f"\n  >>> Best single-topic stacking: k={best_stacking_k} "
      f"(F1-macro={best_stacking_f1m:.4f})")


# ============================================================
# 3C. TF-IDF Baseline (No LSA) + Stacking + SMOTE
# ============================================================
print("\n" + "=" * 60)
print("3C. TF-IDF Baseline (No LSA)")
print("=" * 60)

Xtr_tfidf = sparse.load_npz(os.path.join(OUTPUT_DIR, 'X_tfidf_train.npz'))
Xte_tfidf = sparse.load_npz(os.path.join(OUTPUT_DIR, 'X_tfidf_test.npz'))

smote_t = SMOTE(random_state=RANDOM_STATE, k_neighbors=3)
Xtr_tfidf_s, ytr_tfidf_s = smote_t.fit_resample(Xtr_tfidf, y_train)

stk_t = make_stacking_sparse(RANDOM_STATE)
t0 = time.time()
stk_t.fit(Xtr_tfidf_s, ytr_tfidf_s)
yp_t = stk_t.predict(Xte_tfidf)
m_t = metrics_dict(y_test, yp_t, 'Stacking_TF-IDF', time.time()-t0)
m_t['n_topics'] = 0
m_t['feature_type'] = 'tfidf_baseline'
m_t['smote'] = 'yes'
all_results.append(m_t)

print(f"  TF-IDF: acc={m_t['accuracy']:.4f}, f1w={m_t['f1_weighted']:.4f}, "
      f"f1m={m_t['f1_macro']:.4f}")


# ============================================================
# 3D. SMOTE Analysis: Stacking WITHOUT SMOTE × 5 Granularities
# ============================================================
print("\n" + "=" * 60)
print("3D. Stacking WITHOUT SMOTE (SMOTE Effectiveness Analysis)")
print("=" * 60)

for k in TOPIC_GRANULARITIES:
    Xtr = np.load(os.path.join(OUTPUT_DIR, f'X_train_lsa_{k}.npy'))
    Xte = np.load(os.path.join(OUTPUT_DIR, f'X_test_lsa_{k}.npy'))

    stk = make_stacking(RANDOM_STATE)
    t0 = time.time()
    stk.fit(Xtr, y_train)
    yp = stk.predict(Xte)
    elapsed = time.time() - t0

    m = metrics_dict(y_test, yp, f'Stacking_k{k}_NoSMOTE', elapsed)
    m['n_topics'] = k
    m['feature_type'] = 'single_topic_stacking_nosmote'
    m['smote'] = 'no'
    all_results.append(m)

    print(f"  k={k:>3d}: acc={m['accuracy']:.4f}, f1w={m['f1_weighted']:.4f}, "
          f"f1m={m['f1_macro']:.4f}")


# ============================================================
# 3E. Multi-Topic Concatenation + Stacking + SMOTE
# ============================================================
print("\n" + "=" * 60)
print("3E. Multi-Topic Concatenation + Stacking + SMOTE")
print("=" * 60)

X_train_mt = np.load(os.path.join(OUTPUT_DIR, 'X_train_mt.npy'))
X_test_mt = np.load(os.path.join(OUTPUT_DIR, 'X_test_mt.npy'))

smote_mt = SMOTE(random_state=RANDOM_STATE, k_neighbors=3)
Xtr_mt_s, ytr_mt_s = smote_mt.fit_resample(X_train_mt, y_train)

stk_mt = make_stacking(RANDOM_STATE)
t0 = time.time()
stk_mt.fit(Xtr_mt_s, ytr_mt_s)
yp_mt = stk_mt.predict(X_test_mt)
yproba_mt = stk_mt.predict_proba(X_test_mt)
elapsed_mt = time.time() - t0

m_mt = metrics_dict(y_test, yp_mt, 'MultiTopic_Concat_SMOTE', elapsed_mt)
m_mt['n_topics'] = 190
m_mt['feature_type'] = 'multitopic_concat'
m_mt['smote'] = 'yes'
all_results.append(m_mt)

print(f"  Multi-topic (190 dims): acc={m_mt['accuracy']:.4f}, "
      f"f1w={m_mt['f1_weighted']:.4f}, f1m={m_mt['f1_macro']:.4f}")

print(f"\n  Classification Report:")
print(classification_report(y_test, yp_mt, target_names=le.classes_))


# ============================================================
# 3F. Multi-Topic Concatenation WITHOUT SMOTE
# ============================================================
print("=" * 60)
print("3F. Multi-Topic Concatenation WITHOUT SMOTE")
print("=" * 60)

stk_mt_ns = make_stacking(RANDOM_STATE)
t0 = time.time()
stk_mt_ns.fit(X_train_mt, y_train)
yp_mt_ns = stk_mt_ns.predict(X_test_mt)
elapsed_mt_ns = time.time() - t0

m_mt_ns = metrics_dict(y_test, yp_mt_ns, 'MultiTopic_Concat_NoSMOTE', elapsed_mt_ns)
m_mt_ns['n_topics'] = 190
m_mt_ns['feature_type'] = 'multitopic_concat_nosmote'
m_mt_ns['smote'] = 'no'
all_results.append(m_mt_ns)

print(f"  Multi-topic (no SMOTE): acc={m_mt_ns['accuracy']:.4f}, "
      f"f1w={m_mt_ns['f1_weighted']:.4f}, f1m={m_mt_ns['f1_macro']:.4f}")


# ============================================================
# Save all results
# ============================================================
results_df = pd.DataFrame(all_results)
results_df.to_csv(os.path.join(OUTPUT_DIR, 'all_results.csv'), index=False)

# Save best model predictions for evaluation
# Best model = Stacking with best single-topic k
np.save(os.path.join(OUTPUT_DIR, 'y_pred_best.npy'), best_stacking_ypred)
np.save(os.path.join(OUTPUT_DIR, 'y_pred_proba_best.npy'), best_stacking_yproba)

# Also save multi-topic predictions for comparison figures
np.save(os.path.join(OUTPUT_DIR, 'y_pred_multitopic.npy'), yp_mt)
np.save(os.path.join(OUTPUT_DIR, 'y_pred_proba_multitopic.npy'), yproba_mt)


# ============================================================
# Summary Tables
# ============================================================
print("\n" + "=" * 70)
print("EXPERIMENT SUMMARY")
print("=" * 70)

# Granularity comparison (stacking + SMOTE)
print("\n--- LSA Granularity Comparison (Stacking + SMOTE) ---")
stk_results = results_df[results_df['feature_type'] == 'single_topic_stacking']
print(stk_results[['model','accuracy','f1_weighted','f1_macro']].to_string(index=False))

# SMOTE effect
print("\n--- SMOTE Effectiveness (Stacking, paired by k) ---")
smote_data = []
for k in TOPIC_GRANULARITIES:
    row_yes = results_df[(results_df['model'] == f'Stacking_k{k}')].iloc[0]
    row_no = results_df[(results_df['model'] == f'Stacking_k{k}_NoSMOTE')].iloc[0]
    smote_data.append({
        'k': k,
        'F1m_SMOTE': row_yes['f1_macro'],
        'F1m_NoSMOTE': row_no['f1_macro'],
        'F1m_Delta': round(row_yes['f1_macro'] - row_no['f1_macro'], 4),
        'Acc_SMOTE': row_yes['accuracy'],
        'Acc_NoSMOTE': row_no['accuracy'],
        'Acc_Delta': round(row_yes['accuracy'] - row_no['accuracy'], 4)
    })

smote_df = pd.DataFrame(smote_data)
print(smote_df.to_string(index=False))
smote_df.to_csv(os.path.join(OUTPUT_DIR, 'smote_analysis.csv'), index=False)

# Multi-topic vs single-topic
print("\n--- Multi-Topic vs Best Single-Topic ---")
print(f"  Best single-topic:     Stacking_k{best_stacking_k} "
      f"F1m={best_stacking_f1m:.4f}")
print(f"  Multi-topic concat:    {m_mt['model']} "
      f"F1m={m_mt['f1_macro']:.4f}")
delta = m_mt['f1_macro'] - best_stacking_f1m
print(f"  Delta: {delta:+.4f} ({delta*100:+.2f}%)")

# Top 5 overall
print("\n--- Top 5 Models by F1-Macro ---")
top5 = results_df.nlargest(5, 'f1_macro')
print(top5[['model','accuracy','f1_weighted','f1_macro','smote']].to_string(index=False))

print(f"\n[SAVED] all_results.csv")
print(f"[SAVED] smote_analysis.csv")
print(f"[SAVED] y_pred_best.npy, y_pred_proba_best.npy")
print("[DONE] Step 3 completed.")
