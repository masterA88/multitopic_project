"""
================================================================================
STEP 2: TF-IDF & Multi-Topic LSA Feature Engineering
================================================================================
Added: Class consolidation (anger+fear+sadness → negative) per standard
practice in Indonesian emotion classification (Saputri et al., 2018).

References:
  [1] Saputri et al. (2018). Emotion Classification on Indonesian Twitter.
  [2] Deerwester et al. (1990). Indexing by Latent Semantic Analysis.
  [8] Bradford (2008). Required Dimensionality for LSA Applications.
  [9] Dumais (2004). Latent semantic analysis. ARIST, 38(1).
================================================================================
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import pickle
import time
import warnings
warnings.filterwarnings('ignore')

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

OUTPUT_DIR = "outputs"
FIGURES_DIR = os.path.join(OUTPUT_DIR, "figures")
MODELS_DIR = os.path.join(OUTPUT_DIR, "models")
RANDOM_STATE = 42
TOPIC_GRANULARITIES = [5, 10, 25, 50, 100]

os.makedirs(MODELS_DIR, exist_ok=True)

print("=" * 70)
print("STEP 2: TF-IDF & MULTI-TOPIC LSA FEATURE ENGINEERING")
print("=" * 70)

# ============================================================
# 1. Load & Consolidate Classes
# ============================================================
df = pd.read_csv(os.path.join(OUTPUT_DIR, "data_cleaned.csv"))
print(f"[INFO] Loaded: {df.shape[0]} samples")

# --- CLASS CONSOLIDATION ---
# Merge anger, fear, sadness → negative (standard in Indonesian NLP literature)
# Justification: fear has only 40 samples, sadness 84 — insufficient for
# independent classification. Consolidation preserves emotional valence
# while ensuring statistical validity.
print("\n--- Class Consolidation ---")
print(f"Original distribution:\n{df['label_emosi'].value_counts().to_string()}")

label_map_3class = {
    'joy': 'joy',
    'netral': 'netral',
    'anger': 'negative',
    'fear': 'negative',
    'sadness': 'negative'
}
df['label_3class'] = df['label_emosi'].map(label_map_3class)
print(f"\n3-class distribution:\n{df['label_3class'].value_counts().to_string()}")

# Visualize both
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
colors5 = ['#2ecc71', '#e74c3c', '#3498db', '#f39c12', '#9b59b6']
colors3 = ['#2ecc71', '#3498db', '#e74c3c']

df['label_emosi'].value_counts().plot(kind='bar', ax=axes[0], color=colors5)
axes[0].set_title('Original 5-Class Distribution', fontweight='bold')
axes[0].set_ylabel('Count')

df['label_3class'].value_counts().plot(kind='bar', ax=axes[1], color=colors3)
axes[1].set_title('Consolidated 3-Class Distribution', fontweight='bold')
axes[1].set_ylabel('Count')

plt.tight_layout()
plt.savefig(os.path.join(FIGURES_DIR, 'fig4a_class_consolidation.png'), dpi=150, bbox_inches='tight')
plt.close()
print("[SAVED] fig4a_class_consolidation.png")

# ============================================================
# 2. Label Encoding (3-class)
# ============================================================
X_text = df['final_text'].astype(str)
y_labels = df['label_3class']

le = LabelEncoder()
y_encoded = le.fit_transform(y_labels)
label_mapping = dict(zip(le.classes_, le.transform(le.classes_)))
print(f"\n[INFO] Label mapping: {label_mapping}")

with open(os.path.join(MODELS_DIR, 'label_encoder.pkl'), 'wb') as f:
    pickle.dump(le, f)

# ============================================================
# 3. TF-IDF Vectorization
# ============================================================
print("\n--- TF-IDF Vectorization ---")
tfidf = TfidfVectorizer(
    max_features=5000, min_df=3, max_df=0.90,
    ngram_range=(1, 2), sublinear_tf=True
)
X_tfidf = tfidf.fit_transform(X_text)
print(f"[INFO] TF-IDF shape: {X_tfidf.shape}")

with open(os.path.join(MODELS_DIR, 'tfidf_vectorizer.pkl'), 'wb') as f:
    pickle.dump(tfidf, f)

# ============================================================
# 4. Single-Topic LSA
# ============================================================
print("\n--- Single-Topic LSA ---")
for n_topics in TOPIC_GRANULARITIES:
    t0 = time.time()
    lsa = TruncatedSVD(n_components=n_topics, n_iter=100, random_state=RANDOM_STATE)
    X_lsa = lsa.fit_transform(X_tfidf)
    ev = lsa.explained_variance_ratio_.sum()
    print(f"  k={n_topics:>3d}: explained_var={ev:.4f}, shape={X_lsa.shape}, time={time.time()-t0:.2f}s")

    with open(os.path.join(MODELS_DIR, f'lsa_{n_topics}.pkl'), 'wb') as f:
        pickle.dump(lsa, f)
    np.save(os.path.join(OUTPUT_DIR, f'X_lsa_{n_topics}.npy'), X_lsa)

# ============================================================
# 5. Multi-Topic LSA Concatenation (NOVELTY)
# ============================================================
print("\n--- Multi-Topic LSA Feature Construction (NOVELTY) ---")
lsa_list = [np.load(os.path.join(OUTPUT_DIR, f'X_lsa_{k}.npy')) for k in TOPIC_GRANULARITIES]
X_multitopic = np.hstack(lsa_list)
total_features = sum(TOPIC_GRANULARITIES)
print(f"[INFO] Multi-topic shape: {X_multitopic.shape} ({total_features} features)")
np.save(os.path.join(OUTPUT_DIR, 'X_multitopic.npy'), X_multitopic)

# ============================================================
# 6. Topic Terms Display (k=10)
# ============================================================
print("\n--- Top Terms per Topic (k=10) ---")
with open(os.path.join(MODELS_DIR, 'lsa_10.pkl'), 'rb') as f:
    lsa_10 = pickle.load(f)
terms = tfidf.get_feature_names_out()
for i, comp in enumerate(lsa_10.components_):
    top = [terms[j] for j in comp.argsort()[:-8:-1]]
    print(f"  Topic {i+1}: {', '.join(top)}")

# ============================================================
# 7. Explained Variance Plot
# ============================================================
ev_data = []
for k in TOPIC_GRANULARITIES:
    with open(os.path.join(MODELS_DIR, f'lsa_{k}.pkl'), 'rb') as f:
        m = pickle.load(f)
    ev_data.append((k, m.explained_variance_ratio_.sum()))

fig, ax = plt.subplots(figsize=(10, 5))
ks, evs = zip(*ev_data)
ax.plot(ks, evs, 'bo-', linewidth=2, markersize=8)
for x, y in zip(ks, evs):
    ax.annotate(f'{y:.3f}', (x, y), textcoords="offset points",
                xytext=(0, 12), ha='center', fontsize=10, fontweight='bold')
ax.set_title('Explained Variance vs LSA Topics', fontsize=14, fontweight='bold')
ax.set_xlabel('Number of Topics (k)')
ax.set_ylabel('Cumulative Explained Variance')
ax.set_xticks(list(ks))
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(FIGURES_DIR, 'fig4_lsa_explained_variance.png'), dpi=150, bbox_inches='tight')
plt.close()
print("\n[SAVED] fig4_lsa_explained_variance.png")

# ============================================================
# 8. Train-Test Split (stratified)
# ============================================================
print("\n--- Train-Test Split (80/20, stratified) ---")

X_train_mt, X_test_mt, y_train, y_test = train_test_split(
    X_multitopic, y_encoded, test_size=0.2,
    random_state=RANDOM_STATE, stratify=y_encoded
)
print(f"[INFO] Multi-topic train: {X_train_mt.shape}, test: {X_test_mt.shape}")

np.save(os.path.join(OUTPUT_DIR, 'X_train_mt.npy'), X_train_mt)
np.save(os.path.join(OUTPUT_DIR, 'X_test_mt.npy'), X_test_mt)
np.save(os.path.join(OUTPUT_DIR, 'y_train.npy'), y_train)
np.save(os.path.join(OUTPUT_DIR, 'y_test.npy'), y_test)

for k in TOPIC_GRANULARITIES:
    X_k = np.load(os.path.join(OUTPUT_DIR, f'X_lsa_{k}.npy'))
    Xtr, Xte, _, _ = train_test_split(
        X_k, y_encoded, test_size=0.2,
        random_state=RANDOM_STATE, stratify=y_encoded
    )
    np.save(os.path.join(OUTPUT_DIR, f'X_train_lsa_{k}.npy'), Xtr)
    np.save(os.path.join(OUTPUT_DIR, f'X_test_lsa_{k}.npy'), Xte)

# TF-IDF split
from scipy import sparse
train_idx, test_idx = train_test_split(
    np.arange(X_tfidf.shape[0]), y_encoded, test_size=0.2,
    random_state=RANDOM_STATE, stratify=y_encoded
)[0], None
train_idx = train_test_split(
    np.arange(X_tfidf.shape[0]), y_encoded, test_size=0.2,
    random_state=RANDOM_STATE, stratify=y_encoded
)[0]
test_idx = np.setdiff1d(np.arange(X_tfidf.shape[0]), train_idx)
sparse.save_npz(os.path.join(OUTPUT_DIR, 'X_tfidf_train.npz'), X_tfidf[train_idx])
sparse.save_npz(os.path.join(OUTPUT_DIR, 'X_tfidf_test.npz'), X_tfidf[test_idx])

print(f"\n[INFO] Train distribution:")
for u, c in zip(*np.unique(y_train, return_counts=True)):
    print(f"  {le.inverse_transform([u])[0]}: {c} ({c/len(y_train)*100:.1f}%)")
print(f"[INFO] Test distribution:")
for u, c in zip(*np.unique(y_test, return_counts=True)):
    print(f"  {le.inverse_transform([u])[0]}: {c} ({c/len(y_test)*100:.1f}%)")

np.save(os.path.join(OUTPUT_DIR, 'y_encoded_all.npy'), y_encoded)

metadata = {
    'total_samples': df.shape[0], 'n_train': len(y_train), 'n_test': len(y_test),
    'n_classes': len(le.classes_), 'classes': le.classes_.tolist(),
    'topic_granularities': TOPIC_GRANULARITIES,
    'total_multitopic_features': total_features, 'random_state': RANDOM_STATE,
    'class_consolidation': label_map_3class
}
with open(os.path.join(OUTPUT_DIR, 'metadata.pkl'), 'wb') as f:
    pickle.dump(metadata, f)

print("\n[DONE] Step 2 completed.")
