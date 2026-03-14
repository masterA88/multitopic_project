# LSA-based Stacking Ensemble with Granularity Analysis for Emotion Classification on Indonesian Social Media

## Research Overview

This project investigates the effect of **LSA topic granularity** on stacking ensemble performance for emotion classification on Indonesian social media comments (Instagram). The study provides a systematic analysis of how the number of LSA topics (k=5, 10, 25, 50, 100) affects classification quality, and evaluates whether multi-topic feature concatenation improves over single-topic representations.

### Contributions

1. **Class Consolidation Strategy** — Demonstrates that 5-class emotion classification is infeasible when minority classes (fear: 40, sadness: 84 samples) yield 0.00 F1-score, and proposes a principled 3-class consolidation (joy, negative, netral) validated by the literature.

2. **LSA Granularity Analysis** — Systematic evaluation of k∈{5,10,25,50,100} with stacking ensemble, identifying the optimal topic dimensionality (k=25) for short Indonesian social media text.

3. **Stacking Ensemble Effectiveness** — Shows stacking (RF+SVM+XGB → Logistic Regression meta-learner) outperforms all individual classifiers across all granularities.

4. **SMOTE Trade-off Analysis** — Quantifies the accuracy vs F1-macro trade-off: SMOTE improves minority class recall (+5-10% F1-macro) but reduces accuracy (−10-15%), providing practical guidance for imbalanced Indonesian NLP tasks.

5. **Multi-Granularity Finding** — Demonstrates that concatenating LSA features across multiple k does NOT outperform optimal single-topic (k=25) on short social media text (median 3 words), due to feature redundancy in LSA components — a useful negative result for the community.

6. **Regional Dialect Dataset** — Evaluated on a Javanese-Indonesian dialect dataset from Instagram, extending emotion classification to underrepresented language variants.

---

## Dataset

- **Source**: Instagram comments from @JTV_REK (Jawa Timur Television)
- **Size**: 3,700 labeled comments
- **Original Labels**: 5 classes (joy: 637, anger: 235, sadness: 84, fear: 40, netral: 2,704)
- **Consolidated Labels**: 3 classes (joy: 637, negative: 359, netral: 2,704)
- **Language**: Indonesian (Bahasa Indonesia) with Javanese regional dialect
- **Preprocessing**: Text cleaning, normalization, tokenization, stopword removal, stemming

---

## Project Structure

```
multitopic_project/
├── preprocessing_instagram_xlsx_-_JASMINE.csv   # Dataset
├── run_all.py                    # Master runner
├── step1_eda_preprocessing.py    # EDA & data cleaning
├── step2_feature_engineering.py  # TF-IDF + LSA (3-class consolidation)
├── step3_experiments.py          # ALL experiments
├── step4_evaluation.py           # ALL figures & tables
├── requirements.txt
├── README.md
└── outputs/
    ├── figures/      (13 publication figures)
    ├── tables/       (7 tables + LaTeX)
    └── all_results.csv
```

---

## How to Run

### Prerequisites
- Python 3.8+ (tested: 3.10)
- 4GB+ RAM, ~200MB disk

### Setup
```bash
python -m venv venv
# Windows: venv\Scripts\activate
# macOS/Linux: source venv/bin/activate
pip install -r requirements.txt
```

### Run
```bash
# Copy dataset into project directory
cp /path/to/preprocessing_instagram_xlsx_-_JASMINE.csv .

# Run everything (~15-20 minutes)
python run_all.py

# Or step by step:
python step1_eda_preprocessing.py
python step2_feature_engineering.py
python step3_experiments.py
python step4_evaluation.py
```

---

## Methodology

```
Instagram Comments (3,700 samples)
         │
         ▼
    Preprocessing (pre-done)
         │
         ▼
    Class Consolidation
    (anger+fear+sadness → negative)
    → 3 classes: joy, negative, netral
         │
         ▼
    TF-IDF (max_features=5000, ngram=(1,2), sublinear_tf)
         │
         ▼
    LSA at k ∈ {5, 10, 25, 50, 100}
         │
         ├── SMOTE (k_neighbors=3) for class balance
         │
         ▼
    For EACH k:
    ┌────────────────────────────┐
    │  Stacking Ensemble (5-CV)  │
    │  ┌──────┬──────┬──────┐   │
    │  │  RF  │ SVM  │ XGB  │   │  Base Learners
    │  └──┬───┴──┬───┴──┬───┘   │  (class_weight='balanced')
    │     └──────┼──────┘       │
    │            ▼              │
    │   Logistic Regression     │  Meta-Learner
    └────────────────────────────┘
         │
         ▼
    Emotion Classification → {joy, negative, netral}
```

---

## Experiments

| ID | Experiment | Description |
|----|-----------|-------------|
| 3A | Individual × k | RF, SVM, XGB per granularity + SMOTE |
| 3B | Stacking × k | Stacking ensemble per granularity + SMOTE |
| 3C | TF-IDF baseline | Stacking on raw TF-IDF (no LSA) |
| 3D | No-SMOTE analysis | Stacking per granularity without SMOTE |
| 3E | Multi-topic + SMOTE | Concatenated 190-dim features + Stacking |
| 3F | Multi-topic no SMOTE | Concatenated features without SMOTE |

---

## Key Findings

1. **Optimal granularity k=25** achieves best F1-macro among single-topic configurations
2. **Stacking consistently outperforms** individual models across all k
3. **SMOTE improves F1-macro** at the cost of accuracy — important trade-off for practitioners
4. **Multi-topic concatenation does NOT improve** over optimal single-topic due to LSA component redundancy on short texts
5. **TF-IDF without LSA performs worst**, confirming LSA's value for dimensionality reduction and semantic smoothing

---

## References

[1] Saputri, M. S., Mahendra, R., & Adriani, M. (2018). Emotion Classification on Indonesian Twitter Dataset. *IALP 2018*.

[2] Deerwester, S., Dumais, S. T., Furnas, G. W., Landauer, T. K., & Harshman, R. (1990). Indexing by Latent Semantic Analysis. *JASIS*, 41(6), 391-407.

[3] Mienye, I. D., & Sun, Y. (2022). A Survey of Ensemble Learning. *IEEE Access*, 10, 99129-99149.

[4] Junianto, E. et al. (2024). Klasifikasi Emosi pada Teks Menggunakan Pendekatan Ensemble Bagging. *JNTETI*, 13(4), 272-281.

[5] Glenn, A., LaCasse, P., & Cox, B. (2023). Emotion Classification of Indonesian Tweets using Bidirectional LSTM. *Neural Computing and Applications*, 35, 345-360.

[6] Wolpert, D. H. (1992). Stacked Generalization. *Neural Networks*, 5(2), 241-259.

[7] Chawla, N. V. et al. (2002). SMOTE. *JAIR*, 16, 321-357.

[8] Bradford, R. B. (2008). Required Dimensionality for Large-Scale LSA. *CIKM 2008*.

[9] Dumais, S. T. (2004). Latent Semantic Analysis. *ARIST*, 38(1), 188-230.

[10] Breiman, L. (2001). Random Forests. *Machine Learning*, 45(1), 5-32.

[11] Cortes, C. & Vapnik, V. (1995). Support-Vector Networks. *Machine Learning*, 20(3), 273-297.

[12] Chen, T. & Guestrin, C. (2016). XGBoost. *KDD 2016*.

[13] Sokolova, M. & Lapalme, G. (2009). Performance Measures for Classification. *IPM*, 45(4), 427-437.

[14] Powers, D. M. (2011). Evaluation: Precision, Recall and F-Measure to ROC. *JMLT*, 2(1), 37-63.

[15] Sentiment Analysis Using Stacking Ensemble After the 2024 Indonesian Election. *JINITA*, 7(1), 2025.

[16] Hybrid Models for Emotion Classification in Indonesian Language. *ACIS&C*, 2024.

[17] Xu, C., Tao, D., & Xu, C. (2013). A Survey on Multi-view Learning. *arXiv:1304.5634*.

---


