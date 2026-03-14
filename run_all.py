"""
================================================================================
MASTER RUNNER
================================================================================
Research: LSA-based Stacking Ensemble with Granularity Analysis
          for Emotion Classification on Indonesian Social Media
================================================================================
"""
import subprocess, sys, time, os

def run_step(script, label):
    print(f"\n{'#' * 70}")
    print(f"# {label}")
    print(f"# Script: {script}")
    print(f"{'#' * 70}\n")
    t0 = time.time()
    r = subprocess.run([sys.executable, script], text=True)
    elapsed = time.time() - t0
    if r.returncode != 0:
        print(f"\n[ERROR] {label} FAILED")
        sys.exit(1)
    print(f"\n[OK] {label} — {elapsed:.1f}s")
    return elapsed

if __name__ == "__main__":
    t0 = time.time()
    print("=" * 70)
    print("LSA-BASED STACKING ENSEMBLE FOR EMOTION CLASSIFICATION")
    print("=" * 70)

    steps = [
        ("step1_eda_preprocessing.py",    "Step 1: EDA & Preprocessing"),
        ("step2_feature_engineering.py",   "Step 2: TF-IDF + LSA Features (3-class)"),
        ("step3_experiments.py",           "Step 3: All Experiments"),
        ("step4_evaluation.py",            "Step 4: Evaluation & Figures"),
    ]

    timings = []
    for s, l in steps:
        timings.append((l, run_step(s, l)))

    total = time.time() - t0
    print(f"\n{'=' * 70}")
    print("ALL STEPS COMPLETED")
    print(f"{'=' * 70}")
    for l, t in timings:
        print(f"  {l}: {t:.1f}s")
    print(f"  TOTAL: {total:.1f}s ({total/60:.1f} min)")
    print(f"\n  Figures: outputs/figures/")
    print(f"  Tables:  outputs/tables/")
    print(f"  Results: outputs/all_results.csv")
