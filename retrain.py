"""
retrain.py
==========
One-shot retrain of the Meta-Ensemble Pipeline through the package, plus
reporting artefacts written to outputs/.

Outputs written:
  - master_results.csv          (single-row weighted summary)
  - classification_report.csv   (per-class precision/recall/f1/support)
  - classification_report.txt   (console-style formatted table)
  - confusion_matrix.png
  - per_class_f1.png

The trained .joblib models are saved by train_all() into saved_models/ —
this script does NOT duplicate them.

Run from the project root:  python retrain.py
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from config import RISK_CLASSES
from models.training import train_all

OUTPUT_DIR = Path("outputs")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

MODEL_NAME = "Meta-Ensemble Pipeline"


# ---------------------------------------------------------------------------
# Per-class metrics from a confusion matrix (rows=actual, cols=predicted)
# ---------------------------------------------------------------------------
def per_class_metrics_from_cm(cm: np.ndarray):
    """Return (precision, recall, f1, support) arrays per class."""
    cm = np.asarray(cm, dtype=float)
    tp = np.diag(cm)
    fp = cm.sum(axis=0) - tp
    fn = cm.sum(axis=1) - tp
    support = cm.sum(axis=1)

    precision = np.divide(tp, tp + fp, out=np.zeros_like(tp), where=(tp + fp) > 0)
    recall    = np.divide(tp, tp + fn, out=np.zeros_like(tp), where=(tp + fn) > 0)
    denom = precision + recall
    f1 = np.divide(2 * precision * recall, denom,
                   out=np.zeros_like(tp), where=denom > 0)
    return precision, recall, f1, support


# ---------------------------------------------------------------------------
# Reporting helpers
# ---------------------------------------------------------------------------
def write_master_results(comparison: pd.DataFrame, resampling: str) -> None:
    row = comparison[comparison["model"] == MODEL_NAME].iloc[0]
    out = pd.DataFrame([{
        "balancing": resampling,
        "model":     "MetaEnsemblePipeline",
        "accuracy":  row["accuracy"],
        "precision": row["precision"],
        "recall":    row["recall"],
        "f1":        row["f1"],
        "roc_auc":   row["roc_auc"],
    }])
    path = OUTPUT_DIR / "master_results.csv"
    out.to_csv(path, index=False)
    print(f"[Report] Master results saved -> {path}")


def write_classification_report(cm: np.ndarray, class_names: list) -> None:
    """Write classification_report.csv and .txt, derived from the CM."""
    precision, recall, f1, support = per_class_metrics_from_cm(cm)
    total = support.sum()
    accuracy = np.diag(np.asarray(cm, dtype=float)).sum() / total if total else 0.0

    # macro = unweighted mean; weighted = support-weighted mean
    macro = (precision.mean(), recall.mean(), f1.mean())
    w = support / total if total else np.zeros_like(support)
    weighted = ((precision * w).sum(), (recall * w).sum(), (f1 * w).sum())

    # ---- CSV ----
    rows = []
    for i, name in enumerate(class_names):
        rows.append({
            "class": name,
            "precision": round(float(precision[i]), 4),
            "recall":    round(float(recall[i]), 4),
            "f1_score":  round(float(f1[i]), 4),
            "support":   int(support[i]),
        })
    rows.append({"class": "accuracy", "precision": "", "recall": "",
                 "f1_score": round(float(accuracy), 4), "support": int(total)})
    rows.append({"class": "macro avg", "precision": round(macro[0], 4),
                 "recall": round(macro[1], 4), "f1_score": round(macro[2], 4),
                 "support": int(total)})
    rows.append({"class": "weighted avg", "precision": round(weighted[0], 4),
                 "recall": round(weighted[1], 4), "f1_score": round(weighted[2], 4),
                 "support": int(total)})
    csv_path = OUTPUT_DIR / "classification_report.csv"
    pd.DataFrame(rows).to_csv(csv_path, index=False)
    print(f"[Report] Classification report (CSV) saved -> {csv_path}")

    # ---- TXT (console-style) ----
    name_w = max(len(n) for n in class_names + ["weighted avg"]) + 2
    lines = []
    header = f"{'':<{name_w}}{'precision':>10}{'recall':>10}{'f1-score':>10}{'support':>10}"
    lines.append(header)
    lines.append("")
    for i, name in enumerate(class_names):
        lines.append(f"{name:<{name_w}}{precision[i]:>10.2f}{recall[i]:>10.2f}"
                     f"{f1[i]:>10.2f}{int(support[i]):>10}")
    lines.append("")
    lines.append(f"{'accuracy':<{name_w}}{'':>10}{'':>10}{accuracy:>10.2f}{int(total):>10}")
    lines.append(f"{'macro avg':<{name_w}}{macro[0]:>10.2f}{macro[1]:>10.2f}"
                 f"{macro[2]:>10.2f}{int(total):>10}")
    lines.append(f"{'weighted avg':<{name_w}}{weighted[0]:>10.2f}{weighted[1]:>10.2f}"
                 f"{weighted[2]:>10.2f}{int(total):>10}")
    txt_path = OUTPUT_DIR / "classification_report.txt"
    txt_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"[Report] Classification report (TXT) saved -> {txt_path}")


def plot_confusion_matrix(cm: np.ndarray, class_names: list) -> None:
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=class_names, yticklabels=class_names, cbar=False)
    plt.title("Confusion Matrix — Meta-Ensemble Pipeline (SMOTE+Tomek)")
    plt.ylabel("Actual"); plt.xlabel("Predicted")
    plt.tight_layout()
    path = OUTPUT_DIR / "confusion_matrix.png"
    plt.savefig(path); plt.close()
    print(f"[Report] Confusion matrix saved -> {path}")


def plot_class_f1(f1s: np.ndarray, class_names: list) -> None:
    fig, ax = plt.subplots(figsize=(8, 4))
    colors = ["#185FA5" if v >= 0.7 else "#E24B4A" for v in f1s]
    ax.bar(class_names, f1s, color=colors)
    ax.axhline(0.7, color="gray", linestyle="--", linewidth=1, label="0.70 threshold")
    ax.set_title("Per-class F1 — Meta-Ensemble Pipeline")
    ax.set_ylabel("F1 score"); ax.set_ylim(0, 1.05)
    ax.legend()
    plt.tight_layout()
    path = OUTPUT_DIR / "per_class_f1.png"
    fig.savefig(path); plt.close(fig)
    print(f"[Report] Per-class F1 saved -> {path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    resampling = "SMOTE+Tomek"
    print("Retraining Meta-Ensemble Pipeline (SMOTE+Tomek) ...\n")
    result = train_all(
        resampling=resampling,
        progress_cb=lambda p, msg: print(f"[{p:3d}%] {msg}"),
    )

    comparison = result["comparison"]
    cms = result["confusion_matrices"]

    print("\n" + "=" * 60)
    print("Done. Model comparison:")
    print("=" * 60)
    print(comparison.to_string(index=False))

    # ---- Reporting artefacts into outputs/ ----
    print()
    cm = np.array(cms[MODEL_NAME])
    write_master_results(comparison, resampling)
    write_classification_report(cm, RISK_CLASSES)
    plot_confusion_matrix(cm, RISK_CLASSES)
    _, _, f1s, _ = per_class_metrics_from_cm(cm)
    plot_class_f1(f1s, RISK_CLASSES)

    print(f"\nModels were saved to saved_models/ by train_all().")
    print(f"Reports written to outputs/.")


if __name__ == "__main__":
    main()