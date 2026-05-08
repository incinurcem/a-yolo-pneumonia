"""
Ablation sonuçlarını topla → markdown + LaTeX tablosu + bar chart.
Kaynak: evaluate.py'nin ürettiği detailed_eval_results.csv dosyaları.
"""

import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    precision_recall_fscore_support,
)

VARIANTS = [
    ("none",     "No-SSL (detection only)"),
    ("random",   "Random MAE + Detection"),
    ("gaussian", "Gaussian Anatomy-Aware (Ours)"),
]


def metrics_from_csv(csv_path: Path) -> dict:
    df = pd.read_csv(csv_path)
    y_true = df["target"].astype(int).values
    y_pred = df["pred"].astype(int).values
    y_prob = df["prob"].values
    ious   = df.loc[df["target"] == 1, "iou"].values

    prec, rec, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="binary", zero_division=0)
    acc = float((y_true == y_pred).mean())
    auc = float(roc_auc_score(y_true, y_prob)) if len(set(y_true)) > 1 else float("nan")

    map50_y = ((df["target"] == 1) & (df["iou"] >= 0.5)).astype(int).values
    map50   = float(average_precision_score(map50_y, y_prob))

    mean_iou = float(ious.mean()) if len(ious) else 0.0
    iou50    = float((ious >= 0.5).mean()) if len(ious) else 0.0

    return dict(acc=acc, prec=prec, rec=rec, f1=f1,
                auc=auc, map50=map50, miou=mean_iou, iou50=iou50)


def make_markdown(rows) -> str:
    cols = ["acc", "prec", "rec", "f1", "auc", "map50", "miou", "iou50"]
    head = "| Variant | Acc | Prec | Recall | F1 | AUC | mAP@50 | mIoU | IoU≥.5 |"
    sep  = "|" + "---|" * 9
    out  = [head, sep]
    for name, m in rows:
        out.append("| " + name + " | "
                   + " | ".join(f"{m[c]:.3f}" for c in cols) + " |")
    return "\n".join(out)


def make_latex(rows) -> str:
    cols = ["acc", "prec", "rec", "f1", "auc", "map50", "miou", "iou50"]
    out  = [
        "\\begin{tabular}{lcccccccc}",
        "\\toprule",
        "Variant & Acc & Prec & Recall & F1 & AUC & mAP@50 & mIoU & IoU$\\geq$.5 \\\\",
        "\\midrule",
    ]
    for name, m in rows:
        out.append(name + " & "
                   + " & ".join(f"{m[c]:.3f}" for c in cols) + " \\\\")
    out += ["\\bottomrule", "\\end{tabular}"]
    return "\n".join(out)


def make_chart(rows, save_path: Path):
    keys   = ["rec", "f1", "auc", "map50"]
    labels = ["Recall", "F1", "AUC", "mAP@50"]
    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.arange(len(rows))
    width = 0.20
    for i, (k, lab) in enumerate(zip(keys, labels)):
        ax.bar(x + i * width, [r[1][k] for r in rows], width, label=lab)
    ax.set_xticks(x + 1.5 * width)
    ax.set_xticklabels([r[0] for r in rows], rotation=15, ha="right")
    ax.set_ylabel("Score")
    ax.set_title("A-YOLO Ablation: Anatomy-Aware vs Random vs No-SSL")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    ax.set_ylim(0, 1)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--base", type=str,
        default="/content/drive/MyDrive/Spring Semester/deep learning project/outputs/a_yolo/ablation")
    args = p.parse_args()
    base = Path(args.base)

    rows = []
    for slug, name in VARIANTS:
        csv = base / slug / "eval" / "detailed_eval_results.csv"
        if not csv.exists():
            print(f"⚠️  Eksik: {csv}")
            continue
        m = metrics_from_csv(csv)
        rows.append((name, m))
        print(f"✅ {name}: " + "  ".join(f"{k}={v:.3f}" for k, v in m.items()))

    if not rows:
        print("❌ Hiçbir varyant bulunamadı.")
        return

    md_str    = make_markdown(rows)
    latex_str = make_latex(rows)

    print("\n=== MARKDOWN TABLE ===\n" + md_str)
    print("\n=== LATEX TABLE ===\n"    + latex_str)

    out_md  = base / "ablation_table.md"
    out_tex = base / "ablation_table.tex"
    out_md.write_text(md_str)
    out_tex.write_text(latex_str)
    print(f"\n💾 {out_md}")
    print(f"💾 {out_tex}")

    chart = base / "ablation_chart.png"
    make_chart(rows, chart)
    print(f"📊 {chart}")


if __name__ == "__main__":
    main()