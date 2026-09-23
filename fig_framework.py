"""Figure 1: framework overview."""
from pathlib import Path
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

FIG = Path(__file__).resolve().parent
plt.rcParams.update({"font.family": "DejaVu Serif"})

boxes = [
    ("Data", "4 public clinical datasets\nBC · Pima · Cleveland (UCI)\nDiabetes-130 (130 hospitals)\nfold-internal imputation", "#eaeef2", "#34495e"),
    ("Models", "LR · RF · XGBoost\nSVM · MLP · Naïve Bayes\ndeterminism verified\nnested-CV tuning", "#eaeef2", "#34495e"),
    ("Performance", "5×10 repeated\nstratified CV\nAUC · MCC · Bal. acc · F1\nBrier · cal. slope/intercept", "#eaeef2", "#34495e"),
    ("Comparison", "corrected resampled t\n(Nadeau–Bengio) + Holm\nBayesian correlated &\nhierarchical tests", "#e8f5e9", "#1b5e20"),
    ("Explanation\nstability", "SHAP: seed vs bootstrap\nτ · weighted τ · RBO · J5\nfeature & grouped level\npermutation-source split", "#e8f5e9", "#1b5e20"),
]
fig, ax = plt.subplots(figsize=(13.5, 3.6)); ax.set_xlim(0, 13.5); ax.set_ylim(0, 3.6); ax.axis("off")
w, h, gap = 2.5, 2.55, 0.22
for i, (title, body, fc, ec) in enumerate(boxes):
    x = 0.1 + i * (w + gap)
    ax.add_patch(FancyBboxPatch((x, 0.75), w, h, boxstyle="round,pad=0.02,rounding_size=0.08",
                                fc=fc, ec=ec, lw=2))
    ax.text(x + w / 2, 0.75 + h - 0.45, title, ha="center", va="center", fontsize=12.5,
            fontweight="bold", color=ec)
    ax.text(x + w / 2, 0.75 + h / 2 - 0.35, body, ha="center", va="center", fontsize=8.8, color="#222", linespacing=1.35)
    if i < len(boxes) - 1:
        ax.annotate("", xy=(x + w + gap - 0.02, 0.75 + h / 2), xytext=(x + w + 0.02, 0.75 + h / 2),
                    arrowprops=dict(arrowstyle="-|>", color="#546e7a", lw=1.8))
ax.text(6.75, 0.3, "Questions answered: Is a performance difference real (after accounting for fold dependence)? "
        "Are the probabilities calibrated?\nWhich perturbation (algorithmic seed, training sample, "
        "attribution procedure) changes the explanation?", ha="center", va="center",
        fontsize=9.5, style="italic", color="#37474f")
fig.savefig(FIG / "fig1_framework.png", dpi=300, bbox_inches="tight")
