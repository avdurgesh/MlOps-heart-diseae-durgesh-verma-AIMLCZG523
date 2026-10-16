"""Exploratory data analysis: saves figures to reports/figures."""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import seaborn as sns  # noqa: E402

from src import config  # noqa: E402
from src.data import load_clean  # noqa: E402


def _save(fig, name: str) -> None:
    config.FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(config.FIGURES_DIR / name, dpi=150)
    plt.close(fig)


def run() -> None:
    df = load_clean()
    sns.set_theme(style="whitegrid")
    print(df.describe().T.to_string())
    print("\nMissing values:\n", df.isna().sum().to_string())

    fig, ax = plt.subplots(figsize=(5, 4))
    sns.countplot(data=df, x=config.TARGET, ax=ax)
    ax.set_title("Class balance (0 = no disease, 1 = disease)")
    _save(fig, "class_balance.png")

    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    for ax, col in zip(axes.ravel(), config.NUMERIC_FEATURES):
        sns.histplot(data=df, x=col, hue=config.TARGET, kde=True, ax=ax)
        ax.set_title(f"Distribution of {col}")
    axes.ravel()[-1].axis("off")
    _save(fig, "numeric_histograms.png")

    fig, ax = plt.subplots(figsize=(11, 9))
    sns.heatmap(df.corr(), annot=True, fmt=".2f", cmap="coolwarm", ax=ax)
    ax.set_title("Correlation heatmap")
    _save(fig, "correlation_heatmap.png")

    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    for ax, col in zip(axes.ravel(), config.CATEGORICAL_FEATURES):
        sns.countplot(data=df, x=col, hue=config.TARGET, ax=ax)
        ax.set_title(col)
    _save(fig, "categorical_vs_target.png")

    fig, axes = plt.subplots(1, 5, figsize=(18, 4))
    for ax, col in zip(axes, config.NUMERIC_FEATURES):
        sns.boxplot(data=df, x=config.TARGET, y=col, ax=ax)
    _save(fig, "numeric_boxplots.png")
    print(f"\nFigures saved to {config.FIGURES_DIR}")


if __name__ == "__main__":
    run()
