"""
Plotting utilities for coral segmentation model evaluation.
Generates and saves visualizations such as accuracy vs depth,
histograms of prediction accuracy, coral cover distributions,
and coral cover bias.
"""

import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import pandas as pd

from config import VERSION

def plot_segmentation_summary(save=True, version=None):

    version = version or VERSION

    save_dir = Path(f"figures/segmentation.v.{version}")
    save_dir.mkdir(parents=True, exist_ok=True)

    predictions_data = pd.read_csv(f"data/performance/coral_segmenter_predictions.v.{version}.csv")

    # Plot: Accuracy vs Depth
    plt.figure()
    sns.scatterplot(data=predictions_data, x='Depth_WaterSurface', y='accuracy')
    plt.xlabel("Depth (m)")
    plt.ylabel("Accuracy")
    plt.title("Accuracy vs Depth")
    plt.grid(True)
    plt.tight_layout()
    if save:
        plt.savefig(save_dir / "accuracy_vs_depth.png")
    plt.show()

    # Plot: Histogram of Accuracies with Median Line
    plt.figure()
    sns.histplot(predictions_data['accuracy'], bins=30, kde=False, color='skyblue')
    plt.axvline(predictions_data['accuracy'].median(), color='black', linestyle='--', label='Median')
    plt.xlabel("Accuracy")
    plt.ylabel("Count")
    plt.title("Histogram of Accuracies")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    if save:
        plt.savefig(save_dir / "accuracy_histogram.png")
    plt.show()

    # Plot: Density plots of True vs Predicted Coral Cover
    plt.figure()
    sns.kdeplot(predictions_data['coral_cover'], label='True Coral Cover', fill=True, clip=(0, 1))
    sns.kdeplot(predictions_data['coral_cover_pred'], label='Predicted Coral Cover', fill=True, clip=(0, 1))
    plt.xlabel("Coral Cover")
    plt.ylabel("Density")
    plt.title("True vs Predicted Coral Cover")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    if save:
        plt.savefig(save_dir / "coral_cover_density.png")
    plt.show()

    # Side-by-side histograms of coral cover
    fig, axes = plt.subplots(1, 2, figsize=(16, 9), sharey=True)
    sns.histplot(predictions_data['coral_cover'], bins=30, ax=axes[0], color='skyblue')
    axes[0].set_title("True Coral Cover")
    axes[0].set_xlabel("Coral Cover")
    axes[0].set_ylabel("Count")
    axes[0].set_xlim(0, 1)
    axes[0].grid(True)

    sns.histplot(predictions_data['coral_cover_pred'], bins=30, ax=axes[1], color='salmon')
    axes[1].set_title("Predicted Coral Cover")
    axes[1].set_xlabel("Coral Cover")
    axes[1].set_xlim(0, 1)
    axes[1].grid(True)

    plt.suptitle("Side-by-Side Histograms of Coral Cover (True vs Predicted)")
    plt.tight_layout()
    if save:
        plt.savefig(save_dir / "coral_cover_histograms.png")
    plt.show()

    # Plot: Histogram of Coral Cover Bias (Predicted - True)
    bias = predictions_data['coral_cover_pred'] - predictions_data['coral_cover']
    plt.figure()
    sns.histplot(bias, bins=30, kde=False, color='salmon')
    plt.axvline(bias.median(), color='black', linestyle='--', label='Median Bias')
    plt.xlabel("Bias (Predicted - True)")
    plt.ylabel("Count")
    plt.title("Histogram of Coral Cover Bias")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    if save:
        plt.savefig(save_dir / "coral_cover_bias.png")
    plt.show()

if __name__ == "__main__":
    plot_segmentation_summary()