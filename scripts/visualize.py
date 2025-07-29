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

from matplotlib.colors import Normalize
import contextily as ctx
import geopandas as gpd
from shapely.geometry import Point

from config import VERSION, FIG_SIZE
from train import load_data

def plot_segmentation_summary(save=True, version=None, fig_size=None):

    version = version or VERSION
    fig_size = fig_size or FIG_SIZE

    save_dir = Path(f"figures/segmentation.v.{version}")
    save_dir.mkdir(parents=True, exist_ok=True)

    predictions_data = pd.read_csv(f"data/performance/coral_segmenter_predictions.v.{version}.csv")

    # Plot: Accuracy vs Depth
    plt.figure(figsize=fig_size)
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
    plt.figure(figsize=fig_size)
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
    plt.figure(figsize=fig_size)
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
    fig, axes = plt.subplots(1, 2, figsize=fig_size, sharey=True)
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
    plt.figure(figsize=fig_size)
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

def plot_coral_cover(eval_file=None, save=True, version=None, fig_size=None, col="coral_cover_pred", title="Precited Coral Cover"):

    fig_size = fig_size or FIG_SIZE
    version = version or VERSION

    save_dir = Path(f"figures/segmentation.v.{version}")
    save_dir.mkdir(parents=True, exist_ok=True)

    if eval_file is None:
        predictions_data = load_data(f"data/performance/coral_segmenter_predictions.v.{version}.csv")
    else:
        predictions_data = load_data(eval_file)

    gdf = gpd.GeoDataFrame(
        predictions_data,
        geometry=gpd.points_from_xy(predictions_data["lonPhoto"], predictions_data["latPhoto"]),
        crs="EPSG:4326"  # WGS84
    )

    gdf_web = gdf.to_crs(epsg=3857)

    fig, ax = plt.subplots(figsize=fig_size)

    norm = Normalize(vmin=0, vmax=1)
    gdf_web.plot(
        ax=ax,
        column=col,
        cmap="viridis",
        markersize=50,
        legend=True,
        alpha=0.8,
        norm=norm
    )
    ctx.add_basemap(ax, source=ctx.providers.Esri.WorldImagery)
    ax.set_axis_off()
    plt.title(title, fontsize=14)
    plt.tight_layout()

    if save:
        plt.savefig(save_dir / f"coral_cover_{title.strip().replace(' ', '')}.png")

    plt.show()

if __name__ == "__main__":
    plot_segmentation_summary()
    plot_coral_cover(col="coral_cover_pred", title="Predicted Coral Cover")