import matplotlib.pyplot as plt
import numpy as np

# -----------------------------
# Data
# -----------------------------
gpus = np.array(["1", "2", "4"])
load_base = np.array([1613.73, 3035.49, 5117.7])
cuda_base = np.array([1098.98, 2129.1, 3802.06])
load_tiny = np.array([1958.62, 3438.67, 6311.83])
cuda_tiny = np.array([2264.31, 4253.41, 7496.89])

# Common plot style settings
plt.rcParams.update(
    {
        "font.size": 8,
        "font.family": "sans-serif",
        "axes.linewidth": 0.8,
        "figure.dpi": 300,
    }
)


def plot_performance(
    ylog=False, filename="performance_vs_vertices.pdf", show_values=True
):
    """Generate either linear or log-scale performance plot."""
    fig, ax = plt.subplots(figsize=(3.3, 2.5))  # IEEE column width ~3.3 in

    # Plot lines (print-safe grayscale + shape differentiation)
    ax.plot(
        gpus,
        load_base,
        marker="o",
        linestyle="--",
        color="gray",
        linewidth=1.0,
        markersize=4,
        label="LOAD+Base",
    )
    ax.plot(
        gpus,
        cuda_base,
        marker="s",
        linestyle="-",
        color="gray",
        linewidth=1.0,
        markersize=4,
        label="BAT+Base",
    )
    ax.plot(
        gpus,
        load_tiny,
        marker="o",
        linestyle="--",
        color="black",
        linewidth=1.0,
        markersize=4,
        label="LOAD+Tiny",
    )
    ax.plot(
        gpus,
        cuda_tiny,
        marker="s",
        linestyle="-",
        color="black",
        linewidth=1.0,
        markersize=4,
        label="BAT+Tiny",
    )
    # Axis labels
    ax.set_xlabel("Number of GPUs", fontsize=9)
    ax.set_ylabel("Throughput (images/s)", fontsize=9)

    # Optional log scale
    if ylog:
        ax.set_yscale("log")
        filename = filename.replace(".pdf", "_log.pdf")

    if show_values:
        for x, y in zip(gpus, load_base):
            ax.text(
                x,
                y * 1.3,
                f"{y:.2f}",
                fontsize=6,
                ha="center",
                va="bottom",
                color="dimgray",
            )
        for x, y in zip(gpus, load_tiny):
            ax.text(
                x,
                y * 1.3,
                f"{y:.3f}",
                fontsize=6,
                ha="center",
                va="bottom",
                color="black",
            )
        for x, y in zip(gpus, cuda_base):
            ax.text(
                x,
                y * 1.3 if ylog else y * 1.15,
                f"{y:.3f}",
                fontsize=6,
                ha="center",
                va="bottom",
                color="gray",
            )
        for x, y in zip(gpus, cuda_tiny):
            ax.text(
                x,
                y * 1.3 if ylog else y * 1.15,
                f"{y:.3f}",
                fontsize=6,
                ha="center",
                va="bottom",
                color="gray",
            )

    # Legend and layout
    legend = ax.legend(
        ncol=2, frameon=False, fontsize=7, loc="upper left", title="Algorithm + Model"
    )
    ax.margins(x=0.07, y=0.1)
    ax.minorticks_off()
    ax.tick_params(axis="both", labelsize=8)
    ax.grid(False)
    plt.tight_layout(pad=0.1)

    # Save figures
    plt.savefig(filename, bbox_inches="tight", pad_inches=0.01)
    plt.savefig(
        filename.replace(".pdf", ".png"), dpi=600, bbox_inches="tight", pad_inches=0.01
    )
    plt.close(fig)


# -----------------------------
# Generate both plots
# -----------------------------
plot_performance(ylog=False, filename="training/training.pdf", show_values=False)
