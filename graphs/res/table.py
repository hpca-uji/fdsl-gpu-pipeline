import matplotlib.pyplot as plt
import numpy as np

# -----------------------------
# Data
# -----------------------------
res = np.array([256, 512, 1024])
baseline = np.array([1.144, 1.1637, 1.1715])
batched = np.array([0.010, 0.019, 0.026])
python_cpu = np.array([209.754, 210.603, 211.2118])

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
        res,
        python_cpu,
        marker="^",
        linestyle="-",
        color="dimgray",
        linewidth=1.0,
        markersize=4,
        label="Python-CPU",
    )
    ax.plot(
        res,
        baseline,
        marker="o",
        linestyle="-",
        color="black",
        linewidth=1.0,
        markersize=4,
        label="CUDA-Baseline",
    )
    ax.plot(
        res,
        batched,
        marker="s",
        linestyle="--",
        color="gray",
        linewidth=1.0,
        markersize=4,
        label="CUDA-Batched",
    )

    # Axis labels
    ax.set_xlabel("Image Resolution", fontsize=9)
    ax.set_ylabel("Time (ms)", fontsize=9)

    # Optional log scale
    if ylog:
        ax.set_yscale("log")
        filename = filename.replace(".pdf", "_log.pdf")

    if show_values:
        for x, y in zip(res, python_cpu):
            ax.text(
                x,
                y * 1.3,
                f"{y:.2f}",
                fontsize=6,
                ha="center",
                va="bottom",
                color="dimgray",
            )
        for x, y in zip(res, baseline):
            ax.text(
                x,
                y * 1.3,
                f"{y:.3f}",
                fontsize=6,
                ha="center",
                va="bottom",
                color="black",
            )
        for x, y in zip(res, batched):
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
    ax.legend(frameon=False, fontsize=7, loc="center", bbox_to_anchor=(0.8, 0.7))
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
plot_performance(ylog=True, filename="res/res.pdf", show_values=True)
