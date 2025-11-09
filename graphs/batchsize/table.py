import matplotlib.pyplot as plt
import numpy as np

# -----------------------------
# Data
# -----------------------------
batch = np.array([1, 2, 4, 8, 16, 32, 64, 128, 256, 512])
time = np.array(
    [
        0.1122,
        0.0775,
        0.0491,
        0.0334,
        0.0253,
        0.0215,
        0.0202,
        0.0194,
        0.0193,
        0.0179,
    ]
)

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
    xlog=False, filename="performance_vs_vertices.pdf", show_values=True
):
    """Generate either linear or log-scale performance plot."""
    fig, ax = plt.subplots(figsize=(4.0, 2.5))  # IEEE column width ~3.3 in

    # Plot lines (print-safe grayscale + shape differentiation)
    ax.plot(
        batch,
        time,
        marker="^",
        linestyle="-",
        color="dimgray",
        linewidth=1.0,
        markersize=4,
        label="Batched-GPU",
    )

    # Axis labels
    ax.set_xlabel("Batch size", fontsize=9)
    ax.set_ylabel("Time (ms)", fontsize=9)

    # Optional log scale
    if xlog:
        ax.set_xscale("log", base=2)
        ax.set_xticks(batch)
        ax.get_xaxis().set_major_formatter(
            plt.FuncFormatter(lambda val, _: f"{int(val)}")
        )
        filename = filename.replace(".pdf", "_log.pdf")

    if show_values:
        for x, y in zip(batch, time):
            if x < 2:
                scalex = x
            elif x < 10:
                scalex = x * 1.25
            else:
                scalex = x
            if y > 0.07:
                scaley = y * 1.02
            else:
                scaley = y * 1.07
            ax.text(
                scalex,
                scaley,
                f"{y:.3f}",
                fontsize=6,
                ha="center",
                va="bottom",
                color="dimgray",
            )

    # Legend and layout
    # ax.legend(frameon=False, fontsize=7, loc="center", bbox_to_anchor=(0.8, 0.7))
    ax.margins(x=0.07, y=0.1)
    # ax.minorticks_off()
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
plot_performance(xlog=True, filename="batchsize/batchsize.pdf", show_values=True)
