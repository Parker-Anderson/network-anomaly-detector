import matplotlib.pyplot as plt


def main():

    metrics = [
        "Precision",
        "Recall",
        "F1 Score"
    ]

    values = [
        0.7468,
        0.7699,
        0.7581
    ]

    # Create the chart.
    plt.figure(figsize=(8, 6))

    bars = plt.bar(
        metrics,
        values
    )

    plt.ylim(0, 1)

    plt.title(
        "Isolation Forest Performance Metrics"
    )

    plt.ylabel("Score")

    # Add percentage labels above each bar.
    for bar, value in zip(bars, values):

        plt.text(
            bar.get_x() + bar.get_width() / 2,
            value + 0.02,
            f"{value:.1%}",
            ha="center"
        )

    plt.tight_layout()

    output_file = "results/performance_metrics.png"

    plt.savefig(
        output_file,
        dpi=300,
        bbox_inches="tight"
    )

    print(
        f"Performance metrics chart saved to:\n"
        f"{output_file}"
    )

    plt.show()


if __name__ == "__main__":
    main()