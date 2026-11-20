"""Draw the held-out outcome of every seed and method from the released CSV."""

import argparse
import csv
from pathlib import Path
from statistics import mean

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


TASKS = (
    ("TR02-CTRL-002", "MountainCar"),
    ("TR02-CTRL-003", "Pendulum"),
    ("TR02-CTRL-004", "CartPole"),
)
METHODS = (("no_edit", "No edit"), ("random", "Random"), ("greedy", "Greedy"), ("es", "Evolution"))
CONDITIONS = (
    ("original", "Original", "#17649b", "o", -0.17),
    ("permutation_relu_scale", "Transformed", "#a04a24", "D", 0.17),
)


def plot(input_csv, output):
    with input_csv.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "svg.hashsalt": "weightspace-baseline-2026"})
    figure, axes = plt.subplots(1, 3, sharey=True, figsize=(11.25, 4.05))
    for axes_for_task, (task, label) in zip(axes, TASKS, strict=True):
        axes_for_task.set_title(label, fontweight="bold", loc="left", pad=11)
        axes_for_task.set_ylim(0, 1)
        axes_for_task.set_xlim(-0.48, 3.48)
        axes_for_task.set_xticks(range(len(METHODS)), [label for _, label in METHODS])
        axes_for_task.set_yticks([0, 0.25, 0.50, 0.75, 1])
        axes_for_task.grid(axis="y", color="#e1e5ea", linewidth=0.8)
        axes_for_task.set_axisbelow(True)
        axes_for_task.spines[["top", "right"]].set_visible(False)
        axes_for_task.spines[["left", "bottom"]].set_color("#9ba5b0")
        axes_for_task.tick_params(length=0, pad=7, labelcolor="#26303d")
        for index, (method, _) in enumerate(METHODS):
            for condition, _, color, marker, offset in CONDITIONS:
                panel = sorted(
                    (int(row["seed"]), float(row["heldout_objective"]))
                    for row in rows if row["task"] == task and row["method"] == method
                    and row["condition"] == condition and row["status"] == "completed"
                )
                if len(panel) != 5 or len({seed for seed, _ in panel}) != 5:
                    raise ValueError(f"expected five completed seeds for {task}, {method}, {condition}")
                values = [value for _, value in panel]
                center = index + offset
                x = [center + jitter for jitter in (-0.104, -0.052, 0, 0.052, 0.104)]
                key = f"{task}-{method}-{condition}"
                axes_for_task.plot([center - 0.13, center + 0.13], [mean(values)] * 2, gid=f"{key}-mean",
                                   color=color, linewidth=2.4, solid_capstyle="round", zorder=2)
                axes_for_task.scatter(x, values, s=24, c=color, marker=marker, gid=f"{key}-seeds",
                                      edgecolor="white", linewidth=0.55, zorder=3)
    axes[0].set_ylabel("Held-out objective", labelpad=10, color="#26303d")
    figure.suptitle("Policy repair across five paired seeds", x=0.08, ha="left", fontsize=17,
                    fontweight="bold", color="#192431")
    handles = [Line2D([0], [0], color=color, marker=marker, markersize=5, label=label)
               for _, label, color, marker, _ in CONDITIONS]
    figure.legend(handles=handles, loc="lower center", ncols=2, frameon=False,
                  bbox_to_anchor=(0.5, 0.041), columnspacing=1.8)
    figure.subplots_adjust(left=0.08, right=0.98, bottom=0.22, top=0.76, wspace=0.20)
    figure.savefig(output, facecolor="white", dpi=180,
                   metadata={"Date": None} if output.suffix == ".svg" else None)
    plt.close(figure)
    if output.suffix == ".svg":
        output.write_text("\n".join(line.rstrip() for line in output.read_text().splitlines()) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input_csv", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    plot(args.input_csv, args.output)


if __name__ == "__main__":
    main()
