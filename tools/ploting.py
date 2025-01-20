import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
import argparse
import os


def get_args():
    parser = argparse.ArgumentParser(description="Plot results")
    parser.add_argument(
        "--src",
        type=str,
        default="./results/results.csv",
        help="Path to the results file",
    )
    parser.add_argument(
        "--dst", type=str, default="./figures/", help="Path to the results file"
    )
    return parser.parse_args()


def main():
    args = get_args()

    sns.set_theme(style="darkgrid")

    df = pd.read_csv(args.src, sep=";")

    plt.figure(figsize=(8, 4))
    sns.lineplot(x="epoch", y="train_acc", data=df, label="train", marker="o")
    sns.lineplot(x="epoch", y="test_acc", data=df, label="test", marker="o")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.tight_layout()
    path = os.path.join(args.dst, "Acc.png")
    plt.savefig(path, dpi=300, bbox_inches="tight")

    plt.figure(figsize=(8, 4))
    sns.lineplot(x="epoch", y="train_loss", data=df, label="train", marker="o")
    sns.lineplot(x="epoch", y="test_loss", data=df, label="test", marker="o")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.tight_layout()
    path = os.path.join(args.dst, "Loss.png")
    plt.savefig(path, dpi=300, bbox_inches="tight")


if __name__ == "__main__":
    main()
