import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt

sns.set_theme(style="darkgrid")

df = pd.read_csv("../results/results.csv", sep=";")

plt.figure(figsize=(8, 4))
sns.lineplot(x='epoch', y='train_acc', data=df, label='train', marker='o')
sns.lineplot(x='epoch', y='test_acc', data=df, label='test', marker='o')
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.tight_layout()
plt.savefig("../figures/Acc.png", dpi=300, bbox_inches="tight")


plt.figure(figsize=(8, 4))
sns.lineplot(x='epoch', y='train_loss', data=df, label='train', marker='o')
sns.lineplot(x='epoch', y='test_loss', data=df, label='test', marker='o')
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.tight_layout()
plt.savefig("../figures/Loss.png", dpi=300, bbox_inches="tight")
