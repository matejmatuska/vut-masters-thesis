import sys

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

VECTOR_ATTRIBUTES = [
    "PPI_PKT_LENGTHS",
    "PPI_PKT_TIMES",
    "PPI_PKT_DIRECTIONS",
]


df = pd.read_parquet(sys.argv[1])
print(df.columns)
df["LABEL_ENCODED"], _ = pd.factorize(df["family"])
new_feats = [
    "mean_len",
    "std_len",
    "q1_len",
    "q2_len",
    "q3_len",
    "fwd_bytes",
    "bwd_bytes",
    "fwd_bwd_ratio",
    "direction_switches",
    "mean_iat",
    "std_iat",
    "q1_iat",
    "q2_iat",
    "q3_iat",
    "packets_per_second",
    # "dir_entropy": dir_entropy,
]
scalar_feats = [
    "BYTES",
    "BYTES_REV",
    "PACKETS",
    "PACKETS_REV",
    "TCP_FLAGS",
    "TCP_FLAGS_REV",
    "DST_PORT",
    "SRC_PORT",
    "PROTOCOL",
]


df = df[scalar_feats + VECTOR_ATTRIBUTES + new_feats]


def explode_to_columns(df, col):
    """
    Explode a column of lists into separate columns.
    """
    exploded = df[col].apply(pd.Series)
    exploded.columns = [f"{col}_{i}" for i in range(exploded.shape[1])]
    df = pd.concat([df.drop(columns=[col]), exploded], axis=1)
    return df


print(f"Exploding columns: {VECTOR_ATTRIBUTES}")
# make these aggregable
for attr in VECTOR_ATTRIBUTES:
    df = explode_to_columns(df, attr)

# df = df.drop(columns=[f"PPI_PKT_TIMES_{i}" for i in range(15, 25)])
# df = df.drop(columns=[f"PPI_PKT_LENGTHS_{i}" for i in range(15, 25)])
# df = df.drop(columns=[f"PPI_PKT_DIRECTIONS_{i}" for i in range(15, 25)])


df_corr = df.corr()


def plot_2x2_diag_blocks(corr_df, blocks, titles, figsize=(14, 14)):
    """
    Plot 4 diagonal blocks in a 2x2 layout.

    corr_df: Full correlation matrix (DataFrame)
    blocks: List of ((row_start, row_end), (col_start, col_end)) for each block
    titles: Titles for each subplot
    """
    fig, axes = plt.subplots(2, 2, figsize=figsize, constrained_layout=True)
    axes = axes.flatten()

    for ax, ((r0, r1), (c0, c1)), title in zip(axes, blocks, titles):
        block = corr_df.iloc[r0:r1, c0:c1]
        sns.heatmap(
            block,
            ax=ax,
            cmap="coolwarm",
            center=0,
            xticklabels=block.columns,
            yticklabels=block.index,
            cbar=False,
            square=True,
        )
        ax.set_title(title, fontsize=12)
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45, fontsize=8)
        ax.set_yticklabels(ax.get_yticklabels(), fontsize=8, rotation=0)

    # Shared colorbar
    cbar_ax = fig.add_axes([0.92, 0.3, 0.015, 0.4])  # [left, bottom, width, height]
    sns.heatmap(
        corr_df.iloc[0:1, 0:1] * 0,
        cbar_ax=cbar_ax,
        cbar=True,
        cmap="coolwarm",
        center=0,
        xticklabels=False,
        yticklabels=False,
    )
    cbar_ax.set_ylabel("Correlation", rotation=270, labelpad=15)

    plt.suptitle("Diagonal Correlation Blocks", fontsize=16)
    plt.show()


# Example: define 4 diagonal blocks (for a 60x60 matrix)
blocks = [
    ((0, 20), (0, 20)),
    ((15, 35), (15, 35)),
    ((30, 50), (30, 50)),
    ((45, 60), (45, 60)),
]

titles = [
    "Diagonal Block 1 (Features 1-20)",
    "Diagonal Block 2 (Features 16-35)",
    "Diagonal Block 3 (Features 31-50)",
    "Diagonal Block 4 (Features 46-60)",
]


def plot_full_corr_matrix(
    corr_df,
    figsize,
    path="full_corr_matrix.pdf",
    font_size=5,
    bar_font_size=6,
):
    plt.figure(figsize=figsize)
    ax = sns.heatmap(
        corr_df,
        cmap="coolwarm",
        center=0,
        square=True,
        xticklabels=True,
        yticklabels=True,
        cbar=True,
        vmin=-1,
        vmax=1,
        cbar_kws={"shrink": 0.5},
    )
    ax.tick_params(axis="both", which="both", length=0)

    cbar = ax.collections[0].colorbar
    cbar.ax.tick_params(labelsize=bar_font_size)
    plt.xticks(rotation=90, fontsize=font_size)
    plt.yticks(fontsize=font_size)
    plt.title("Full Correlation Matrix", fontsize=12)
    plt.tight_layout()
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.show()


# Usage (assuming corr is your full correlation DataFrame):
plot_full_corr_matrix(df_corr, figsize=(10, 10), font_size=4, bar_font_size=6)

df_corr = df[scalar_feats + new_feats].corr()
plot_full_corr_matrix(
    df_corr,
    figsize=(6, 6),
    path="scalar_corr_matrix.pdf",
    font_size=10,
    bar_font_size=11,
)

# correlations = df.corr()['family'].drop('family').sort_values(key=abs, ascending=False)
#
# # Plot as 1D heatmap
# plt.figure(figsize=(12, 1.5))
# sns.heatmap([correlations.values],
#             cmap='coolwarm',
#             center=0,
#             cbar=True,
#             xticklabels=correlations.index,
#             yticklabels=['Correlation'],
#             linewidths=0.5, linecolor='gray')
#
# plt.xticks(rotation=90)
# plt.title('Feature Correlation with Label (1D Heatmap)', pad=20)
# plt.tight_layout()
# plt.show()
