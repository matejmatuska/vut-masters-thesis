import sys
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from ast import literal_eval
import numpy as np

DATE_FORMAT = "%Y-%m-%dT%H:%M:%S.%f"
VECTOR_ATTRIBUTES = [
    'PPI_PKT_LENGTHS',
    'PPI_PKT_TIMES',
    'PPI_PKT_DIRECTIONS',
    # 'PPI_PKT_FLAGS',
]


def _parse_ppi(df):
    """
    Parse the per-packet information (PPI) fields into Python lists.
    """
    df["PPI_PKT_LENGTHS"] = df["PPI_PKT_LENGTHS"].str.replace("|", ",")
    df["PPI_PKT_LENGTHS"] = df["PPI_PKT_LENGTHS"].apply(literal_eval)

    df["PPI_PKT_DIRECTIONS"] = df["PPI_PKT_DIRECTIONS"].str.replace("|", ",")
    df["PPI_PKT_DIRECTIONS"] = df["PPI_PKT_DIRECTIONS"].apply(literal_eval)

    df["PPI_PKT_FLAGS"] = df["PPI_PKT_FLAGS"].str.replace("|", ",")
    df["PPI_PKT_FLAGS"] = df["PPI_PKT_FLAGS"].apply(literal_eval)

    df["PPI_PKT_TIMES"] = df["PPI_PKT_TIMES"].str[1:-1].str.split("|")
    df["PPI_PKT_TIMES"] = df["PPI_PKT_TIMES"].apply(pd.to_datetime, format=DATE_FORMAT)
    return df

def load_csv(file_path):
    print("loading data...")
    df = pd.read_csv(sys.argv[1])
    print("parsing data...")
    df = _parse_ppi(df)

    print("padding data...")
    def pad_or_truncate(lst, to_len, value=0):
        lst = lst[:to_len]
        pad_width = max(0, to_len - len(lst))
        return np.pad(lst, (0, pad_width), "constant", constant_values=value)

    # Ensure fixed-length arrays (padding or truncating to 30)
    for col in VECTOR_ATTRIBUTES:
        df[col] = df[col].apply(pad_or_truncate, args=(30, 0))
    return df

if True:
    df = pd.read_parquet(sys.argv[1])
    print(df.columns)
    df["LABEL_ENCODED"], _ = pd.factorize(df["family"])
    new_feats = [
        "min_len",
        "max_len",
        "mean_len",
        "std_len",
        "q1_len",
        "q2_len",
        "q3_len",
        "range_len",
        # 'skew_len',
        # 'kurt_len',
        "num_fwd",
        "num_bwd",
        "fwd_bytes",
        "bwd_bytes",
        "fwd_bwd_ratio",
        "mean_fwd_len",
        "mean_bwd_len",
        "direction_switches",
        "mean_iat",
        "std_iat",
        "min_iat",
        "max_iat",
        "q1_iat",
        "q2_iat",
        "q3_iat",
        "packets_per_second",
        # "dir_entropy": dir_entropy,
    ]

    df = df[
        [
            "LABEL_ENCODED",
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
        + VECTOR_ATTRIBUTES
        + new_feats
    ]

else:
    df = load_csv(sys.argv[1])
    df["LABEL_ENCODED"], _ = pd.factorize(df["family"])
    df = df[
        [
            "LABEL_ENCODED",
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
        + VECTOR_ATTRIBUTES
    ]


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

print("Computing correlations...")
corr_with_label = df.corr()['LABEL_ENCODED'].drop('LABEL_ENCODED')

print("Plotting correlations...")
# Convert to 2D DataFrame
heatmap_data = pd.DataFrame(corr_with_label).T  # Single row heatmap

plt.figure(figsize=(20, 1.5))  # Wide but short
sns.heatmap(heatmap_data, cmap='coolwarm', annot=True, cbar=True)
plt.title('Feature Correlation with Label')
plt.yticks([])  # Optional: hide the single y-label
plt.tight_layout()
plt.show()

df_corr = df.corr()
print("Plotting full correlation matrix...")
plt.figure(figsize=(15, 15))
sns.heatmap(df_corr, cmap='coolwarm', annot=False, cbar=True)
plt.title('Full Feature Correlation Matrix')
plt.tight_layout()
plt.show()
