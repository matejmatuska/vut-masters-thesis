"""
This script prepares the dataset for training and evaluation.
"""
import os
import sys
from ast import literal_eval

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

DATE_FORMAT = "%Y-%m-%dT%H:%M:%S.%f"

PPI_PAD_LEN = 30
VAL_SIZE = 0.2
TEST_SIZE = 0.1

TOO_FEW_SAMPLES = 300
TOO_MANY_SAMPLES = 2000


def print_flows_and_samples(df):
    print(f"Samples {len(df['sample'].unique())}, Flows {len(df)}")


def sample_count(df):
    return len(df["sample"].unique())


def _parse_ppi(df):
    df["PPI_PKT_LENGTHS"] = df["PPI_PKT_LENGTHS"].apply(literal_eval)
    df["PPI_PKT_DIRECTIONS"] = df["PPI_PKT_DIRECTIONS"].apply(literal_eval)

    df["PPI_PKT_TIMES"] = df["PPI_PKT_TIMES"].apply(literal_eval)
    df["PPI_PKT_TIMES"] = df["PPI_PKT_TIMES"].apply(
        lambda lst: [pd.to_datetime(x, format="ISO8601").value // 10**6 for x in lst]
    )
    print("notna", df["PPI_PKT_TIMES"].notna().sum())
    print("isna", df["PPI_PKT_TIMES"].isna().sum())

    return df


def cap_samples_per_fam(df, min_samples, max_samples) -> pd.DataFrame:
    """
    Keep the number of samples per family between samples[0] and samples[1].

    :param df: The DataFrame containing the dataset.
    :type df: pandas.DataFrame
    :param min_samples: Minimum number of samples per family.
    :type min_samples: int
    :param max_samples: Maximum number of samples per family.
    :type max_samples: int
    :return: The capped dataset.
    """
    # remove families with small saples < samples[0]
    sample_counts = df[["family", "sample"]].drop_duplicates().groupby("family").size()
    enough_samples = sample_counts[sample_counts >= min_samples].index
    df_enough = df[df["family"].isin(enough_samples)]

    # cap the number of samples per family to samples[1]
    has_more = (
        df_enough[["family", "sample"]]
        .drop_duplicates()
        .groupby("family", group_keys=False)
        .apply(lambda x: x.sample(n=min(len(x), max_samples)))
    )
    return df_enough.merge(has_more, on=["family", "sample"])


def normalize_all(df, per_packet_len=25) -> pd.DataFrame:
    """
    Normalize the dataset.
    - log1p + std transform on PACKETS, PACKETS_REV, BYTES, BYTES_REV
    - categorical encoding on PROTOCOL

    - log1p + std transform on PPI_PKT_LENGTHS
    - convert to relative -> log1p + std transform on PPI_PKT_TIMES
    - pad/truncate PPI_PKT_TIMES, PPI_PKT_LENGTHS, PPI_PKT_DIRECTIONS to 30

    :param df: The DataFrame containing the dataset.
    :type df: pandas.DataFrame
    :return: The normalized dataset.
    :rtype: pandas.DataFrame
    """
    df = df.copy()

    def times_to_relative(timestamps):
        base = timestamps[0]
        relative_times = [(ts - base).total_seconds() for ts in timestamps]
        # Exclude the first timestamp since it's always 0 - can use 0 for padding
        return relative_times[1:]

    df["PPI_PKT_TIMES"] = df["PPI_PKT_TIMES"].apply(times_to_relative)

    def pad_or_truncate(lst, to_len, value=0):
        lst = lst[:to_len]
        pad_width = max(0, to_len - len(lst))
        return np.pad(lst, (0, pad_width), "constant", constant_values=value)

    # Ensure fixed-length arrays (padding or truncating to 30)
    df["PPI_PKT_TIMES"] = df["PPI_PKT_TIMES"].apply(pad_or_truncate, args=(25, 0))
    df["PPI_PKT_DIRECTIONS"] = df["PPI_PKT_DIRECTIONS"].apply(
        pad_or_truncate, args=(25, 0)
    )
    df["PPI_PKT_LENGTHS"] = df["PPI_PKT_LENGTHS"].apply(pad_or_truncate, args=(25, 0))
    return df


def extract_flow_features(row):
    lens = np.array(row["PPI_PKT_LENGTHS"])
    times = np.array(row["PPI_PKT_TIMES"])
    dirs = np.array(row["PPI_PKT_DIRECTIONS"])

    # Basic stats
    len_mean = lens.mean()
    len_std = lens.std()
    len_q1 = np.percentile(lens, 25)
    len_q2 = np.percentile(lens, 50)
    len_q3 = np.percentile(lens, 75)

    fwd_mask = dirs == 1
    bwd_mask = dirs == -1
    fwd_num = fwd_mask.sum()
    num_bwd = bwd_mask.sum()
    bytes_fwd = lens[fwd_mask].sum() if fwd_num > 0 else 0
    bytes_bwd = lens[bwd_mask].sum() if num_bwd > 0 else 0
    ratio_fwd_bwd = bytes_fwd / bytes_bwd if bytes_bwd > 0 else 0
    direction_switches = np.sum(np.diff(dirs) != 0)

    if len(times) >= 2:
        duration = times[-1] - times[0]
        pps = len(times) / duration if duration > 0 else 0

    else:
        duration = pps = 0

    return pd.Series(
        {
            "mean_len": len_mean,
            "std_len": len_std,
            "q1_len": len_q1,
            "q2_len": len_q2,
            "q3_len": len_q3,
            "fwd_bytes": bytes_fwd,
            "bwd_bytes": bytes_bwd,
            "fwd_bwd_ratio": ratio_fwd_bwd,
            "direction_switches": direction_switches,
            "packets_per_second": pps,
        }
    )


def new_features(df):
    new_feats_cols = None
    new_feats_df = df.apply(extract_flow_features, axis=1)

    if new_feats_cols is None:
        new_feats_cols = new_feats_df.columns.tolist()

    df = pd.concat([df, new_feats_df], axis=1)
    return df, new_feats_cols


def normalize_splits(
    df_train, df_val, df_test, new_feats_cols
) -> (pd.DataFrame, pd.DataFrame, pd.DataFrame):
    """
    Normalize the train, validation, and test splits.

    Applied normalization:
    - log1p + std transform on PACKETS, PACKETS_REV, BYTES, BYTES_REV
    - binning + one hot encoding on PROTOCOL
    - log1p + std transform on PPI_PKT_LENGTHS packet wise
    - log1p + std transform on PPI_PKT_TIMES packet wise
    Other columns are not normalized.

    Scalers are fit on the train set and applied to the val and test sets.
    :return: A normalized copy training, validation, and test DataFrames.
    :rtype: tuple
    """
    df_train, df_val, df_test = df_train.copy(), df_val.copy(), df_test.copy()

    scalars = [
        "PACKETS",
        "PACKETS_REV",
        "BYTES",
        "BYTES_REV",
        "DURATION",
    ] + new_feats_cols

    for col in scalars:
        df_train[col] = np.log1p(df_train[col])
        df_val[col] = np.log1p(df_val[col])
        df_test[col] = np.log1p(df_test[col])

    scaler = StandardScaler()
    df_train[scalars] = scaler.fit_transform(df_train[scalars])
    df_val[scalars] = scaler.transform(df_val[scalars])
    df_test[scalars] = scaler.transform(df_test[scalars])

    def one_hot_encode_protocol(df):
        def map_protocol(proto):
            if proto == 6:
                return "TCP"
            elif proto == 17:
                return "UDP"
            elif proto == 1:
                return "ICMP"
            else:
                return "OTHER"

        # Map protocols to categories
        df["PROTOCOL_CAT"] = df["PROTOCOL"].apply(map_protocol)
        protocol_onehot = pd.get_dummies(df["PROTOCOL_CAT"], prefix="PROTO")
        df = pd.concat([df, protocol_onehot], axis=1)
        df.drop(columns=["PROTOCOL_CAT"], inplace=True)
        return df

    df_train = one_hot_encode_protocol(df_train)
    df_val = one_hot_encode_protocol(df_val)
    df_test = one_hot_encode_protocol(df_test)

    scaler = StandardScaler()
    for col in ["PPI_PKT_LENGTHS", "PPI_PKT_TIMES"]:
        df_train[col] = list(
            scaler.fit_transform(np.vstack(df_train[col].values))
        )  # shape: (num_rows, 30)
        df_val[col] = list(scaler.transform(np.vstack(df_val[col].values)))
        df_test[col] = list(scaler.transform(np.vstack(df_test[col].values)))

    df_train["DURATION"] = df_train["DURATION"].fillna(0)
    df_val["DURATION"] = df_val["DURATION"].fillna(0)
    df_test["DURATION"] = df_test["DURATION"].fillna(0)

    return df_train, df_val, df_test


def train_val_test_split(
    df, val_size, test_size, random_state=83
) -> (pd.DataFrame, pd.DataFrame, pd.DataFrame):
    """
    Splits the dataset into train, validation, and test sets.

    The split is stratified by family and sample size.
    """
    sample_sizes = df.groupby("sample").size().rename("sample_size")
    sample_families = df[["sample", "family"]].drop_duplicates().set_index("sample")

    sample_df = sample_families.join(sample_sizes)

    sample_df["size_bin"] = pd.qcut(sample_df["sample_size"], q=5, duplicates="drop")

    # stratify by key and size bin
    sample_df["stratify_key"] = (
        sample_df["family"].astype(str) + "__" + sample_df["size_bin"].astype(str)
    )

    train_samples, temp_samples = train_test_split(
        sample_df,
        test_size=val_size + test_size,
        stratify=sample_df["stratify_key"],
        random_state=random_state,
    )
    val_samples, test_samples = train_test_split(
        temp_samples,
        test_size=test_size / (val_size + test_size),
        stratify=temp_samples["stratify_key"],
        random_state=random_state,
    )

    train_df = df[df["sample"].isin(train_samples.index)]
    val_df = df[df["sample"].isin(val_samples.index)]
    test_df = df[df["sample"].isin(test_samples.index)]

    return train_df, val_df, test_df


def prepare_features(df, new_feats_cols) -> (pd.DataFrame, pd.DataFrame, pd.DataFrame):
    print("Splitting dataset...")
    train, val, test = train_val_test_split(df, VAL_SIZE, TEST_SIZE)
    print(
        f"Split: Train: {sample_count(train)}, Val: {sample_count(val)}, Test: {sample_count(test)}"
    )

    print("Normalizing and scaling dataset...")
    train, val, test = normalize_splits(train, val, test, new_feats_cols)
    print("Normalization and scaling done:")
    print(train.head())
    return train, val, test


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python prep_dataset.py <dset_path> <output_dir>")
        print("       dset_path: Path to the stitched dataset .parquet file")
        print("       output_dir: Path to the output directory")
        sys.exit(1)

    path = sys.argv[1]
    if not os.path.exists(path):
        print(f"Dataset path {path} does not exist.")
        sys.exit(1)

    output_dir = sys.argv[2]
    if not os.path.exists(path):
        print(f"Output directory {output_dir} does not exist.")
        sys.exit(1)

    print(f"Loading dataset from {path}...")
    df = pd.read_parquet(path)

    if len(df) == 0:
        print("Empty dataset, nothing to do. Make sure there is enough samples.")
        sys.exit(1)

    # df = _parse_ppi(df)

    print("Capping samples per family...")
    # WARN: there is no check if there are enough samples
    df = cap_samples_per_fam(df, TOO_FEW_SAMPLES, TOO_MANY_SAMPLES)
    print(df[["family", "sample"]].drop_duplicates()["family"].value_counts())
    print(df[["family", "sample"]].drop_duplicates()["family"].value_counts().sum())
    print(f"Samples after capping: {sample_count(df)}")
    print_flows_and_samples(df)

    print("Adding new features...")
    df, new_feats_cols = new_features(df)
    new_feats_path = os.path.join(output_dir, "dataset-all-feats.parquet")
    df.to_parquet(new_feats_path, index=False)

    print("Normalizing dataset")
    df = normalize_all(df, per_packet_len=PPI_PAD_LEN)
    print("Normalizing done:")
    print(df.head())
    new_feats_path = os.path.join(output_dir, "dataset-all-feats-normed.parquet")
    df.to_parquet(new_feats_path, index=False)

    train, val, test = prepare_features(df, new_feats_cols)

    print(
        f"Storing train dataset to {output_dir}/train.parquet and {output_dir}/train.csv..."
    )
    train.to_parquet(os.path.join(output_dir, "train.parquet"), index=False)
    train.to_csv(os.path.join(output_dir, "train.csv"), index=False)
    print(
        f"Storing val dataset to {output_dir}/val.parquet and {output_dir}/val.csv..."
    )
    val.to_parquet(os.path.join(output_dir, "val.parquet"), index=False)
    val.to_csv(os.path.join(output_dir, "val.csv"), index=False)
    print(
        f"Storing test dataset to {output_dir}/test.parquet and {output_dir}/test.csv..."
    )
    test.to_parquet(os.path.join(output_dir, "test.parquet"), index=False)
    test.to_csv(os.path.join(output_dir, "test.csv"), index=False)
    print(
        f"Train: {sample_count(train)}, Val: {sample_count(val)}, Test: {sample_count(test)}"
    )
    print(train.head())

    print(f"Done. Outputs stored in {output_dir}")
