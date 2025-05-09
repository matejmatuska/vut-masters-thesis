import os
import sys
from ast import literal_eval

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler

from stitch_dns import stitch_dns


def _parse_ppi(df):
    # TODO better docstring
    """
    Parse PPI_PKT_LENGTHS and PPI_PKT_TIMES in the [x|y|z|...] format into Python lists.
    """
    df["PPI_PKT_LENGTHS"] = df["PPI_PKT_LENGTHS"].str.replace("|", ",")
    df["PPI_PKT_LENGTHS"] = df["PPI_PKT_LENGTHS"].apply(literal_eval)

    df["PPI_PKT_DIRECTIONS"] = df["PPI_PKT_DIRECTIONS"].str.replace("|", ",")
    df["PPI_PKT_DIRECTIONS"] = df["PPI_PKT_DIRECTIONS"].apply(literal_eval)

    df["PPI_PKT_TIMES"] = df["PPI_PKT_TIMES"].str[1:-1].str.split("|")
    df["PPI_PKT_TIMES"] = df["PPI_PKT_TIMES"].apply(
        pd.to_datetime, format="%Y-%m-%dT%H:%M:%S.%f"
    )
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
    print("DGB: assuming there is enough samples")
    # remove families with small saples < samples[0]
    sample_counts = df[["family", "sample"]].drop_duplicates().groupby("family").size()
    enough_samples = sample_counts[sample_counts >= min_samples].index
    df_enough = df[df["family"].isin(enough_samples)]

    # cap the number of samples per family to samples[1]
    print(df_enough[["family", "sample"]].head())
    has_more = (
        df_enough[["family", "sample"]]
        .drop_duplicates()
        .groupby("family", group_keys=False)
        .apply(lambda x: x.sample(n=min(len(x), max_samples)))
    )
    print(has_more.head())
    return df_enough.merge(has_more, on=["family", "sample"])


def stitch_dns_uniflows(df) -> pd.DataFrame:
    """
    Stitches DNS uniflows back into biflows.
    """
    print("Stitched DNS:")
    dns = df[df["DNS_NAME"].notna()]
    nondns = df[df["DNS_NAME"].isna()]
    dns = stitch_dns(dns).reset_index()
    return pd.concat([dns, nondns])


def remove_extreme_packet_count_samples(df, min=4, max=1e6) -> pd.DataFrame:
    """
    Remove samples with very few or very many packets.
    """
    group_sums = df.groupby("sample")[["PACKETS", "PACKETS_REV"]].transform("sum")
    return df[
        (group_sums.sum(axis=1) > min) & (group_sums.sum(axis=1) < max)
    ]


def remove_dns_only_samples(df) -> pd.DataFrame:
    """
    Filter out samples with only DNS packets.
    """
    df["is_dns"] = df["DNS_NAME"].notna()
    dns_only = df.groupby(["family", "sample"])["is_dns"].transform("all")
    return df[~dns_only]


def normalize(df, per_packet_len=30, scalerf=MinMaxScaler) -> pd.DataFrame:
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
    scalar_features = ["PACKETS", "PACKETS_REV", "BYTES", "BYTES_REV"]
    port_features = ["SRC_PORT", "DST_PORT"]

    for col in scalar_features:
        df[col] = np.log1p(df[col])

    scalar_scaler = scalerf()
    df[scalar_features] = scalar_scaler.fit_transform(df[scalar_features])

    # one hot encode PROTOCOL
    df["PROTOCOL_CAT"] = df["PROTOCOL"].astype("category").cat.codes

    def normalize_times(timestamps):
        base = timestamps[0]
        relative_times = [(ts - base).total_seconds() for ts in timestamps]
        #deltas = np.diff(timestamps)
        #return deltas
        # TODO maybe?? Measure. Exclude the first timestamp since it's always 0 - can use 0 for padding
        return relative_times[1:]

    df["PPI_PKT_TIMES"] = df['PPI_PKT_TIMES'].apply(normalize_times)

    def pad_or_truncate(lst, to_len, value=0):
        lst = lst[:to_len]
        pad_width = max(0, 30 - len(lst))
        return np.pad(lst, (0, pad_width), "constant", constant_values=value)

    # Ensure fixed-length arrays (padding or truncating to 30)
    print(df["PPI_PKT_TIMES"].head())
    for col in ["PPI_PKT_TIMES", "PPI_PKT_LENGTHS", "PPI_PKT_DIRECTIONS"]:
        df[col] = df[col].apply(pad_or_truncate, args=(per_packet_len, 0))

    def norm_arr(arr):
        scaler = scalerf()
        return scaler.fit_transform(arr.reshape(-1, 1)).flatten()

    def norm_arr_log(arr):
        arr = np.log1p(np.array(arr))
        scaler = scalerf()
        return scaler.fit_transform(arr.reshape(-1, 1)).flatten()

    def norm_2d(col):
        X = np.vstack(col.values)  # shape: (num_rows, 30)
        scaled = StandardScaler().fit_transform(X)
        return scaled.tolist()


    df["PPI_PKT_LENGTHS"] = df["PPI_PKT_LENGTHS"].apply(norm_arr)
    df["PPI_PKT_TIMES"] = norm_2d(df["PPI_PKT_TIMES"])

    # def normalize_times(timestamps):
    #     deltas = np.diff(timestamps)
    #     return norm_arr_logstd(deltas)
    return df


def train_val_test_split(
    df, val_size, test_size, random_state=42
) -> (pd.DataFrame, pd.DataFrame, pd.DataFrame):
    """
    Split the dataset into train, validation, and test sets in a stratified manner.
    """
    train_df, temp_df = train_test_split(
        df, test_size=val_size + test_size, stratify=df["family"], random_state=42
    )
    val_df, test_df = train_test_split(
        temp_df,
        test_size=test_size / (val_size + test_size),
        stratify=temp_df["family"],
        random_state=random_state,
    )
    return train_df, val_df, test_df


def main(path):
    """
    Prepare the dataset for training.

    :param path: Path to the CSV dataset.
    :type path: str
    :return: The loaded and prepared CSV dataset.
    :rtype: pandas.DataFrame
    """
    print("Loading dataset...")
    df = pd.read_csv(path)
    print(f"Loaded {len(df)} samples")
    # df = df[~df['family'].isin(['LOKIBOT', 'XWORM', 'NETWIRE', 'SLIVER', 'AGENTTESLA', 'WARZONERAT', 'COBALTSTRIKE'])]
    print("Removing extreme packet count samples...")
    df = remove_extreme_packet_count_samples(df, min=4, max=1e6)
    print(f"Samples after removing extreme packet counts: {len(df)}")

    keep_cols = [
        "family",
        "sample",
        "DNS_ID",
        "DNS_NAME",
        "SRC_IP",
        "DST_IP",
        "SRC_PORT",
        "DST_PORT",
        "PROTOCOL",
        "PACKETS",
        "PACKETS_REV",
        "BYTES",
        "BYTES_REV",
        "TIME_FIRST",
        "TIME_LAST",
        "PPI_PKT_LENGTHS",
        "PPI_PKT_DIRECTIONS",
        "PPI_PKT_TIMES",
    ]
    # these are the cols that stitch_dns uses and can aggr right now
    df = df[keep_cols]
    # has to be done before stitch_dns to be able to aggregate PPI cols
    df = _parse_ppi(df)
    print(df['PPI_PKT_TIMES'].head())

    print(f"Stitching DNS samples...")
    df = stitch_dns_uniflows(df)
    print(f"Samples after stitching: {len(df)}")

    print("Removing DNS-only samples...")
    df = remove_dns_only_samples(df)
    print(f"Samples after removing DNS-only: {len(df)}")

    print("Capping samples per family...")
    # WARN: there is no check of whether there is enough samples
    df = cap_samples_per_fam(df, 500, 2000)
    print(df[["family", "sample"]].drop_duplicates()["family"].value_counts())
    print(df[["family", "sample"]].drop_duplicates()["family"].value_counts().sum())
    print(f"Samples after capping: {len(df)}")

    if len(df) == 0:
        print("Empty dataset, nothing to do. Make sure there is enough samples.")
        sys.exit(1)

    df["label_encoded"], _ = pd.factorize(df["family"])
    print(f'Encoded families: {df.groupby("family")["label_encoded"].first()}')

    print("Normalizing dataset...")
    df = normalize(df, per_packet_len=30)

    print("Splitting dataset...")
    train, val, test = train_val_test_split(df, 0.15, 0.15)

    return train, val, test


def parse_args():
    """
    Parse command line arguments.
    """
    import argparse

    parser = argparse.ArgumentParser(description="Prepare dataset for training.")
    parser.add_argument(
        type=str, required=True, help="Path to the dataset CSV file."
    )
    parser.add_argument(
        "--output_dir", type=str, required=True, help="Directory to save the output files."
    )
    parser.add_argument(
        "--min_samples",
        type=int,
        default=50,
        help="Minimum number of samples per family.",
    )
    parser.add_argument(
        "--max_samples",
        type=int,
        default=100,
        help="Maximum number of samples per family.",
    )
    parser.add_argument(
        "--per_packet_len",
        type=int,
        default=30,
        help="Length to pad/truncate PPI_PKT_* fields.",
    )
    parser.add_argument(
        "--val_size",
        type=float,
        default=0.15,
        help="Proportion of the dataset to include in the validation split.",
    )
    parser.add_argument(
        "--test_size",
        type=float,
        default=0.15,
        help="Proportion of the dataset to include in the test split.",
    )
    parser.add_argument(
        "--random_state",
        type=int,
        default=42,
        help="Random state for reproducibility.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python prep_dataset.py <dset_path> <output_dir>")
        sys.exit(1)

    path = sys.argv[1]
    if not os.path.exists(path):
        print(f"Dataset path {path} does not exist.")
        sys.exit(1)

    output_dir = sys.argv[2]
    if not os.path.exists(path):
        print(f"Output directory {output_dir} does not exist.")
        sys.exit(1)

    train, val, test = main(path)
    train.to_csv(os.path.join(output_dir, "train.csv"), index=False)
    val.to_csv(os.path.join(output_dir, "val.csv"), index=False)
    test.to_csv(os.path.join(output_dir, "test.csv"), index=False)
    print(f"Train: {len(train)}, Val: {len(val)}, Test: {len(test)}")

    print(train.head())
