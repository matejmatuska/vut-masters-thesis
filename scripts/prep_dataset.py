import os
import sys
from ast import literal_eval

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder, StandardScaler
from stitch_dns import stitch_dns

DATE_FORMAT = "%Y-%m-%dT%H:%M:%S.%f"


def sample_count(df):
    return len(df["sample"].unique())


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
        pd.to_datetime, format=DATE_FORMAT
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
    return df[(group_sums.sum(axis=1) > min) & (group_sums.sum(axis=1) < max)]


def remove_dns_only_samples(df) -> pd.DataFrame:
    """
    Filter out samples with only DNS packets.
    """
    df["is_dns"] = df["DNS_NAME"].notna()
    dns_only = df.groupby(["family", "sample"])["is_dns"].transform("all")
    return df[~dns_only]


def normalize_all(df, per_packet_len=30) -> pd.DataFrame:
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
        # TODO maybe?? Measure. Exclude the first timestamp since it's always 0 - can use 0 for padding
        return relative_times[1:]

    df["PPI_PKT_TIMES"] = df["PPI_PKT_TIMES"].apply(times_to_relative)

    def pad_or_truncate(lst, to_len, value=0):
        lst = lst[:to_len]
        pad_width = max(0, to_len - len(lst))
        return np.pad(lst, (0, pad_width), "constant", constant_values=value)

    # Ensure fixed-length arrays (padding or truncating to 30)
    for col in ["PPI_PKT_TIMES", "PPI_PKT_LENGTHS", "PPI_PKT_DIRECTIONS"]:
        df[col] = df[col].apply(pad_or_truncate, args=(per_packet_len, 0))

    return df


def normalize_scale(
    df_train, df_val, df_test
) -> (pd.DataFrame, pd.DataFrame, pd.DataFrame):

    df_train, df_val, df_test = df_train.copy(), df_val.copy(), df_test.copy()

    scalar_features = ["PACKETS", "PACKETS_REV", "BYTES", "BYTES_REV", "DURATION"]
    port_features = ["SRC_PORT", "DST_PORT"]

    for col in scalar_features:
        df_train[col] = np.log1p(df_train[col])
        df_val[col] = np.log1p(df_val[col])
        df_test[col] = np.log1p(df_test[col])

    scaler = StandardScaler()
    df_train[scalar_features] = scaler.fit_transform(df_train[scalar_features])
    df_val[scalar_features] = scaler.transform(df_val[scalar_features])
    df_test[scalar_features] = scaler.transform(df_test[scalar_features])

    scaler = StandardScaler()
    for col in ["PPI_PKT_LENGTHS", "PPI_PKT_TIMES"]:
        df_train[col] = list(scaler.fit_transform(np.vstack(df_train[col].values)))  # shape: (num_rows, 30)
        df_val[col] = list(scaler.transform(np.vstack(df_val[col].values)))
        df_test[col] = list(scaler.transform(np.vstack(df_test[col].values)))

    df_train['DURATION'] = df_train['DURATION'].fillna(0)
    df_val['DURATION'] = df_val['DURATION'].fillna(0)
    df_test['DURATION'] = df_test['DURATION'].fillna(0)
    return df_train, df_val, df_test


def train_val_test_split(
    df, val_size, test_size, random_state=42
) -> (pd.DataFrame, pd.DataFrame, pd.DataFrame):
    """
    Split the dataset into train, validation, and test sets in a stratified manner.
    """
    # Get unique samples and their family labels
    sample_df = df[["sample", "family"]].drop_duplicates()

    # First split: train and temp (val + test)
    train_samples, temp_samples = train_test_split(
        sample_df,
        test_size=val_size + test_size,
        stratify=sample_df["family"],
        random_state=random_state,
    )

    # Second split: validation and test from temp
    val_samples, test_samples = train_test_split(
        temp_samples,
        test_size=test_size / (val_size + test_size),
        stratify=temp_samples["family"],
        random_state=random_state,
    )

    # Map back to full DataFrame
    train_df = df[df["sample"].isin(train_samples["sample"])]
    val_df = df[df["sample"].isin(val_samples["sample"])]
    test_df = df[df["sample"].isin(test_samples["sample"])]

    return train_df, val_df, test_df


def clean(df):
    print("Removing extreme packet count samples...")
    df = remove_extreme_packet_count_samples(df, min=4, max=1e6)
    print(f"Samples after removing extreme packet counts: {sample_count(df)}")

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
    df['TIME_FIRST'] = pd.to_datetime(df['TIME_FIRST'], format=DATE_FORMAT)
    df['TIME_LAST'] = pd.to_datetime(df['TIME_LAST'], format=DATE_FORMAT)
    df['DURATION'] = (df['TIME_LAST'] - df['TIME_FIRST']).dt.total_seconds()

    print("Stitching DNS samples...")
    df = stitch_dns_uniflows(df)
    print(f"Stitched samples stitching: {sample_count(df)}")

    print("Removing DNS-only samples...")
    df = remove_dns_only_samples(df)
    print(f"Samples after removing DNS-only: {sample_count(df)}")

    print("Capping samples per family...")
    # WARN: there is no check if there are enough samples
    df = cap_samples_per_fam(df, 500, 2000)
    print(df[["family", "sample"]].drop_duplicates()["family"].value_counts())
    print(df[["family", "sample"]].drop_duplicates()["family"].value_counts().sum())
    print(f"Samples after capping: {sample_count(df)}")

    print("Normalizing dataset")
    df = normalize_all(df, per_packet_len=30)
    print("Normalizing done:")
    print(df.head())
    return df


def prepare_features(df) -> (pd.DataFrame, pd.DataFrame, pd.DataFrame):
    # df["label_encoded"], _ = pd.factorize(df["family"])
    # print(f'Encoded families: {df.groupby("family")["label_encoded"].first()}')
    print("Splitting dataset...")
    train, val, test = train_val_test_split(df, 0.15, 0.15)
    print(f"Split: Train: {sample_count(train)}, Val: {sample_count(val)}, Test: {sample_count(test)}")

    print("Normalizing and scaling dataset...")
    train, val, test = normalize_scale(train, val, test)
    print("Normalization and scaling done:")
    print(train.head())

    return train, val, test


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

    ready_path = os.path.join(output_dir, "dataset-ready.parquet")
    if os.path.exists(ready_path):
        print(f"Clean dataset found at {ready_path}, loading - skipping cleaning.")
        df = pd.read_parquet(ready_path)
    else:
        print("Loading dataset...")
        df = pd.read_csv(path)
        print(f"Loaded {sample_count(df)} samples")

        print("Cleaning dataset...")
        df = clean(df)
        print(f"Storing cleaned dataset to {ready_path}...")
        df.to_parquet(ready_path, index=False)

    if len(df) == 0:
        print("Empty dataset, nothing to do. Make sure there is enough samples.")
        sys.exit(1)

    train, val, test = prepare_features(df)

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
