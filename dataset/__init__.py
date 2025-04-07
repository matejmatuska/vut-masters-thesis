import os
from ast import literal_eval

import pandas as pd
import torch

from torch_geometric.data import InMemoryDataset
from torch_geometric.utils import from_networkx

from model.baseline import row_to_graph

from stitch_dns import stitch_dns


def sample_to_graph(df):
    """
    Convert a sample DataFrame to a graph.

    A sample is a DataFrame of flows with the same family and sample name,
    aggregated from a PCAP file of a single malware sample execution.
    """
    graph = row_to_graph(df, draw=False)
    label = df["label_encoded"].iloc[0]

    data = from_networkx(graph)
    data.edge_attr = torch.tensor(
        [list(graph.edges[edge].values()) for edge in graph.edges], dtype=torch.float32
    )
    data.y = torch.tensor([label], dtype=torch.long)
    return data


def parse_ppi(df):
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


def load_dataset_csv(path, samples=(500, 1500)):
    """
    Load a dataset from a CSV file.

    :param path: Path to the CSV file.
    :type path: str
    :param samples: A tuple (min_samples, max_samples) to filter the dataset.
    :return: The loaded and prepared CSV dataset.
    :rtype: pandas.DataFrame
    """

    def cap_samples_per_fam(df):
        # remove families with small number of samples
        sample_counts = (
            df[["family", "sample"]].drop_duplicates().groupby("family").size()
        )

        enough_samples = sample_counts[sample_counts >= samples[0]].index
        df_filtered = df[df["family"].isin(enough_samples)]
        print(df_filtered)

        # cap the number of samples per family
        selected_samples = (
            df_filtered[["family", "sample"]]
            .drop_duplicates()
            .groupby("family", group_keys=False)
            .apply(lambda x: x.sample(n=min(len(x), samples[1])))
        )
        return df_filtered.merge(selected_samples, on=["family", "sample"])

    df = pd.read_csv(path)
    # df = df[~df['family'].isin(['LOKIBOT', 'XWORM', 'NETWIRE', 'SLIVER', 'AGENTTESLA', 'WARZONERAT', 'COBALTSTRIKE'])]

    # remove samples with small number of packets
    group_sums = df.groupby("sample")[["PACKETS", "PACKETS_REV"]].transform("sum")
    df = df[(group_sums.sum(axis=1) > 4) & (group_sums.sum(axis=1) < 1e6)]

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

    # has to be done before stitch_dns to be able to aggr PPI cols
    df = parse_ppi(df)

    # stitch DNS uniflows
    print("Stitched DNS:")
    dns = df[df["DNS_NAME"].notna()]
    nondns = df[df["DNS_NAME"].isna()]
    dns = stitch_dns(dns).reset_index()
    df = pd.concat([dns, nondns])
    print(df)

    print("Filtering out DNS only samples")
    df["is_dns"] = df["DNS_NAME"].notna()
    dns_only = df.groupby(["family", "sample"])["is_dns"].transform("all")
    df = df[~dns_only]
    print(df)

    df = cap_samples_per_fam(df)
    print(df[["family", "sample"]].drop_duplicates()["family"].value_counts())
    print(df[["family", "sample"]].drop_duplicates()["family"].value_counts().sum())

    def convert_to_relative_times(timestamps):
        base = timestamps[0]
        relative_times = [int((ts - base).total_seconds()) for ts in timestamps]
        return relative_times[1:]  # TODO maybe?? Measure. Exclude the first timestamp since it's always 0 - can use 0 for padding

    df["PPI_PKT_TIMES"] = df["PPI_PKT_TIMES"].apply(convert_to_relative_times)

    df["label_encoded"], _ = pd.factorize(df["family"])
    print(f'Encoded familites: {df.groupby("family")["label_encoded"].first()}')
    return df


class GraphDataset(InMemoryDataset):

    def __init__(self, root, transform=None, pre_transform=None, pre_filter=None):
        super().__init__(root, transform, pre_transform, pre_filter)
        self.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return ["dataset.csv"]

    @property
    def processed_file_names(self):
        return ["dataset.pt"]

    def process(self):
        df = load_dataset_csv(self.raw_paths[0])
        df.to_csv(os.path.join(self.root, "processed", "used.csv"), index=False)

        data_list = []
        for sample_name, group in df.groupby("sample"):
            data_list.append(sample_to_graph(group))
            # print(f"Processed sample: {sample_name}")

        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]

        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]

        self.save(data_list, self.processed_paths[0])
