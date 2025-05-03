import os
from abc import ABC, abstractmethod
from ast import literal_eval
from typing import override

import networkx as nx
import numpy as np
import pandas as pd
import torch

from torch_geometric.data import Data, HeteroData, InMemoryDataset
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


class BaseGraphDataset(InMemoryDataset, ABC):

    def __init__(self, root, transform=None, pre_transform=None, pre_filter=None):
        super().__init__(root, transform, pre_transform, pre_filter)
        self.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return ["dataset.csv"]

    @property
    def processed_file_names(self):
        return ["dataset.pt"]

    @abstractmethod
    def sample_to_graph(self, df) -> Data|HeteroData|None:
        """
        Convert a sample DataFrame to a graph.

        A sample is a DataFrame of flows with the same family and sample name,
        aggregated from a PCAP file of a single malware sample execution.

        :param df: A DataFrame representing a single sample.
        :return: A PyTorch Geometric Data or HetetoData object representing the graph.
        """
        pass

    def process(self):
        df = load_dataset_csv(self.raw_paths[0])
        df.to_csv(os.path.join(self.root, "processed", "used.csv"), index=False)

        data_list = []
        for sample_name, group in df.groupby("sample"):
            print(f"Processing sample: {sample_name}")
            if df.empty:
                print(f"Sample {sample_name} is empty. Skipping.")
                continue
            if df.shape[0] < 2:
                print(f"Sample {sample_name} is < 2. Skipping.")
                continue

            graph = self.sample_to_graph(group)
            if graph:
                data_list.append(graph)
            else:
                print(f"Sample {sample_name} has no nodes. Skipping.")

        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]

        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]

        self.save(data_list, self.processed_paths[0])


class SunDataset(BaseGraphDataset):

    @override
    def sample_to_graph(self, df):
        return sample_to_graph(df)


class ChronoDataset(BaseGraphDataset):

    def row_to_graph(
        self,
        df,
        attributes=[
            'PACKETS',
            'PACKETS_REV',
            'BYTES',
            'BYTES_REV',
            # TODO 'TCP_FLAGS',
            # TODO 'TCP_FLAGS_REV',
            'PROTOCOL',
            'SRC_PORT',
            'DST_PORT',
            # TODO experiment with these more
        ],
    ):
        """
        Convert a row of dataset DataFrame to a digraph.
        """

        def pad_ppi(series, to_len=30, value=0):
            return np.pad(series, (0, to_len - len(series)), 'constant', constant_values=value)

        def node_key(row):
            # TODO just some unique key for the node
            return f"{row['SRC_IP']}:{row['SRC_PORT']}\n{row['DST_IP']}:{row['DST_PORT']}-{row['PROTOCOL']}"

        G = nx.DiGraph()

        # first make the forward edges
        df = df.sort_values(by='TIME_FIRST', ascending=True).reset_index(drop=True)
        prev = df.iloc[0]
        for _, curr in df.iloc[1:].iterrows():
            prev_node = node_key(prev)
            curr_node = node_key(curr)

            prev_attrs = prev[attributes].to_dict()
            prev_attrs['PPI_PKT_LENGTHS'] = pad_ppi(prev['PPI_PKT_LENGTHS'], value=0)
            prev_attrs['PPI_PKT_DIRECTIONS'] = pad_ppi(prev['PPI_PKT_DIRECTIONS'], value=0)
            prev_attrs['PPI_PKT_TIMES'] = pad_ppi(prev['PPI_PKT_TIMES'], value=0)

            curr_attrs = curr[attributes].to_dict()
            curr_attrs['PPI_PKT_LENGTHS'] = pad_ppi(curr['PPI_PKT_LENGTHS'], value=0)
            curr_attrs['PPI_PKT_DIRECTIONS'] = pad_ppi(curr['PPI_PKT_DIRECTIONS'], value=0)
            curr_attrs['PPI_PKT_TIMES'] = pad_ppi(curr['PPI_PKT_TIMES'], value=0)

            G.add_node(prev_node, **prev_attrs)
            G.add_node(curr_node, **curr_attrs)
            G.add_edge(prev_node, curr_node)  # no edge attributes

            prev = curr

        # now the reverse edges
        reverse = df.sort_values(by='TIME_LAST', ascending=True)
        prev = reverse.iloc[0]
        for _, curr in reverse.iloc[1:].iterrows():
            prev_node = node_key(prev)
            curr_node = node_key(curr)

            if curr.name - prev.name > 1:
                G.add_edge(node_key(curr), node_key(prev))

            if curr.name - prev.name < -1:
                G.add_edge(node_key(prev), node_key(curr))

            prev = curr
        return G

    @override
    def sample_to_graph(self, df):
        graph = self.row_to_graph(df)
        print(graph)
        label = df["label_encoded"].iloc[0]

        if len(graph) == 0:
            return None

        data = from_networkx(graph, group_node_attrs='all')
        print(data.x)
        #data.edge_attr = torch.tensor(
        #    [list(graph.edges[edge].values()) for edge in graph.edges], dtype=torch.float32
        #)
        data.y = torch.tensor([label], dtype=torch.long)
        print(data)
        return data


class Repr1Dataset(BaseGraphDataset):

    def row_to_graph(
        self,
        df,
        attributes=[
            'PACKETS',
            'PACKETS_REV',
            'BYTES',
            'BYTES_REV',
            # TODO 'TCP_FLAGS',
            # TODO 'TCP_FLAGS_REV',
            'PROTOCOL',
            'SRC_PORT',
            'DST_PORT',
            # TODO experiment with these more
        ],
    ):
        """
        Convert a row of dataset DataFrame to HeteroData.
        """

        def pad_ppi(series, to_len=30, value=0):
            return np.pad(series, (0, to_len - len(series)), 'constant', constant_values=value)

        flow_attrs = []
        host_ip_to_id = {}
        edges_host_to_flow = [[], []]  # [host_idx, flow_idx]
        edges_flow_to_host = [[], []]  # [flow_idx, host_idx]
        edges_flow_to_flow = [[], []]  # [flow_idx_src, flow_idx_dst]

        current_host_id = 0
        current_flow_id = 0

        for _, row in df.iloc[0:].iterrows():  # TODO do we need sorting?
            for ip in [row['SRC_IP'], row['DST_IP']]:
                if ip not in host_ip_to_id:
                    host_ip_to_id[ip] = current_host_id
                    current_host_id += 1

            src_id = host_ip_to_id[row['SRC_IP']]
            dst_id = host_ip_to_id[row['DST_IP']]

            attrs = row[attributes].to_dict()
            attr_vals = list(attrs.values())
            attr_vals.extend(pad_ppi(row['PPI_PKT_LENGTHS'], value=0))
            attr_vals.extend(pad_ppi(row['PPI_PKT_DIRECTIONS'], value=0))
            attr_vals.extend(pad_ppi(row['PPI_PKT_TIMES'], value=0))
            flow_attrs.append(attr_vals)

            edges_host_to_flow[0] += [src_id, dst_id]
            edges_host_to_flow[1] += [current_flow_id, current_flow_id]

            edges_flow_to_host[0] += [current_flow_id, current_flow_id]
            edges_flow_to_host[1] += [src_id, dst_id]
            current_flow_id += 1

        for i in range(current_flow_id - 2):
            edges_flow_to_flow[0] += [i, i + 1]
            edges_flow_to_flow[1] += [i + 1, i]

        data = HeteroData()
        data["NetworkFlow"].x = torch.tensor(flow_attrs, dtype=torch.float)
        data["Host"].node_ids = torch.arange(len(host_ip_to_id), dtype=torch.long)
        data[("Host", "communicates", "NetworkFlow")].edge_index = torch.tensor(
            edges_host_to_flow, dtype=torch.long
        )
        data[("NetworkFlow", "related", "NetworkFlow")].edge_index = torch.tensor(
            edges_flow_to_flow, dtype=torch.long
        )
        data[("NetworkFlow", "communicates", "Host")].edge_index = torch.tensor(
            edges_flow_to_host, dtype=torch.long
        )
        return data

    @override
    def sample_to_graph(self, df):
        graph = self.row_to_graph(df)
        print(graph)
        if len(graph) == 0:
            return None

        label = df["label_encoded"].iloc[0]

        # data = from_networkx(graph, group_node_attrs='all')
        # print(data.x)
        # data.edge_attr = torch.tensor(
        #    [list(graph.edges[edge].values()) for edge in graph.edges], dtype=torch.float32
        #)
        graph.y = torch.tensor([label], dtype=torch.long)
        # print(data)
        return graph
