import os
from abc import ABC, abstractmethod
from typing import override

import networkx as nx
import numpy as np
import pandas as pd
import torch
from torch_geometric.data import Data, HeteroData, InMemoryDataset
from torch_geometric.utils import from_networkx

_DEFAULT_ATTRIBUTES = [
    "PACKETS",
    "PACKETS_REV",
    "BYTES",
    "BYTES_REV",
    # TODO 'TCP_FLAGS',
    # TODO 'TCP_FLAGS_REV',
    "PROTOCOL",
    "SRC_PORT",
    "DST_PORT",
]


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


def load_dataset_csv(path) -> pd.DataFrame:
    """
    Load a dataset from a CSV file.

    :param path: Path to the CSV file.
    :type path: str
    :param samples: A tuple (min_samples, max_samples) to filter the dataset.
    :return: The loaded and prepared CSV dataset.
    :rtype: pandas.DataFrame
    """
    df = pd.read_csv(path)
    df["label_encoded"], _ = pd.factorize(df["family"])
    print(f'Encoded familites: {df.groupby("family")["label_encoded"].first()}')
    return df


class BaseGraphDataset(InMemoryDataset, ABC):

    def __init__(
        self, root, split, transform=None, pre_transform=None, pre_filter=None
    ):
        """
        Initialize the dataset.

        :param root: Root directory where the dataset is stored.
        :param split: Split of the dataset to load ('train', 'test', 'val').
        """
        if split not in ["train", "test", "val"]:
            raise ValueError(
                f"Invalid split: {split}. Must be 'train', 'test', or 'val'."
            )
        self.split = split

        super().__init__(root, transform, pre_transform, pre_filter)
        self.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return ["train.parquet", "test.parquet", "val.parquet"]

    @property
    def processed_file_names(self):
        return [f"{self.split}.pt"]

    @abstractmethod
    def sample_to_graph(self, df) -> Data | HeteroData | None:
        """
        Convert a sample DataFrame to a graph.

        A sample is a DataFrame of flows with the same family and sample name,
        aggregated from a PCAP file of a single malware sample execution.

        :param df: A DataFrame representing a single sample.
        :return: A PyTorch Geometric Data or HetetoData object representing the graph.
        """
        pass

    def process(self):
        df = load_dataset_csv(os.path.join(self.raw_dir, f"{self.split}.parquet"))

        data_list = []
        for sample_name, group in df.groupby("sample"):
            print(f"Processing sample: '{sample_name}'")

            # TODO these should no longer be needed
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

    def __init__(
        self, root, split, transform=None, pre_transform=None, pre_filter=None
    ):
        """
        Initialize the dataset.

        :param root: Root directory where the dataset is stored.
        :param split: Split of the dataset to load ('train', 'test', 'val').
        """
        super().__init__(root, split, transform, pre_transform, pre_filter)

    def _sample_to_graph(
        self,
        df,
        attributes=_DEFAULT_ATTRIBUTES,
        aggfunc=np.mean,
    ):
        """
        Convert a row of dataset DataFrame to a digraph.

        :param attributes: A list of attributes to use as edge attributes.
        :type attributes: list
        :param aggfunc: A function to aggregate edge attributes.
        :type aggfunc: function
        """
        G = nx.MultiDiGraph()
        for _, row in df.iterrows():
            src_ip = row["SRC_IP"]
            dst_ip = row["DST_IP"]

            edge_attr = row[attributes].to_list()
            edge_attr.extend(row["PPI_PKT_LENGTHS"])
            edge_attr.extend(row["PPI_PKT_TIMES"])
            edge_attr.extend(row["PPI_PKT_DIRECTIONS"])
            G.add_edge(src_ip, dst_ip, feature=edge_attr)

        def aggregate_edges(graph, aggfunc=aggfunc) -> nx.DiGraph:
            """
            Aggregate edges of a multi-digraph to a standard di-graph.
            """
            G = nx.DiGraph()
            edge_data = defaultdict(lambda: defaultdict(list))

            for u, v, data in graph.edges(data=True):
                for key, val in data.items():
                    edge_data[(u, v)][key].append(val)

            for (u, v), data in edge_data.items():
                agg_data = {}
                for key, val_list in data.items():
                    stacked = np.vstack(val_list)  # shape: (num_edges, feature_dim)
                    agg_data[key] = aggfunc(stacked, axis=0)  # mean across edges
                G.add_edge(u, v, **agg_data)
            return G

        agg_G = aggregate_edges(G)
        # assert not any(nx.isolates(agg_G))
        return agg_G

    @override
    def sample_to_graph(self, df):
        graph = self._sample_to_graph(df)
        label = df["label_encoded"].iloc[0]

        data = from_networkx(graph)
        data.edge_attr = torch.tensor(
            [e["feature"] for _, _, e in graph.edges(data=True)],
            dtype=torch.float
        )
        # data.edge_attr = torch.tensor(
        #     [list(graph.edges[edge].values()) for edge in graph.edges],
        #     dtype=torch.float32,
        # )
        data.y = torch.tensor([label], dtype=torch.long)
        return data


class ChronoDataset(BaseGraphDataset):

    def __init__(
        self, root, split, transform=None, pre_transform=None, pre_filter=None
    ):
        """
        Initialize the dataset.

        :param root: Root directory where the dataset is stored.
        :param split: Split of the dataset to load ('train', 'test', 'val').
        """
        super().__init__(root, split, transform, pre_transform, pre_filter)


    def _sample_to_graph(
        self,
        df,
        attributes=_DEFAULT_ATTRIBUTES,
    ):
        """
        Convert a row of dataset DataFrame to a digraph.
        """

        def node_key(row):
            # TODO just some unique key for nodes
            return f"{row['SRC_IP']}:{row['SRC_PORT']}\n{row['DST_IP']}:{row['DST_PORT']}-{row['PROTOCOL']}"

        G = nx.DiGraph()
        # first make the forward edges
        df = df.sort_values(by="TIME_FIRST", ascending=True).reset_index(drop=True)
        prev = df.iloc[0]
        for _, curr in df.iloc[1:].iterrows():
            prev_node = node_key(prev)
            curr_node = node_key(curr)

            prev_attrs = prev[attributes].to_dict()
            prev_attrs["PPI_PKT_LENGTHS"] = prev["PPI_PKT_LENGTHS"]
            prev_attrs["PPI_PKT_DIRECTIONS"] = prev["PPI_PKT_DIRECTIONS"]
            prev_attrs["PPI_PKT_TIMES"] = prev["PPI_PKT_TIMES"]

            curr_attrs = curr[attributes].to_dict()
            curr_attrs["PPI_PKT_LENGTHS"] = curr["PPI_PKT_LENGTHS"]
            curr_attrs["PPI_PKT_DIRECTIONS"] = curr["PPI_PKT_DIRECTIONS"]
            curr_attrs["PPI_PKT_TIMES"] = curr["PPI_PKT_TIMES"]

            G.add_node(prev_node, **prev_attrs)
            G.add_node(curr_node, **curr_attrs)
            G.add_edge(prev_node, curr_node)  # no edge attributes
            prev = curr

        # now the reverse edges
        reverse = df.sort_values(by="TIME_LAST", ascending=True)
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
        graph = self._sample_to_graph(df)
        if len(graph) == 0:
            return None

        label = df["label_encoded"].iloc[0]
        data = from_networkx(graph, group_node_attrs="all")
        # data.edge_attr = torch.tensor(
        #    [list(graph.edges[edge].values()) for edge in graph.edges], dtype=torch.float32
        # )
        data.y = torch.tensor([label], dtype=torch.long)
        return data


class Repr1Dataset(BaseGraphDataset):

    def __init__(
        self, root, split, transform=None, pre_transform=None, pre_filter=None
    ):
        """
        Initialize the dataset.

        :param root: Root directory where the dataset is stored.
        :param split: Split of the dataset to load ('train', 'test', 'val').
        """
        super().__init__(root, split, transform, pre_transform, pre_filter)

    def _sample_to_graph(
        self,
        df,
        attributes=_DEFAULT_ATTRIBUTES,
    ):
        """
        Convert a row of dataset DataFrame to HeteroData.
        """
        flow_attrs = []
        host_ip_to_id = {}
        edges_host_to_flow = [[], []]  # [host_idx, flow_idx]
        edges_flow_to_host = [[], []]  # [flow_idx, host_idx]
        edges_flow_to_flow = [[], []]  # [flow_idx_src, flow_idx_dst]

        current_host_id = 0
        current_flow_id = 0

        for _, row in df.iloc[0:].iterrows():  # TODO do we need sorting?
            for ip in [row["SRC_IP"], row["DST_IP"]]:
                if ip not in host_ip_to_id:
                    host_ip_to_id[ip] = current_host_id
                    current_host_id += 1

            src_id = host_ip_to_id[row["SRC_IP"]]
            dst_id = host_ip_to_id[row["DST_IP"]]

            attrs = row[attributes].to_dict()
            attr_vals = list(attrs.values())
            attr_vals.extend(row["PPI_PKT_LENGTHS"])
            attr_vals.extend(row["PPI_PKT_DIRECTIONS"])
            attr_vals.extend(row["PPI_PKT_TIMES"])
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
        graph = self._sample_to_graph(df)
        if len(graph) == 0:
            return None

        label = df["label_encoded"].iloc[0]
        graph.y = torch.tensor([label], dtype=torch.long)
        return graph
