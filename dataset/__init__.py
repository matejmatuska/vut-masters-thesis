import os
from abc import ABC, abstractmethod
from typing import override

import networkx as nx
import numpy as np
import pandas as pd
import torch
from torch_geometric.data import Data, HeteroData, InMemoryDataset
from torch_geometric.utils import from_networkx

SCALAR_ATTRIBUTES = [
    "PACKETS",
    "PACKETS_REV",
    "BYTES",
    "BYTES_REV",
    "DURATION",
]
PROTO_ONE_HOT = [
    "PROTO_TCP",
    "PROTO_UDP",
    "PROTO_ICMP",
    # "PROTO_OTHER", currently empty
]
VECTOR_ATTRIBUTES = [
    "PPI_PKT_LENGTHS",
    "PPI_PKT_TIMES",
    "PPI_PKT_DIRECTIONS",
]
EMBED_COLS = [
    "DST_PORT",
    "TCP_FLAGS",
    "TCP_FLAGS_REV",
]


class BaseGraphDataset(InMemoryDataset, ABC):
    """
    Base class for creating graph datasets.

    This class is designed to be subclassed for specific graph representations.
    A subclass should implement the `sample_to_graph` method to convert a sample DataFrame into a graph.
    """

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

        out = torch.load(self.processed_paths[0])
        assert isinstance(out, tuple)
        assert len(out) == 4
        data, self.slices, data_cls, extra_attrs = out
        self.data = data_cls.from_dict(data)
        self._label_map = extra_attrs["label_map"]

    @property
    def raw_file_names(self):
        return ["train.parquet", "test.parquet", "val.parquet"]

    @property
    def processed_file_names(self):
        return [f"{self.split}.pt"]

    @property
    def label_map(self) -> dict[int, str]:
        """
        Returns a mapping of class indices to class names.

        :return: A dictionary mapping class indices to class names.
        """
        return self._label_map

    @abstractmethod
    def sample_to_graph(self, df) -> Data | HeteroData | None:
        """
        Convert a sample DataFrame to a graph.

        Subclasses should implement this method to convert a DataFrame representing a single
        sample into a graph representation.
        None can be returned if the DataFrame doesn't contain enough data to create a graph.

        Note that a label also needs to be set on the graph, the following can be used:
            label = df["label_encoded"].iloc[0]
            built_graph.y = torch.tensor([label], dtype=torch.long)

        :param df: A DataFrame representing a single sample.
        :return: A PyTorch Geometric Data or HetetoData object representing the graph.
        """
        pass

    def preprocess_all(self, df):
        return df

    def _load_dataset_parquet(self, path) -> pd.DataFrame:
        """
        Load a dataset from a .parquet file.

        :param path: Path to the .parquet file.
        :type path: str
        :param samples: A tuple (min_samples, max_samples) to filter the dataset.
        :return: The loaded and prepared CSV dataset.
        :rtype: pandas.DataFrame
        """
        df = pd.read_parquet(path)
        df["label_encoded"], _ = pd.factorize(df["family"])
        print(f"Encoded familites: {df.groupby('family')['label_encoded'].first()}")
        return df

    def process(self):
        df = self._load_dataset_parquet(
            os.path.join(self.raw_dir, f"{self.split}.parquet")
        )
        df = self.preprocess_all(df)

        data_list = []
        for sample_name, group in df.groupby("sample"):
            print(f"Processing sample: '{sample_name}'")

            graph = self.sample_to_graph(group)
            if graph:
                data_list.append(graph)
            else:
                print(f"Sample {sample_name} has no nodes. Skipping.")

        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]

        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]

        self._label_map = df.groupby("label_encoded")["family"].first().to_dict()
        extra_attrs = {
            "label_map": self._label_map,
        }
        data, slices = self.collate(data_list)
        torch.save(
            (data.to_dict(), slices, data.__class__, extra_attrs),
            self.processed_paths[0],
        )


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

    def _sample_to_graph(self, df, aggfunc="mean"):
        """
        Convert a row of dataset DataFrame to a digraph.

        :param aggfunc: A function to aggregate edge attributes.
        :type aggfunc: function
        """

        def explode_to_columns(df, col):
            """
            Explode a column of lists into separate columns.
            """
            exploded = df[col].apply(pd.Series)
            exploded.columns = [f"{col}_{i}" for i in range(exploded.shape[1])]
            df = pd.concat([df.drop(columns=[col]), exploded], axis=1)
            return df

        # only aggregable (and IPs for grouping)
        keep_cols = (
            ["SRC_IP", "DST_IP"]
            + SCALAR_ATTRIBUTES
            + PROTO_ONE_HOT
            + VECTOR_ATTRIBUTES
            + EMBED_COLS
        )
        df = df[keep_cols]
        # make these aggregable
        for attr in VECTOR_ATTRIBUTES:
            df = explode_to_columns(df, attr)

        df = df.groupby(["SRC_IP", "DST_IP"], as_index=False).agg(aggfunc)

        edge_attr_cols = df.columns.difference(["SRC_IP", "DST_IP"] + EMBED_COLS)
        G = nx.DiGraph()
        for _, row in df.iterrows():
            src_ip = row["SRC_IP"]
            dst_ip = row["DST_IP"]

            edge_attr = row[edge_attr_cols].to_list()

            G.add_edge(
                src_ip,
                dst_ip,
                features=edge_attr,
                port=row["DST_PORT"],
                tcp_flags=row["TCP_FLAGS"],
                tcp_flags_rev=row["TCP_FLAGS_REV"],
            )
        return G

    @override
    def sample_to_graph(self, df):
        graph = self._sample_to_graph(df)
        label = df["label_encoded"].iloc[0]

        data = from_networkx(graph)
        data.edge_attr = torch.tensor(
            [e["features"] for _, _, e in graph.edges(data=True)], dtype=torch.float
        )
        data.dst_ports = torch.tensor(
            [e["port"] for _, _, e in graph.edges(data=True)], dtype=torch.long
        )
        data.tcp_flags = torch.tensor(
            [e["tcp_flags"] for _, _, e in graph.edges(data=True)], dtype=torch.long
        )
        data.tcp_flags_rev = torch.tensor(
            [e["tcp_flags_rev"] for _, _, e in graph.edges(data=True)], dtype=torch.long
        )
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

    def _sample_to_graph(self, df):
        """
        Convert a row of dataset DataFrame to a digraph.
        """

        def node_key(row):
            # just some unique key for nodes, required for networkx NOT used as a feature
            return f"{row['SRC_IP']}:{row['SRC_PORT']}\n{row['DST_IP']}:{row['DST_PORT']}-{row['TIME_FIRST']}"

        G = nx.DiGraph()
        # first make the forward edges
        df = df.sort_values(by="TIME_FIRST", ascending=True).reset_index(drop=True)

        dst_ports = torch.tensor(df["DST_PORT"].values, dtype=torch.long)
        tcp_flags = torch.tensor(df["TCP_FLAGS"].values, dtype=torch.long)
        tcp_flags_rev = torch.tensor(df["TCP_FLAGS_REV"].values, dtype=torch.long)

        prev = df.iloc[0]
        prev_node = node_key(prev)

        prev_attrs = prev[SCALAR_ATTRIBUTES + PROTO_ONE_HOT].to_dict()
        for attr in VECTOR_ATTRIBUTES:
            prev_attrs[attr] = prev[attr]
        G.add_node(prev_node, **prev_attrs)

        for _, curr in df.iloc[1:].iterrows():
            curr_node = node_key(curr)

            curr_attrs = curr[SCALAR_ATTRIBUTES + PROTO_ONE_HOT].to_dict()
            for attr in VECTOR_ATTRIBUTES:
                curr_attrs[attr] = curr[attr]

            G.add_node(curr_node, **curr_attrs)
            G.add_edge(prev_node, curr_node)  # no edge attributes
            prev_node = curr_node

        # now the reverse edges
        reverse = df.sort_values(by="TIME_LAST", ascending=True)
        prev = reverse.iloc[0]
        prev_node = node_key(prev)
        for _, curr in reverse.iloc[1:].iterrows():
            curr_node = node_key(curr)
            # .name is the index of the row
            # FIXME rename the index to something more meaningful
            if curr.name - prev.name > 1:
                G.add_edge(curr_node, prev_node)

            if curr.name - prev.name < -1:
                G.add_edge(prev_node, curr_node)

            prev = curr
            prev_node = curr_node

        return G, dst_ports, tcp_flags, tcp_flags_rev

    @override
    def sample_to_graph(self, df):
        graph, dst_ports, tcp_flags, tcp_flags_rev = self._sample_to_graph(df)
        if len(graph) == 0:
            return None

        data = from_networkx(graph, group_node_attrs="all")
        data.dst_ports = dst_ports
        data.tcp_flags = tcp_flags
        data.tcp_flags_rev = tcp_flags_rev

        label = df["label_encoded"].iloc[0]
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

    def _sample_to_graph(self, df):
        """
        Convert a row of dataset DataFrame to HeteroData.
        """
        flow_attrs = []
        host_ip_to_id = {}
        edges_host_to_flow = [[], []]  # [host_idx, flow_idx]
        edges_flow_to_host = [[], []]  # [flow_idx, host_idx]
        edges_flow_to_flow = [[], []]  # [flow_idx_src, flow_idx_dst]

        dst_ports = torch.tensor(df["DST_PORT"].values, dtype=torch.long)
        tcp_flags = torch.tensor(df["TCP_FLAGS"].values, dtype=torch.long)
        tcp_flags_rev = torch.tensor(df["TCP_FLAGS_REV"].values, dtype=torch.long)

        current_host_id = 0
        current_flow_id = 0

        for _, row in df.iloc[0:].iterrows():  # TODO do we need sorting?
            for ip in [row["SRC_IP"], row["DST_IP"]]:
                if ip not in host_ip_to_id:
                    host_ip_to_id[ip] = current_host_id
                    current_host_id += 1

            src_id = host_ip_to_id[row["SRC_IP"]]
            dst_id = host_ip_to_id[row["DST_IP"]]

            attrs = row[SCALAR_ATTRIBUTES + PROTO_ONE_HOT].to_dict()
            attr_vals = list(attrs.values())
            for attr in VECTOR_ATTRIBUTES:
                attr_vals.extend(row[attr])
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
        data["NetworkFlow"].dst_ports = dst_ports
        data["NetworkFlow"].tcp_flags = tcp_flags
        data["NetworkFlow"].tcp_flags_rev = tcp_flags_rev
        return data

    @override
    def sample_to_graph(self, df):
        graph = self._sample_to_graph(df)
        if len(graph) == 0:
            return None

        label = df["label_encoded"].iloc[0]
        graph.y = torch.tensor([label], dtype=torch.long)
        return graph
