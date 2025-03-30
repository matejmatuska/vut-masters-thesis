import os
from ast import literal_eval

import networkx as nx
import pandas as pd
import torch

from torch_geometric.data import Data, InMemoryDataset
from torch_geometric.utils import from_networkx

from model.baseline import row_to_graph


def sample_to_graph(df):
    """
    Convert a sample DataFrame to a graph.

    A sample is a DataFrame of flows with the same family and sample name,
    aggregated from a PCAP file of a single malware sample execution.
    """
    graph = row_to_graph(df, draw=False)
    label = df['label_encoded'].iloc[0]

    data = from_networkx(graph)
    data.edge_attr = torch.tensor([list(graph.edges[edge].values()) for edge in graph.edges], dtype=torch.float32)
    data.y = torch.tensor([label], dtype=torch.long)
    return data


def load_dataset_csv(path, samples=(500, 1500)):
    """
    Load a dataset from a CSV file.

    :param path: Path to the CSV file.
    :param samples: A tuple (min_samples, max_samples) to filter the dataset.
    :return: The loaded and prepared CSV dataset.
    :rtype: pandas.DataFrame
    """
    df = pd.read_csv(path)
    #df = df[~df['family'].isin(['LOKIBOT', 'XWORM', 'NETWIRE', 'SLIVER', 'AGENTTESLA', 'WARZONERAT', 'COBALTSTRIKE'])]

    sample_counts = (
        df[['family', 'sample']]
            .drop_duplicates()
            .groupby('family')
            .size()
    )

    enough_samples = sample_counts[sample_counts >= samples[0]].index
    df_filtered = df[df['family'].isin(enough_samples)]
    print(df_filtered)

    selected_samples = (
        df_filtered[['family', 'sample']].drop_duplicates()
            .groupby('family', group_keys=False)
            .apply(lambda x: x.sample(n=min(len(x), samples[1])))
    )

    df = df_filtered.merge(selected_samples, on=['family', 'sample'])
    print(df[['family', 'sample']].drop_duplicates()['family'].value_counts())
    print(df[['family', 'sample']].drop_duplicates()['family'].value_counts().sum())

    df['label_encoded'], _ = pd.factorize(df['family'])
    print(f'Encoded familites: {df.groupby("family")["label_encoded"].first()}')

    df['PPI_PKT_LENGTHS'] = df['PPI_PKT_LENGTHS'].str.replace('|', ',')
    df['PPI_PKT_LENGTHS'] = df['PPI_PKT_LENGTHS'].apply(literal_eval)

    #df['PPI_PKT_TIMES'] = df['PPI_PKT_TIMES'].str[1:-1]
    #df['PPI_PKT_TIMES'] = df['PPI_PKT_TIMES'].str.split('|')
    #df['PPI_PKT_TIMES'] = df['PPI_PKT_TIMES'].apply(lambda times: [pd.to_datetime(t).value for t in times])
    return df


class GraphDataset(InMemoryDataset):

    def __init__(self, root, transform=None, pre_transform=None, pre_filter=None):
        super().__init__(root, transform, pre_transform, pre_filter)
        self.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return ['dataset.csv']

    @property
    def processed_file_names(self):
        return ['dataset.pt']

    def process(self):
        df = load_dataset_csv(self.raw_paths[0])
        df.to_csv(os.path.join(self.root, 'processed', 'used.csv'), index=False)

        data_list = []
        for sample_name, group in df.groupby('sample'):
            data_list.append(sample_to_graph(group))
            print(f"Processed sample: {sample_name}")

        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]

        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]

        self.save(data_list, self.processed_paths[0])
