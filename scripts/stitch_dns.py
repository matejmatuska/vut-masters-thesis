"""
Stitch DNS uniflow records together into biflow records.
"""
import functools
import itertools
import sys

import pandas as pd

STITCH_COLS = ["family", "sample", "DNS_ID", "DNS_NAME"]
"""
The columns to group by when stitching DNS records together.
Used together with FLOW_TUPLE_COLS to create a unique key for each flow.
"""

FLOW_TUPLE_COLS = [
    "SRC_IP",
    "SRC_PORT",
    "DST_IP",
    "DST_PORT",
    "PROTOCOL",
]
"""
Flow tuple columns to group by when stitching DNS records together.
Used together with STITCH_COLS to create a unique key for each flow.
"""

def _concat_lists(lists):
    return list(itertools.chain(*lists))

COL_AGG_FUNCS = {
    "BYTES": "sum",
    "BYTES_REV": "sum",
    "PACKETS": "sum",
    "PACKETS_REV": "sum",
    "TIME_FIRST": "min",
    "TIME_LAST": "max",
    "TCP_FLAGS": lambda x: functools.reduce(lambda a, b: a | b, x.astype(int)),
    "TCP_FLAGS_REV": lambda x: functools.reduce(lambda a, b: a | b, x.astype(int)),
    "PPI_PKT_LENGTHS": _concat_lists,
    "PPI_PKT_DIRECTIONS": _concat_lists,
    "PPI_PKT_TIMES": _concat_lists,
}


def stitch_dns(df):
    """
    Stitch DNS records together based on the family, sample, DNS_ID, and DNS_NAME.

    The PPI fields must be Python lists
    """
    ordered_cols = [
        "SRC_IP",
        "DST_IP",
        "SRC_PORT",
        "DST_PORT",
        "BYTES",
        "BYTES_REV",
        "PACKETS",
        "PACKETS_REV",
        "TCP_FLAGS",
        "TCP_FLAGS_REV",
    ]
    reverse_cols = [
        "DST_IP",
        "SRC_IP",
        "DST_PORT",
        "SRC_PORT",
        "BYTES_REV",
        "BYTES",
        "PACKETS_REV",
        "PACKETS",
        "TCP_FLAGS_REV",
        "TCP_FLAGS",
    ]
    groupcols = STITCH_COLS + FLOW_TUPLE_COLS

    def flip_reversed_flows(group):
        # NOTE: hopefully there is no weird case where the first packet is not sent by the host
        # this maybe could happen if the capture was start after the request and before the response
        group = group.sort_values(by="TIME_FIRST")
        hostip = group.iloc[0]["SRC_IP"]

        # swap the reversed columns
        isreversed = group["DST_IP"] == hostip
        # NOTE: this is ugly but recommended in the pandas docs
        # > The correct way to swap column values is by using raw values
        group.loc[isreversed, ordered_cols] = group.loc[isreversed, reverse_cols].values

        # "flip" directions from 1 to 0
        group.loc[isreversed, 'PPI_PKT_DIRECTIONS'] = group.loc[isreversed, 'PPI_PKT_DIRECTIONS'].map(lambda lst: [-1] * len(lst))
        return group


    # first aggregate the flows in the same direction
    # print("Uniflow Aggregated:")
    df = df.groupby(groupcols).agg(COL_AGG_FUNCS).reset_index()
    # print(df)
    # print(df.index)
    # print(df.columns)

    # print("Reversed columns:")
    df = df.groupby(STITCH_COLS, group_keys=False).apply(flip_reversed_flows)
    # print(df)
    # print(df.index)
    # print(df.columns)

    # print('Reset index')
    # df = df.reset_index()
    # print(df.index)
    # print(df.columns)

    # agg to biflow
    print("Reversed columns and biflows merged")
    df = df.groupby(groupcols).agg(COL_AGG_FUNCS)
    # print(df)
    return df


if __name__ == "__main__":
    df = pd.read_csv(sys.argv[1])

    # df = parse_ppi(df)

    # lets drop some unused columns
    keep_cols = set(COL_AGG_FUNCS.keys()) | set(STITCH_COLS) | set(FLOW_TUPLE_COLS)
    df = df[list(keep_cols)]

    dns = df[df["DNS_NAME"].notna()]
    nondns = df[df["DNS_NAME"].isna()]

    dns = stitch_dns(dns)
    print("Final DNS:")
    pd.concat([nondns, dns]).to_csv("stitched.csv", index=True)
