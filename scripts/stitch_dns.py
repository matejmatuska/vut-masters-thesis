import itertools
import sys
from ast import literal_eval

import pandas as pd

_STITCH_COLS = ["family", "sample", "DNS_ID", "DNS_NAME"]
_TUPLE_COLS = [
    "SRC_IP",
    "SRC_PORT",
    "DST_IP",
    "DST_PORT",
    "PROTOCOL",
]

def concat_lists(lists):
    return list(itertools.chain(*lists))

_agg_funcs = {
    "BYTES": "sum",
    "BYTES_REV": "sum",
    "PACKETS": "sum",
    "PACKETS_REV": "sum",
    "TIME_FIRST": "min",
    "TIME_LAST": "max",
    # "PROTOCOL": "first",
    "PPI_PKT_LENGTHS": concat_lists,
    "PPI_PKT_DIRECTIONS": concat_lists,
    "PPI_PKT_TIMES": concat_lists,
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
    ]
    groupcols = _STITCH_COLS + _TUPLE_COLS

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
    print("Uniflow Aggregated:")
    df = df.groupby(groupcols).agg(_agg_funcs).reset_index()
    print(df)
    print(df.index)
    print(df.columns)

    print("Reversed columns:")
    df = df.groupby(_STITCH_COLS, group_keys=False).apply(flip_reversed_flows)
    print(df)
    print(df.index)
    print(df.columns)

    print('Reset index')
    df = df.reset_index()
    print(df.index)
    print(df.columns)

    # agg to biflow
    print("Reversed columns and biflow merge:")
    df = df.groupby(groupcols).agg(_agg_funcs)
    print(df)
    return df


if __name__ == "__main__":
    df = pd.read_csv(sys.argv[1])

    df = parse_ppi(df)

    # lets drop some unused columns
    keep_cols = set(_agg_funcs.keys()) | set(_STITCH_COLS) | set(_TUPLE_COLS)
    df = df[list(keep_cols)]

    dns = df[df["DNS_NAME"].notna()]
    nondns = df[df["DNS_NAME"].isna()]

    dns = stich_dns(dns)
    print("Final DNS:")
    pd.concat([nondns, dns]).to_csv("stitched.csv", index=True)
