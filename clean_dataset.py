import os
import sys

import pandas as pd

_DNS_PTR_QTYPE = 12


def load_dataset(path) -> pd.DataFrame:
    dfs = []

    for subdir, _, files in os.walk(path):
        csv_files = [f for f in files if f.endswith(".csv")]
        fam = os.path.basename(subdir)

        for file in csv_files:
            fpath = os.path.join(subdir, file)
            try:
                df = pd.read_csv(fpath)
            except pd.errors.EmptyDataError:
                print("Empty file:", fpath)
                continue
            # must be done before adding new columns
            df.columns = df.columns.str.split().str[1]
            df["family"] = fam
            df["sample"] = file

            dfs.append(df)

    dataset = pd.concat(dfs, ignore_index=True)
    return dataset


def get_host_ip(df):
    all_ips = pd.concat([df["SRC_IP"], df["DST_IP"]])
    ip_counts = all_ips.value_counts()

    # Find IPs that occur in every flow
    ip_in_every_flow = ip_counts[ip_counts == len(df)].idxmax()
    return pd.Series({"host_ip": ip_in_every_flow})


def get_common_ips(df):
    unique_dst_ips = df.groupby(["family", "sample"])["DST_IP"].unique()
    host_ips = df.groupby(["family", "sample"], group_keys=True)[
        ["SRC_IP", "DST_IP"]
    ].apply(get_host_ip)

    unique_dst_ips = pd.merge(
        unique_dst_ips, host_ips, how="outer", on=["family", "sample"]
    ).explode("DST_IP")
    # filtered_unique_dst_ips.to_csv('hosts_and_dsts.csv')

    before = len(unique_dst_ips)
    unique_dst_ips = unique_dst_ips[
        unique_dst_ips["DST_IP"] != unique_dst_ips["host_ip"]
    ]
    print(f"Removed {before - len(unique_dst_ips)} host IPs")

    result = (
        unique_dst_ips.reset_index().groupby("DST_IP")["family"].nunique().reset_index()
    )
    result = result[result["family"] > 1]
    print(result["DST_IP"])
    # result.to_csv('uniq_dst_ips.v2.csv', index=False)


def filter_common_ips(df, common_ips):
    return df[
        (~df["DST_IP"].isin(set(common_ips))) & (~df["SRC_IP"].isin(set(common_ips)))
    ]


def filter_DNS(df, filter_names, output_dir):
    top1m = pd.read_csv(filter_names, names=["index", "domain"], usecols=["domain"])[
        "domain"
    ]
    keep_domains = [
        r"^mail\..*",
        r"^smtp\..*",
        r"^webmail\..*",
        r"drive\.google\.com$",
        r"steamcommunity\.com$",
        r"t\.me$",
        r"api\.ip\.sb$",
        r"api\.ipify\.org$",
        r"api\.myip\.com$",
        r"api\.steampowered\.com$",
        r"api\.telegram\.org$",
        r"apis\.roblox\.com$",
        r"bitbucket\.org$",
        r"discord\.(com|gg)$",
        r"drive\.usercontent\.google\.com$",
        r"example\.org$",
        r"freegeoip\.app$",
        r"g\.api\.mega\.co\.nz$",
        r"gateway\.discord\.gg$",
        r"geolocation-db\.com$",
        r"geolocation\.onetrust\.com$",
        r"geoplugin\.net$",
        r"gofile\.io$",
        r"hbx\.media\.net$",
        r"i\.imgur\.com$",
        r"ip-api\.com$",
        r"ip-info\.ff\.avast\.com$",
        r"ipbase\.com$",
        r"ipinfo\.io$",
        r"iplogger\.org$",
        r"mediafire\.com$",
        r"mega\.nz$",
        r"onedrive\.live\.com$",
        r"pastebin\.com$",
        r"pool\.hashvault\.pro$",
        r"tinyurl\.com$",
        r".*tlauncher\.org$",
        r"whatismyipaddress\.com$",
        r"www\.dropbox\.com$",
        r"www\.mediafire\.com$",
        r"www\.mediafiredls\.com$",
        r"www\.myexternalip\.com$",
        r"2makestorage\.com$",
        # from URLhaus database
        r"activetykes\.shop$",
        r"distro\.ibiblio\.org$",
        r"dl\.dropboxusercontent\.com$",
        r"(.*\.)contabostorage\.com$$",
        r"firebasestorage\.googleapis\.com$",
        # r'github\.com$',
        r"ia803402\.us\.archive\.org$",
        r"paste\.ee$",
        r"res\.cloudinary\.com$",
        r"sin1\.contabostorage\.com$",
        r"static1\.squarespace\.com$",
        r"update\.drp\.su$",
        r"update\.itopvpn\.com$",
        r"web\.archive\.org$",
    ]
    # htlb.casalemedia.com
    # TODO keep dns server names?

    combined_regex = "|".join(f"({pattern})" for pattern in keep_domains)
    top1m = top1m[~top1m.str.contains(combined_regex, regex=True)]
    top1m.to_csv(
        os.path.join(output_dir, "domains-to-remove.csv"), index=False
    )  # TODO put to some dated folder

    index = df["DNS_NAME"].isin(set(top1m))

    removed = df[index]
    removed["DNS_NAME"].to_csv(os.path.join(output_dir, "removed_dns.csv"), index=False)
    return df[~index]


def filter_rDNS(df, output_dir) -> pd.DataFrame:
    """
    Filter reverse DNS requests occurring in multiple families
    """
    df["DNS_NAME"] = df["DNS_NAME"].map(
        lambda x: x.rstrip(".in-addr.arpa"), na_action="ignore"
    )

    rdnsdf = df[df["DNS_QTYPE"] == _DNS_PTR_QTYPE]
    common_names = rdnsdf.groupby("DNS_NAME")["family"].nunique()

    index = (df["DNS_QTYPE"] == _DNS_PTR_QTYPE) & (
        df["DNS_NAME"].isin(set(common_names[common_names > 1].index))
    )

    removed = df[index]
    removed.to_csv(os.path.join("removed_rdns.csv"), index=False)
    return df[~index]


def run(dset_path, output_dir) -> pd.DataFrame:
    print("Loading dataset")
    df = load_dataset(dset_path)
    print(f"Loaded {len(df)} rows (flows)")
    print(df.columns)

    prev_len = len(df)

    print("Filtering common IPs")
    common_ips = pd.read_csv(
        "data/filter_common_ips.csv"
    )  # TODO: this should be computed!!!
    df = filter_common_ips(df, common_ips["DST_IP"])
    # print(df.groupby(['family', 'sample']).size())
    print(f"Removed {prev_len - len(df)} rows")

    prev_len = len(df)

    print("Filtering DNS requests")
    df = filter_DNS(df, "data/top-1m.csv", output_dir)
    print(f"Removed {prev_len - len(df)} rows")

    prev_len = len(df)

    print("Filtering common rDNS requests")
    df = filter_rDNS(df, output_dir)
    print(f"Removed {prev_len - len(df)} rows")
    return df


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python clean_dataset.py <dataset_path> <output_dir>")
        sys.exit(1)

    path = sys.argv[1]
    df = run(path, sys.argv[2])
    df.to_csv(os.path.join(sys.argv[2], "dataset.csv"), index=False)
