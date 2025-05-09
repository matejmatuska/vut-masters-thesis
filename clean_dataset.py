import os
import sys

import pandas as pd

_DNS_PTR_QTYPE = 12


def load_csvs_dataset(path) -> pd.DataFrame:
    """
    Load the CSVs dataset from the given path

    :param path: path to the dataset
    :return: the loaded dataset
    :rtype: pd.DataFrame
    """
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


def sample_get_host_ip(df) -> pd.Series:
    """
    Find host IP in a sample

    Host IP is the one that occurs in every flow of the sample, either as
    source or destination IP.

    :param df: the sample
    :type df: pd.DataFrame
    :return: the found host IP
    :rtype: pd.Series
    """
    all_ips = pd.concat([df["SRC_IP"], df["DST_IP"]])
    ip_counts = all_ips.value_counts()

    # Find IPs that occur in every flow
    ip_in_every_flow = ip_counts[ip_counts == len(df)].idxmax()
    return pd.Series({"host_ip": ip_in_every_flow})


def get_common_ips(df) -> pd.Series:
    """
    Get IPs common IPs across multiple families

    :param df: the dataset
    :type df: pd.DataFrame
    :return: a list of common IPs
    :rtype: pd.Series
    """
    unique_dst_ips = df.groupby(["family", "sample"])["DST_IP"].unique()
    host_ips = df.groupby(["family", "sample"], group_keys=True)[
        ["SRC_IP", "DST_IP"]
    ].apply(sample_get_host_ip)

    unique_dst_ips = pd.merge(
        unique_dst_ips, host_ips, how="outer", on=["family", "sample"]
    ).explode("DST_IP")
    # filtered_unique_dst_ips.to_csv('hosts_and_dsts.csv')

    # do not remove the host IPs
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


def filter_common_ips(df, common_ips) -> pd.DataFrame:
    """
    Filter common IPs from the dataset
    :param df: the dataset
    :param common_ips: the common IPs to filter out
    :return: the filtered dataset
    :rtype: pd.DataFrame
    """
    return df[
        (~df["DST_IP"].isin(set(common_ips))) & (~df["SRC_IP"].isin(set(common_ips)))
    ]


def load_and_filter_top1m(top1m_path) -> pd.DataFrame:
    """
    Load the top-1m domains and remove the ones that could be used maliciously

    :param top1m_path: path to the original top-1m domains file
    :return: the filtered top-1m domains
    :rtype: pd.DataFrame
    """
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

    top1m = pd.read_csv(top1m_path, names=["index", "domain"], usecols=["domain"])[
        "domain"
    ]
    return top1m[~top1m.str.contains(combined_regex, regex=True)]


def filter_DNS(df, filter) -> (pd.DataFrame, pd.Series):
    """
    Filter DNS requests present in filter

    :param df: the dataset
    :param filter: List of domains to filter out
    :return: the filtered dataset and a series of removed domains
    :rtype: (pd.DataFrame, pd.Series)
    """
    index = df["DNS_NAME"].isin(set(filter))
    removed = df[index]
    return (df[~index], removed["DNS_NAME"])


def filter_rDNS(df) -> (pd.DataFrame, pd.DataFrame):
    """
    Filter reverse DNS requests occurring among multiple families

    :param df: the dataset DataFrame
    :return: the filtered dataset and the removed dataset
    :rtype: (pd.DataFrame, pd.DataFrame)
    """
    df["DNS_NAME"] = df["DNS_NAME"].map(
        lambda x: x.rstrip(".in-addr.arpa"), na_action="ignore"
    )

    rdnsdf = df[df["DNS_QTYPE"] == _DNS_PTR_QTYPE]
    common_names = rdnsdf.groupby("DNS_NAME")["family"].nunique()

    index = (df["DNS_QTYPE"] == _DNS_PTR_QTYPE) & (
        df["DNS_NAME"].isin(set(common_names[common_names > 1].index))
    )
    return (df[~index], d[index])


def run(dset_path, output_dir) -> pd.DataFrame:
    # inputs
    top1m_path = os.path.join("data", "top-1m.csv")  # TODO find some place for this
    common_ips_path = os.path.join(
        "data", "filter_common_ips.csv"
    )  # TODO find some place for this

    # outputs
    top1m_mod_path = os.path.join(output_dir, "domains-to-remove.csv")
    removed_dns_path = os.path.join(output_dir, "removed_dns.csv")
    removed_rdns_path = os.path.join(output_dir, "removed_rdns.csv")

    print("Loading dataset")
    df = load_csvs_dataset(dset_path)
    print(f"Loaded {len(df)} rows (flows)")

    prev_len = len(df)

    print("Filtering common IPs")
    if True:  # TODO: this should be computed!!!
        common_ips = pd.read_csv(common_ips_path)
    else:
        common_ips = get_common_ips(df)
        # common_ips.to_csv(common_ips_path, index=False)
    df = filter_common_ips(df, common_ips["DST_IP"])
    # print(df.groupby(['family', 'sample']).size())
    print(f"Removed {prev_len - len(df)} rows")

    prev_len = len(df)

    print("Filtering DNS requests")
    top1m = load_and_filter_top1m(top1m_path)
    top1m.to_csv(top1m_mod_path, index=False)
    df, removed_domains = filter_DNS(df, top1m)
    removed_domains.to_csv(removed_dns_path, index=False)
    print(f"Removed {prev_len - len(df)} rows")

    prev_len = len(df)

    print("Filtering common rDNS requests")
    df, removed = filter_rDNS(df)
    removed.to_csv(removed_rdns_path, index=False)
    print(f"Removed {prev_len - len(df)} rows")
    return df


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python clean_dataset.py <dataset_path> <output_dir>")
        sys.exit(1)

    path = sys.argv[1]
    if not os.path.exists(path):
        print(f"Dataset path {path} does not exist")
        sys.exit(1)

    output_dir = sys.argv[2]
    if not os.path.exists(output_dir):
        print(f"Output directory {output_dir} does not exist")
        sys.exit(1)

    df = run(path, output_dir)
    df.to_csv(os.path.join(output_dir, "dataset-clean.csv"), index=False)
