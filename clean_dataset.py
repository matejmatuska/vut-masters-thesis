import os
import sys

import ipaddress
import pandas as pd

def load_dataset(path, store=False) -> pd.DataFrame:
    dataset = pd.DataFrame()

    for root, families, _ in os.walk(path):
        for family in families:
            family_path = os.path.join(root, family)
            for _, _, files in os.walk(family_path):
                for file in files:
                    if file.endswith('.csv'):
                        file_path = os.path.join(family_path, file)
                        #print(file_path)
                        try:
                            df = pd.read_csv(file_path)
                        except pd.errors.EmptyDataError:
                            print('Empty file:', file_path)
                            continue

                        df.columns = df.columns.str.split().str[1]

                        df['family'] = family
                        df['sample'] = file

                        dataset = pd.concat([dataset, df], ignore_index=True)
    return dataset


def get_host_ip(df):
    all_ips = pd.concat([df['SRC_IP'], df['DST_IP']])
    ip_counts = all_ips.value_counts()

    # Find IPs that occur in every flow
    ip_in_every_flow = ip_counts[ip_counts == len(df)].idxmax()
    return pd.Series({'host_ip': ip_in_every_flow})


def get_common_ips(df):
    unique_dst_ips = df.groupby(['family', 'sample'])['DST_IP'].unique()
    host_ips = df.groupby(['family', 'sample'], group_keys=True)[['SRC_IP', 'DST_IP']].apply(get_host_ip)

    unique_dst_ips = pd.merge(unique_dst_ips, host_ips, how='outer', on=['family', 'sample']).explode('DST_IP')
    #filtered_unique_dst_ips.to_csv('hosts_and_dsts.csv')

    before = len(unique_dst_ips)
    unique_dst_ips = unique_dst_ips[unique_dst_ips['DST_IP'] != unique_dst_ips['host_ip']]
    print(f'Removed {before - len(unique_dst_ips)} host IPs')

    result = unique_dst_ips.reset_index().groupby('DST_IP')['family'].nunique().reset_index()
    result = result[result['family'] > 1]
    print(result['DST_IP'])
    #result.to_csv('uniq_dst_ips.v2.csv', index=False)


def filter_common_ips(df, common_ips):
    df = df[(~df['DST_IP'].isin(set(common_ips))) & (~df['SRC_IP'].isin(set(common_ips)))]
    return df


def filter_DNS(df, filter_names):
    top1m = pd.read_csv(filter_names, names=['index', 'domain'], usecols=['domain'])['domain']
    keep_domains = [
        r'^mail\..*',
        r'^smtp\..*',
        r'^webmail\..*',

        r'drive\.google\.com$',
        r'steamcommunity\.com$',
        r't\.me$',
        r'api\.ip\.sb$',
        r'api\.ipify\.org$',
        r'api\.myip\.com$',
        r'api\.steampowered\.com$',
        r'api\.telegram\.org$',
        r'apis\.roblox\.com$',
        r'bitbucket\.org$',
        r'discord\.(com|gg)$',
        r'drive\.usercontent\.google\.com$',
        r'example\.org$',
        r'freegeoip\.app$',
        r'g\.api\.mega\.co\.nz$',
        r'gateway\.discord\.gg$',
        r'geolocation-db\.com$',
        r'geolocation\.onetrust\.com$',
        r'geoplugin\.net$',
        r'gofile\.io$',
        r'hbx\.media\.net$',
        r'i\.imgur\.com$',
        r'ip-api\.com$',
        r'ip-info\.ff\.avast\.com$',
        r'ipbase\.com$',
        r'ipinfo\.io$',
        r'iplogger\.org$',
        r'mediafire\.com$',
        r'mega\.nz$',
        r'onedrive\.live\.com$',
        r'pastebin\.com$',
        r'pool\.hashvault\.pro$',
        r'tinyurl\.com$',
        r'.*tlauncher\.org$',
        r'whatismyipaddress\.com$',
        r'www\.dropbox\.com$',
        r'www\.mediafire\.com$',
        r'www\.mediafiredls\.com$',
        r'www\.myexternalip\.com$',

        r'2makestorage\.com$',

        # from URLhaus database
        r'activetykes\.shop$',
        r'distro\.ibiblio\.org$',
        r'dl\.dropboxusercontent\.com$',
        r'(.*\.)contabostorage\.com$$',
        r'firebasestorage\.googleapis\.com$',
        #r'github\.com$',
        r'ia803402\.us\.archive\.org$',
        r'paste\.ee$',
        r'res\.cloudinary\.com$',
        r'sin1\.contabostorage\.com$',
        r'static1\.squarespace\.com$',
        r'update\.drp\.su$',
        r'update\.itopvpn\.com$',
        r'web\.archive\.org$',
    ]
    # htlb.casalemedia.com
    # TODO keep dns server names?

    combined_regex = '|'.join(f'({pattern})' for pattern in keep_domains)
    top1m = top1m[~top1m.str.contains(combined_regex, regex=True)]
    top1m.to_csv('domains-to-remove.csv', index=False)

    index = df['DNS_NAME'].isin(set(top1m))

    removed = df[index]
    removed['DNS_NAME'].to_csv('out/removed_dns.csv', index=False)
    return df[~index]


def filter_rDNS(df) -> pd.DataFrame:
    """
    Filter reverse DNS requests occurring in multiple families
    """
    df['DNS_NAME'] = df['DNS_NAME'].map(lambda x: x.rstrip('.in-addr.arpa'), na_action='ignore')

    rdnsdf = df[df['DNS_QTYPE'] == 12]
    common_names = rdnsdf.groupby('DNS_NAME')['family'].nunique()

    index = (df['DNS_QTYPE'] == 12) & (df['DNS_NAME'].isin(set(common_names[common_names > 1].index)))

    removed = df[index]
    removed.to_csv('out/removed_rdns.csv', index=False)
    return df[~index]


def run(dset_path) -> pd.DataFrame:
    print('Loading dataset')
    df = load_dataset(dset_path, store=True)
    print(f'Loaded {len(df)} rows (flows)')
    print(df.columns)

    prev_len = len(df)

    print('Filtering common IPs')
    common_ips = pd.read_csv('filter_ips.csv')
    df = filter_common_ips(df, common_ips['DST_IP'])
    #print(df.groupby(['family', 'sample']).size())
    print(f'Removed {prev_len - len(df)} rows')

    prev_len = len(df)

    print('Filtering DNS requests')
    df = filter_DNS(df, '../data/top-1m.csv')
    print(f'Removed {prev_len - len(df)} rows')

    prev_len = len(df)

    print('Filtering common rDNS requests')
    df = filter_rDNS(df)
    print(f'Removed {prev_len - len(df)} rows')
    return df


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print('Usage: python clean_dataset.py <in-path> <out-path>')
        sys.exit(1)

    path = sys.argv[1]
    df = run(path)
    df.to_csv('out/dataset.csv', index=False)
