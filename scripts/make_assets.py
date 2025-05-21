"""
Make assets for the thesis

The first argument is the path to the dataset CSV file.
TODO Right now the output path is hardcoded to out/
"""

import sys

import matplotlib.pyplot as plt
import networkx as nx
import pandas as pd


columns = {
    'AGENTTESLA' : 'Agent Tesla',
    'ARDAMAX' : 'Ardamax',
    'ASYNCRAT' : 'AsyncRAT',
    'AZORULT' : 'Azorult',
    'BAZARBACKDOOR' : 'BazarBackdoor',
    'BLACKMOON' : 'Blackmoon',
    'COBALTSTRIKE' : 'Cobalt Strike',
    'CYBERGATE' : 'CyberGate',
    'DARKCOMET' : 'DeathRansom',
    'DISCORDRAT' : 'DiscordRat',
    'DRIDEX' : 'Dridex',
    'EMOTET' : 'Emotet',
    'FLOXIF' : 'Flofix',
    'FORMBOOK' : 'FormBook',
    'GOZI_IFSB' : 'Gozi IFSB',
    'GULOADER' : 'GuLoader',
    'GURCU' : 'gurcu',
    'HAWKEYE' : 'HawkEye',
    'HAWKEYE_REBORN' : 'HawkEye Reborn',
    'ICEDID' : 'IcedID',
    'KPOT' : 'KPOT',
    'LOKIBOT' : 'LokiBot',
    'MASSLOGGER' : 'MASS Logger',
    'MATIEX' : 'Matiex',
    'METASPLOIT' : 'Metasploit',
    'MODILOADER' : 'ModiLoader',
    'NANOCORE' : 'Nanocore',
    'NETWIRE' : 'NetWire',
    'NJRAT' : 'NjRAT',
    'PHOBOS' : 'Phobos',
    'PONY' : 'Pony',
    'QAKBOT' : 'QakBot',
    'QNODESERVICE' : 'QNodeService',
    'RACCOON' : 'Raccoon',
    'REMCOS' : 'Remcos',
    'REVENGERAT' : 'Revenge RAT',
    'SALITY' : 'Sality',
    'SLIVER' : 'Sliver',
    'SMOKELOADER' : 'SmokeLoader',
    'SODINOKIBI' : 'Sodinokibi (REvil)',
    'TRICKBOT' : 'TrickBot',
    'UPATRE' : 'Upatre',
    'VIDAR' : 'Vidar',
    'WANNACRY' : 'WannaCry',
}

type_map = {
    'Agent Tesla' : 'spyware/trojan',
    'Ardamax' : 'keylogger',
    'AsyncRAT' : 'RAT',
    'Azorult' : 'RAT',
    'BazarBackdoor' : 'backdoor',
    'Blackmoon' : 'banker trojan',
    'Cobalt Strike' : 'keylogger',
    'CyberGate' : 'RAT',
    'DarkCommet' : 'RAT',
    'DeathRansom' : 'ransomware',
    'DiscordRat' : 'RAT',
    'Dridex' : 'credential stealer',
    'Emotet' : 'banking Trojan',
    'Flofix' : 'virus, backdoor',
    'FormBook' : 'stealer',
    'Gozi IFSB' : 'banking trojan',
    'GuLoader' : 'downloader',
    'gurcu' : 'stealer',
    'HawkEye' : 'keylogger/stealer',
    'HawkEye Reborn' : 'keylogger/stealer',
    'IcedID' : 'banking trojan/RAT',
    'KPOT' : 'stealer/trojan',
    'LokiBot' : 'ransomware/stealer',
    'MASS Logger' : 'credential stealer',
    'Matiex' : 'keylogger',
    'Metasploit' : 'backdoor, Trojan',
    'ModiLoader' : 'loader',
    'Nanocore' : 'RAT',
    'NetWire' : 'RAT',
    'NjRAT' : 'RAT',
    'Phobos' : 'ransomware',
    'Pony' : 'password stealer',
    'QakBot' : 'infostealer',
    'QNodeService' : 'stealer, RCE',
    'Raccoon' : 'stealer',
    'Remcos' : 'backdoor, remote access',
    'Revenge RAT' : 'RAT',
    'Sality' : 'virus, multiple',
    'Sliver' : 'C2 system, commercial',
    'SmokeLoader' : 'backdoor',
    'Sodinokibi (REvil)' : 'ransomware, multiple',
    'TrickBot' : 'banker trojan',
    'Upatre' : 'downloader',
    'Vidar' : 'stealer',
    'WannaCry' : 'ransomware',
}

common_style = {'font_size': 12}


def make_latex_dset_tab(df, path='out/tab.tex'):
    by_fam = df.groupby(['family']).size().rename('flows')
    samples = df.groupby('family')['sample'].nunique().rename('samples')
    tab = pd.concat([by_fam, samples], axis=1)
    tab = tab[tab['flows'] > 100]
    tab.sort_values(by='family', inplace=True, key=lambda col: col.str.lower())
    tab = tab.reset_index()
    # tab['type'] = tab['family'].map(type_map)
    tab = tab.reindex(columns=['family', 'samples', 'flows'])
    tab.loc['Total'] = tab.sum(numeric_only=True)
    tab = tab.astype({'samples': 'int', 'flows': 'int'})
    #print(tab['flows'] / tab['samples'])
    tab.to_latex(path, index=False)


flow_id = ['SRC_IP', 'DST_IP', 'SRC_PORT', 'DST_PORT', 'PROTOCOL']


def node_key(row):
    return f"{row['SRC_IP']}:{row['SRC_PORT']}\n{row['DST_IP']}:{row['DST_PORT']}-{row['PROTOCOL']}"


ip2hostname = {
    '8.8.8.8': '8.8.8.8',
    '149.154.167.99': 't.me',
    '104.131.166.122': 'C2-A',
    '77.91.101.71' : 'C2-B',
    '10.127.1.149' : 'host',
}


def make_sun_repr(df):
    G = nx.Graph()

    for _, row in df.iterrows():
        G.add_edge(ip2hostname.get(row['SRC_IP'], row['SRC_IP']), ip2hostname.get(row['DST_IP'], row['DST_IP']))

    plt.figure(figsize=(4, 4))
    pos = nx.circular_layout(G)
    nx.draw(G, pos, with_labels=True, node_color='lightsalmon', node_size=2500, **common_style)
    plt.savefig("sun_repr.pdf", format="pdf", bbox_inches='tight', pad_inches=0)
    #plt.show()


def make_repr_1(df):
    G = nx.DiGraph()
    node_color = []

    def add_host_node(node):
        if node not in G:
            G.add_node(node)
            node_color.append('lightsalmon')

    def add_flow_node(node):
        G.add_node(node)
        node_color.append('lightblue')

    prev = None

    for _, row in df.iloc[1:].iterrows():
        add_host_node(row['SRC_IP'])
        add_host_node(row['DST_IP'])

        flow = node_key(row)
        add_flow_node(flow)
        G.add_edge(row['SRC_IP'], flow)
        G.add_edge(flow, row['DST_IP'])

        if prev:
            G.add_edge(flow, prev)
            G.add_edge(prev, flow)

        prev = flow

    plt.figure(figsize=(6, 6))
    pos = nx.kamada_kawai_layout(G)
    nx.draw(G, pos, node_size=2500, **common_style)
    #nx.draw(G, pos, with_labels=False, node_color=node_color, node_size=2500, **common_style)
    # nx.draw_networkx_labels(G, pos, labels=ip2hostname, font_color='black', **common_style)
    # nx.draw_networkx_labels(G, pos, font_color='black', **common_style)
    plt.savefig("repr1.pdf", format="pdf", dpi=300, bbox_inches='tight', pad_inches=0)
    #plt.show()



def make_repr_2(df):
    G = nx.DiGraph()
    labels = {}

    def add_label(node):
        if node['DST_IP'].startswith('10.127.1.'):
            labels[node_key(node)] = ip2hostname.get(node['SRC_IP'], node['SRC_IP']) + f'\n:{node["SRC_PORT"]}'
        else:
            labels[node_key(node)] = ip2hostname.get(node['DST_IP'], node['DST_IP']) + f'\n:{node["DST_PORT"]}'

    df = df.sort_values(by='TIME_FIRST', ascending=True).reset_index(drop=True)
    print(df)

    prev = df.iloc[0]
    add_label(prev)
    for _, curr in df.iloc[1:].iterrows():
        G.add_edge(node_key(prev), node_key(curr), length=3)
        add_label(curr)
        prev = curr

    curved_edges = []
    reverse = df.sort_values(by='TIME_LAST', ascending=True)
    prev = reverse.iloc[0]
    for _, curr in reverse.iloc[1:].iterrows():
        # I forgot what .name is. It's probably the index of the row in the dataframe
        # and it's used as a comparator of index (position) of the row in the dataframe
        if curr.name - prev.name > 1:
            curved_edges.append((node_key(curr), node_key(prev)))
            G.add_edge(node_key(curr), node_key(prev))

        if curr.name - prev.name < -1:
            curved_edges.append((node_key(prev), node_key(curr)))
            G.add_edge(node_key(prev), node_key(curr))
        prev = curr

    plt.figure(figsize=(12, 2))
    #pos = nx.shell_layout(G)
    pos = {node : (i, 0) for i, node in zip(range(0, len(G.nodes()) * 3, 3), G.nodes())}
    print(pos)
    #nx.draw(G, pos, with_labels=False, node_color='lightblue', node_size=2500)
    nx.draw_networkx_nodes(G, pos, node_color='lightblue', node_size=2500)
    nx.draw_networkx_labels(G, pos, labels=labels, font_color='black', **common_style)

    straight_edges = list(set(G.edges()) - set(curved_edges))
    nx.draw_networkx_edges(G, pos, edgelist=straight_edges, arrows=True, node_size=2500)
    arc_rad = 0.45
    nx.draw_networkx_edges(G, pos, edgelist=curved_edges, connectionstyle=f'arc3, rad = {arc_rad}', arrows=True, node_size=2500)

    plt.box(False)
    plt.tight_layout()
    plt.savefig("repr2.pdf", format="pdf", dpi=300, bbox_inches='tight', pad_inches=0)
    plt.show()


if __name__ == '__main__':
    path = sys.argv[1]
    df = pd.read_csv(path)
    # df['family'].replace(columns, inplace=True)
    make_latex_dset_tab(df)
    sys.exit(0)

    #df = df[(df['family'] == 'Vidar') & (df['sample'] == '240720-fvt33sxfnq.behavioral1.csv')]
    df = df[df['sample'] == '240822-hpsdeaxcjm.behavioral2.csv']
    print(df)

    # FIXME: remove the DNS dups, workaround
    # graph_df = df.copy()
    graph_df = df
    # graph_df['IPS'] = graph_df.apply(lambda row: ''.join(sorted([row['DST_IP'], row['SRC_IP']])), axis=1)
    # graph_df['PORTS'] = graph_df.apply(lambda row: ''.join(sorted([str(row['SRC_PORT']), str(row['DST_PORT'])])), axis=1)
    # graph_df = graph_df.drop_duplicates(subset=['IPS', 'PORTS', 'DNS_NAME'])
    # print(graph_df[['SRC_IP', 'DST_IP', 'SRC_PORT', 'DST_PORT', 'DNS_NAME', 'PROTOCOL']])


    make_sun_repr(graph_df)
    make_repr_1(graph_df)
    make_repr_2(graph_df)

    graph_df = graph_df.sort_values(by='TIME_FIRST', ascending=True).reset_index(drop=True)
    graph_df['label'] = graph_df.apply(lambda x: ip2hostname.get(x['DST_IP'], x['DST_IP']), axis=1)
    graph_df['Start time'] = graph_df.index
    graph_df['End time'] = graph_df.sort_values(by='TIME_LAST', ascending=True).index
    graph_df.reindex(columns=['Start time', 'End time','DST_IP', 'DST_PORT', 'label'])
    graph_df[['Start time', 'End time','DST_IP', 'DST_PORT', 'label']].to_latex('out/repr-sample.tex', index=False)
