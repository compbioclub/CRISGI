import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from pyvis.network import Network
import networkx as nx
import matplotlib.cm as cm
import matplotlib.colors as mcolors
from src.util import set_adata_obs

def relation_score(sniee_obj, groupby=None):
    edata = sniee_obj.edata
    df = edata.obs.copy()
    df['score'] = edata[:, edata.uns['common_trend_relations']].layers['pearson_entropy'].sum(axis=1)
    if groupby is None:
        groupby = sniee_obj.groupby
    sns.lineplot(df, x='time', y='score', hue=groupby)
    plt.show()

def investigate_relation(sniee_obj, relation, groupby='group', 
                         score_types=['entropy', 'prob', 'bg_net'],
                         figsize = (10, 12)):
    adata = sniee_obj.adata
    edata = sniee_obj.edata
    methods = sniee_obj.relation_methods

    nrow, ncol= len(score_types)*2+1, len(methods)
    if ncol < 3:
        ncol = 3
    fig = plt.figure(figsize=figsize)
    fig.suptitle(relation)

    sorted_samples = adata.obs.sort_values(by=[groupby, 'time']).index.tolist()

    header = 'log1p'
    genes = relation.split('_')
    plt.subplot(nrow, ncol, 1)
    df = pd.DataFrame(adata[sorted_samples, genes].layers[header].T, index=genes)
    sns.heatmap(df, cmap='coolwarm')

    df_list = []
    for i, gene in enumerate(genes):
        sadata = adata[:, gene]
        
        val = sadata.layers[header].reshape(-1).tolist()
        df = sadata.obs[[groupby, 'time']]
        df[header] = val
        df['gene'] = gene
        df_list.append(df)
    df = pd.concat(df_list)

    plt.subplot(nrow, ncol, 2)
    sns.boxplot(df, x='gene', y=header, hue=groupby, legend=False)

    plt.subplot(nrow, ncol, 3)
    sns.lineplot(df, x='time', y=header, hue=groupby, units='gene', estimator=None, legend=False)


    for i, method in enumerate(methods):
        sedata = edata[:, relation]

        for j, score_type in enumerate(score_types):
            header = f'{method}_{score_type}'
            val = sedata.layers[header].reshape(-1).tolist()
            set_adata_obs(sedata, header, val)

            n = i+1+ncol+j*2*ncol
            plt.subplot(nrow, ncol, n)
            sns.boxplot(sedata.obs, x=groupby, y=header, hue=groupby, legend=False)

            n += ncol
            plt.subplot(nrow, ncol, n)
            sns.lineplot(sedata.obs, x='time', y=header, hue=groupby, legend=False)
            #sns.lineplot(sedata.obs, x='time', y=relation, units='Sample', hue=groupby, estimator=None)


    plt.tight_layout()
    plt.show()


def pathway_dynamic(sniee_obj, method='all', p_cutoff=0.05, n_top_pathway=20):
    df = sniee_obj.edata.uns[f'{method}_enrich_df'] 
    df = df[df['Adjusted P-value'] < p_cutoff]

    for gene_set in df['Gene_set'].unique():
        tmp = df[df['Gene_set'] == gene_set]
        tmp = pd.pivot(tmp[['top_n', 'Term', 'top_n_ratio']].drop_duplicates(), index='Term', columns='top_n', values='top_n_ratio')
        idx = tmp.sum(axis=1).sort_values(ascending=False).index
        tmp = tmp.T[idx].T.head(n_top_pathway)
        sns.heatmap(tmp, yticklabels=1)
        plt.title(gene_set)
        plt.show()  


def draw_gene_network(sniee_obj, method='pearson', n_top_relations=100,
                      cmap='viridis', html_fn="gene_network.html"):
    net = Network(notebook=True, cdn_resources='in_line')
    
    df = sniee_obj.edata.uns['rank_genes_groups']
    df = df[df.method == method]
    df.index = df.names
    gene2pvals_adj = df['pvals_adj'].to_dict()
    gene2pvals = df['pvals'].to_dict()
    gene2scores = df['scores'].to_dict()
    gene2logfoldchanges = df['logfoldchanges'].to_dict()
    
    relation_list = sniee_obj.edata.uns[f'{method}_trend_relations'][:n_top_relations]

    weights = [ gene2scores[relation] for relation in relation_list]
    norm = mcolors.Normalize(vmin=min(weights), vmax=max(weights), clip=True)
    cmap = cm.get_cmap(cmap)

    for relation in relation_list:
        gene1, gene2 = relation.split('_')

        net.add_node(gene1, label=gene1)
        net.add_node(gene2, label=gene2)        

        score = gene2scores[relation]
        rgba_color = cmap(norm(score))
        hex_color = mcolors.rgb2hex(rgba_color)
        title = f'scores: {gene2scores[relation]:.2f}\n'
        title += f'logfoldchanges: {gene2logfoldchanges[relation]:.2f}\n'
        title += f'pvals: {gene2pvals[relation]:.2f}\n'
        title += f'pvals_adj: {gene2pvals_adj[relation]:.2f}\n'
        net.add_edge(gene1, gene2, value=score, 
                     title=title, 
                     color=hex_color)
    # Generate the network
    net.show(html_fn)
