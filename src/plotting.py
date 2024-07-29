import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from pyvis.network import Network
import networkx as nx
import matplotlib.cm as cm
import matplotlib.colors as mcolors
from src.util import set_adata_obs, print_msg
import os, random


def relation_score(sniee_obj, method='pearson', test_type='TER', groupby=None,
                   relations=None, unit_header='subject',
                   title='', out_prefix='test',
                   ax=None):
    edata = sniee_obj.edata
    df = edata.obs.copy()
    if relations is None:
        relations = edata.uns[f'{method}_{test_type}']
    else:
        relations = [x for x in relations if x in edata.var_names]

    df['score'] = edata[:, relations].layers[f'{method}_entropy'].sum(axis=1)
    if groupby is None:
        groupby = sniee_obj.groupby
    sns.lineplot(df, x='time', y='score', hue=groupby, #units=unit_header, estimator=None,
                 ax=ax)
    
    title += f'{sniee_obj.dataset} {method} entropy score of {len(relations)} {test_type}s '
    if ax is not None:
        ax.set_title(title)
    elif out_prefix:
        plt.title(title)
        plt.savefig(f'{out_prefix}_{title}_relation_score.png')
        plt.show()
    else:
        plt.title(title)
        plt.show()

def get_landscape_score(sniee_obj, relations=None, 
                        method='pearson', test_type='TER',
                        subject_header='subject'):
    edata = sniee_obj.edata

    if relations is None:
        relations = edata.uns[f'{method}_{test_type}']
    else:
        relations = [x for x in relations if x in edata.var_names]

    subjects = edata.obs[subject_header].unique()
    times = np.sort(edata.obs['time'].unique())
    time2i = {x:i for i, x in enumerate(times)}
    print('subjects', len(subjects), 'times', len(times))
    for i, subject in enumerate(subjects):
        sedata = edata[edata.obs[subject_header] == subject]

        t_is = [time2i[t] for t in sedata.obs['time']]
        X = np.empty((len(relations), len(times)))
        X[:] = np.nan
        X[:, t_is] = sedata[:, relations].layers[f'{method}_entropy'].T
        df = pd.DataFrame(X, columns=times, index=relations)
        df.to_csv(f'{sniee_obj.out_dir}/{subject}_{method}_{test_type}{len(relations)}_landscape.csv')

        #sns.heatmap(df, cmap='RdYlBu_r', robust=True)
        #plt.title(f'delta entropy score for subject {subject}')
        #plt.show()


def generate_landscape_images(folder_path, output_path, robust=False, scale=False,
                              rep_n=10, random_seed=0,
                              figsize=(5,5), dpi=500):
    """
    Parameters:
    folder_path (str): Path to the folder containing CSV files.
    output_path (str): Path to the folder where PNG files will be saved.
    robust (bool): If True, use a robust colormap. Default is False.
    scale (bool): If True, scale all heatmaps to the same color range. Default is False.
    """
    # Get all CSV file names in the folder
    csv_files = [f for f in os.listdir(folder_path) if f.endswith('.csv')]

    if scale:
        # Initialize variables to store global min and max values
        global_min = float('inf')
        global_max = float('-inf')

        # Read each CSV file to determine the global min and max values
        data_frames = []
        for file_name in csv_files:
            file_path = os.path.join(folder_path, file_name)
            df = pd.read_csv(file_path)
            data_frames.append(df)

            # Flatten the values to determine the global min and max
            df_values = df.drop(columns=['Unnamed: 0']).values.flatten()
            current_min = df_values.min()
            current_max = df_values.max()

            if current_min < global_min:
                global_min = current_min
            if current_max > global_max:
                global_max = current_max

    # Read and process each CSV file
    for file_name in csv_files:
        file_path = os.path.join(folder_path, file_name)
        df = pd.read_csv(file_path, index_col=0)

        df = df.dropna(how='all', axis=0)
        df = df.dropna(how='all', axis=1)

        random.seed(random_seed)
        for i in range(rep_n):
            if i > 0:
                df = df.sample(frac=1)
            # Create heatmap without displaying data values
            plt.figure(figsize=figsize)
            if scale:
                heatmap = sns.heatmap(df.astype(float), annot=False, cmap='RdYlBu_r', cbar=False, robust=robust, vmin=global_min, vmax=global_max)
            else:
                heatmap = sns.heatmap(df.astype(float), annot=False, cmap='RdYlBu_r', cbar=False, robust=robust)
            
            heatmap.set_xticks([])
            heatmap.set_yticks([])
            heatmap.set_xlabel('')
            heatmap.set_ylabel('')
            
            # Remove title
            plt.title('')
            
            # Save the heatmap as a PNG file with a transparent background
            output_file = os.path.join(output_path, os.path.splitext(file_name)[0] + f'_rep{i}.png')
            plt.savefig(output_file, transparent=True, bbox_inches='tight', pad_inches=0, dpi=dpi)
            plt.close()




def relation_score_boxplot(sniee_obj, groupby=None, 
                           relation_type='common_TER',
                           method='pearson',
                           title=''):
    edata = sniee_obj.edata
    df = edata.obs.copy()
    df['score'] = edata[:, edata.uns[relation_type]].layers[f'{method}_entropy'].sum(axis=1)
    if groupby is None:
        groupby = sniee_obj.groupby
    sns.lineplot(df, x='time', y='score', hue=groupby)
    plt.title(title)
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


def pathway_dynamic(sniee_obj, method='pearson', test_type='TER',
                    p_adjust=True, p_cutoff=0.05, n_top_pathway=30, 
                    n_top_relations=500):
    df = sniee_obj.edata.uns[f'{method}_{test_type}_enrich_df'] 

    if p_adjust:
        df = df[df['Adjusted P-value'] < p_cutoff]
    else:
        df = df[df['P-value'] < p_cutoff]
    df = df[df['top_n'] <= n_top_relations]

    for gene_set in set(df['Gene_set'].unique()):
        tmp = df[df['Gene_set'] == gene_set]
        tmp = pd.pivot(tmp[['top_n', 'Term', 'top_n_ratio']].drop_duplicates(), index='Term', columns='top_n', values='top_n_ratio')
        idx = tmp.sum(axis=1).sort_values(ascending=False).index
        tmp = tmp.T[idx].T.head(n_top_pathway)
        sns.heatmap(tmp, yticklabels=1)
        plt.title(gene_set)
        plt.show()  


def draw_gene_network(sniee_obj, per_group,
                      method='pearson', test_type='TER',
                      n_top_relations=100,
                    cmap='viridis'):
    net = Network(notebook=True, cdn_resources='in_line')
    df = sniee_obj.edata.uns[f'{per_group}_DER']
    df = df[(df.method == method) & df['DER']]
    df.index = df.names
    gene2pvals_adj = df['pvals_adj'].to_dict()
    gene2pvals = df['pvals'].to_dict()
    gene2scores = df['scores'].to_dict()
    gene2logfoldchanges = df['logfoldchanges'].to_dict()
    
    relation_list = sniee_obj.edata.uns[f'{method}_{test_type}'][:n_top_relations]

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

    html_fn = f'{sniee_obj.out_dir}/{method}_{test_type}{len(relation_list)}_gene_network.html'
    print_msg(f'[Output] Relation network has saved to {html_fn}.')

    net.show(html_fn)


def draw_relation_heatmap(sniee_obj, method='pearson', 
                          n_top_relations=100,
                          cmap='viridis'):    
    
    relation_list = sniee_obj.edata.uns[f'{method}_TER'][:n_top_relations]
    for relation in relation_list:
        gene1, gene2 = relation.split('_')
        # to be continued