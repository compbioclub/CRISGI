import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from pyvis.network import Network
import matplotlib.cm as cm
import matplotlib.colors as mcolors
from src.util import set_adata_obs, print_msg



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


def pathway_dynamic(sniee_obj, per_group,method="pearson", test_type="TER",
                    p_adjust=True, p_cutoff=0.05, n_top_pathway=10, n_top_relations=500,
                    # Available options for piority_term: None, list of terms(specific pathway names)
                    piority_term=None,
                    # Available options for eval_para: 'top_n_ratio', 'overlap_ratio, 'P-value', 'Adjusted P-value', 'Odds Ratio', 'Combined Score', '-logP'
                    eval_para='top_n_ratio',
                    dataset_name=None,):
    df = sniee_obj.edata.uns[f"{method}_{sniee_obj.groupby}_{per_group}_{test_type}_enrich_df"]

    if p_adjust:
        df = df[df["Adjusted P-value"] < p_cutoff]
    else:
        df = df[df["P-value"] < p_cutoff]
    df = df[df["top_n"] <= n_top_relations]
    
    df["overlap_ratio"] = df["Overlap"].apply(
    lambda x: float(x.split("/")[0]) / float(x.split("/")[1])
    )
    
    if eval_para in ['Adjusted P-value', 'P-value']:
        ascending=True
        if piority_term is None:
            df["selected"] = 0
        else:
            df["selected"] = df["Term"].apply(
                lambda x: x
                not in piority_term
            )
    elif eval_para in ['Odds Ratio', 'top_n_ratio', 'overlap_ratio', 'Combined Score']:
        ascending=False
        if piority_term is None:
            df["selected"] = 1
        else:
            df["selected"] = df["Term"].apply(
                lambda x: x
                in piority_term
            )
    elif eval_para == '-logP':
        ascending=False
        if p_adjust == True:
            df['-logP'] = -np.log10(df['Adjusted P-value'])
        else:
            df['-logP'] = -np.log10(df['P-value'])
            
        if piority_term is None:
            df["selected"] = 1
        else:
            df["selected"] = df["Term"].apply(
                lambda x: x
                in piority_term
            )
    else:
        print_msg(f"Evaluation parameter {eval_para} is not supported.")
        return None
    
    for gene_set in set(df["Gene_set"].unique()):
        tmp = df[df["Gene_set"] == gene_set]
        tmp = pd.pivot(
            tmp[["top_n", "Term", eval_para, "selected"]].drop_duplicates(),
            index=["Term", "selected"],
            columns=["top_n"],
            values=[eval_para]
        )[eval_para]
        
        tmp_weight = tmp * np.array(np.sum(tmp.columns)/tmp.columns)
        
        # weights
        if ascending == True:
            tmp_weight = pd.DataFrame(tmp_weight.sum(axis=1) / (len(tmp.columns) - tmp.isna().sum(axis=1)),columns=['sum'])
        else:
            tmp_weight = pd.DataFrame(tmp_weight.sum(axis=1) * (len(tmp.columns) - tmp.isna().sum(axis=1)),columns=['sum'])
    
        tmp_weight = tmp_weight.sort_values(by=['sum'], ascending=ascending)
        tmp_weight['rank'] = np.arange(1,len(tmp_weight)+1)
        tmp['rank'] = tmp_weight['rank']
        tmp_weight = tmp_weight.sort_values(by=['selected','sum'], ascending=ascending)
        idx = tmp_weight.index.get_level_values(0)
        tmp = tmp.T[idx].T.head(n_top_pathway).reset_index().set_index('Term').drop(columns='selected')
        tmp['rank'] = tmp['rank'].astype(int)
        tmp.index = tmp.apply(lambda row: f"{row.name} (Rank {int(row['rank'])})", axis=1)
        tmp.drop(columns='rank', inplace=True)
        
        sns.heatmap(tmp, yticklabels=1)
        plt.title(f"{dataset_name}_{method}_{sniee_obj.groupby}_{per_group}_{test_type}s\n" + f"{gene_set} with {eval_para}")
        fn = f"{sniee_obj.out_dir}/{method}_{sniee_obj.groupby}_{per_group}_{test_type}_{gene_set}_{eval_para}_enrich_top_n.png"
        plt.savefig(fn,dpi=300,format='png',bbox_inches='tight')
        print_msg(f"[Output] The {method} {sniee_obj.groupby} {per_group} {test_type} {gene_set} {eval_para} top_n enrichment is saved to:\n{fn}")
        plt.show()

        
def draw_gene_network(sniee_obj, per_group,
                      method='pearson', test_type='TER',
                      n_top_relations=100,
                    cmap='viridis'):
    net = Network(notebook=True, cdn_resources='in_line')
    print(sniee_obj.edata)
    df = sniee_obj.edata.uns[f'{method}_{sniee_obj.groupby}_{per_group}_DER_df']
    df = df[(df.method == method) & df['DER']]
    df.index = df.names
    gene2pvals_adj = df['pvals_adj'].to_dict()
    gene2pvals = df['pvals'].to_dict()
    gene2scores = df['scores'].to_dict()
    gene2logfoldchanges = df['logfoldchanges'].to_dict()
    
    relation_list = sniee_obj.edata.uns[f'{method}_{sniee_obj.groupby}_{per_group}_{test_type}'][:n_top_relations]

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
    print_msg(f'[Output] Relation network has saved to:\n{html_fn}')

    net.show(html_fn)
