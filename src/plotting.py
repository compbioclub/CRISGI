import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

from src.util import set_adata_obs

def investigate_relation(dghub_obj, relation, groupby='group', 
                         methods=['pearson', 'spearman', 'pos_coexp', 'neg_coexp'],
                         score_types=['entropy', 'prob', 'bg_net'],
                         figsize = (10, 12)):
    adata = dghub_obj.adata
    edata = dghub_obj.edata

    nrow, ncol= len(score_types)*2+1, len(methods)
    
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

            n = i+5+j*2*ncol
            plt.subplot(nrow, ncol, n)
            sns.boxplot(sedata.obs, x=groupby, y=header, hue=groupby, legend=False)

            n += 4
            plt.subplot(nrow, ncol, n)
            sns.lineplot(sedata.obs, x='time', y=header, hue=groupby, legend=False)
            #sns.lineplot(sedata.obs, x='time', y=relation, units='Sample', hue=groupby, estimator=None)


    plt.tight_layout()
    plt.show()