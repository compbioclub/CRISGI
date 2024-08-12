import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

import src.plotting as pl
from src.util import print_msg

def relation_score_line(sniee_obj, method='prod', test_type='DER',
                   relations=None, cmap='Spectral',
                   title='', out_prefix='test'):
    edata = sniee_obj.edata
    
    df_list = []
    for per_group in sniee_obj.groups:
        key = f'{method}_{sniee_obj.groupby}_{per_group}_{test_type}'
        if relations is None:
            per_relations = edata.uns[key]
        else:
            per_relations = [x for x in relations if x in edata.var_names]

        if not per_relations:
            continue

        df = edata.obs.copy()
        df['avg(score)'] = edata[:, per_relations].layers[f'{method}_entropy'].mean(axis=1)
        df['per_group'] = f'{per_group} ({len(per_relations)} {test_type}s)'
        df_list.append(df)      

        sub_title = title + f'{sniee_obj.dataset}\n{method} entropy score of {per_group} {len(per_relations)} {test_type}s'
        ax = sns.lineplot(df, x=sniee_obj.groupby, y='avg(score)')
        plt.axvline(x=per_group, color='purple', linestyle=':')
        plt.title(sub_title)
        plt.show()
        if out_prefix:
            plt.savefig(f'{out_prefix}_{title}_relation_score_boxplot.png'.replace('\n', ' '))


        
    df = pd.concat(df_list)
    title += f'{sniee_obj.dataset}\n{method} entropy score of {sniee_obj.groupby} {test_type}s'
    ax = sns.lineplot(df, x=sniee_obj.groupby, hue='per_group', y='avg(score)')
    ax.get_legend().remove()

    plt.title(title)
    plt.show()
    if out_prefix:
        plt.savefig(f'{out_prefix}_{title}_relation_score_lineplot.png'.replace('\n', ' '))

def relation_score_box(sniee_obj, method='prod', test_type='DER',
                   relations=None,  cmap='Spectral',
                   title='', out_prefix='test'):
    edata = sniee_obj.edata
    

    for per_group in sniee_obj.groups:
        key = f'{method}_{sniee_obj.groupby}_{per_group}_{test_type}'
        if relations is None:
            per_relations = edata.uns[key]
        else:
            per_relations = [x for x in relations if x in edata.var_names]

        if not per_relations:
            continue
        
        df = pd.DataFrame()
        df[sniee_obj.groupby] = edata.obs_names.tolist()*len(per_relations)
        df['avg(score)'] = edata[:, per_relations].layers[f'{method}_entropy'].toarray().T.reshape(-1)
        df['per_group'] = f'{per_group} ({len(per_relations)} {test_type}s)'

        sub_title = title + f'{sniee_obj.dataset}\n{method} entropy score of {per_group} {len(per_relations)} {test_type}s'
        ax = sns.boxenplot(df, x=sniee_obj.groupby, y='avg(score)',
                           hue=sniee_obj.groupby, cmap=cmap)
        plt.title(sub_title)
        plt.show()
        if out_prefix:
            plt.savefig(f'{out_prefix}_{title}_relation_score_boxplot.png'.replace('\n', ' '))

def get_relation_score(sniee_obj, relations=None, groups=None,
                        method='prod', test_type='DER',
                        out_dir=None):
    edata = sniee_obj.edata

    if groups is None:
        groups = sniee_obj.groups
    for per_group in groups:
        key = f'{method}_{sniee_obj.groupby}_{per_group}_{test_type}'
        if relations is None:
            per_relations = edata.uns[key]
        else:
            per_relations = [x for x in relations if x in edata.var_names]

        if not per_relations:
            print(per_group)
            continue

        X = edata[:, per_relations].layers[f'{method}_entropy'].T
        sns.heatmap(pd.DataFrame(
            X, columns=edata.obs_names,
            index=per_relations
                                 ))
        #print_msg(f'[Output] The subject {subject} {method} {test_type}{len(relations)} entropy scores are saved to:\n{fn}')

        #sns.heatmap(df, cmap='RdYlBu_r', robust=True)
        plt.title(f'delta entropy score for {sniee_obj.groupby} {per_group} {len(per_relations)} {test_type}s')
        plt.show()

def draw_gene_network(*args, **kwargs):
    pl.draw_gene_network(*args, **kwargs)


def pathway_dynamic(sniee_obj, groups=None, *args, **kwargs):
    if groups is None:
        groups = sniee_obj.groups
    for per_group in groups:
        pl.pathway_dynamic(sniee_obj, per_group=per_group, *args, **kwargs)