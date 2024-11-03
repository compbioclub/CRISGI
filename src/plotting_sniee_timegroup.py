import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import random

import src.plotting as pl
from src.util import print_msg

def relation_score_bar(obj, per_group=None, method='prod', test_type='DER',
                   relations=None, cmap='Spectral',
                   title='', out_prefix='test'):
    edata = obj.edata
    myper_group = per_group

    df_list = []
    for per_group in obj.groups:
        if myper_group is not None and myper_group != per_group:
            continue
        key = f'{method}_{obj.groupby}_{per_group}_{test_type}'
        if relations is None:
            per_relations = edata.uns[key]
        else:
            per_relations = [x for x in relations if x in edata.var_names]

        if not per_relations:
            continue
        for ref_group in obj.groups:
            if ref_group == per_group:
                continue
            df = edata.obs.copy()
            df['avg(score)'] = edata[:, per_relations].layers[f'{obj.ref_time}_{method}_entropy'].mean(axis=1)
            df['sum(score)'] = edata[:, per_relations].layers[f'{obj.ref_time}_{method}_entropy'].sum(axis=1)
            df['per_group'] = f'{ref_group}->{per_group} ({len(per_relations)} {test_type}s)'
            df = df.sort_values(by='sum(score)')
            df_list.append(df)      

            sub_title = title + f'{obj.dataset} ref {ref_group} per {per_group}\n{method} entropy score of {len(per_relations)} {test_type}s'
            ax = sns.barplot(df, x='test', hue=obj.groupby, y='sum(score)')
            ax.set_xticks([])
            ax.set_xlabel('')            
            plt.title(sub_title)
            plt.show()
            if out_prefix:
                plt.savefig(f'{out_prefix}_{title}_relation_score_barplot.png'.replace('\n', ' '))
        
def relation_score_line(obj, per_group=None, method='prod', test_type='DER',
                   relations=None, cmap='Spectral',
                   title='', out_prefix='test'):
    edata = obj.edata
    myper_group = per_group

    df_list = []
    for per_group in obj.groups:
        if myper_group is not None and myper_group != per_group:
            continue
        key = f'{method}_{obj.groupby}_{per_group}_{test_type}'
        if relations is None:
            per_relations = edata.uns[key]
        else:
            per_relations = [x for x in relations if x in edata.var_names]

        if not per_relations:
            continue
        for ref_group in obj.groups:
            if ref_group == per_group:
                continue
            df = edata.obs.copy()
            df['sum(score)'] = edata[:, per_relations].layers[f'{obj.ref_time}_{method}_entropy'].sum(axis=1)
            df['per_group'] = f'{ref_group}->{per_group} ({len(per_relations)} {test_type}s)'
            df_list.append(df)      

            sub_title = title + f'{obj.dataset} ref {ref_group} per {per_group}\n{method} entropy score of {len(per_relations)} {test_type}s'
            ax = sns.lineplot(df, x=obj.groupby, y='sum(score)')
            plt.axvline(x=per_group, color='purple', linestyle=':')
            plt.title(sub_title)
            plt.show()
            if out_prefix:
                plt.savefig(f'{out_prefix}_{title}_relation_score_boxplot.png'.replace('\n', ' '))
        
    df = pd.concat(df_list)
    title += f'{obj.dataset} per_group\n{method} entropy score of {obj.groupby} {test_type}s'
    ax = sns.boxplot(df, x=obj.groupby, hue='per_group', y='sum(score)')
    ax.get_legend().remove()

    plt.title(title)
    plt.show()
    if out_prefix:
        plt.savefig(f'{out_prefix}_{title}_relation_score_lineplot.png'.replace('\n', ' '))

def relation_score_box(obj, per_group=None, method='prod', test_type='DER',
                   relations=None,  cmap='Spectral',
                   title='', out_prefix='test'):
    edata = obj.edata
    myper_group = per_group
    
    for per_group in obj.groups:
        if myper_group is not None and myper_group != per_group:
            continue        
        key = f'{method}_{obj.groupby}_{per_group}_{test_type}'
        if relations is None:
            per_relations = edata.uns[key]
        else:
            per_relations = [x for x in relations if x in edata.var_names]
        if not per_relations:
            continue
        
        for ref_group in obj.groups:
            if ref_group == per_group:
                continue
            df = pd.DataFrame()
            df[obj.groupby] = edata.obs[obj.groupby].tolist()*len(per_relations)
            df['score'] = edata[:, per_relations].layers[f'{obj.ref_time}_{method}_entropy'].toarray().T.reshape(-1)
            sub_title = title + f'{obj.dataset} ref {ref_group} per {per_group}\n {method} entropy score of {len(per_relations)} {test_type}s'
            ax = sns.boxenplot(df, x=obj.groupby, y='score', hue=obj.groupby,
                            cmap=cmap)
            plt.title(sub_title)
            plt.show()
            if out_prefix:
                plt.savefig(f'{out_prefix}_{title}_relation_score_boxplot.png'.replace('\n', ' '))

def get_relation_score(obj, per_group=None, relations=None, groups=None,
                        method='prod', test_type='DER',
                        out_dir=None):
    edata = obj.edata
    myper_group = per_group
    if groups is None:
        groups = obj.groups
    for per_group in groups:
        if myper_group is not None and myper_group != per_group:
            continue  
        key = f'{method}_{obj.groupby}_{per_group}_{test_type}'
        if relations is None:
            per_relations = edata.uns[key]
        else:
            per_relations = [x for x in relations if x in edata.var_names]

        if not per_relations:
            continue

        from sklearn.preprocessing import normalize
        import numpy as np

        samples = edata.obs[edata.obs[obj.groupby] == per_group].index
        X = edata[samples, per_relations].layers[f'{obj.ref_time}_{method}_entropy'].T.toarray()
        X = normalize(X, axis=1)
        sns.heatmap(pd.DataFrame(
            X, columns=samples+per_group,
            index=per_relations
            ))
        #print_msg(f'[Output] The subject {subject} {method} {test_type}{len(relations)} entropy scores are saved to:\n{fn}')
        if out_dir is None:
            out_dir = obj.out_dir
        fn = f'{out_dir}/{subject}_{method}_{groupby}_{per_group}_{test_type}{len(relations)}_relation_score.csv'
        df.to_csv(fn)
        
        #sns.heatmap(df, cmap='RdYlBu_r', robust=True)
        plt.title(f'delta entropy score for {obj.groupby} {per_group} {len(per_relations)} {test_type}s')
        plt.show()


def generate_relation_score_images(folder_path, output_path, robust=False, scale=False,
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
    csv_files = [f for f in os.listdir(folder_path) if f.endswith('relation_score.csv')]

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
            print_msg(f'[Output] The relation score image is saved to:\n{output_file}')

            plt.close()


def draw_gene_network(*args, **kwargs):
    pl.draw_gene_network(*args, **kwargs)


def pathway_dynamic(obj, groups=None, *args, **kwargs):
    if groups is None:
        groups = obj.groups
    for per_group in groups:
        pl.pathway_dynamic(obj, per_group=per_group, *args, **kwargs)