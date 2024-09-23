import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os, random

import src.plotting as pl
from src.util import print_msg


def relation_score_line(sniee_obj, per_group=None, 
                        method='pearson', test_type='TER', 
                        relations=None, unit_header='subject',
                   title='', out_prefix='test',
                   ax=None):
    edata = sniee_obj.edata
    myper_group = per_group
    for per_group in sniee_obj.groups:
        if myper_group is not None and per_group != myper_group:
            continue
        df = edata.obs.copy()
        if relations is None:
            key = f'{method}_{sniee_obj.groupby}_{per_group}_{test_type}'
            print(key)
            myrelations = edata.uns[key]
        else:
            myrelations = [x for x in relations if x in edata.var_names]
        print(relations)
        df['avg(score)'] = edata[:, myrelations].layers[f'{sniee_obj.ref_time}_{method}_entropy'].mean(axis=1)

        if unit_header is not None:
            print(df)
            sns.lineplot(df, x='time', y='avg(score)', hue=sniee_obj.groupby, 
                        units=unit_header, estimator=None,
                        ax=ax)
        else:
            sns.lineplot(df, x='time', y='avg(score)', hue=sniee_obj.groupby, 
                        ax=ax)
                    
        mytitle = title + f'{sniee_obj.dataset} per {per_group}\n{method} entropy score of {len(myrelations)} {test_type}s '
        if ax is not None:
            ax.set_title(mytitle)
        elif out_prefix:
            plt.title(mytitle)
            plt.savefig(f'{out_prefix}_{mytitle}_relation_score.png'.replace('\n', ' '))
            plt.show()
        else:
            plt.title(mytitle)
            plt.show()

def get_relation_score(sniee_obj, per_group, groupby=None, relations=None, 
                       method='pearson', test_type='TER',
                        subject_header='subject',
                        out_dir=None):
    edata = sniee_obj.edata

    if groupby is None:
        groupby = sniee_obj.groupby

    if relations is None:
        relations = edata.uns[f'{method}_{groupby}_{per_group}_{test_type}']
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
        if out_dir is None:
            out_dir = sniee_obj.out_dir
        fn = f'{out_dir}/{subject}_{method}_{groupby}_{per_group}_{test_type}{len(relations)}_relation_score.csv'
        df.to_csv(fn)
        print_msg(f'[Output] The subject {subject} {method} {groupby} {per_group} {test_type}{len(relations)} entropy scores are saved to:\n{fn}')

        #sns.heatmap(df, cmap='RdYlBu_r', robust=True)
        #plt.title(f'delta entropy score for subject {subject}')
        #plt.show()


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


def pathway_dynamic(*args, **kwargs):
    pl.pathway_dynamic(*args, **kwargs)