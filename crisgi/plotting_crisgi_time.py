import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os, random

import crisgi.plotting as pl
from crisgi.util import print_msg


def interaction_score_line(crisgi_obj, target_group=None, 
                        method='pos_coexp', test_type='TER', 
                        interactions=None, unit_header='subject',
                   title='', out_prefix='test',
                   ax=None):
    edata = crisgi_obj.edata
    mytarget_group = target_group
    for target_group in crisgi_obj.groups:
        if mytarget_group is not None and target_group != mytarget_group:
            continue
        df = edata.obs.copy()
        if interactions is None:
            key = f'{method}_{crisgi_obj.groupby}_{target_group}_{test_type}'
            print(key)
            myinteractions = edata.uns[key]
        else:
            myinteractions = [x for x in interactions if x in edata.var_names]
        print(interactions)
        df['avg(score)'] = edata[:, myinteractions].layers[f'{crisgi_obj.ref_time}_{method}_entropy'].mean(axis=1)

        if unit_header is not None:
            print(df)
            sns.lineplot(df, x='time', y='avg(score)', hue=crisgi_obj.groupby, 
                        units=unit_header, estimator=None,
                        ax=ax)
        else:
            sns.lineplot(df, x='time', y='avg(score)', hue=crisgi_obj.groupby, 
                        ax=ax)
                    
        mytitle = title + f'{crisgi_obj.dataset} {target_group} (target)\n{method} entropy score of {len(myinteractions)} {test_type}s '
        if ax is not None:
            ax.set_title(mytitle)
        elif out_prefix:
            plt.title(mytitle)
            plt.savefig(f'{out_prefix}_{mytitle}_interaction_score.png'.replace('\n', ' '))
            plt.show()
        else:
            plt.title(mytitle)
            plt.show()

def get_interaction_score(crisgi_obj, target_group, groupby=None, interactions=None, 
                       method='pos_coexp', test_type='TER',
                        subject_header='subject',
                        out_dir=None):
    edata = crisgi_obj.edata

    if groupby is None:
        groupby = crisgi_obj.groupby

    if interactions is None:
        interactions = edata.uns[f'{method}_{groupby}_{target_group}_{test_type}']
    else:
        interactions = [x for x in interactions if x in edata.var_names]

    subjects = edata.obs[subject_header].unique()
    times = np.sort(edata.obs['time'].unique())
    time2i = {x:i for i, x in enumerate(times)}
    print('subjects', len(subjects), 'times', len(times))
    for i, subject in enumerate(subjects):
        sedata = edata[edata.obs[subject_header] == subject]

        t_is = [time2i[t] for t in sedata.obs['time']]
        X = np.empty((len(interactions), len(times)))
        X[:] = np.nan
        X[:, t_is] = sedata[:, interactions].layers[f'{crisgi_obj.ref_time}_{method}_entropy'].T
        df = pd.DataFrame(X, columns=times, index=interactions)
        if out_dir is None:
            out_dir = crisgi_obj.out_dir
        fn = f'{out_dir}/{subject}_{method}_{groupby}_{target_group}_{test_type}{len(interactions)}_interaction_score.csv'
        df.to_csv(fn)
        print_msg(f'[Output] The subject {subject} {method} {groupby} {target_group} {test_type}{len(interactions)} entropy scores are saved to:\n{fn}')

        #sns.heatmap(df, cmap='RdYlBu_r', robust=True)
        #plt.title(f'delta entropy score for subject {subject}')
        #plt.show()


def generate_interaction_score_images(folder_path, output_path, robust=False, scale=False,
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
    csv_files = [f for f in os.listdir(folder_path) if f.endswith('interaction_score.csv')]

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
            print_msg(f'[Output] The interaction score image is saved to:\n{output_file}')

            plt.close()


def plot_interaction(obj, target_group=None, 
                  layer='log1p',
                  method='pos_coexp', test_type='TER', 
                  interactions=None, subject_header='subject',
                   title='', out_prefix='test',
                   ax=None):
    edata = obj.edata
    adata = obj.adata
    mytarget_group = target_group

    if interactions is None:
        key = f'{method}_{obj.groupby}_{target_group}_{test_type}'
        myinteractions = edata.uns[key]
    else:
        myinteractions = [x for x in interactions if x in edata.var_names]

    genes1 = [x.split('_')[0] for x in myinteractions]
    genes2 = [x.split('_')[0] for x in myinteractions]

    subjects = edata.obs[subject_header].unique()
    times = np.sort(edata.obs['time'].unique())
    time2i = {x:i for i, x in enumerate(times)}
    print('subjects', len(subjects), 'times', len(times))
    for i, subject in enumerate(subjects):
        if mytarget_group is not None:
            sub_adata = adata[(adata.obs[subject_header] == subject) & (adata.obs[obj.groupby] == mytarget_group)]
        else:
            sub_adata = adata[adata.obs[subject_header] == subject]
        if sub_adata.shape[0] == 0:
            continue
        print(subject)
        t_is = [time2i[t] for t in sub_adata.obs['time']]
        X = np.zeros((len(genes1), len(times)))
        
        X[:, t_is] = np.multiply(sub_adata[:, genes1].layers[layer], 
                                 sub_adata[:, genes2].layers[layer]).T
        df = pd.DataFrame(X, columns=times, index=myinteractions)
        print(X)
        #if out_dir is None:
        #    out_dir = obj.out_dir
        #fn = f'{out_dir}/{subject}_{method}_{obj.groupby}_{target_group}_{test_type}{len(interactions)}_interaction_score.csv'
        #df.to_csv(fn)
        #print_msg(f'[Output] The subject {subject} {method} {groupby} {target_group} {test_type}{len(interactions)} entropy scores are saved to:\n{fn}')
        sns.clustermap(df, col_cluster=False)
        plt.suptitle(f'Subject {subject}')
        plt.show()


def draw_gene_network(*args, **kwargs):
    pl.draw_gene_network(*args, **kwargs)


def pheno_level_accumulated_top_n_ORA(*args, **kwargs):
    pl.pheno_level_accumulated_top_n_ORA(*args, **kwargs)



def plot_interaction_score(obj,output_path='./out',label = False ,robust=False, scale=False,
                              rep_n=10, random_seed=0,
                              figsize=(5,5), dpi=500):
    
    edata = obj.edata

    os.makedirs(output_path, exist_ok=True)

    if scale:
        # Initialize variables to store global min and max values
        global_min = edata.X.min()
        global_max = edata.X.max()
        print(f"Global min: {global_min}, Global max: {global_max}")    

    label_records = []

    for sample in edata.obs['subject'].unique():
        sample_edata = edata[edata.obs['subject'] == sample]
        df = sample_edata.to_df()
        df.index = sample_edata.obs['time'].values

        df = df.dropna(how='all', axis=0)
        df = df.dropna(how='all', axis=1)

        if df.shape[0] * df.shape[1] == 0:
            print("DataFrame is empty after dropping rows and columns.")

        df = df.T

        random.seed(random_seed)
        for i in range(rep_n):
            if i > 0:
                df = df.sample(frac=1, random_state= random_seed)
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
            output_file = os.path.join(output_path, f'{sample}_rep{i}.png')
            plt.savefig(output_file, transparent=True, bbox_inches='tight', pad_inches=0, dpi=dpi)
            # print_msg(f'[Output] The interaction score image is saved to:\n{output_file}')

            plt.close()

            if label:
                symptom = sample_edata.obs['symptom'].iloc[0]
                label_records.append({'filename': f'{sample}_rep{i}.png', 'label': symptom})

    if label:
        label_df = pd.DataFrame(label_records)
        label_csv_path = os.path.join(output_path, 'labels.csv')
        label_df.to_csv(label_csv_path, index=False)
        print(f"[Output] Labels saved to: {label_csv_path}")