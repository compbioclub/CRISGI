
import numpy as np
import scanpy as sc
import pandas as pd
from scipy.sparse import csr_matrix

from src.util import print_msg, get_array

class SNIEE():

    def __init__(self, adata, R_cutoff=0.1, bg_net=None):
        adata = adata.copy()
        self.adata = adata
        if bg_net is None and 'bg_net' not in adata.varm.keys():
            self._pearsonr(adata) 
            R = adata.varm['pearsonr']
            print_msg(f'The number of edge for pearsonr is {R.count_nonzero()}')
            row, col = (np.abs(R) > R_cutoff).nonzero()
            val = np.array(R[row, col]).reshape(row.shape)
            bg_net = csr_matrix((val, (row, col)), shape = R.shape)
            print_msg(f'The number of edge for bg_net is {bg_net.count_nonzero()}, with R cutoff {R_cutoff}.')
            adata.varm['bg_net'] = bg_net
        elif 'bg_net' not in adata.varm.keys():
            bg_net = csr_matrix(bg_net)
            adata.varm['bg_net'] = bg_net
        else:
            adata.varm['bg_net'] = csr_matrix(adata.varm['bg_net'])
            bg_net = adata.varm['bg_net']
        print_msg(f'The number of edge for bg_net is {bg_net.count_nonzero()}.')


    def _scale(self, adata, axis=0):
        X = get_array(adata, layer='log1p')
        N = X.shape[axis]
        mean = X.mean(axis=axis, keepdims=True)
        X = X - mean
        std = np.sqrt(np.power(X, 2).sum(axis=axis, keepdims=True)/N)
        std[std == 0] = 1
        X = X / std
        adata.X = X
        adata.layers['scale'] = X.copy()
        adata.var['mean'] = mean.reshape(-1)
        adata.var['std'] = std.reshape(-1)

    def _pearsonr(self, adata):
        self._scale(adata)
        print('std', adata.var['std'].to_numpy()[7748])

        X = get_array(adata)
        R = np.dot(X.T, X)/X.shape[1]
        # adata.varm['pearsonr'] = R
        print_msg(f'correlation min { R.min()} max {R.max()}')
        return R
    
    def calculate_entropy(self, ref_obs, per_obss):
        adata_dict = {}
        adata = self._calculate_cluster_entropy(ref_obs)
        adata_dict[f'{ref_obs}'] = adata        
        for per_obs in per_obss:
            adata = self._calculate_cluster_entropy(ref_obs + per_obs)
            adata_dict[','.join(per_obs)] = adata
        self.adata_dict = adata_dict

    def calculate_dynamic_entropy(self, ref_obs, per_obss):
        adata_dict = self.adata_dict 
        entropy_dict = {}
        for per_obs in per_obss:
            ref_adata = adata_dict[f'{ref_obs}']
            adata = adata_dict[','.join(per_obs)]    

            delta_entropy = np.abs(adata.varm['entropy'] - ref_adata.varm['entropy'])
            delta_std = np.abs(adata.varm['std'] - ref_adata.varm['std'])

            delta_entropy = delta_entropy * delta_std
            entropy_dict[','.join(per_obs)] = delta_entropy     

        self.entropy_dict = entropy_dict  

    def report_top_entropy_relation(self, top_n=10):
        df_list = []
        for key, entropy in self.entropy_dict.items():
            #print_msg(f'{key} {entropy.min()} {entropy.max()}')
            col, row = entropy.nonzero()
            top_n_indices = np.argpartition(entropy.data, -top_n)[-top_n:]
            sorted_indices = top_n_indices[np.argsort(-entropy.data[top_n_indices])]
            top_row = row[sorted_indices]
            top_col = col[sorted_indices]
            top_val = entropy.data[sorted_indices]   

            top_entropy = csr_matrix((top_val, (top_row, top_col)), shape = entropy.shape)     
            df = pd.DataFrame({
                'key': key,
                'gene1_i': top_row,
                'gene2_i': top_col,
                'gene1': self.adata.var_names[top_row],
                'gene2': self.adata.var_names[top_col],
                'delta entropy': top_val
                })
            df_list.append(df)

        df = pd.concat(df_list)
        df = df[df.gene1 != df.gene2]
        return df 


    def _calculate_cluster_entropy(self, obs):
        print_msg(f'Calculating the cluster entropy for {obs}')
        adata = self.adata[obs, :]
        self._calculate_node_entropy(adata)
        self._calculate_edge_entropy(adata)
        return adata
    
    def _calculate_node_entropy(self, adata):
        bg_net = self.adata.varm['bg_net']
        row, col = bg_net.nonzero()

        R = self._pearsonr(adata) 
        val = np.array(R[col, row]).reshape(-1)
        R = csr_matrix((np.abs(val), (row, col)), shape = bg_net.shape)
        adata.varm['pearsonr*bg_net'] = R

        prob = R/(R.sum(axis=0))    
        print_msg(f'prob min {prob.min()} max {prob.max()}')
 
        adata.varm['prob'] = prob
        row, col = prob.nonzero()
        tmp = np.array(prob.todense()[row, col]).reshape(-1)
        val = tmp * np.log(tmp)
        entropy_matrix = csr_matrix((val, (row, col)), shape = prob.shape)
        print_msg(f'entroy min {entropy_matrix.min()} max {entropy_matrix.max()}')
        adata.varm['init_entropy'] = entropy_matrix
        n_neighbors = np.array((R != 0).sum(axis=0))
        norm = np.log(n_neighbors)
        norm[n_neighbors == 0] = 1
        norm[n_neighbors == 1] = 1
        entropy = - np.array(entropy_matrix.sum(axis=0))/norm
        adata.var['entropy'] = entropy.reshape(-1)

    def _calculate_edge_entropy(self, adata):
        row, col = adata.varm['bg_net'].nonzero()

        for key in ['entropy', 'std']:
            val = (adata.var[key].to_numpy()[row]+adata.var[key].to_numpy()[col])/2
            adata.varm[key] = csr_matrix((val, (row, col)), shape = adata.varm['bg_net'].shape)





