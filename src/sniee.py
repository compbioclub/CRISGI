
import numpy as np
import pandas as pd
import scanpy as sc
import anndata as ad
from scipy.sparse import csr_matrix
from scipy.stats import spearmanr
from pyseat.SEAT import SEAT
import seaborn as sns
import matplotlib.pyplot as plt
import pickle

from src.util import print_msg, get_array, set_adata_var, set_adata_obs

class SNIEE():

    def __init__(self, adata, R_cutoff=0.1, bg_net=None,
                 relation_methods=['pearson', 'spearman', 'pos_coexp', 'neg_coexp']
                 ):

        adata = adata.copy()
        adata.obs['i'] = range(adata.shape[0])
        adata.var['i'] = range(adata.shape[1])
        self.adata = adata
        self.relation_methods = relation_methods
        self.load_bg_net(bg_net)


    def load_bg_net(self, bg_net):
        
        if bg_net is None:
            ref_gb_net = pickle.load(open("src/stringdb_human_v12_gb_net.pk","rb"))
            ref_genes = pickle.load(open("src/stringdb_human_v12_genes.pk","rb"))
            self.adata.uns['stringdb_genes'] = ref_genes
            ref_genes2i = {g:i for i, g in enumerate(ref_genes)}

            genes = self.adata.var_names
            bg_net = np.zeros((len(genes), len(genes)))
            for i, gene1 in enumerate(genes):
                if gene1 not in ref_genes2i:
                    continue
                for j, gene2 in enumerate(genes):
                    if gene2 not in ref_genes2i:
                        continue
                    if i < j:
                        continue
                    ref_gene1_i = ref_genes2i[gene1]
                    ref_gene2_i = ref_genes2i[gene2]
                    bg_net[i, j] = ref_gb_net[ref_gene1_i, ref_gene2_i]
            bg_net = np.tril(bg_net)
            bg_net = csr_matrix(bg_net)

        self.adata.varm['bg_net'] = bg_net
        print_msg(f'The number of edge for bg_net is {bg_net.count_nonzero()}.')

    def _std(self, adata):
        X = get_array(adata, layer='log1p')
        set_adata_var(adata, 'std', X.std(axis=0))

    def _relation_score(self, adata, method):
        X = get_array(adata, layer='log1p')
        if 'pearsonr' == method:  # split pos and neg in the future
            R = np.corrcoef(X.T)
        if 'spearman' == method:
            R = spearmanr(X.T)
        if 'pos_coexp':
            R = np.dot(X.T, X)/X.shape[0]
        if 'neg_coexp':
            R = np.dot(X.T, X.max() - X)/X.shape[0]
        print_msg(f'{method} {R.shape} min { R.min()} max {R.max()}')
        return R
    
    def calculate_entropy(self, ref_obs, per_obss                        
                          ):
        adata_dict = {}
        print_msg(f'---Calculating the group entropy for reference')
        adata = self._calculate_group_entropy(ref_obs)
        adata_dict[f'{ref_obs}'] = adata        
        for per_obs in per_obss:
            print_msg(f'---Calculating the group entropy for {per_obs}')
            adata = self._calculate_group_entropy(ref_obs + per_obs)
            adata_dict[','.join(per_obs)] = adata
        self.adata_dict = adata_dict

    def calculate_delta_entropy(self, ref_obs, per_obss):
        adata_dict = self.adata_dict 
        entropy_dict = {}
        entropy_matrix = np.zeros((len(per_obss), self.adata.varm['bg_net'].count_nonzero()))

        row, col = self.adata.varm['bg_net'].nonzero()
        edata = ad.AnnData(entropy_matrix)
        edata.obs_names = [x[0] for x in per_obss]
        edata = edata[self.adata.obs_names]
        edata.obs = self.adata.obs
        edata.var_names = self.adata.var_names[row] + '_' + self.adata.var_names[col]
        edata.var['gene1'] = self.adata.var_names[row]
        edata.var['gene2'] = self.adata.var_names[col]
        edata.var['gene1_i'] = row
        edata.var['gene2_i'] = col

        for method in self.relation_methods:
            edata.layers[method] = entropy_matrix.copy()

        for i, per_obs in enumerate(per_obss):
            ref_adata = adata_dict[f'{ref_obs}']
            adata = adata_dict[','.join(per_obs)]    
            delta_std = np.abs(adata.varm['std'] - ref_adata.varm['std'])

            for method in self.relation_methods:
                delta_entropy = np.abs(adata.varm[f'{method}_entropy'] - ref_adata.varm[f'{method}_entropy'])
                val = (np.array(delta_entropy[row, col]) * np.array(delta_std[row, col])).reshape(-1)
                # direct dot product will raise much more entry, further investigate   
                edata.layers[method][i, :] = val



        self.entropy_dict = entropy_dict  
        self.edata = edata

    def _calculate_group_entropy(self, obs):
        adata = self.adata[obs, :]
        for method in self.relation_methods:
            self._calculate_node_entropy(adata, method)
            self._calculate_edge_entropy(adata, method)
        return adata
    
    def _calculate_node_entropy(self, adata, method):
        bg_net = self.adata.varm['bg_net']
        row, col = bg_net.nonzero()

        self._std(adata)
        R = self._relation_score(adata, method) 
        val = np.array(R[row, col]).reshape(-1)
        R = csr_matrix((np.abs(val), (row, col)), shape = bg_net.shape)
        adata.varm[f'{method}_bg_net'] = R
        print_msg(f'{method}_bg_net min {R.min()} max {R.max()}')

        prob = R/(R.sum(axis=0))    
        print_msg(f'{method}_prob min {prob.min()} max {prob.max()}')
 
        adata.varm[f'{method}_prob'] = prob
        row, col = prob.nonzero()
        tmp = np.array(prob.todense()[row, col]).reshape(-1)
        val = tmp * np.log(tmp)
        entropy_matrix = csr_matrix((val, (row, col)), shape = prob.shape)
        print_msg(f'{method}_init_entropy min {entropy_matrix.min()} max {entropy_matrix.max()}')
        adata.varm[f'{method}_init_entropy'] = entropy_matrix
        n_neighbors = np.array((R != 0).sum(axis=0))
        norm = np.log(n_neighbors)
        norm[n_neighbors == 0] = 1
        norm[n_neighbors == 1] = 1
        entropy = - np.array(entropy_matrix.sum(axis=0))/norm
        adata.var[f'{method}_entropy'] = entropy.reshape(-1)

    def _calculate_edge_entropy(self, adata, method):
        row, col = adata.varm['bg_net'].nonzero()

        for key in [f'{method}_entropy', 'std']:
            val = (adata.var[key].to_numpy()[row]+adata.var[key].to_numpy()[col])/2
            adata.varm[key] = csr_matrix((val, (row, col)), shape = adata.varm['bg_net'].shape)

    def find_dynamic_entropy(self, n_cluster=2, n_top_genes=5000,
                             random_state=0, n_pcs=30, n_neighbors=10,
                             plot=True, plot_label=[]):
        adata = self.adata
        edata = self.edata

        # cluster with expression
        sc.pp.scale(adata)
        sc.pp.highly_variable_genes(adata, n_top_genes=n_top_genes)
        sc.tl.pca(adata)
        sc.pp.neighbors(adata, n_pcs=n_pcs, n_neighbors=n_neighbors)
        sc.tl.umap(adata, random_state=random_state)

        seat = SEAT(affinity="gaussian_kernel",
                    sparsification="knn_neighbors",
                    objective="SE",
                    n_neighbors=n_neighbors,
                    strategy="bottom_up")
        seat.fit_predict(adata.obsm['X_umap'])
        clusters = ('c' + seat.ks_clusters[f'K={n_cluster}'].astype(str)).tolist()
        adata.obs['seat_cluster'] = clusters

        if plot:
            sc.pl.umap(adata, color=['seat_cluster', *plot_label])

        count_df = pd.DataFrame(self.adata.obs[['seat_cluster', 'type']].value_counts())
        count_df = count_df.reset_index()
        print(count_df)
        ref_count = count_df[count_df['type'] == 'Ref']['count'].max()
        self.ref_like_cluster = count_df[count_df['count'] == ref_count]['seat_cluster'].tolist()[0]
        print(self.ref_like_cluster)
        edata.obs = adata[edata.obs_names].obs
        edata.obsm['X_umap'] = adata[edata.obs_names].obsm['X_umap']

        for method in self.relation_methods:
            sddd
            sc.tl.rank_genes_groups(edata, groupby="seat_cluster", method="wilcoxon")
        return

    def find_localnet_against_ref(self, n_top_relations=50, var_quantile=0.75, p_cutoff=0.05,
                                    plot=True):
        edata = self.edata
        group = self.ref_like_cluster
        df = sc.get.rank_genes_groups_df(edata, group=group)
        df = df[df['pvals_adj'] < p_cutoff].head(n_top_relations)
        local_net = df['names']

        val = edata[edata.obs['seat_cluster'] == group, local_net].X.var(axis=0)
        cutoff = np.quantile(val, var_quantile)

        if plot:
            sns.histplot(val)
            plt.axvline(x=cutoff, color='r')

        self.local_net = local_net[val < cutoff].tolist()
        return
    
    def annot_perputation(self, n_neighbors=10, n_cluster=2, plot_label=[],
                          plot=True):

        seat = SEAT(affinity="gaussian_kernel",
                    sparsification="knn_neighbors",
                    objective="SE",
                    n_neighbors=n_neighbors,
                    strategy="bottom_up")
        seat.fit_predict(self.edata[:, self.local_net].X)
        clusters = ('c' + seat.ks_clusters[f'K={n_cluster}'].astype(str)).tolist()
        self.adata.obs['per_seat_cluster'] = clusters
        self.edata.obs = self.adata.obs
        if True:
            sc.pl.umap(self.adata, color=['per_seat_cluster', *plot_label])


