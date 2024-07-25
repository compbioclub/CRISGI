
import numpy as np
import pandas as pd
import scanpy as sc
import anndata as ad
from scipy.sparse import csr_matrix
from scipy.stats import spearmanr, ttest_1samp
from pyseat.SEAT import SEAT
import seaborn as sns
import matplotlib.pyplot as plt
import pickle
import gseapy as gp
import pymannkendall as mk
from multiprocessing import Pool
import warnings
warnings.filterwarnings('ignore')

pd.options.mode.copy_on_write = True

from src.util import print_msg, get_array, set_adata_var, set_adata_obs

class SNIEE():

    def __init__(self, adata, bg_net=None, bg_net_score_cutoff=850,
                 n_top_genes=5000, n_threads=5,
                 relation_methods=['pearson', 'spearman', 'pos_coexp', 'neg_coexp']
                 ):

        adata = adata.copy()
        adata.obs['i'] = range(adata.shape[0])
        adata.var['i'] = range(adata.shape[1])
        self.adata = adata
        self.relation_methods = relation_methods
        self.bg_net_score_cutoff = bg_net_score_cutoff
        self.n_threads = n_threads
        self.preprocess_adata(n_top_genes=n_top_genes)

        if 'bg_net' not in adata.varm:
            if bg_net is None:
                genes = self.adata.var_names[self.adata.var['highly_variable']].sort_values()
                self.adata = self.adata[:, genes]
                print(self.adata)
                bg_net, _ = self.load_bg_net(genes)
        else:
            bg_net = csr_matrix(np.triu(self.adata.varm['bg_net']))
        self.adata.varm['bg_net'] = bg_net
        print_msg(f'The number of edge for bg_net is {bg_net.count_nonzero()}.')


    def analysis(self, ref_obs, per_obss, ref_groupby, ref_group, plot_label, method,
                 n_top_relations=100, plot=True):
        self.calculate_entropy(ref_obs, per_obss)
        self.calculate_delta_entropy(ref_obs, per_obss)
        self.test_diff_entropy(groupby=None, ref_groupby=ref_groupby, ref_group=ref_group,
                               plot_label=plot_label)
        self.get_diff_relations(per_group=self.per_like_group, n_top_relations=n_top_relations, plot=plot)
        self.test_trend_entropy()

        #self.pathway_enrich(n_top_relations=n_top_relations, method=method)


    def load_bg_net(self, genes):
        
        if (genes != np.sort(genes)).any():    
            raise ValueError('The genes should be sorted!')
        
        print('input gene', len(genes))
        ref_gb_net = pickle.load(open("src/stringdb_human_v12_gb_net.pk","rb"))
        ref_genes = pickle.load(open("src/stringdb_human_v12_genes.pk","rb"))
        self.adata.uns['stringdb_genes'] = ref_genes
        
        if ref_gb_net.count_nonzero() > len(genes)*len(genes):
            bg_net, relations = self._loop_gene(genes, ref_genes, ref_gb_net)
        else:
            bg_net, relations = self._loop_bg_net(genes, ref_genes, ref_gb_net)

        bg_net = np.triu(bg_net)
        bg_net = csr_matrix(bg_net)        
        print('output relations after bg_net', len(relations))  
        return bg_net, relations

    def _loop_gene(self, genes, ref_genes, ref_gb_net):
        print_msg('load bg_net by looping genes.')
        ref_genes2i = {g:i for i, g in enumerate(ref_genes)}
        relations = []
        bg_net = np.zeros((len(genes), len(genes)))
        for i, gene1 in enumerate(genes):
            if gene1 not in ref_genes:
                continue
            for j, gene2 in enumerate(genes):
                if gene2 not in ref_genes:
                    continue
                if i > j:
                    continue
                ref_gene1_i = ref_genes2i[gene1]
                ref_gene2_i = ref_genes2i[gene2]
                if ref_gb_net[ref_gene1_i, ref_gene2_i] >= self.bg_net_score_cutoff:
                    bg_net[i, j] = ref_gb_net[ref_gene1_i, ref_gene2_i]
                    relations.append(f'{gene1}_{gene2}')
        return bg_net, relations

    def _loop_bg_net(self, genes, ref_genes, ref_gb_net):
        print_msg('load bg_net by looping bg_net.')
        genes2i = {g:i for i, g in enumerate(genes)}
        relations = []
        bg_net = np.zeros((len(genes), len(genes)))
        row, col = ref_gb_net.nonzero()
        for ref_gene1_i, ref_gene2_i in zip(row, col):
            gene1, gene2 = ref_genes[ref_gene1_i], ref_genes[ref_gene2_i]
            if (gene1 not in genes) or (gene2 not in genes):
                continue
            if ref_gb_net[ref_gene1_i, ref_gene2_i] >= self.bg_net_score_cutoff:
                i, j = genes2i[gene1], genes2i[gene2]
                bg_net[i, j] = ref_gb_net[ref_gene1_i, ref_gene2_i]
                relations.append(f'{gene1}_{gene2}')
        return bg_net, relations

    def preprocess_adata(self, n_top_genes=5000, random_state=0, n_pcs=30, n_neighbors=10):
        adata = self.adata
        #sc.pp.scale(adata)
        sc.pp.highly_variable_genes(adata, n_top_genes=n_top_genes, flavor='cell_ranger')
        sc.tl.pca(adata)
        sc.pp.neighbors(adata, n_pcs=n_pcs, n_neighbors=n_neighbors)
        sc.tl.umap(adata, random_state=random_state)

    def _std(self, adata):
        X = get_array(adata, layer='log1p')
        set_adata_var(adata, 'std', X.std(axis=0))


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
        X = get_array(adata, layer='scale')
        R = np.dot(X.T, X)/X.shape[1]
        return R

    def _relation_score(self, adata, method):
        X = get_array(adata, layer='log1p')
        self._std(adata)

        if 'pearson' == method:  # split pos and neg in the future
            R = self._pearsonr(adata) # faster
            #R = np.corrcoef(X.T)
        if 'spearman' == method: # include p-val
            R = spearmanr(X).statistic
        if 'pos_coexp' == method:
            R = np.dot(X.T, X)/X.shape[0]
        if 'neg_coexp' == method:
            R = np.dot(X.T, X.max() - X)/X.shape[0]
        print_msg(f'{method} size {R.shape} min { R.min()} max {R.max()}')
        return R
    
    def calculate_entropy(self, ref_obs, per_obss):
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
        self.per_obss = per_obss
        self.ref_obs = ref_obs
        adata_dict = self.adata_dict 
        entropy_matrix = np.zeros((len(per_obss), self.adata.varm['bg_net'].count_nonzero()))

        row, col = self.adata.varm['bg_net'].nonzero()
        edata = ad.AnnData(entropy_matrix)
        edata.obs_names = [x[0] for x in per_obss]
        edata.obs = self.adata[edata.obs_names, :].obs
        edata.var_names = self.adata.var_names[row] + '_' + self.adata.var_names[col]
        edata.var['gene1'] = self.adata.var_names[row]
        edata.var['gene2'] = self.adata.var_names[col]
        edata.var['gene1_i'] = row
        edata.var['gene2_i'] = col

        for method in self.relation_methods:
            edata.layers[f'{method}_entropy'] = entropy_matrix.copy()
            edata.layers[f'{method}_prob'] = entropy_matrix.copy()
            edata.layers[f'{method}_bg_net'] = entropy_matrix.copy()

        for i, per_obs in enumerate(per_obss):
            ref_adata = adata_dict[f'{ref_obs}']
            adata = adata_dict[','.join(per_obs)]    
            delta_std = np.abs(adata.varm['std'] - ref_adata.varm['std'])

            for method in self.relation_methods:
                delta_entropy = np.abs(adata.varm[f'{method}_entropy'] - ref_adata.varm[f'{method}_entropy'])
                val = (np.array(delta_entropy[row, col]) * np.array(delta_std[row, col])).reshape(-1)
                # direct dot product will raise much more entry, further investigate   
                edata.layers[f'{method}_entropy'][i, :] = val

                val = adata.varm[f'{method}_bg_net'][row, col].reshape(-1)
                edata.layers[f'{method}_bg_net'][i, :] = val

                val = csr_matrix(adata.varm[f'{method}_prob'])[row, col].reshape(-1)
                edata.layers[f'{method}_prob'][i, :] = val

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

        R = self._relation_score(adata, method) 
        val = np.array(R[row, col]).reshape(-1)
        R = csr_matrix((np.abs(val), (row, col)), shape = bg_net.shape)
        adata.varm[f'{method}_bg_net'] = R
        print_msg(f'{method}_bg_net min {R.min()} max {R.max()}')

        R_sum = R.sum(axis=0)
        R_sum[R_sum == 0] = 1  # TBD, speed up
        prob = R/R_sum
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



    def find_ref_like_group(self, ref_groupby, ref_group, n_cluster=2, n_neighbors=10, 
                             plot=True, plot_label=[]):
        adata = self.adata

        # cluster with expression
        seat = SEAT(affinity="gaussian_kernel",
                    sparsification="knn_neighbors",
                    objective="SE",
                    n_neighbors=n_neighbors,
                    strategy="bottom_up")
        seat.fit_predict(adata.obsm['X_umap'])
        clusters = ('c' + seat.ks_clusters[f'K={n_cluster}'].astype(str)).tolist()
        adata.obs['seat_cluster'] = clusters

        if plot:
            plot_label = [x for x in plot_label if x in adata.obs.columns]
            sc.pl.umap(adata, color=['seat_cluster', *plot_label], show=False)

        count_df = pd.DataFrame(self.adata.obs[['seat_cluster', ref_groupby]].value_counts())
        count_df = count_df.reset_index()
        print(count_df)
        ref_count_df = count_df[count_df[ref_groupby] == ref_group]

        ref_count = ref_count_df['count'].max()
        self.ref_like_group = count_df[count_df['count'] == ref_count]['seat_cluster'].tolist()[0]
        self.per_like_group = [x for x in count_df['seat_cluster'] if x != self.ref_like_group][0]
        print('ref_like_group is', self.ref_like_group)
        print('per_like_group is', self.per_like_group)

    def test_diff_entropy(self, groupby=None, plot=True, plot_label=[], 
                          ref_groupby=None, ref_group=None):
        if groupby is None:
            self.find_ref_like_group(ref_groupby, ref_group, n_cluster=2, n_neighbors=10,
                                     plot=plot, plot_label=plot_label)
            groupby = 'seat_cluster'

        self.groupby = groupby
        adata = self.adata
        edata = self.edata
        edata.obs = adata[edata.obs_names].obs
        edata.obsm['X_umap'] = adata[edata.obs_names].obsm['X_umap']        
        for method in self.relation_methods:
            sc.tl.rank_genes_groups(edata, layer=f'{method}_entropy',
                                    groupby=groupby, method="wilcoxon")
            edata.uns[f'{method}_rank_genes_groups'] = edata.uns['rank_genes_groups']
        return

    def get_diff_relations(self, per_group, n_top_relations=100, 
                           p_adjust=True, p_cutoff=0.05, fc_cutoff=1, sortby='pvals_adj',
                           plot=True):
        edata = self.edata
        df_list = []

        for i, method in enumerate(self.relation_methods):
            df = sc.get.rank_genes_groups_df(edata, group=per_group, 
                                             key=f'{method}_rank_genes_groups')
            if p_adjust:
                p_method = 'pvals_adj'
            else:
                p_method = 'pvals'
            df = df[(df[p_method] < p_cutoff) & (df['logfoldchanges'] > fc_cutoff)]
            if sortby == 'logfoldchanges':
                df = df.sort_values(by=sortby, ascending=False)
            df = df.head(n_top_relations)
            
            if df.empty:
                continue
            df['method'] = method
            relations = df['names'].tolist()
            edata.uns[f'{method}_diff_relations'] = relations
            df_list.append(df)

            if i == 0:
                common_diff_relations = set(relations)
                all_diff_relations = set(relations)
            else:
                common_diff_relations = common_diff_relations & set(relations)
                all_diff_relations = all_diff_relations | set(relations)

        if df.empty:
            return
        
        df = pd.concat(df_list)
        if sortby in ['logfoldchanges', 'scores']:
            df = df.sort_values(by=sortby, ascending=False)        
        else:
            df = df.sort_values(by=sortby, ascending=True)        

        edata.uns['rank_genes_groups'] = df
        edata.uns[f'common_diff_relations'] = list(common_diff_relations)
        edata.uns[f'all_diff_relations'] = list(all_diff_relations)
        # sort the all and common relations, to be continued
        return
    
    def _test_up_trend(self, relation, adata, edata, method='pearson', p_cutoff=0.05):
        sorted_samples = adata.obs.sort_values(by=['time']).index.tolist()
        val = edata[sorted_samples, relation].layers[f'{method}_entropy'].reshape(-1)
        result = mk.original_test(val, alpha=p_cutoff)
        return result.trend == 'increasing'
    
    def _test_zero_trend(self, relation, adata, edata, method='pearson', p_cutoff=0.05):
        sorted_samples = adata.obs.sort_values(by=['time']).index.tolist()
        val = edata[sorted_samples, relation].layers[f'{method}_entropy'].reshape(-1)
        _, p_value = ttest_1samp(val, 0)
        return p_value < p_cutoff 

    def test_trend_entropy(self, per_group=None, p_cutoff=0.05):
        adata = self.adata
        edata = self.edata

        for i, method in enumerate(self.relation_methods):
            relations = set()
            for j, relation in enumerate(edata.uns[f'{method}_diff_relations']):
                per_adata = adata[adata.obs[self.groupby] == per_group, :]
                is_per_up = self._test_up_trend(relation, per_adata, edata, method=method, p_cutoff=p_cutoff)

                ref_adata = adata[adata.obs[self.groupby] != per_group, :]
                is_ref_zero = self._test_zero_trend(relation, ref_adata, edata, method=method, p_cutoff=p_cutoff)

                if is_per_up and is_ref_zero:
                    relations.add(relation)

            print(method, 'diff', j+1, 'trend', len(relations))
            edata.uns[f'{method}_trend_relations'] = list(relations)
            if i == 0:
                common_relations = set(relations)
                all_relations = set(relations)
            else:
                common_relations = common_relations & set(relations)
                all_relations = all_relations | set(relations)

        #edata.uns['rank_genes_groups'] = pd.concat(df_list)
        edata.uns[f'common_trend_relations'] = list(common_relations)
        edata.uns[f'all_trend_relations'] = list(all_relations)
        return

    def test_val_trend_entropy(self, relations, method='pearson', p_cutoff=0.05):
        adata = self.adata
        edata = self.edata
        candidates = []
        for relation in relations:
            if relation not in self.edata.var_names:
                continue
            is_up = self._test_up_trend(relation, adata, edata, method=method, p_cutoff=p_cutoff)
            is_zero = self._test_zero_trend(relation, adata, edata, method=method, p_cutoff=p_cutoff)

            if is_up or is_zero:
                candidates.append(relation)    
        print(method, 'val trend before', len(relations), 'after', len(candidates))        
        return candidates
    
    def _enrich_for_top_n(self, args):
        top_n, relation_list, gene_sets, organism = args
        print('_enrich_for_top_n', top_n, gene_sets)

        gene_list = list(set(np.array([x.split('_') for x in relation_list[:top_n]]).reshape(-1)))
        enr = gp.enrichr(gene_list=gene_list, gene_sets=gene_sets,
                         background=self.adata.var_names, organism=organism, outdir=None)
        df = enr.results
        df['n_gene'] = df['Genes'].apply(lambda x: len(x.split(';')))
        df['top_n'] = top_n
        df['top_n_ratio'] = df['n_gene'] / top_n
        return top_n, enr, df

    def pathway_enrich(self, n_top_relations=100, n_space=10, 
                       method='pearson', test_type='trend',
                       gene_sets=['KEGG_2021_Human', 
                                  'GO_Molecular_Function_2023', 'GO_Cellular_Component_2023', 'GO_Biological_Process_2023',
                                  'MSigDB_Hallmark_2020'],
                       organism='human', plot=True):
        relation_list = self.edata.uns[f'{method}_{test_type}_relations']
        enr_dict = {}
        df_list = []
        print(relation_list)
        # Prepare arguments for parallel processing
        args_list = [(top_n, relation_list, gene_sets, organism) for top_n in range(10, n_top_relations + 1, n_space)]

        # Use multiprocessing Pool to run the enrichment in parallel
        with Pool(processes=self.n_threads) as pool:
            results = pool.map(self._enrich_for_top_n, args_list)

        # Collect results
        for top_n, enr, df in results:
            enr_dict[top_n] = enr
            df_list.append(df)

        df = pd.concat(df_list)
        self.edata.uns[f'{method}_enrich_res'] = enr_dict
        self.edata.uns[f'{method}_enrich_df'] = df
          

    def annot_perputation(self, n_neighbors=10, n_cluster=2, plot_label=[],
                          method='all',
                          plot=True):
        # need improvement
        seat = SEAT(affinity="gaussian_kernel",
                    sparsification="knn_neighbors",
                    objective="SE",
                    n_neighbors=n_neighbors,
                    strategy="bottom_up")
        print(self.edata.uns[f'{method}_gene_hub'])
        seat.fit_predict(self.edata[:, self.edata.uns[f'{method}_gene_hub']].layers[f'{method}_entropy'])
        clusters = ('c' + seat.ks_clusters[f'K={n_cluster}'].astype(str)).tolist()
        print(clusters)
        self.adata.obs['per_seat_cluster'] = clusters
        self.edata.obs = self.adata.obs
        if True:
            sc.pl.umap(self.adata, color=['per_seat_cluster', *plot_label])


