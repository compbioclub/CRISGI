
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
import os
import pickle
import gseapy as gp
import pymannkendall as mk
from multiprocessing import Pool
from concurrent.futures import ProcessPoolExecutor, as_completed
import warnings
warnings.filterwarnings('ignore')

pd.options.mode.copy_on_write = True

from src.util import print_msg, get_array, set_adata_var, set_adata_obs

class SNIEE():

    def __init__(self, adata, bg_net=None, bg_net_score_cutoff=850,
                 n_top_genes=5000, n_threads=5,
                 relation_methods=['pearson', 'spearman', 'pos_coexp', 'neg_coexp'],
                 dataset='test',
                 out_dir='./out'
                 ):

        adata = adata.copy()
        adata.obs['i'] = range(adata.shape[0])
        adata.var['i'] = range(adata.shape[1])
        self.adata = adata
        self.relation_methods = relation_methods
        self.bg_net_score_cutoff = bg_net_score_cutoff
        self.n_threads = n_threads
        self.dataset = dataset

        if not os.path.exists(out_dir):
            os.makedirs(out_dir)
        self.out_dir = out_dir
        self.preprocess_adata(n_top_genes=n_top_genes)

        if 'bg_net' not in adata.varm:
            if bg_net is None:
                genes = self.adata.var_names[self.adata.var['highly_variable']].sort_values()
                self.adata = self.adata[:, genes]
                bg_net, _ = self.load_bg_net(genes)
        else:
            bg_net = csr_matrix(np.triu(self.adata.varm['bg_net']))
        self.adata.varm['bg_net'] = bg_net
        print_msg(f'The number of edge for bg_net is {bg_net.count_nonzero()}.')

    '''
    def analysis(self, ref_obs, per_obss, ref_groupby, ref_group, plot_label, method,
                 n_top_relations=100, plot=True):
        self.calculate_entropy(ref_obs, per_obss)
        self.calculate_delta_entropy(ref_obs, per_obss)
        self.test_diff_entropy(groupby=None, ref_groupby=ref_groupby, ref_group=ref_group,
                               plot_label=plot_label)
        self.get_diff_relations(per_group=self.per_like_group, n_top_relations=n_top_relations, plot=plot)
        self.test_trend_entropy(per_group=self.per_like_group)

        #self.pathway_enrich(n_top_relations=n_top_relations, method=method)
    '''

    def save(self):
        pk_fn = f'{self.out_dir}/{self.dataset}_sniee_obj.pk'
        pickle.dump(self, open(pk_fn, 'wb'))        
        print_msg(f'[Output] SNIEE object has saved to:\n{pk_fn}')

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
                             plot=True, plot_label=[], out_prefix=None):
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
            plt.savefig(f'{out_prefix}_umap.png')
            plt.show()

        count_df = pd.DataFrame(self.adata.obs[['seat_cluster', ref_groupby]].value_counts())
        count_df = count_df.reset_index()
        print(count_df)
        ref_count_df = count_df[count_df[ref_groupby] == ref_group]

        ref_count = ref_count_df['count'].max()
        self.ref_like_group = count_df[count_df['count'] == ref_count]['seat_cluster'].tolist()[0]
        self.per_like_group = [x for x in count_df['seat_cluster'] if x != self.ref_like_group][0]
        print('ref_like_group is', self.ref_like_group)
        print('per_like_group is', self.per_like_group)

    def test_DER(self, groupby=None, plot=True, plot_label=[], 
                          ref_groupby=None, ref_group=None,
                          out_prefix=None):
        
        if groupby is None:
            self.find_ref_like_group(ref_groupby, ref_group, n_cluster=2, n_neighbors=10,
                                     plot=plot, plot_label=plot_label, out_prefix=out_prefix)
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

    def get_DER(self, per_group, n_top_relations=None, 
                p_adjust=True, p_cutoff=0.05, fc_cutoff=1, sortby='pvals_adj',
                ):
        edata = self.edata
        df_list = []

        for i, method in enumerate(self.relation_methods):
            df = sc.get.rank_genes_groups_df(edata, group=per_group, 
                                             key=f'{method}_rank_genes_groups')
            df['method'] = method

            if p_adjust:
                p_method = 'pvals_adj'
            else:
                p_method = 'pvals'

            df['DER'] = (df[p_method] < p_cutoff) & (df['logfoldchanges'] > fc_cutoff)
            if sortby in ['logfoldchanges', 'scores']:
                df = df.sort_values(by=['DER', sortby], ascending=False)
            if sortby == 'pvals_adj':
                df = df.sort_values(by=['DER', sortby], ascending=[False, True])

            df_list.append(df)

            if n_top_relations:
                top_df = df[df['DER']].head(n_top_relations)        
            else:
                top_df = df[df['DER']]    
            if top_df.empty:
                continue
            relations = top_df['names'].tolist()
            edata.uns[f'{method}_DER'] = relations

            if i == 0:
                common_DERs = set(relations)
                all_DERs = set(relations)
            else:
                common_DERs = common_DERs & set(relations)
                all_DERs = all_DERs | set(relations)
        
        df = pd.concat(df_list)
        if sortby in ['logfoldchanges', 'scores']:
            df = df.sort_values(by=['DER', sortby], ascending=False)
        else:
            df = df.sort_values(by=['DER', sortby], ascending=[False, True])

        fn = f'{self.out_dir}/{per_group}_DER.csv'
        df.to_csv(fn, index=False)
        print_msg(f'[Output] The differential expressed relation (DER) statistics are saved to:\n{fn}')
        edata.uns[f'{per_group}_DER'] = df
        edata.uns[f'common_DER'] = list(common_DERs)
        edata.uns[f'all_DER'] = list(all_DERs)
        # sort the all and common relations, to be continued
        return
    
    def _test_up_trend(self, relation, adata, edata, method='pearson', p_cutoff=0.05):
        sorted_samples = adata.obs.sort_values(by=['time']).index.tolist()
        val = edata[sorted_samples, relation].layers[f'{method}_entropy'].reshape(-1)
        res = mk.original_test(val, alpha=p_cutoff)
        # https://pypi.org/project/pymannkendall/
        res = {
            'relation': relation, 'method': method,
            'trend': res.trend, 'h': res.h, 'p': res.p, 'z': res.z,
            'Tau': res.Tau, 's': res.s, 'var_s': res.var_s, 'slope': res.slope, 'intercept': res.intercept
        }
        return res
    
    def _test_zero_trend(self, relation, adata, edata, method='pearson', p_cutoff=0.05):
        sorted_samples = adata.obs.sort_values(by=['time']).index.tolist()
        val = edata[sorted_samples, relation].layers[f'{method}_entropy'].reshape(-1)
        t_statistic, p_value = ttest_1samp(val, 0)
        # https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.ttest_1samp.html
        res = {'relation': relation, 'method': method,
               't_statistic': t_statistic, 'p_value': p_value}
        return res 

    def test_TER(self, per_group, p_cutoff=0.05):
        adata = self.adata
        edata = self.edata

        trend_list = []
        for i, method in enumerate(self.relation_methods):
            relations = set()
            for j, relation in enumerate(edata.uns[f'{method}_DER']):
                per_adata = adata[adata.obs[self.groupby] == per_group, :]
                up_res = self._test_up_trend(relation, per_adata, edata, method=method, p_cutoff=p_cutoff)
                is_per_up = up_res['trend'] == 'increasing'

                ref_adata = adata[adata.obs[self.groupby] != per_group, :]
                zero_res = self._test_zero_trend(relation, ref_adata, edata, method=method, p_cutoff=p_cutoff)
                is_ref_zero = zero_res['p_value'] < p_cutoff 
                up_res.update(zero_res)
                
                is_TER = is_per_up and is_ref_zero    
                up_res['TER'] = is_TER
                if is_TER:
                    relations.add(relation)
                trend_list.append(up_res)

            print(method, 'diff', j+1, 'trend', len(relations))
            edata.uns[f'{method}_TER'] = list(relations)
            if i == 0:
                common_TERs = set(relations)
                all_TERs = set(relations)
            else:
                common_TERs = common_TERs & set(relations)
                all_TERs = all_TERs | set(relations)

        df = pd.DataFrame(trend_list)
        fn = f'{self.out_dir}/{per_group}_TER.csv'
        df.to_csv(fn, index=False)
        print_msg(f'[Output] The trend expressed relation (TER) statistics are saved to:\n{fn}')
        edata.uns[f'{per_group}_TER'] = df
        edata.uns[f'common_TER'] = list(common_TERs)
        edata.uns[f'all_TER'] = list(all_TERs)
        return

    def test_val_trend_entropy(self, relations, method='pearson', p_cutoff=0.05,
                               out_prefix='./test'):
        adata = self.adata
        edata = self.edata
        candidates = []
        trend_list = []
        for relation in relations:
            if relation not in self.edata.var_names:
                continue
            up_res = self._test_up_trend(relation, adata, edata, method=method, p_cutoff=p_cutoff)
            is_up = up_res['trend'] == 'increasing'
            zero_res = self._test_zero_trend(relation, adata, edata, method=method, p_cutoff=p_cutoff)
            is_zero = zero_res['p_value'] < p_cutoff 
            up_res.update(zero_res)

            is_TER = is_up or is_zero    
            up_res['TER'] = is_TER

            trend_list.append(up_res)
            if is_TER:
                candidates.append(relation)  

        print(method, 'val trend before', len(relations), 'after', len(candidates))        

        df = pd.DataFrame(trend_list)
        fn = f'{out_prefix}_TER.csv'
        df.to_csv(fn, index=False)
        print_msg(f'[Output] The validation trend expressed relation (TER) statistics are saved to:\n{fn}')
                       
        return candidates
          
    def _enrich_for_top_n(self, top_n, relation_list, gene_sets, organism, background):
        print('_enrich_for_top_n', top_n)
        gene_list = list(set(np.array([x.split('_') for x in relation_list[:top_n]]).reshape(-1)))
        enr = gp.enrichr(gene_list=gene_list, gene_sets=gene_sets,
                         background=background, 
                         organism=organism, 
                         outdir=None)
        df = enr.results
        df['n_gene'] = df['Genes'].apply(lambda x: len(x.split(';')))
        df['top_n'] = top_n
        df['top_n_ratio'] = df['n_gene'] / top_n
        return top_n, enr, df

    def pathway_enrich(self, n_top_relations=None, n_space=10, 
                       method='pearson', test_type='TER',
                       gene_sets=['KEGG_2021_Human', 
                                  'GO_Molecular_Function_2023', 'GO_Cellular_Component_2023', 'GO_Biological_Process_2023',
                                  'MSigDB_Hallmark_2020'],
                       background=None,
                       organism='human', plot=True):
        relation_list = self.edata.uns[f'{method}_{test_type}']
        enr_dict = {}
        df_list = []

        if n_top_relations is None:
            n_top_relations = len(relation_list)

        args_list = [(top_n, relation_list, gene_sets, organism, background) for top_n in range(10, n_top_relations + 1, n_space)]

        with ProcessPoolExecutor(max_workers=self.n_threads) as executor:
            futures = {executor.submit(self._enrich_for_top_n, *args): args[0] for args in args_list}

            for future in as_completed(futures):
                top_n = futures[future]
                try:
                    top_n, enr, df = future.result()
                    enr_dict[top_n] = enr
                    df_list.append(df)
                except Exception as exc:
                    print(f'Top_n {top_n} generated an exception: {exc}')

        df = pd.concat(df_list)
        fn = f'{self.out_dir}/{method}_{test_type}_enrich.csv'
        df.to_csv(fn, index=False)
        print_msg(f'[Output] The {method} {test_type} enrich statistics are saved to:\n{fn}')

        self.edata.uns[f'{method}_{test_type}_enrich_res'] = enr_dict
        self.edata.uns[f'{method}_{test_type}_enrich_df'] = df

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


