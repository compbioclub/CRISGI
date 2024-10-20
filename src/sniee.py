
import numpy as np
import pandas as pd
import scanpy as sc
import anndata as ad
from scipy.sparse import csr_matrix, coo_matrix
from scipy.stats import spearmanr, ttest_1samp, wilcoxon
from pyseat.SEAT import SEAT
import seaborn as sns
import matplotlib.pyplot as plt
import pickle
import os
import pickle
from sksurv.nonparametric import kaplan_meier_estimator
from sksurv.compare import compare_survival
from itertools import chain, permutations
import gseapy as gp
from gseapy import gseaplot
import pymannkendall as mk
from multiprocessing import Pool
from concurrent.futures import ProcessPoolExecutor, as_completed
import warnings
warnings.filterwarnings('ignore')

pd.options.mode.copy_on_write = True

from src.util import print_msg, get_array, set_adata_var, set_adata_obs


def load_sniee(pk_fn):
    sniee_obj = pickle.load(open(pk_fn, 'rb'))
    print_msg(f'[Input] SNIEE object stored at {pk_fn} has been loaded.')
    return sniee_obj

class SNIEE():

    # SNIEE
    def __init__(self, adata, bg_net=None, bg_net_score_cutoff=850,
                 genes=None,
                 n_hvg=5000, n_pcs=30,
                 relations=None,
                 n_threads=5,
                 relation_methods=['pearson', 'spearman', 'pos_coexp', 'neg_coexp'],
                 organism='human',
                 class_type='time',
                 dataset='test',
                 out_dir='./out'
                 ):

        adata = adata.copy()
        adata.obs['i'] = range(adata.shape[0])
        adata.var['i'] = range(adata.shape[1])
        self.adata = adata
        self.relation_methods = relation_methods
        self.organism = organism
        self.n_threads = n_threads
        self.dataset = dataset
        self.class_type = class_type
        if not os.path.exists(out_dir):
            os.makedirs(out_dir)
        self.out_dir = out_dir

        if n_hvg is not None:
            self.preprocess_adata(n_hvg=n_hvg, n_pcs=n_pcs)

        self.load_bg_net(bg_net, bg_net_score_cutoff, relations, genes)

    def load_bg_net(self, bg_net, bg_net_score_cutoff, relations, genes):
        self.bg_net_score_cutoff = bg_net_score_cutoff

        if relations is not None:
            bg_net = self.load_bg_net_from_relations(relations)
            self.adata.varm['bg_net'] = bg_net

        if 'bg_net' not in self.adata.varm:
            if bg_net is None:
                if genes is None:
                    if 'highly_variable' in self.adata.var:
                        genes = self.adata.var_names[self.adata.var['highly_variable']]
                    else:
                        genes = self.adata.var_names
                    
                self.adata = self.adata[:, self.adata.var_names.isin(genes)]
                genes = np.array(list(set(genes).intersection(self.adata.var_names)))
                genes = np.sort(genes)
                self.adata = self.adata[:, genes]
                self.adata.var['i'] = range(self.adata.shape[1])
                bg_net, _ = self.load_bg_net_from_genes(genes)
        else:
            bg_net = csr_matrix(np.triu(self.adata.varm['bg_net']))
        self.adata.varm['bg_net'] = bg_net 
        print_msg(f'The number of edge for bg_net is {bg_net.count_nonzero()}.')

    # SNIEE
    def init_edata(self, per_obss, headers):
        adata = self.adata
        per_obss_is = [adata[per_obs, :].obs.i.tolist() for per_obs in per_obss]
        adata = self.adata
        bg_net = adata.varm['bg_net']
        row, col = bg_net.nonzero()
        per_n = len(per_obss_is)
        relation_n = bg_net.count_nonzero()
        edata = ad.AnnData(np.zeros((per_n, relation_n)))
        per_obs = list(chain.from_iterable(per_obss))
        df = adata[per_obs, :].obs[headers].copy().drop_duplicates()
        df.index = df['test']
        edata.obs = df
        edata.var_names = adata.var_names[row].astype(str) + '_' + adata.var_names[col].astype(str)
        edata.var['gene1'] = adata.var_names[row]
        edata.var['gene2'] = adata.var_names[col]
        edata.var['gene1_i'] = row
        edata.var['gene2_i'] = col
        edata.var['i'] = range(relation_n)
        self.edata = edata
        print_msg(f'Init edata with obs {edata.shape[0]} and relation {edata.shape[1]}')

    # SNIEE
    def save(self):
        pk_fn = f'{self.out_dir}/{self.dataset}_sniee_obj.pk'
        pickle.dump(self, open(pk_fn, 'wb'))
        print_msg(f'[Output] SNIEE object has benn saved to:\n{pk_fn}')

    # SNIEE
    def load_bg_net_from_relations(self, relations):
        print_msg('load bg_net by looping relations.')
        genes = np.array([r.split('_') for r in relations]).reshape(-1)
        genes = list(set([g for g in genes if g in self.adata.var_names]))
        if not genes:
            raise ValueError('The genes in given relations do not exists in adata!')

        genes = np.sort(genes)
        self.adata = self.adata[:, genes]
        gene2i = {g: i for i, g in enumerate(genes)}

        bg_net = np.zeros((len(genes), len(genes)))
        for r in relations:
            gene1, gene2 = r.split('_')
            if gene1 in genes and gene2 in genes:
                bg_net[gene2i[gene1], gene2i[gene2]] = 1
        return bg_net

    # SNIEE
    def load_bg_net_from_genes(self, genes):
        if (genes != np.sort(genes)).any():
            raise ValueError('The genes should be sorted!')

        print('input gene', len(genes))
        dir = os.path.abspath( os.path.dirname( __file__ ) )
        ref_gb_net = pickle.load(open(f"{dir}/stringdb_{self.organism}_v12_gb_net.pk","rb"))
        ref_genes = pickle.load(open(f"{dir}/stringdb_{self.organism}_v12_genes.pk","rb"))
        self.adata.uns['stringdb_genes'] = ref_genes

        if ref_gb_net.count_nonzero() > len(genes)*len(genes):
            bg_net, relations = self._loop_gene(genes, ref_genes, ref_gb_net)
        else:
            bg_net, relations = self._loop_bg_net(genes, ref_genes, ref_gb_net)

        bg_net = np.triu(bg_net)
        bg_net = csr_matrix(bg_net)
        print('output relations after bg_net', len(relations))
        return bg_net, relations

    # SNIEE
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

    # SNIEE
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

    # SNIEE
    def preprocess_adata(self, n_hvg=5000, random_state=0, n_pcs=30, n_neighbors=10):
        adata = self.adata
        #sc.pp.scale(adata)
        sc.pp.highly_variable_genes(adata, n_top_genes=n_hvg, flavor='cell_ranger')
        sc.tl.pca(adata)
        sc.pp.neighbors(adata, n_pcs=n_pcs, n_neighbors=n_neighbors)
        sc.tl.umap(adata, random_state=random_state)

    # SNIEE
    def _prod(self, X, obs_is, row, col, obs_cutoff=100):
        _, M = X.shape
        if len(obs_is) > obs_cutoff:
            X_tmp = X[obs_is,:]
            R = np.dot(X_tmp.T, X_tmp)
        else:
            R = np.zeros((M, M))
            for i in obs_is:
                for j, k in zip(row, col):
                    R[j, k] += X[i, j] * X[i, k]
        return R

    # SNIEE
    def sparseR2entropy(self, R, row, col):
        R_sum = R.sum(axis=0)
        R_sum[R_sum == 0] = 1  # TBD, speed up
        prob = R/R_sum
        prob_row, prob_col = prob.nonzero()
        tmp = np.array(prob.todense()[prob_row, prob_col]).reshape(-1)
        val = - tmp * np.log(tmp)
        entropy_matrix = csr_matrix((val, (prob_row, prob_col)), shape = R.shape)
        n_neighbors = np.array((R != 0).sum(axis=0))
        norm = np.log(n_neighbors)
        norm[n_neighbors == 0] = 1
        norm[n_neighbors == 1] = 1
        gene_entropy = (np.array(entropy_matrix.sum(axis=0))/norm).reshape(-1)
        relation_entropy = (gene_entropy[row]+gene_entropy[col])/2
        return relation_entropy

    # SNIEE
    def _std(self, adata):
        X = get_array(adata, layer='log1p')
        set_adata_var(adata, 'std', X.std(axis=0))

    # SNIEE
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

    # SNIEE
    def _pearsonr(self, adata):
        self._scale(adata)
        X = get_array(adata, layer='scale')
        R = np.dot(X.T, X)/X.shape[1]
        return R

    # SNIEE
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

    # SNIEE
    def test_DER(self, groupby, per_group=None, test_method="wilcoxon", method='prod'):
        myper_group = per_group
        groups = self.adata.obs[groupby].unique()
        self.groups = groups
        self.groupby = groupby
        edata = self.edata

        df_list = []
        mean_df = pd.DataFrame(index=edata.var_names)
        for ref_group in groups:
            if self.class_type == 'time':
                edata_layer_ref = self.ref_time
            else:
                edata_layer_ref = ref_group
            group_X = edata[edata.obs[groupby] == ref_group].layers[f'{edata_layer_ref}_{method}_entropy']
            mean_df[ref_group] = np.nansum(group_X, axis=0)
        for ref_group in groups:
            if self.class_type == 'time':
                edata_layer_ref = self.ref_time
            else:
                edata_layer_ref = ref_group
            sc.tl.rank_genes_groups(edata, layer=f'{edata_layer_ref}_{method}_entropy',
                                        groupby=groupby, reference=ref_group,
                                        method=test_method)
            for per_group in groups:
                if myper_group is not None and myper_group != per_group:
                    continue
                if ref_group == per_group:
                    continue

                df = sc.get.rank_genes_groups_df(edata, group=per_group)
                df['ref_group'] = ref_group
                df['per_group'] = per_group
                df['method'] = method
                df['ref_group_mean'] = df['names'].apply(lambda x: mean_df[ref_group].to_dict()[x])
                df['per_group_mean'] = df['names'].apply(lambda x: mean_df[per_group].to_dict()[x])
                edata.uns[f'{method}_rank_genes_groups_{ref_group}_{per_group}'] = edata.uns['rank_genes_groups']
                df_list.append(df)

        df = pd.concat(df_list)
        edata.uns[f'rank_genes_groups_df'] = df
        return df

    #SNIEE
    def get_DER(self, per_group=None, n_top_relations=None, method='prod', 
                p_adjust=True, p_cutoff=0.05, fc_cutoff=1, sortby='scores',
                ):
        myper_group = per_group
        edata = self.edata
        df_list = []

        for ref_group, per_group in permutations(self.groups, 2):
            if myper_group is not None and myper_group != per_group:
                continue
            df = sc.get.rank_genes_groups_df(edata, group=per_group,
                                             key=f'{method}_rank_genes_groups_{ref_group}_{per_group}')
            df['per_group'] = per_group
            df['ref_group'] = ref_group
            df['method'] = method
            if p_adjust:
                p_method = 'pvals_adj'
            else:
                p_method = 'pvals'

            df['DER'] = (df[p_method] < p_cutoff) & (df['logfoldchanges'] > fc_cutoff)
            df_list.append(df)

        df = pd.concat(df_list)
        tmp = df[df['DER']][['names', 'per_group', 'method']]
        count_df = tmp.value_counts()
        count_df = count_df[count_df == (len(self.groups) - 1)]
        count_df = count_df.reset_index()
        for per_group in self.groups:
            if myper_group is not None and myper_group != per_group:
                continue
            if per_group not in count_df['per_group'].unique():
                relations = []
                tmp_df = pd.DataFrame()
            else:
                tmp = count_df[(count_df['per_group'] == per_group) & (count_df['method'] == method)]
                relations = tmp.names.tolist()
                tmp_df = df[(df['per_group'] == per_group) & df['names'].isin(relations)]
                if sortby in ['logfoldchanges', 'scores']:
                    tmp_df = tmp_df.sort_values(by=['DER', sortby], ascending=False)
                else:
                    tmp_df = tmp_df.sort_values(by=['DER', sortby], ascending=[False, True])
                relations = tmp_df.names.drop_duplicates().tolist()
                if n_top_relations is not None:
                    relations = relations[: n_top_relations]
            edata.uns[f'{method}_{self.groupby}_{per_group}_DER'] = relations
            edata.uns[f'{method}_{self.groupby}_{per_group}_DER_df'] = tmp_df
            fn = f'{self.out_dir}/{method}_{self.groupby}_{per_group}_DER.csv'
            tmp_df.to_csv(fn, index=False)
            print_msg(f'[Output] The differential expressed relation (DER) {len(relations)} statistics are saved to:\n{fn}')
        return

    # SNIEE
    def _test_trend(self, relation, edata, layer, p_cutoff=0.05):
        sorted_samples = edata.obs.sort_values(by=['time']).index.tolist()
        val = edata[sorted_samples, relation].layers[layer].reshape(-1)
        res = mk.original_test(val, alpha=p_cutoff)
        # https://pypi.org/project/pymannkendall/
        res = {
            'relation': relation, 'layer': layer,
            'trend': res.trend, 'h': res.h, 'p': res.p, 'z': res.z,
            'Tau': res.Tau, 's': res.s, 'var_s': res.var_s, 'slope': res.slope, 'intercept': res.intercept
        }
        return res

    # SNIEE
    def _test_zero_trend(self, relation, edata, layer):
        #sorted_samples = edata.obs.sort_values(by=['time']).index.tolist()
        val = edata[:, relation].layers[layer].reshape(-1)
        t_statistic, p_value = ttest_1samp(val, 0)
        # https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.ttest_1samp.html
        res = {'relation': relation, 'layer': layer,
               't_statistic': t_statistic, 'p_value': p_value}
        return res

    # SNIEE
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

    # SNIEE
    def pathway_enrich(self, per_group, n_top_relations=None, n_space=10,
                       method='pearson', test_type='TER',
                       gene_sets=['KEGG_2021_Human',
                                  'GO_Molecular_Function_2023', 'GO_Cellular_Component_2023', 'GO_Biological_Process_2023',
                                  'MSigDB_Hallmark_2020'],
                       background=None,
                       organism='human', plot=True):
        relation_list = self.edata.uns[f'{method}_{self.groupby}_{per_group}_{test_type}']
        enr_dict = {}
        df_list = []

        if n_top_relations is None:
            n_top_relations = len(relation_list)

        for top_n in range(10, n_top_relations + 1, n_space):
            try:
                top_n, enr, df = self._enrich_for_top_n(top_n, relation_list, gene_sets, organism, background)
                enr_dict[top_n] = enr
                df_list.append(df)
            except Exception as exc:
                print(f'Top_n {top_n} generated an exception: {exc}')

        df = pd.concat(df_list)
        fn = f'{self.out_dir}/{method}_{self.groupby}_{per_group}_{test_type}_enrich.csv'
        df.to_csv(fn, index=False)
        print_msg(f'[Output] The {method} {self.groupby} {per_group} {test_type} enrich statistics are saved to:\n{fn}')

        self.edata.uns[f'{method}_{self.groupby}_{per_group}_{test_type}_enrich_res'] = enr_dict
        self.edata.uns[f'{method}_{self.groupby}_{per_group}_{test_type}_enrich_df'] = df

    # SNIEE
    def prerank_enrich_gene(self, sortby='pvals_adj', n_top_relations=None,
                       gene_sets=['KEGG_2021_Human',
                                  'GO_Molecular_Function_2023', 'GO_Cellular_Component_2023', 'GO_Biological_Process_2023',
                                  'MSigDB_Hallmark_2020'], 
                        prefix = 'test',                            
                       min_size=5, max_size=1000, permutation_num=1000, seed=0,
                       ):
        if n_top_relations is None:
            n_top_relations = len(self.edata.uns['rank_genes_groups_df'])
        if n_top_relations > len(self.edata.uns['rank_genes_groups_df']):
            n_top_relations = len(self.edata.uns['rank_genes_groups_df'])
        if n_top_relations < 1:
            n_top_relations = 1
                
        df = self.edata.uns['rank_genes_groups_df'][0:n_top_relations]
        df = df[['names', sortby]]
        df['gene1'] = df['names'].apply(lambda x: x.split('_')[0])
        df['gene2'] = df['names'].apply(lambda x: x.split('_')[1])
        df1 = df[['gene1', sortby]]
        df2 = df[['gene2', sortby]]
        df1.columns = ['gene', sortby]
        df2.columns = ['gene', sortby]
        df = pd.concat([df1, df2])
        df = df.groupby('gene').sum()
        df = df.reset_index()
        rank_df = df.sort_values(sortby, ascending=False)
        rank_df = rank_df.reset_index()
        del rank_df['index']
        res = gp.prerank(rnk=rank_df,
                         gene_sets=gene_sets,
                         threads=self.n_threads,
                         min_size=min_size, max_size=max_size,
                         permutation_num=permutation_num,
                         outdir=f'{self.out_dir}/{prefix}',
                         seed=seed,
                         verbose=True
                         )
        self.gp_res = res
        
        for term in res.results.keys():
            rank_df[f'{term} hits'] = [1 if x in res.results[term]['hits'] else 0 for x in rank_df.index]
            rank_df[f'{term} RES'] = res.results[term]['RES']
            # gseaplot(rank_metric=res.ranking, term=term, ofname=f'{self.out_dir}/{prefix}/{term}.pdf', **res.results[term])
        rank_df.to_csv(f'{self.out_dir}/{prefix}/rank.csv')
        
    
    # SNIEE
    def prerank_gsva_relation(self, gene_sets, prefix='test',
                              min_size=5,
                       ):
        df = pd.DataFrame(self.edata.X.T, columns=self.edata.obs_names,
                        index=self.edata.var_names)
        es = gp.gsva(data=df, gene_sets=gene_sets, min_size=min_size,
                     outdir=self.out_dir + '/' + prefix)
        df = es.res2d
        if 'groupby' in self.__dict__:
            df[self.groupby] = df['Name'].apply(lambda x: self.edata.obs[self.groupby].to_dict()[x])
        #df['subject'] = df['Name'].apply(lambda x: x.split(' ')[0])
        #df['time'] = df['Name'].apply(lambda x: int(x.split(' ')[1]))
        df = df.sort_values(by=['ES'])
        df.to_csv(f'./{self.out_dir}/{prefix}/prerank_gsva_relation.csv')
        
        self.gp_es = es        
        '''
        for term in gene_sets.keys():
            term_df = df[df['Term'] == term]
            sns.lineplot(term_df, x='time', y='ES', hue=self.groupby, 
                        units='subject', estimator=None)
          
            plt.title(term)
            plt.show()
            sns.barplot(term_df, x='Name', y='ES', hue=self.groupby)
            plt.title(term)
            plt.xticks([])
            plt.xlabel(None)
            plt.show()
        '''
        return df
    
    # SNIEE
    def check_common_diff(self, top_n, per_group, layer='log1p', 
                          method='prod', test_type='TER', 
                          relations=None, unit_header='subject',
                          out_dir=None):
        edata = self.edata
        adata = self.adata
        myper_group = per_group

        if relations is None:
            key = f'{method}_{self.groupby}_{per_group}_{test_type}'
            myrelations = edata.uns[key]
        else:
            myrelations = [x for x in relations if x in edata.var_names]
        
        if self.class_type == 'time':
            edata_layer_ref = self.ref_time
        #else:
            #edata_layer_ref = ref_group

        X = edata.layers[f'{edata_layer_ref}_{method}_entropy']
        top_indices = np.argsort(X, axis=1)[:, -top_n:]
        top_relations = np.array(edata.var_names)[top_indices]
        top_values = np.take_along_axis(X, top_indices, axis=1)
        #X = edata[:, myrelations].layers[f'{edata_layer_ref}_{method}_entropy']
        #plt.show()
        #sns.heatmap(X)
        overlap_stats = []
        for row in top_relations:
            overlap_count = len(set(row).intersection(myrelations))
            overlap_stats.append(overlap_count)

        edata.obs[f'top_{top_n}_overlap'] = overlap_stats
        edata.obs[f'top_{top_n}_overlap_ratio'] = np.array(overlap_stats) / top_n

    # SNIEE
    def find_relation_module(self, per_group, layer='log1p',
                          method='prod', test_type='TER', 
                          relations=None, unit_header='subject',
                          out_dir=None, label_df=None,
                          n_neighbors=10, strategy='bottom_up'):
        edata = self.edata
        adata = self.adata
        myper_group = per_group

        if relations is None:
            key = f'{method}_{self.groupby}_{per_group}_{test_type}'
            myrelations = edata.uns[key]
        else:
            myrelations = [x for x in relations if x in edata.var_names]

        genes1 = [x.split('_')[0] for x in myrelations]
        genes2 = [x.split('_')[1] for x in myrelations]

        sub_adata = adata[(adata.obs[self.groupby] == myper_group)]
        sub_adata = sub_adata[sub_adata.obs.sort_values(by=[unit_header, 'time']).index]

        X = np.multiply(sub_adata[:, genes1].layers[layer], 
                                    sub_adata[:, genes2].layers[layer]).T
        df = pd.DataFrame(X, columns=sub_adata.obs.test, index=myrelations)

        seat = SEAT(affinity="gaussian_kernel",
                    sparsification="knn_neighbors_from_X",
                    objective="SE",
                    n_neighbors=n_neighbors,
                    strategy=strategy,
                    verbose=False)
        seat.fit_predict(X)

        if label_df is None:
            label_df = pd.DataFrame()
            label_df.index = myrelations
        label_df['community'] = seat.labels_
        label_df['module'] = seat.clubs
        for col in label_df.columns:
            label = label_df[col]
            label_colors = dict(zip(set(label), sns.color_palette('Spectral', len(set(label)))))
            label_colors = [label_colors[l] for l in label]
            label_df[col] = label_colors

        units = sub_adata.obs[unit_header].unique()
        col_colors = dict(zip(units, sns.color_palette('Spectral', len(units))))
        col_colors = [col_colors[x] for x in sub_adata.obs[unit_header]]
        
        sns.clustermap(df,
                    row_linkage=seat.Z_,
                    col_cluster=False,
                    row_colors=label_df,
                    col_colors=col_colors,
                    cmap='YlGnBu')

        if out_dir is None:
            out_dir = self.out_dir
        fn = f'{out_dir}/{method}_{self.groupby}_{per_group}_{test_type}{len(myrelations)}_relation_matrix.csv'
        df.to_csv(fn)
        print_msg(f'[Output] The {method} {self.groupby} {per_group} {test_type}{len(myrelations)} relation matrix are saved to:\n{fn}')

        df = pd.DataFrame({'relation':myrelations,
                          'community':seat.labels_,
                          'module':seat.clubs})

        df.index = df.relation
        fn = f'{out_dir}/{method}_{self.groupby}_{per_group}_{test_type}{len(myrelations)}_relation_community_module.csv'
        df.to_csv(fn)
        print_msg(f'[Output] The {method} {self.groupby} {per_group} {test_type}{len(myrelations)} relation community & module are saved to:\n{fn}')

        fn = f'{out_dir}/{method}_{self.groupby}_{per_group}_{test_type}{len(myrelations)}_relation_hierarchy.nwk'        
        with open(fn, 'w') as f:
            f.write(seat.newick)
        print_msg(f'[Output] The {method} {self.groupby} {per_group} {test_type}{len(myrelations)} relation hierarchy are saved to:\n{fn}')
        return df

    def network_analysis(self, per_group, layer='log1p',
                          method='prod', test_type='TER', 
                          relations=None, unit_header='subject',
                          out_dir=None,    
                          n_neighbors=10, strategy='bottom_up'):
        edata = self.edata
        adata = self.adata
        myper_group = per_group

        if relations is None:
            key = f'{method}_{self.groupby}_{per_group}_{test_type}'
            myrelations = edata.uns[key]
        else:
            myrelations = [x for x in relations if x in edata.var_names]

        print(len(myrelations))

        sedata = edata[:, myrelations]
        print(sedata)

    # SNIEE
    def _assign_score_group(self, df, x, by='mean'):
        if by == 'median':
            cutoff = df['score'].quantile(0.5)
        else:
            cutoff = df['score'].mean()
        if x <= cutoff:
            return f'<= {by}'
        return f'> {by}'

    # SNIEE
    def survival_analysis(self, ref_group,
                        per_group,
                        relations=None,
                        groupbys=[],
                        survival_types = ['os', 'pfs'],
                        time_unit = 'time',
                        test_type='DER', method='prod',
                        title=''):

        edata = self.edata
        if relations is None:
            relations = edata.uns[f'{method}_{self.groupby}_{per_group}_{test_type}']
        else:
            relations = [x for x in relations if x in edata.var_names]
        if len(relations) == 0:
            return
        edata.obs['score'] = np.nansum(edata[:, relations].layers[f'{ref_group}_{method}_entropy'], axis=1)  # check OV nan value
        edata.obs['score_group'] = edata.obs['score'].apply(lambda x: self._assign_score_group(edata.obs, x))
        df = edata.obs.copy()
        for survival in survival_types:
            if survival not in df.columns:
                continue
            df = df[~df[f'{survival}_status'].isna()]
            df = df[~df[survival].isna()]
            df[f"{survival}_status"] = df[f"{survival}_status"].astype(bool)
            if df[[f"{survival}_status", 'score_group']].value_counts().shape[0] < 2:
                continue

            for groupby in ['score_group'] + groupbys:
                for score_group in df[groupby].unique():
                    mask_group = df[groupby] == score_group
                    time_treatment, survival_prob_treatment, conf_int = kaplan_meier_estimator(
                        df[f"{survival}_status"][mask_group],
                        df[survival][mask_group],
                        conf_type="log-log",
                    )
                    if groupby == 'score_group':
                        if score_group.startswith('<'):
                            color = 'steelblue'
                        else:
                            color = 'red'
                    else:
                        color = None
                    plt.step(time_treatment, survival_prob_treatment, where="post", label=score_group, color=color)
                    plt.fill_between(time_treatment, conf_int[0], conf_int[1], alpha=0.25, step="post", color=color)

                dt = np.dtype([(f"{survival}_status", bool), (survival, float)])
                y = [(df.iloc[i][f"{survival}_status"], df.iloc[i][survival]) for i in range(df.shape[0])]
                y = np.array(y, dtype=dt)
                chi2, p_value = compare_survival(y, df[groupby])

                plt.ylim(0, 1)
                plt.ylabel(r"est. probability of survival $\hat{S}(t)$")
                plt.xlabel(f"{survival.upper()} {time_unit} $t$")
                plt.legend(loc="best")
                plt.title(f'{title} {method}_{self.groupby}_{per_group}_{len(relations)}{test_type}s\nlog-rank test\nchi2: {round(chi2, 2)}, p-value: {p_value}')
                plt.show()
                fn = f'{self.out_dir}/{method}_{self.groupby}_{per_group}_{len(relations)}{test_type}s_{survival.upper()}_surv.png'
                print_msg(f'[Output] The survival plot are saved to:\n{fn}')

    
class SNIEETime(SNIEE):

    def __init__(self, adata, **kwargs):
        super().__init__(adata, class_type='time', **kwargs)
        

    # SNIEETime
    def calculate_entropy(self, ref_obs, per_obss, groupby, ref_time, layer='log1p'):
        print('reference observations', len(ref_obs))
        print('perputation groups', len(per_obss))
        self.ref_time = ref_time
        self.init_edata(per_obss, headers=['test', 'subject', 'time', groupby])

        if 'prod' in self.relation_methods:
            self._calculate_entropy_by_prod(ref_obs, per_obss, layer=layer)
        else:
            self._calculate_entropy(ref_obs, per_obss)

    # SNIEETime
    def _calculate_entropy_by_prod(self, ref_obs, per_obss, layer='log1p',
                                   obs_cutoff=100):
        adata = self.adata
        edata = self.edata

        bg_net = adata.varm['bg_net']
        row, col = bg_net.nonzero()

        ref_obs_is = adata[ref_obs, :].obs.i.tolist()
        ref_n = len(ref_obs_is)
        per_obss_is = [adata[per_obs, :].obs.i.tolist() for per_obs in per_obss]
        X = get_array(adata, layer=layer)
        N, M = X.shape

        gene_std = X[ref_obs_is].std(axis=0)
        ref_relation_std = (gene_std[row]+gene_std[col])/2

        print_msg(f'---Calculating the entropy for reference group')
        ref_R_sum = self._prod(X, ref_obs_is, row, col, obs_cutoff=obs_cutoff)
        ref_R_sparse = csr_matrix((ref_R_sum[row, col]/(ref_n), (row, col)), shape = (M, M))
        ref_relation_entropy = self.sparseR2entropy(ref_R_sparse, row, col)

        for per_i, per_obs_is in enumerate(per_obss_is):
            print_msg(f'---Calculating the entropy for pertutation group {per_i}, observations {len(per_obs_is)}')
            R = self._prod(X, per_obs_is, row, col, obs_cutoff=obs_cutoff)
            R = csr_matrix(((R[row, col]+ref_R_sum[row, col])/(ref_n + len(per_obs_is)), (row, col)), shape = (M, M))
            per_relation_entropy = self.sparseR2entropy(R, row, col)
            delta_entropy = np.abs(per_relation_entropy - ref_relation_entropy)

            gene_std = X[ref_obs_is + per_obs_is].std(axis=0)
            per_relation_std = (gene_std[row]+gene_std[col])/2
            delta_std = np.abs(per_relation_std - ref_relation_std)
            edata.X[per_i, :] = delta_entropy * delta_std

        edata.layers[f'{self.ref_time}_prod_entropy'] = edata.X
        self.edata = edata

    # SNIEETime
    def _calculate_entropy(self, ref_obs, per_obss):
        adata_dict = {}
        print_msg(f'---Calculating the group entropy for reference')
        adata = self._calculate_group_entropy(ref_obs)
        adata_dict[f'{ref_obs}'] = adata

        for per_obs in per_obss:
            print_msg(f'---Calculating the group entropy for {per_obs}')
            adata = self._calculate_group_entropy(ref_obs + per_obs)
            adata_dict[','.join(per_obs)] = adata
        self.adata_dict = adata_dict

        self._calculate_delta_entropy(ref_obs, per_obss)

    # SNIEETime
    def _calculate_delta_entropy(self, ref_obs, per_obss):
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
            edata.layers[f'{self.ref_time}_{method}_entropy'] = entropy_matrix.copy()
            edata.layers[f'{self.ref_time}_{method}_prob'] = entropy_matrix.copy()
            edata.layers[f'{self.ref_time}_{method}_bg_net'] = entropy_matrix.copy()

        for i, per_obs in enumerate(per_obss):
            ref_adata = adata_dict[f'{ref_obs}']
            adata = adata_dict[','.join(per_obs)]
            delta_std = np.abs(adata.varm['std'] - ref_adata.varm['std'])

            for method in self.relation_methods:
                delta_entropy = np.abs(adata.varm[f'{method}_entropy'] - ref_adata.varm[f'{method}_entropy'])
                val = (np.array(delta_entropy[row, col]) * np.array(delta_std[row, col])).reshape(-1)
                # direct dot product will raise much more entry, further investigate
                edata.layers[f'{self.ref_time}_{method}_entropy'][i, :] = val

                val = adata.varm[f'{method}_bg_net'][row, col].reshape(-1)
                edata.layers[f'{self.ref_time}_{method}_bg_net'][i, :] = val

                val = csr_matrix(adata.varm[f'{method}_prob'])[row, col].reshape(-1)
                edata.layers[f'{self.ref_time}_{method}_prob'][i, :] = val

        self.edata = edata

    # SNIEETime
    def _calculate_group_entropy(self, obs):
        adata = self.adata[obs, :]

        for method in self.relation_methods:
            self._calculate_node_entropy(adata, method)
            self._calculate_edge_entropy(adata, method)
        return adata

    # SNIEETime
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

    # SNIEETime
    def _calculate_edge_entropy(self, adata, method):
        row, col = adata.varm['bg_net'].nonzero()

        for key in [f'{method}_entropy', 'std']:
            val = (adata.var[key].to_numpy()[row]+adata.var[key].to_numpy()[col])/2
            adata.varm[key] = csr_matrix((val, (row, col)), shape = adata.varm['bg_net'].shape)

    # SNIEETime
    def test_TER(self, per_group=None, p_cutoff=0.05, method='prod', groups=None):
        myper_group = per_group
        edata = self.edata

        if groups is None:
            groups = self.groups
        for per_group in groups:
            if myper_group is not None and myper_group != per_group:
                continue
            trend_list = []
            relations = edata.uns[f'{method}_{self.groupby}_{per_group}_DER']
            filtered_relations = []
            for relation in relations:
                per_edata = edata[edata.obs[self.groupby] == per_group, :]
                layer = f'{self.ref_time}_{method}_entropy'
                trend_res = self._test_trend(relation, per_edata, layer, p_cutoff=p_cutoff)
                is_per_trend = trend_res['trend'] != 'no trend'

                ref_edata = edata[edata.obs[self.groupby] != per_group, :]
                zero_res = self._test_zero_trend(relation, ref_edata, layer)
                is_ref_zero = zero_res['p_value'] < p_cutoff
                trend_res.update(zero_res)

                is_TER = is_per_trend and is_ref_zero
                trend_res['TER'] = is_TER
                trend_res['per_group'] = per_group
                if is_TER:
                    filtered_relations.append(relation)
                trend_list.append(trend_res)

            print(f'{method}_{self.groupby}_{per_group}', 'DER', len(relations), 'TER', len(filtered_relations))
            edata.uns[f'{method}_{self.groupby}_{per_group}_TER'] = filtered_relations

            df = pd.DataFrame(trend_list)
            fn = f'{self.out_dir}/{method}_{self.groupby}_{per_group}_TER.csv'
            df.to_csv(fn, index=False)
            print_msg(f'[Output] The trend expressed relation (TER) statistics are saved to:\n{fn}')
            edata.uns[f'{method}_{self.groupby}_{per_group}_TER_df'] = df
        return

    # SNIEETime
    def test_val_trend_entropy(self, relations, method='pearson',
                               p_cutoff=0.05,
                               out_prefix='./test'):
        edata = self.edata
        candidates = []
        trend_list = []
        for relation in relations:
            if relation not in self.edata.var_names:
                continue            
            layer = f'{self.ref_time}_{method}_entropy'
            trend_res = self._test_trend(relation, edata, layer, p_cutoff=p_cutoff)
            is_trend = trend_res['trend'] != 'no trend'
            zero_res = self._test_zero_trend(relation, edata, layer)
            is_zero = zero_res['p_value'] < p_cutoff
            trend_res.update(zero_res)

            is_TER = is_trend or is_zero
            trend_res['TER'] = is_TER

            trend_list.append(trend_res)
            if is_TER:
                candidates.append(relation)

        print(method, 'val trend before', len(relations), 'after', len(candidates))

        df = pd.DataFrame(trend_list)
        fn = f'{out_prefix}_TER.csv'
        df.to_csv(fn, index=False)
        print_msg(f'[Output] The validation trend expressed relation (TER) statistics are saved to:\n{fn}')

        return candidates

    # SNIEETime
    def pathway_enrich(self, per_group, **kwargs):
        super(SNIEETime, self).pathway_enrich(per_group=per_group, **kwargs)


class SNIEEGroup(SNIEE):

    # SNIEEGroup
    def __init__(self, adata, **kwargs):
        super().__init__(adata, **kwargs)
        self.class_type = 'group'

    # SNIEEGroup
    def calculate_ref_entropy(self, groupby, groups, layer='log1p', obs_cutoff=100):
        adata = self.adata
        X = get_array(adata, layer=layer)
        N, M = X.shape
        bg_net = adata.varm['bg_net']
        row, col = bg_net.nonzero()

        self.ref_R_sum_dict = {}
        self.ref_obs_is_dict = {}
        self.ref_entropy_dict = {}
        self.ref_std_dict = {}

        for group in groups:
            ref_obs_is = adata.obs[adata.obs[groupby] == group].i.tolist()
            self.ref_obs_is_dict[group] = ref_obs_is
            ref_R_sum = self._prod(X, ref_obs_is, row, col, obs_cutoff=obs_cutoff)
            self.ref_R_sum_dict[group] = ref_R_sum

            ref_R_sparse = csr_matrix((ref_R_sum[row, col]/len(ref_obs_is), (row, col)), shape = (M, M))
            self.ref_entropy_dict[group] = self.sparseR2entropy(ref_R_sparse, row, col)

            gene_std = X[ref_obs_is].std(axis=0)
            self.ref_std_dict[group] = (gene_std[row]+gene_std[col])/2

    # SNIEEGroup
    def load_ref_entropy(self, train_sniee_obj):
        ref_edata = train_sniee_obj.edata
        edata = self.edata
        relations = sorted(list(set(ref_edata.var_names) & set(edata.var_names)))
        ref_edata = ref_edata[:, relations]
        edata = edata[:, relations]
        _, M = self.adata.shape

        ref_row, ref_col = ref_edata.var.gene1_i, ref_edata.var.gene2_i
        row, col = edata.var.gene1_i, edata.var.gene2_i
        self.ref_R_sum_dict = {}
        self.ref_entropy_dict = {}
        self.ref_std_dict = {}
        for group in train_sniee_obj.groups:
            ref_R_sum = train_sniee_obj.ref_R_sum_dict[group]
            val = ref_R_sum[ref_row, ref_col]
            ref_R_sum = csr_matrix((val, (row, col)), shape = (M, M)).toarray()
            self.ref_R_sum_dict[group] = ref_R_sum

            ref_entropy = train_sniee_obj.ref_entropy_dict[group]
            ref_entropy = ref_entropy[ref_edata.var.i]
            self.ref_entropy_dict[group] = ref_entropy

            ref_std = train_sniee_obj.ref_std_dict[group]
            ref_std = ref_std[ref_edata.var.i]
            self.ref_std_dict[group] = ref_std
        self.ref_obs_is_dict = train_sniee_obj.ref_obs_is_dict

    # SNIEEGroup
    def calculate_entropy(self, groupby, layer='log1p', obs_cutoff=100,
                          method='prod', train_sniee_obj=None,
                          headers=[]):
        adata = self.adata
        X = get_array(adata, layer=layer)
        _, M = X.shape

        if groupby not in adata.obs:
            raise KeyError(f'{groupby} not in adata.obs.')
        if method != 'prod':
            raise KeyError(f'method "{method}" should be "prod".')
        
        groups = adata.obs[groupby].unique()
        self.groups = groups
        print('cluster groups', len(groups))

        per_obss = [[x] for x in adata.obs_names]
        self.init_edata(per_obss, headers=['test', 'subject', groupby] + headers)
        edata = self.edata

        if train_sniee_obj is None:
            self.calculate_ref_entropy(groupby, groups, layer=layer, obs_cutoff=obs_cutoff)
        else:
            self.load_ref_entropy(train_sniee_obj)

        row, col = edata.var.gene1_i, edata.var.gene2_i

        per_obss_is = [adata[per_obs, :].obs.i.tolist() for per_obs in per_obss]
        for ref_group in groups:
            for per_i, per_obs_is in enumerate(per_obss_is):
                ref_obs_is = self.ref_obs_is_dict[ref_group]

                per_R_sum = self._prod(X, per_obs_is, row, col, obs_cutoff=obs_cutoff) + self.ref_R_sum_dict[ref_group]
                per_R_sparse = csr_matrix((per_R_sum[row, col]/(len(ref_obs_is) + len(per_obs_is)), (row, col)), shape = (M, M))
                per_relation_entropy = self.sparseR2entropy(per_R_sparse, row, col)
                delta_entropy = np.abs(per_relation_entropy - self.ref_entropy_dict[ref_group])

                if train_sniee_obj is None:
                    gene_std = X[ref_obs_is + per_obs_is].std(axis=0)
                else:
                    ref_X = get_array(train_sniee_obj.adata, layer=layer)
                    ref_X = ref_X[ref_obs_is, :][:, train_sniee_obj.adata[:, self.adata.var_names].var.i]
                    per_X = np.concatenate((ref_X, X), axis=0)
                    gene_std = per_X.std(axis=0)
                per_relation_std = (gene_std[row]+gene_std[col])/2
                delta_std = np.abs(per_relation_std - self.ref_std_dict[ref_group])
                edata.X[per_i, :] = delta_entropy * delta_std
            edata.layers[f'{ref_group}_{method}_entropy'] = edata.X

    # SNIEEGroup
    def test_TER(self, per_group=None, p_cutoff=0.05, method='prod', groups=None):
        edata = self.edata
        myper_group = per_group

        if groups is None:
            groups = self.groups
        for per_group in groups:
            if myper_group is not None and per_group != myper_group:
                continue
            relations = edata.uns[f'{method}_{self.groupby}_{per_group}_DER']
            trend_list = []
            filtered_relations = []
            for relation in relations:
                is_TER = []
                for ref_group in groups:
                    if ref_group == per_group:
                        continue
                    layer = f'{ref_group}_{method}_entropy' 
                    ref_edata = edata[edata.obs[self.groupby] == ref_group, :]
                    zero_res = self._test_zero_trend(relation, ref_edata, layer)
                    is_ref_zero = zero_res['p_value'] < p_cutoff

                    is_TER.append(is_ref_zero)
                    zero_res['is_ref_zero'] = is_ref_zero
                    zero_res['ref_group'] = ref_group
                    zero_res['per_group'] = per_group
                    trend_list.append(zero_res)
                if all(is_TER):
                    filtered_relations.append(relation)
                    
            print(f'{method}_{self.groupby}_{per_group}', 'DER', len(relations), 'TER', len(filtered_relations))
            edata.uns[f'{method}_{self.groupby}_{per_group}_TER'] = list(filtered_relations)

            df = pd.DataFrame(trend_list)
            if not df.empty:
                df['TER'] = df['relation'].apply(lambda x: True if x in filtered_relations else False)
            fn = f'{self.out_dir}/{method}_{self.groupby}_{per_group}_TER.csv'
            df.to_csv(fn, index=False)
            print_msg(f'[Output] The trend expressed relation (TER) {len(filtered_relations)} statistics are saved to:\n{fn}')
            edata.uns[f'{method}_{self.groupby}_{per_group}_TER_df'] = df
        return

    # SNIEEGroup
    def pathway_enrich(self, groups=None, **kwargs):
        if groups is None:
            groups = self.groups
        for per_group in groups:
            super(SNIEEGroup, self).pathway_enrich(per_group=per_group, **kwargs)

    # SNIEEGroup
    def survival_analysis(self, ref_group, groups=None, **kwargs):
        if groups is None:
            groups = self.groups
        for per_group in groups:
            super().survival_analysis(ref_group, per_group=per_group, **kwargs)

class SNIEETimeGroup(SNIEETime):

    def __init__(self, adata, **kwargs):
        super().__init__(adata, **kwargs)
        self.class_type = 'time'

    # SNIEETimeGroup
    def calculate_entropy(self, tgroupby, groupby, ref_time='Ref', layer='log1p',
                          ref_as_per=True,
                          headers=[]):
        self.ref_time = ref_time
        edata_list = []
        for group in self.adata.obs[groupby].unique():
            adata = self.adata[self.adata.obs[groupby] == group]
            ref_obs = adata[adata.obs[tgroupby] == ref_time].obs_names
            if ref_as_per:
                per_obss = [[x] for x in adata.obs_names]
            else:
                per_obss = [[x] for x in adata[adata.obs[tgroupby] != ref_time].obs_names]           
            print('group', group, 'reference observations', len(ref_obs))
            print('group', group, 'perputation groups', len(per_obss))

            self.init_edata(per_obss, headers=['test', 'subject', 'time', groupby]+headers)
            
            if 'prod' in self.relation_methods:
                self._calculate_entropy_by_prod(ref_obs, per_obss, layer=layer)
            else:
                self._calculate_entropy(ref_obs, per_obss)
            
            edata = self.edata
            edata_list.append(edata)

        edata = ad.concat(edata_list)
        self.edata = edata

class SNIEEGroupTime(SNIEETime):

    def __init__(self, adata, **kwargs):
        super().__init__(adata, **kwargs)
        self.class_type = 'grouptime'

    # SNIEEGroupTime
    def calculate_entropy(self, tgroupby, groupby, ref_group, layer='log1p',
                          ref_as_per=True,
                          headers=[]):
        self.ref_time = ref_group
        edata_list = []
        for tgroup in self.adata.obs[tgroupby].unique():
            adata = self.adata[self.adata.obs[tgroupby] == tgroup]
            ref_obs = adata[adata.obs[groupby] == ref_group].obs_names
            if ref_as_per:
                per_obss = [[x] for x in adata.obs_names]
            else:
                per_obss = [[x] for x in adata[adata.obs[groupby] != ref_group].obs_names]           
            print('tgroup', tgroup, 'reference observations', len(ref_obs))
            print('tgroup', tgroup, 'perputation groups', len(per_obss))

            self.init_edata(per_obss, headers=['test', 'subject', tgroupby, groupby]+headers)
            
            if 'prod' in self.relation_methods:
                self._calculate_entropy_by_prod(ref_obs, per_obss, layer=layer)
            else:
                self._calculate_entropy(ref_obs, per_obss)
            
            edata = self.edata
            edata_list.append(edata)

        edata = ad.concat(edata_list)
        self.edata = edata