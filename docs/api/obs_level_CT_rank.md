# obs_level_CT_rank

## Function

```python
crisgi_obj.obs_level_CT_rank(gene_sets, prefix='test', min_size=5)
```

Performs observation-level cell type (CT) ranking using gene set variation analysis (GSVA) on the provided gene sets. This method is designed as a member function of the `crisgi_obj` class. It computes enrichment scores for each observation (cell/sample), ranks them, and saves the results to a CSV file. If group information is available, it annotates the results accordingly.

## Parameters

| Name      | Type    | Description                                                                 |
|-----------|---------|-----------------------------------------------------------------------------|
| gene_sets | object  | Dictionary or compatible object containing gene sets for GSVA analysis.     |
| prefix    | str     | Prefix for output directory and files. Default is `'test'`.                 |
| min_size  | int     | Minimum number of genes required in each gene set. Default is `5`.          |

## Return type

`pandas.DataFrame`

## Returns

A DataFrame with GSVA enrichment scores for each observation, optionally annotated with group information and sorted by enrichment score (ES). The results are also exported as a CSV file in the specified output directory.

## Attributes Set

- `self.gp_es`: Stores the GSVA result object for downstream analysis.

## Example

```python
# Assuming `crisgi_obj` is an instance with `edata`, `out_dir`, and optionally `groupby` attributes
gene_sets = {'Pathway1': ['GeneA', 'GeneB'], 'Pathway2': ['GeneC', 'GeneD']}
result_df = crisgi_obj.obs_level_CT_rank(gene_sets=gene_sets, prefix='experiment1', min_size=10)

# The results are saved to './<out_dir>/experiment1/prerank_gsva_interaction.csv'
# The DataFrame `result_df` contains the ranked enrichment scores for each observation.
```
