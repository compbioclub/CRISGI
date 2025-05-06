# pheno_level_CT_rank

## Function

```python
crisgi_obj.pheno_level_CT_rank(
    ref_group,
    target_group,
    sortby='pvals_adj',
    n_top_interactions=None,
    gene_sets=[
        'KEGG_2021_Human',
        'GO_Molecular_Function_2023',
        'GO_Cellular_Component_2023',
        'GO_Biological_Process_2023',
        'MSigDB_Hallmark_2020'
    ],
    prefix='test',
    min_size=5,
    max_size=1000,
    permutation_num=1000,
    seed=0,
)
```

Performs gene set enrichment analysis (GSEA) on ranked gene interactions between a reference group and a comparison group, using multiple gene set libraries. The function ranks genes based on their adjusted p-values or another specified metric, aggregates scores for each gene, and runs GSEA using the `gseapy` package. Results are saved to disk and stored as an attribute for further analysis.

## Parameters

| Name               | Type            | Description                                                                                  |
|--------------------|-----------------|----------------------------------------------------------------------------------------------|
| `ref_group`        | `str`           | Reference group name for comparison.                                                         |
| `target_group`     | `str`           | Comparison group name.                                                                       |
| `sortby`           | `str`           | Column name to sort interactions by (default: `'pvals_adj'`).                                |
| `n_top_interactions` | `int or None` | Number of top interactions to include (default: all).                                        |
| `gene_sets`        | `list of str`   | List of gene set libraries to use for enrichment analysis.                                   |
| `prefix`           | `str`           | Prefix for output directory and files.                                                       |
| `min_size`         | `int`           | Minimum size of gene sets to include in analysis.                                            |
| `max_size`         | `int`           | Maximum size of gene sets to include in analysis.                                            |
| `permutation_num`  | `int`           | Number of permutations for GSEA.                                                             |
| `seed`             | `int`           | Random seed for reproducibility.                                                             |

## Return type

`None`

## Returns

This function does not return a value. It saves GSEA results and ranked gene data to disk and sets an attribute for further access.

## Attributes Set

- `self.gp_res`: Stores the GSEA results object for further analysis.

## Example

```python
# Assume `obj` is an instance of the class containing pheno_level_CT_rank

obj.pheno_level_CT_rank(
    ref_group='Control',
    target_group='Treatment',
    sortby='pvals_adj',
    n_top_interactions=100,
    gene_sets=[
        'KEGG_2021_Human',
        'GO_Biological_Process_2023'
    ],
    prefix='experiment1',
    min_size=10,
    max_size=500,
    permutation_num=2000,
    seed=42
)

# After execution, results are saved in the specified output directory,
# and the GSEA results are accessible via obj.gp_res.
```