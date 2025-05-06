## Function

```python
crisgi_obj.pheno_level_accumulated_top_n_ORA(
    target_group, 
    n_top_interactions=None, 
    n_space=10, 
    method='prod', 
    test_type='TER', 
    gene_sets=[
        'KEGG_2021_Human',
        'GO_Molecular_Function_2023',
        'GO_Cellular_Component_2023',
        'GO_Biological_Process_2023',
        'MSigDB_Hallmark_2020'
    ],
    background=None, 
    organism='human', 
    plot=True,
)
```

Performs an accumulated over-representation analysis (ORA) for the top-N interactions at the phenotype level. This function iteratively evaluates enrichment for increasing numbers of top interactions, aggregates the results, and saves the enrichment statistics to a CSV file. The results are also stored in the object's attributes for further analysis.


## Parameters

| Name                | Type            | Description                                                                                                         |
|---------------------|-----------------|---------------------------------------------------------------------------------------------------------------------|
| `target_group`      | str             | The group or phenotype to analyze.                                                                                  |
| `n_top_interactions`| int, optional   | The maximum number of top interactions to consider. If `None`, uses all available interactions.                     |
| `n_space`           | int, optional   | Step size for the number of top interactions to include in each enrichment analysis. Default is 10.                 |
| `method`            | str, optional   | Correlation or interaction method used to select interactions. Default is 'prod'.                                   |
| `test_type`         | str, optional   | Type of statistical test applied (e.g., 'TER'). Default is 'TER'.                                                   |
| `gene_sets`         | list of str     | List of gene set databases to use for enrichment analysis.                                                          |
| `background`        | list or None    | Background gene set for enrichment. If `None`, uses default background.                                             |
| `organism`          | str, optional   | Organism name for gene set enrichment (e.g., 'human'). Default is 'human'.                                          |
| `plot`              | bool, optional  | Whether to generate plots for the enrichment results. Default is `True`.                                            |


## Return type

None

## Returns

This function does not return a value. It saves the enrichment results as a CSV file and updates the object's attributes with the enrichment results and dataframes.

## Attributes Set

- `self.edata.uns[f'{method}_{self.groupby}_{target_group}_{test_type}_enrich_res']`:  
    Dictionary mapping top-N values to enrichment results.
- `self.edata.uns[f'{method}_{self.groupby}_{target_group}_{test_type}_enrich_df']`:  
    DataFrame containing concatenated enrichment results for all top-N values.

## Example

```python
# Assume `obj` is an instance of the class containing this method

# Perform accumulated ORA for the 'disease' group using default parameters
obj.pheno_level_accumulated_top_n_ORA(target_group='disease')

# Specify custom parameters, such as using only the top 50 interactions and a different gene set
obj.pheno_level_accumulated_top_n_ORA(
    target_group='tissue',
    n_top_interactions=50,
    n_space=5,
    method='prod',
    test_type='TER',
    gene_sets=['KEGG_2021_Human'],
    organism='human',
    plot=False
)

# After execution, results are saved to a CSV file and stored in:
# obj.edata.uns['pearson_<groupby>_disease_TER_enrich_res']
# obj.edata.uns['pearson_<groupby>_disease_TER_enrich_df']
```

