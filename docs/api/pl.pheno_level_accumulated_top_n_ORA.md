# pl.pheno_level_accumulated_top_n_ORA

## Function

```python
pl.pheno_level_accumulated_top_n_ORA(
    target_group,
    method="prod",
    test_type="TER",
    p_adjust=True,
    p_cutoff=0.05,
    n_top_pathway=10,
    n_top_interactions=500,
    piority_term=None,
    eval_para='top_n_ratio',
    dataset_name=None
)
```

Performs pathway enrichment analysis at the phenotype level using the top-N accumulation strategy. This function evaluates pathway enrichment results across multiple top-N gene sets, ranks pathways based on a specified evaluation parameter, and visualizes the results as a heatmap. It supports prioritization of specific pathways and flexible evaluation metrics.

## Parameters

| Name               | Type                | Description                                                                                                    |
|--------------------|---------------------|----------------------------------------------------------------------------------------------------------------|
| `target_group`     | str                 | Target group for enrichment analysis (e.g., phenotype or cluster name).                                        |
| `method`           | str, optional       | Correlation method used for analysis. Default is `"prod"`.                                                     |
| `test_type`        | str, optional       | Type of statistical test. Default is `"TER"`.                                                                  |
| `p_adjust`         | bool, optional      | Whether to use adjusted p-values for filtering. Default is `True`.                                             |
| `p_cutoff`         | float, optional     | P-value cutoff for significance filtering. Default is `0.05`.                                                  |
| `n_top_pathway`    | int, optional       | Number of top pathways to display in the heatmap. Default is `10`.                                             |
| `n_top_interactions`| int, optional      | Maximum number of top interactions (gene sets) to consider. Default is `500`.                                  |
| `piority_term`     | list or None, optional | List of pathway terms to prioritize or `None` for no prioritization. Default is `None`.                     |
| `eval_para`        | str, optional       | Evaluation parameter for ranking pathways. Options: `'top_n_ratio'`, `'overlap_ratio'`, `'P-value'`, `'Adjusted P-value'`, `'Odds Ratio'`, `'Combined Score'`, `'-logP'`. Default is `'top_n_ratio'`. |
| `dataset_name`     | str or None, optional | Name of the dataset for labeling outputs. Default is `None`.                                                 |

## Return type

`None`

## Returns

- Saves a heatmap plot of the top-N pathway enrichment results to the output directory.
- Displays the heatmap in the current matplotlib session.
- Prints the output file path.

## Attributes Set

- No new attributes are set on the CRISGI object by this function.

## Example

```python
# Assume crisgi_obj is an instance of CRISGI with enrichment results computed

# Perform top-N pathway enrichment analysis for the 'Tumor' group
crisgi_obj.pheno_level_accumulated_top_n_ORA(
    target_group='Tumor',
    method='prod',
    test_type='TER',
    p_adjust=True,
    p_cutoff=0.01,
    n_top_pathway=15,
    n_top_interactions=300,
    piority_term=['Apoptosis', 'Cell Cycle'],
    eval_para='overlap_ratio',
    dataset_name='CancerStudy'
)

# The function will save and display a heatmap of the top 15 pathways
# ranked by overlap ratio for the 'Tumor' group, highlighting prioritized terms.
```
