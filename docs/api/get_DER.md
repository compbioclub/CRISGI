# get_DER

## Function

```python
crisgi_obj.get_DER(
    target_group=None,
    n_top_interactions=None,
    method='prod',
    p_adjust=True,
    p_cutoff=0.05,
    fc_cutoff=1,
    sortby='scores'.
)
```

Identifies Differentially Expressed Reactions (DER) between groups in the dataset. This function compares all pairs of groups, computes statistics for each, and identifies interactions that are significantly differentially expressed according to specified thresholds. Results are saved to CSV files and stored in the `edata.uns` attribute for downstream analysis.

## Parameters

| Name               | Type      | Description                                                                                   |
|--------------------|-----------|-----------------------------------------------------------------------------------------------|
| `target_group`     | str or None | The group to compare against all others. If `None`, all groups are compared.                |
| `n_top_interactions` | int or None | Number of top interactions to return per group. If `None`, returns all.                   |
| `method`           | str       | Method used for ranking genes (e.g., `'prod'`).                                               |
| `p_adjust`         | bool      | Whether to use adjusted p-values (`True`) or raw p-values (`False`).                          |
| `p_cutoff`         | float     | P-value cutoff for significance.                                                              |
| `fc_cutoff`        | float     | Log fold change cutoff for significance.                                                      |
| `sortby`           | str       | Column to sort results by (`'scores'`, `'logfoldchanges'`, or other valid column names).      |

## Return type

`None`

## Returns

This function does not return a value. It saves the results to CSV files and updates the `edata.uns` attribute with DER interactions and their corresponding dataframes for each group.

## Attributes Set

- `edata.uns[f'{method}_{self.groupby}_{target_group}_DER']`:  
  List of DER interaction names for each target group.
- `edata.uns[f'{method}_{self.groupby}_{target_group}_DER_df']`:  
  DataFrame containing DER statistics for each target group.

## Example

```python
# Assume `obj` is an instance of the class containing `get_DER`
# Identify DERs for all groups using default parameters
obj.get_DER()

# Identify DERs for a specific group with custom thresholds
obj.get_DER(
    target_group='GroupA',
    n_top_interactions=20,
    method='prod',
    p_adjust=True,
    p_cutoff=0.01,
    fc_cutoff=1.5,
    sortby='logfoldchanges'
)

# After running, results are saved to CSV files in `obj.out_dir`
# and DER lists/dataframes are available in `obj.edata.uns`
```
