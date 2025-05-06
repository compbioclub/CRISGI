# test_DER

## Function

```python
crisgi_obj.test_DER(
    groupby,
    target_group=None,
    test_method="wilcoxon",
    method='pos_coexp',
)
```

Performs differential entropy ranking (DER) analysis on the provided AnnData object, comparing groups defined by the `groupby` column. The function computes group-wise entropy, applies statistical tests (e.g., Wilcoxon), and aggregates results for downstream analysis.

## Parameters

| Name           | Type     | Description                                                                                     |
|----------------|----------|-------------------------------------------------------------------------------------------------|
| groupby        | str      | The column in `adata.obs` used to define groups for comparison.                                 |
| target_group   | str, optional | Specific group to compare against the reference group. If `None`, all groups are compared. |
| test_method    | str, optional | Statistical test method to use (default: `"wilcoxon"`).                                    |
| method         | str, optional | Entropy calculation method (default: `'pos_coexp'`).                                       |

## Return type

`pandas.DataFrame`

## Returns

A DataFrame containing differential entropy ranking results for each group comparison. Columns include gene names, reference and target groups, method used, and mean entropy values for each group.

## Attributes Set

- `self.groups`: List of unique group names from `groupby`.
- `self.groupby`: The groupby column name.
- `edata.uns[f'{method}_rank_genes_groups_{ref_group}_{target_group}']`: Stores ranking results for each group comparison.
- `edata.uns['rank_genes_groups_df']`: Stores the concatenated DataFrame of all results.

## Example

```python
# Assume `obj` is an instance with an AnnData object as `obj.adata`
# Compare groups in the 'cell_type' column using the default Wilcoxon test and 'pos_coexp' method
result_df = obj.test_DER(groupby='cell_type')

# To compare a specific group, e.g., 'B_cell', against all others:
result_df = obj.test_DER(groupby='cell_type', target_group='B_cell')

# Access the results
print(result_df.head())

# Access attributes set by the function
print(obj.groups)  # List of group names
print(obj.groupby) # The groupby column name
```
