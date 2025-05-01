# test_TER

## Function

```python
crisgi_obj.test_TER(
    per_group=None, 
    p_cutoff=0.05, 
    method="prod", 
    groups=None
)
```

Identifies Trend Expressed Interactions (TER) for each group in the dataset. This function evaluates interactions based on trend analysis and statistical significance, saving the results and statistics for downstream analysis.

## Parameters

| Name        | Type           | Description                                                                                 |
|-------------|----------------|--------------------------------------------------------------------------------------------|
| per_group   | str or None    | Specific group to analyze. If `None`, all groups in `groups` are processed.                |
| p_cutoff    | float          | P-value cutoff for statistical significance (default: 0.05).                               |
| method      | str            | Method used for interaction analysis (e.g., `'prod'`).                                     |
| groups      | list or None   | List of groups to analyze. If `None`, uses `self.groups`.                                  |

## Return type

`None`

## Returns

This function does not return a value. It saves TER results and statistics to the `edata.uns` attribute and outputs CSV files with TER statistics for each group.

## Attributes Set

- `edata.uns[f'{method}_{self.groupby}_{per_group}_TER']`: List of filtered interactions identified as TER.
- `edata.uns[f'{method}_{self.groupby}_{per_group}_TER_df']`: DataFrame containing detailed TER statistics for each interaction.

## Example

```python
# Assume `crisgi` is an instance of the CRISGI class, already initialized with data.

# Run TER analysis for all groups using the default method and p-value cutoff
crisgi.test_TER()

# Run TER analysis for a specific group with a custom p-value cutoff
crisgi.test_TER(per_group='GroupA', p_cutoff=0.01)

# Run TER analysis for a custom list of groups and a different method
custom_groups = ['GroupA', 'GroupB']
crisgi.test_TER(groups=custom_groups, method='sum')
```

After execution, TER results and statistics are saved in the `edata.uns` attribute and as CSV files in the specified output directory.
