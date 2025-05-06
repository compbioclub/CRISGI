# network_analysis

## Function

```python
crisgi_obj.network_analysis(
    target_group,
    layer="log1p",
    method="prod",
    test_type="TER",
    interactions=None,
    unit_header="subject",
    out_dir=None,
    n_neighbors=10,
    strategy="bottom_up",
)
```

Performs network analysis on the dataset using specified parameters and interaction sets. This method extracts and analyzes interaction features from the data, supporting various strategies and test types for flexible network exploration.

## Parameters

| Name          | Type            | Description                                                                                   |
|---------------|-----------------|-----------------------------------------------------------------------------------------------|
| target_group  | str             | The group identifier for which the network analysis is performed.                             |
| layer         | str, optional   | The data layer to use for analysis (default: `'log1p'`).                                      |
| method        | str, optional   | The method for interaction calculation (default: `'prod'`).                                   |
| test_type     | str, optional   | The statistical test type to use (default: `'TER'`).                                          |
| interactions  | list, optional  | List of interaction features to include; if `None`, uses default from `edata.uns`.            |
| unit_header   | str, optional   | The header indicating the unit of analysis (default: `'subject'`).                            |
| out_dir       | str, optional   | Output directory for results; if `None`, results are not saved to disk.                       |
| n_neighbors   | int, optional   | Number of neighbors to consider in the analysis (default: `10`).                              |
| strategy      | str, optional   | Strategy for network construction (default: `'bottom_up'`).                                   |

## Return type

`None`

## Returns

This function does not return a value. It performs network analysis and may print or save intermediate results depending on the parameters.

## Attributes Set

- May access or update `self.edata`, `self.adata`, and related attributes.
- Uses or sets keys in `edata.uns`.

## Example

```python
# Assume `crisgi` is an instance of the CRISGI class

# Perform network analysis for group 'A' with default settings
crisgi.network_analysis(target_group='A')

# Perform network analysis with custom interactions and output directory
custom_interactions = ['gene1', 'gene2', 'gene3']
crisgi.network_analysis(
    target_group='B',
    interactions=custom_interactions,
    out_dir='/path/to/output',
    n_neighbors=15,
    strategy='top_down'
)
```
