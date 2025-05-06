# check_common_diff

## Function

```python
crisgi_obj.check_common_diff(
    top_n,
    target_group,
    layer="log1p",
    method="pos_coexp",
    test_type="TER",
    interactions=None,
    unit_header="subject",
    out_dir=None,
)
```

Identifies and analyzes the overlap between the top N differential features (e.g., genes or interactions) and a reference set within the dataset. This function is useful for evaluating the consistency of differential features across groups or conditions in the CRISGI analysis workflow.

## Parameters

| Name           | Type         | Description                                                                                  |
|----------------|--------------|----------------------------------------------------------------------------------------------|
| `top_n`        | `int`        | Number of top features to consider for overlap analysis.                                     |
| `target_group` | `str`        | The group or condition by which to stratify the analysis.                                    |
| `layer`        | `str`        | Data layer to use for entropy calculation (default: `'log1p'`).                              |
| `method`       | `str`        | Method for entropy calculation (default: `'pos_coexp'`).                                     |
| `test_type`    | `str`        | Statistical test type to use (default: `'TER'`).                                             |
| `interactions` | `list` or `None` | List of features to compare for overlap. If `None`, uses default from `edata.uns`.       |
| `unit_header`  | `str`        | Header indicating the unit of analysis (default: `'subject'`).                               |
| `out_dir`      | `str` or `None`  | Output directory to save results. If `None`, saves to current directory.                 |

## Return type

`None`

## Returns

This function does not return a value. It updates the `obs` attribute of the `edata` object with two new columns:
- `top_{top_n}_overlap`: Number of overlapping features for each observation.
- `top_{top_n}_overlap_ratio`: Ratio of overlapping features to `top_n`.

It also saves a CSV file with these statistics to the specified output directory.

## Attributes Set

- `edata.obs['top_{top_n}_overlap']`
- `edata.obs['top_{top_n}_overlap_ratio'}`

## Example

```python
# Assume crisgi is an instance of the CRISGI class
crisgi.check_common_diff(
    top_n=20,
    target_group='cell_type',
    layer='log1p',
    method='pos_coexp',
    test_type='TER',
    interactions=['GeneA', 'GeneB', 'GeneC'],
    unit_header='subject',
    out_dir='results'
)

# After running, check the overlap statistics:
import pandas as pd
overlap_stats = pd.read_csv('./results/top_20_overlap.csv')
print(overlap_stats.head())
```

This example computes the overlap of the top 20 differential features per cell type, using the specified interactions, and saves the results in the `results` directory.
