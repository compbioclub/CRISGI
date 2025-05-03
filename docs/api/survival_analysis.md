# survival_analysis

## Function

```python
crisgi_obj.survival_analysis(
    ref_group,
    target_group,
    interactions=None,
    groupbys=[],
    survival_types=['os', 'pfs'],
    time_unit='time',
    test_type='DER',
    method='pos_coexp',
    title=''
)
```
Performs survival analysis using Kaplan-Meier estimators and log-rank tests for specified groups and survival types. This method generates survival plots and statistical comparisons between groups, saving the resulting figures and printing output messages.

## Parameters

| Name           | Type            | Description                                                                                  |
|----------------|-----------------|----------------------------------------------------------------------------------------------|
| ref_group      | str             | Reference group name used for entropy calculation.                                            |
| target_group   | str             | Comparison group name for analysis.                                                          |
| interactions   | list or None    | List of interaction features to include; if None, uses default from `edata.uns`.             |
| groupbys       | list            | Additional columns in `obs` to group data by, in addition to score group.                    |
| survival_types | list            | List of survival types to analyze (e.g., `['os', 'pfs']`).                                   |
| time_unit      | str             | Label for the time axis in plots (e.g., 'months', 'days').                                   |
| test_type      | str             | Type of statistical test to use (default: 'DER').                                            |
| method         | str             | Method for entropy calculation (default: 'pos_coexp').                                            |
| title          | str             | Title prefix for the generated plots.                                                        |

## Return type

None

## Returns

This function does not return a value. It generates and saves survival plots for each specified survival type and grouping, and prints the output file paths.

## Attributes Set

- `edata.obs['score']`: Stores the computed score for each observation.
- `edata.obs['score_group']`: Stores the assigned score group for each observation.

## Example

```python
# Assume `crisgi` is an instance of the CRISGI class with loaded data

# Perform survival analysis comparing 'control' and 'treatment' groups
crisgi.survival_analysis(
    ref_group='control',
    target_group='treatment',
    interactions=['geneA', 'geneB'],
    groupbys=['batch'],
    survival_types=['os'],
    time_unit='months',
    test_type='DER',
    method='pos_coexp',
    title='Survival Analysis Example'
)
```

This example compares the overall survival (`'os'`) between 'control' and 'treatment' groups, considering interactions for 'geneA' and 'geneB', and grouping by 'batch'. The resulting survival plot is saved to the output directory, and the file path is printed.
