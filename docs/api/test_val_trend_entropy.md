# test_val_trend_entropy

## Function

```python
crisgi.test_val_trend_entropy(
    interactions,
    method="pos_coexp",
    p_cutoff=0.05,
    out_prefix="./test",
)
```

Performs trend and zero-trend statistical tests on a list of interactions using entropy-based values, identifying validation trend expressed interactions (TERs) based on significance thresholds.

## Parameters

| Name         | Type            | Description                                                                                  |
|--------------|-----------------|----------------------------------------------------------------------------------------------|
| interactions | list of str     | List of interaction names to be tested.                                                      |
| method       | str, optional   | Correlation method used for entropy calculation. Default is `'pos_coexp'`.                   |
| p_cutoff     | float, optional | Significance threshold for trend and zero-trend tests. Default is `0.05`.                    |
| out_prefix   | str, optional   | Prefix for the output CSV file containing TER statistics. Default is `'./test'`.             |

## Return type

`list of str`

## Returns

A list of interaction names that are identified as validation trend expressed interactions (TERs) based on the specified statistical criteria.

## Attributes Set

- Saves a CSV file named `{out_prefix}_TER.csv` containing the trend and zero-trend test results for all tested interactions.

## Example

```python
# Assume `crisgi` is an instance of the CRISGI class and edata is already set up.

interactions = ['geneA', 'geneB', 'geneC']
candidates = crisgi.test_val_trend_entropy(
    interactions=interactions,
    method='pearson',
    p_cutoff=0.01,
    out_prefix='./results/val_trend'
)

print("TER candidates:", candidates)
# Output CSV will be saved as './results/val_trend_TER.csv'
```
