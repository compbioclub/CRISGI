# calculate_entropy

## Function

```python
crisgi_obj.calculate_entropy(
    ref_obs, 
    per_obss, 
    groupby, 
    ref_time, 
    layer='log1p'
)
```

Calculates the entropy changes of gene-gene interactions between a reference group and one or more perturbation groups within a single-cell dataset. This function initializes the entropy data structure, computes entropy for the reference and perturbation groups, and stores the results for downstream analysis. The entropy calculation can be performed using different interaction methods, including a product-based approach if specified.

## Parameters

| Name      | Type            | Description                                                                                   |
|-----------|-----------------|----------------------------------------------------------------------------------------------|
| ref_obs   | list or array   | List of observation (cell) identifiers for the reference group.                              |
| per_obss  | list of lists   | List of perturbation groups, each a list of observation (cell) identifiers.                  |
| groupby   | str             | The column name in `obs` used to group observations (e.g., experimental condition, batch).   |
| ref_time  | str or int      | Reference time point or label, used for naming output layers.                                |
| layer     | str, optional   | Name of the data layer in AnnData to use for calculations (default: `'log1p'`).             |

## Return type

None

## Returns

This function does not return a value. Instead, it updates the `edata` attribute of the CRISGI object with entropy results and related metadata for each perturbation group compared to the reference group.

## Attributes Set

- `self.ref_time`: Stores the reference time or label used in the calculation.
- `self.edata`: An AnnData object containing entropy results and metadata for each perturbation group.
- `self.adata_dict` (if applicable): Dictionary mapping group identifiers to AnnData objects with entropy calculations.
- `self.per_obss`: Stores the list of perturbation groups used in the calculation.
- `self.ref_obs`: Stores the reference group used in the calculation.

## Example

```python
from crisgi import CRISGI

# Assume adata is an AnnData object with required structure and data
crisgi = CRISGI(adata)

# Define reference and perturbation groups
ref_obs = adata.obs[adata.obs['condition'] == 'control'].index.tolist()
per_obss = [
    adata.obs[adata.obs['condition'] == 'treatment1'].index.tolist(),
    adata.obs[adata.obs['condition'] == 'treatment2'].index.tolist()
]

# Calculate entropy changes between reference and perturbation groups
crisgi.calculate_entropy(
    ref_obs=ref_obs,
    per_obss=per_obss,
    groupby='condition',
    ref_time='day0',
    layer='log1p'
)

# Access the results
entropy_results = crisgi.edata
print(entropy_results)
```
