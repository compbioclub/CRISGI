# find_interaction_module

## Function

```python
find_interaction_module(
    target_group,
    layer='log1p',
    method='pos_coexp',
    test_type='TER',
    interactions=None,
    unit_header='subject',
    out_dir=None,
    label_df=None,
    n_neighbors=10,
    strategy='bottom_up'
)
```

Identifies and clusters gene interaction modules within a specified group using a community detection algorithm. The function computes interaction matrices, applies clustering, and outputs results including interaction matrices, community/module assignments, and hierarchical relationships.

---

## Parameters

| Name          | Type            | Description                                                                                  |
|---------------|-----------------|----------------------------------------------------------------------------------------------|
| target_group  | str             | The group label to analyze within the data.                                                  |
| layer         | str, optional   | The data layer to use for calculations (default: `'log1p'`).                                 |
| method        | str, optional   | Method for interaction calculation (default: `'pos_coexp'`).                                 |
| test_type     | str, optional   | Type of statistical test to use (default: `'TER'`).                                          |
| interactions  | list, optional  | List of interaction names to analyze. If `None`, uses all available interactions.            |
| unit_header   | str, optional   | Column in `obs` to use as the unit identifier (default: `'subject'`).                        |
| out_dir       | str, optional   | Output directory for saving results. If `None`, uses the class's `out_dir` attribute.        |
| label_df      | pd.DataFrame, optional | DataFrame for custom labels. If `None`, labels are generated automatically.           |
| n_neighbors   | int, optional   | Number of neighbors for clustering (default: `10`).                                          |
| strategy      | str, optional   | Clustering strategy to use (default: `'bottom_up'`).                                         |

---

## Return type

`pandas.DataFrame`

---

## Returns

A DataFrame indexed by interaction names, containing the assigned community and module for each interaction.

---

## Attributes Set

- Saves the following files to `out_dir`:
  - Interaction matrix CSV
  - Interaction community/module CSV
  - Interaction hierarchy in Newick format

---

## Example

```python
# Assume `crisgi` is an instance of CRISGI with loaded data

result_df = crisgi.find_interaction_module(
    target_group='GroupA',
    layer='log1p',
    method='pos_coexp',
    test_type='TER',
    interactions=None,  # Use all available interactions
    unit_header='subject',
    out_dir='./results',
    n_neighbors=15,
    strategy='bottom_up'
)

# The result_df contains community and module assignments for each interaction.
# Output files are saved in './results' directory.
```
