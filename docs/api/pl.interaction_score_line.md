# pl.interaction_score_line

## Function

```python
pl.interaction_score_line(
    crisgi_obj,
    target_group=None,
    method="pos_coexp",
    test_type="TER",
    interactions=None,
    unit_header="subject",
    title="",
    out_prefix="test",
    ax=None,
)
```

Generates and plots a line chart of interaction scores (entropy-based) for a specified group or all groups within a CRISGI object. The function computes average interaction scores over time for selected interactions and visualizes them using seaborn's lineplot, supporting customization of grouping, statistical method, and output options.

## Parameters

| Name           | Type                | Description                                                                                      |
|----------------|---------------------|--------------------------------------------------------------------------------------------------|
| crisgi_obj     | CRISGI              | The CRISGI object containing experimental data and metadata.                                     |
| target_group   | str or None         | Specific group to plot. If `None`, iterates over all groups in `crisgi_obj.groups`.              |
| method         | str                 | Statistical method for score calculation (e.g., `'pos_coexp'`).                                  |
| test_type      | str                 | Type of test or interaction (e.g., `'TER'`).                                                     |
| interactions   | list or None        | List of interaction names to include. If `None`, uses all interactions for the group and method. |
| unit_header    | str or None         | Column name in `obs` to use as units for repeated measures (e.g., `'subject'`).                  |
| title          | str                 | Custom title prefix for the plot.                                                                |
| out_prefix     | str                 | Prefix for output file name if saving the plot.                                                  |
| ax             | matplotlib.axes.Axes or None | Matplotlib Axes object to plot on. If `None`, creates a new figure.                     |

## Return type

`None`

## Returns

This function does not return a value. It generates and displays (and optionally saves) a line plot of average interaction scores over time for the specified group(s).

## Attributes Set

- No attributes are set by this function. The function operates on the provided `crisgi_obj` and does not modify its state.

## Example

```python
import crisgi.plotting_crisgi_time as pl

# Assume crisgi_obj is a pre-loaded CRISGI object with required data
# Plot with specific interactions and custom unit_header
selected_interactions = ['geneA_geneB', 'geneC_geneD']
pl.interaction_score_line(
    crisgi_obj,
    target_group='Symptomatic',
    method='pearson',
    test_type='TER',
    interactions=selected_interactions,
    unit_header='subject_id',
    title='Selected Interactions ',
    out_prefix='selected_interactions'
)
```
