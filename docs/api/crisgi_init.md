# CRISGI Class Initialization

## Function

The `CRISGI` class initializes and preprocesses gene expression data for downstream analysis, including background network construction and feature selection. It provides methods for data preprocessing, background network loading, and supports various interaction inference methods.

## Parameters

| Name                   | Type            | Description                                                                                   |
|------------------------|-----------------|-----------------------------------------------------------------------------------------------|
| `adata`                | AnnData         | Annotated data matrix (cells × genes) to be analyzed.                                         |
| `bg_net`               | array or None   | Optional. Precomputed background network matrix.                                              |
| `bg_net_score_cutoff`  | int             | Score threshold for filtering background network edges. Default is 850.                       |
| `genes`                | list or None    | Optional. List of gene names to include in the analysis.                                      |
| `n_hvg`                | int or None     | Number of highly variable genes to select. Default is 5000.                                   |
| `n_pcs`                | int             | Number of principal components for dimensionality reduction. Default is 30.                   |
| `interactions`         | array or None   | Optional. Predefined gene-gene interactions to use for background network construction.       |
| `n_threads`            | int             | Number of threads to use for computation. Default is 5.                                       |
| `interaction_methods`  | list            | List of methods for inferring gene interactions. Default: `['prod']`. |
| `organism`             | str             | Organism name (e.g., 'human'). Default is 'human'.                                            |
| `class_type`           | str             | Type of classification task (e.g., 'time'). Default is 'time'.                                |
| `dataset`              | str             | Dataset identifier. Default is 'test'.                                                        |
| `out_dir`              | str             | Output directory for results. Default is './out'.                                             |

## Return Type

`CRISGI` object

## Returns

Initializes a `CRISGI` object with preprocessed data and background network ready for downstream analysis.

## Attributes Set

- `adata`: Preprocessed AnnData object.
- `interaction_methods`: List of interaction inference methods.
- `organism`: Organism name.
- `n_threads`: Number of computation threads.
- `dataset`: Dataset identifier.
- `class_type`: Classification type.
- `out_dir`: Output directory path.
- `bg_net_score_cutoff`: Background network score cutoff.

## Example

```python
import anndata as ad
from crisgi import CRISGI

# Load your single-cell data into an AnnData object
adata = ad.read_h5ad('example_data.h5ad')

# Initialize CRISGI with default parameters
crisgi = CRISGI(
    adata=adata,
    n_hvg=3000,
    n_pcs=20,
    organism='human',
    class_type='time',
    dataset='my_dataset',
    out_dir='./crisgi_output'
)

# The object is now ready for further analysis
print(crisgi.adata)
print(crisgi.interaction_methods)
```
