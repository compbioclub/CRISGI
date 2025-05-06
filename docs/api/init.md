# Initialization

## CRISGI.__init__

### Function

```python
__init__(
    adata,
    bg_net=None,
    bg_net_score_cutoff=850,
    genes=None,
    n_hvg=5000,
    n_pcs=30,
    interactions=None,
    n_threads=5,
    interaction_methods=["prod"],
    organism="human",
    class_type="time",
    dataset="test",
    out_dir="./out",
)
```

Initializes a CRISGI object for single-cell gene interaction analysis. Sets up the AnnData object, prepares the background network, and configures preprocessing and output directories.

### Parameters

| Name                | Type            | Description                                                                                  |
|---------------------|-----------------|----------------------------------------------------------------------------------------------|
| adata               | AnnData         | The annotated data matrix (cells x genes) to be analyzed.                                    |
| bg_net              | Optional        | Precomputed background network (default: None).                                              |
| bg_net_score_cutoff | int             | Score cutoff for background network edges (default: 850).                                    |
| genes               | list or None    | List of gene names to include (default: None, uses highly variable genes if available).      |
| n_hvg               | int or None     | Number of highly variable genes to select (default: 5000).                                   |
| n_pcs               | int             | Number of principal components for PCA (default: 30).                                        |
| interactions        | Optional        | Predefined gene interactions (default: None).                                                |
| n_threads           | int             | Number of threads to use for computation (default: 5).                                       |
| interaction_methods | list            | Methods for interaction inference (default: ["prod"]).                                       |
| organism            | str             | Organism name (default: 'human').                                                            |
| class_type          | str             | Type of analysis class (default: 'time').                                                    |
| dataset             | str             | Dataset identifier (default: 'test').                                                        |
| out_dir             | str             | Output directory for results (default: './out').                                             |

### Return type

`None`

### Returns

Initializes the CRISGI object and prepares it for downstream analysis.

### Attributes Set

- `adata`: Processed AnnData object.
- `interaction_methods`: List of interaction inference methods.
- `organism`: Organism name.
- `n_threads`: Number of threads for computation.
- `dataset`: Dataset identifier.
- `class_type`: Type of analysis class.
- `out_dir`: Output directory path.
- `bg_net_score_cutoff`: Score cutoff for background network.
- `adata.varm['bg_net']`: Background network matrix.

### Example

```python
import scanpy as sc
from crisgi import CRISGI

adata = sc.read_h5ad('example_data.h5ad')
crisgi = CRISGI(
    adata,
    interaction_methods=["prod"],
    n_hvg=3000,
    n_pcs=20,
    organism='human',
    out_dir='./crisgi_results'
)
```

---

## CRISGITime.__init__

### Function

```python
__init__(
    self,
    adata,
    device="cpu",
    model_type="cnn",
    ae_path=None,
    mlp_path=None,
    model_path=None,
    **kwargs
)
```

Initializes a CRISGITime object for time-series or temporal single-cell gene interaction analysis. Inherits from CRISGI and adds model selection and device configuration.

### Parameters

| Name        | Type     | Description                                                                  |
|-------------|----------|------------------------------------------------------------------------------|
| adata       | AnnData  | The annotated data matrix (cells x genes) to be analyzed.                    |
| device      | str      | Device to use for computation (`'cpu'` or `'cuda'`, default: `'cpu'`).       |
| model_type  | str      | Model type to use (`'cnn'`, `'simple_cnn'`, `'logistic'`, default: `'cnn'`). |
| ae_path     | str/None | Path to autoencoder model weights (optional).                                |
| mlp_path    | str/None | Path to MLP model weights (optional).                                        |
| model_path  | str/None | Path to logistic model weights (optional).                                   |
| **kwargs    | dict     | Additional keyword arguments passed to CRISGI.__init__.                      |

### Return type

`None`

### Returns

Initializes the CRISGITime object and sets up the selected model for downstream analysis.

### Attributes Set

- All attributes from `CRISGI`.
- `device`: Computation device.
- `model_type`: Selected model type.
- `model`: Instantiated model object.

### Example

```python
import scanpy as sc
from crisgi import CRISGITime

adata = sc.read_h5ad('example_data.h5ad')
crisgi_time = CRISGITime(
    adata,
    interaction_methods=["prod"],
    device='cuda',
    model_type='cnn',
    out_dir='./crisgi_time_results'
)
```

