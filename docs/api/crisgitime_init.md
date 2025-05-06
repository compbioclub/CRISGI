# CRISGITime Class Initialization

## Function

The `CRISGITime` class extends the `CRISGI` base class to provide time-based modeling for gene expression data. It supports various neural network architectures (3L-CNN, 1L-CNN (simple CNN), logistic regression) for downstream analysis, and manages model selection, initialization, and device assignment.

## Parameters

| Name         | Type     | Description                                                                                  |
|--------------|----------|----------------------------------------------------------------------------------------------|
| `adata`      | AnnData  | Annotated data matrix (typically single-cell gene expression data).                          |
| `device`     | str      | Device to run the model on (`'cpu'` or `'cuda'`). Default is `'cpu'`.                        |
| `model_type` | str      | Type of model to use (`'cnn'`, `'simple_cnn'`, or `'logistic'`). Default is `'cnn'`.         |
| `ae_path`    | str      | Path to a pre-trained autoencoder model (optional, used for CNN-based models).               |
| `mlp_path`   | str      | Path to a pre-trained MLP model (optional, used for CNN-based models).                       |
| `model_path` | str      | Path to a pre-trained logistic regression model (optional, used for logistic model).         |
| `**kwargs`   | dict     | Additional keyword arguments passed to the base `CRISGI` class.                              |

## Return type

`CRISGITime` object

## Returns

An instance of the `CRISGITime` class, initialized with the specified data, model type, and device. The model is ready for downstream analysis.

## Attributes Set

- `adata`: Processed AnnData object.
- `device`: Device used for computation.
- `model_type`: Type of model selected.
- `model`: Instantiated model object (CNNModel, SimpleCNNModel, or LogisticModel).
- `out_dir`: Output directory for results.
- Other attributes inherited from `CRISGI` (e.g., `interaction_methods`, `organism`, `n_threads`, etc.).

## Example

```python
import anndata as ad
from crisgitime import CRISGITime

# Load your single-cell data
adata = ad.read_h5ad('your_data.h5ad')

# Initialize CRISGITime with a CNN model on CPU
crisgi_time = CRISGITime(
    adata=adata,
    device='cpu',
    model_type='cnn',
    ae_path='path/to/autoencoder.pth',
    mlp_path='path/to/mlp.pth',
    out_dir='./results'
)

# The model is now ready for downstream analysis
print(crisgi_time.model_type)  # Output: cnn
print(crisgi_time.device)      # Output: cpu
```
