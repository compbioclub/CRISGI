import numpy as np

def print_msg(msg, echo=True):
    if echo:
        print(msg)

def get_array(adata, layer=None):
    if layer:
        X = adata.layers[layer].copy()
    else:
        X = adata.X.copy()
    if type(X) != np.ndarray:
        X = X.toarray()
    return X