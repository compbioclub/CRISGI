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

def get_top_n(val, adata, top_n, max=True):
    row, col = adata.varm['bg_net'].nonzero()
    if max:
        top_n_indices = np.argpartition(val, -top_n)[-top_n:]
        sorted_indices = top_n_indices[np.argsort(-val[top_n_indices])]
    else:
        top_n_indices = np.argpartition(val, top_n)[:top_n]
        sorted_indices = top_n_indices[np.argsort(val[top_n_indices])]

    top_row = row[sorted_indices]
    top_col = col[sorted_indices]
    return top_row, top_col

