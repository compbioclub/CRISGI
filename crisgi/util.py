import os
import pickle
import numpy as np
import pandas as pd
from torch.utils.data import Dataset
from PIL import Image

def load_crisgi(pk_fn):
    crisgi_obj = pickle.load(open(pk_fn, 'rb'))
    print_msg(f'[Input] CRISGI object stored at {pk_fn} has been loaded.')
    return crisgi_obj


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

def set_adata_var(adata, header, x):
    df = adata.var.copy()
    df[header] = x
    adata.var = df

def set_adata_obs(adata, header, x):
    df = adata.obs.copy()
    df[header] = x
    adata.obs = df


class ImageDataset(Dataset):
    def __init__(self, image_dir, label_csv=None, transform=None, return_label=True, label_map= {'Asymptomatic': 0, 'Symptomatic': 1}):
        self.image_paths = [os.path.join(image_dir, f)
                            for f in os.listdir(image_dir) if f.endswith('.png')]
        self.image_paths.sort()
        self.transform = transform
        self.return_label = return_label
        self.labels = {}
        self.has_labels = False
        self.label_map = label_map

        if label_csv and os.path.exists(label_csv):
            label_df = pd.read_csv(label_csv)
            self.labels = {
                row['filename']: row['label'] for _, row in label_df.iterrows()
            }
            self.has_labels = True

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        image = Image.open(image_path).convert("RGBA")
        if self.transform:
            image = self.transform(image)

        filename = os.path.basename(image_path)

        if self.return_label and self.has_labels:
            label_str= self.labels.get(filename, None)  # 可设 default label 如 0
            label = self.label_map[label_str]
            return image, label
        else:
            return image, []