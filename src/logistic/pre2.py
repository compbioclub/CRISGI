import os
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset


class PatientDataset2(Dataset):
    def __init__(self, data_dir, label_dir, transform=None):
        self.label_df = pd.read_csv(label_dir)
        self.img_dir = [os.path.join(data_dir, fname) for fname in os.listdir(data_dir) if fname.endswith('.png')]
        self.labels = []
        self.patient_ids = []
        self.transform = transform
        part = data_dir.split(os.sep)
        self.dataset_name = part[-2].replace('train_', '').replace('val_', '')

        for x in self.img_dir:
            img_name = os.path.basename(x)
            id = img_name.split('_')[1]
            row = self.label_df[self.label_df['patient'].astype(str) == id]
            row = row['symptom'].values[0]
            y = 0 if row == 'Symptomatic' else 1
            self.labels.append(y)
            self.patient_ids.append(id)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        img_path = self.img_dir[idx]
        image = Image.open(img_path).convert("RGBA")
        label = self.labels[idx]
        if self.transform:
            image = self.transform(image)
        return image, label, self.dataset_name