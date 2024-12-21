import torch
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed


class CustomDataLoader:
    def __init__(self, dataset, batch_size, shuffle=True, num_workers=4):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.num_workers = num_workers
        self.indexes = list(range(len(dataset)))
        if self.shuffle:
            np.random.shuffle(self.indexes)

    def __iter__(self):
        batch_data = []
        batch_labels = []
        batch_datasets = []
        with ThreadPoolExecutor(max_workers=self.num_workers) as executor:
            futures = []
            for idx in self.indexes:
                futures.append(executor.submit(self.dataset.__getitem__, idx))
                if len(futures) == self.batch_size:
                    for future in as_completed(futures):
                        data, label, dataset_name = future.result()
                        batch_data.append(data)
                        batch_labels.append(label)
                        batch_datasets.append(dataset_name)
                    yield (
                        torch.stack(batch_data),
                        torch.tensor(batch_labels),
                        batch_datasets
                    )
                    batch_data, batch_labels, batch_datasets = [], [], []
                    futures = []

            # Process the last batch
            if futures:
                for future in as_completed(futures):
                    data, label, dataset_name = future.result()
                    batch_data.append(data)
                    batch_labels.append(label)
                    batch_datasets.append(dataset_name)
                yield (
                    torch.stack(batch_data),
                    torch.tensor(batch_labels),
                    batch_datasets
                )

    def __len__(self):
        return (len(self.dataset) + self.batch_size - 1) // self.batch_size