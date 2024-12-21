import pandas as pd
import torch
from torch import nn
import torch.optim as optim
import os
import re
from torchvision import transforms
from autoEncoder import AE
from mlp import MLP
from train import train
from test import test
from pre import PatientDataset
from pre2 import PatientDataset2
from customDataLoader import CustomDataLoader



def extract_number(filename):
    match = re.search(r'(\d+)', filename)
    return int(match.group(0)) if match else float('inf')


def main():
    device = torch.device('cuda')
    print(f"Using device: {device}, {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")

    label_dir = 'data/meta_info.csv'
    base_dir = 'data/dataset'
    fold_dirs = os.listdir(base_dir)

    base2_dir = 'data/dataset2'
    fold2_dirs = os.listdir(base2_dir)

    base3_dir = 'data/dataset3'
    fold3_dirs = os.listdir(base3_dir)

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])



    for fold3_dir in fold3_dirs:
        test_loaders = []
        all_infection = []

        print(f"Handling test dataset: {fold3_dir}")
        full_fold_dir = os.path.join(base3_dir, fold3_dir)
        all_infection_dir = os.listdir(full_fold_dir)

        for infection in all_infection_dir:
            print(f"Infection: {infection}")
            all_infection.append(infection)
            full_path = os.path.join(full_fold_dir, infection)
            test_dataset = PatientDataset2(full_path, label_dir, transform)
            test_loader = CustomDataLoader(
                test_dataset,
                batch_size=64,
                shuffle=True,
                num_workers=8,
            )
            test_loaders.append(test_loader)


        log = torch.load("GSE30550_H3N2_log_new_model.pth")


        AE_loss_function = nn.MSELoss().to(device)
        CE_loss_function = nn.CrossEntropyLoss().to(device)
        optimizer = optim.Adam(list(ae.parameters()) + list(mlp.parameters()), lr=0.001, weight_decay=1e-4)


        all_metrics = []
        for test_loader, infection in zip(test_loaders, all_infection):
            print(f"{infection} testing=========")
            metrics = test(log, test_loader)
            # metrics = test(log, test_loader)

            metrics['Infection'] = infection
            all_metrics.append(metrics)

        metrics_result = pd.DataFrame(all_metrics)
        metrics_result.to_csv(f'metrics_summary_{fold3_dir}.csv', index=False)


    print("\nDone!")


if __name__ == '__main__':
    main()