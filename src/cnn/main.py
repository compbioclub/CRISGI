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

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])

    for fold_dir in fold_dirs:
        full_fold_dir = os.path.join(base_dir, fold_dir)
        all_dataset = os.listdir(full_fold_dir)

        train_dir = None
        val_dirs = []
        all_train_loader = []
        all_val_loader = []

        for dataset in all_dataset:
            full_path = os.path.join(full_fold_dir, dataset)
            if dataset.startswith('train_'):
                train_dir = full_path

            elif dataset.startswith('val_'):
                val_dirs.append(full_path)

        all_train_ter = os.listdir(train_dir)
        all_train_ter = sorted(all_train_ter, key=extract_number)
        for ter in all_train_ter:
            full_path = os.path.join(train_dir, ter)
            train_dataset = PatientDataset(full_path, label_dir, transform)
            train_loader = CustomDataLoader(
                train_dataset,
                batch_size=64,
                shuffle=True,
                num_workers=8,
            )
            all_train_loader.append(train_loader)

        for val_dir in val_dirs:
            all_val_ter = os.listdir(val_dir)
            all_val_ter = sorted(all_val_ter, key=extract_number)
            tmp = []
            for ter in all_val_ter:
                full_path = os.path.join(val_dir, ter)
                val_dataset = PatientDataset(full_path, label_dir, transform)
                val_loader = CustomDataLoader(
                    val_dataset,
                    batch_size=64,
                    shuffle=True,
                    num_workers=8,
                )
                tmp.append(val_loader)
            all_val_loader.append(tmp)

        ae = AE().to(device)
        mlp = MLP().to(device)

        AE_loss_function = nn.MSELoss().to(device)
        CE_loss_function = nn.CrossEntropyLoss().to(device)
        optimizer = optim.Adam(list(ae.parameters()) + list(mlp.parameters()), lr=0.001, weight_decay=1e-4)

        print(f"\nTask: {fold_dir} start training")

        all_metrics = []
        for epoch in range(10):
            print(f"\nEpoch {epoch + 1}\n--------------------------------------------------------------")

            for i, (train_loader, ter) in enumerate(zip(all_train_loader, all_train_ter)):
                print(f"\n----------[{ter}] start training and validating----------")
                metrics = train(ae, mlp, train_loader, AE_loss_function, CE_loss_function, optimizer, device)
                if epoch == 9:
                    metrics['dataset'] = os.path.basename(train_dir).replace('train_', '')
                    metrics['ter'] = ter
                    all_metrics.append(metrics)

                counter = 0
                for val_loader in all_val_loader:
                    print(f"[Validation set {counter+1}]")
                    metrics = test(ae, mlp, val_loader[i], AE_loss_function, CE_loss_function, device)
                    if epoch == 9:
                        metrics['dataset'] = os.path.basename(val_dirs[counter]).replace('val_', '')
                        metrics['ter'] = ter
                        all_metrics.append(metrics)
                    counter = counter + 1



        # torch.save(ae, f'{fold_dir}_ae_model.pth')
        # torch.save(mlp, f'{fold_dir}_mlp_model.pth')

        metrics_result = pd.DataFrame(all_metrics)
        metrics_result.to_csv(f'metrics_summary_{fold_dir}.csv', index=False)



    print("\nDone!")


if __name__ == '__main__':
    main()