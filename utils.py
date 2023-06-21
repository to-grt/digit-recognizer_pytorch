import pandas as pd
import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader


def from_csv_to_loader(model, path, is_train=True, prints=False):

    data = pd.read_csv(path).values

    x = data[:, 1:] if is_train else data
    x = np.reshape(x, newshape=(x.shape[0], model.NB_CHANNELS, *model.INPUT_SIZE)) / 255
    np.random.shuffle(x)
    x = torch.from_numpy(x)

    if is_train:
        validation_split = int(model.VALIDATION_SPLIT * x.shape[0])
        x, x_val = x[:-validation_split], x[-validation_split:]
        y = data[:, 0]
        y, y_val = y[:-validation_split], y[-validation_split:]
        y = np.eye(model.OUTPUT_SIZE)[y]
        y_val = np.eye(model.OUTPUT_SIZE)[y_val]
        y = torch.from_numpy(y).to(torch.float64)
        y_val = torch.from_numpy(y_val).to(torch.float64)

        if prints:
            print(f"train_x.shape: {x.shape} mean: {x.mean()} std: {x.std()}")
            print(f"val_x.shape: {x_val.shape} mean: {x_val.mean()} std: {x_val.std()}")
            print(f"train_y.shape: {y.shape} dtype: {y.dtype} min: {y.min()} max: {y.max()}")
            print(f"val_y.shape: {y_val.shape} dtype: {y_val.dtype} min: {y_val.min()} max: {y_val.max()}")

        train_dataset = TensorDataset(x, y)
        val_dataset = TensorDataset(x_val, y_val)
        train_loader = DataLoader(train_dataset, batch_size=model.BATCH_SIZE, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=model.BATCH_SIZE, shuffle=False)
        return train_loader, val_loader
    else:
        if prints:
            print(f"test_x.shape: {x.shape} mean: {x.mean()} std: {x.std()}")
        test_dataset = TensorDataset(x)
        test_loader = DataLoader(test_dataset, batch_size=model.BATCH_SIZE, shuffle=False)
        return test_loader


