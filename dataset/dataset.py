import torch
import pandas as pd

from sklearn.model_selection import train_test_split
from torch.utils.data import TensorDataset, DataLoader


def get_data(batch_size=8):

    """
    Get the data from the csv files and split it into train, validation and test sets.
    Also open the inference file and creates a set with it.
    Prints all the shapes of the created objects and creates dataloader for the concerned sets.
    """
    data = pd.read_csv("./data/train.csv").to_numpy()
    inference = pd.read_csv("./data/test.csv").to_numpy()

    train, test = train_test_split(data, test_size=0.2)
    train, validation = train_test_split(train, test_size=0.2)
    train_y, train_x = train[:, 0], train[:, 1:]/255
    validation_y, validation_x = validation[:, 0], validation[:, 1:]/255
    test_y, test_x = test[:, 0], test[:, 1:]/255
    train_y = pd.get_dummies(train_y).to_numpy()
    validation_y = pd.get_dummies(validation_y).to_numpy()
    test_y = pd.get_dummies(test_y).to_numpy()
    train_x = train_x.reshape(-1, 1, 28, 28)
    validation_x = validation_x.reshape(-1, 1, 28, 28)
    test_x = test_x.reshape(-1, 1, 28, 28)
    inference = inference.reshape(-1, 1, 28, 28)/255

    print(f"-----------------------------------")
    print(f"train_x shape: {train_x.shape}")
    print(f"train_y shape: {train_y.shape}")
    print(f"validation_x shape: {validation_x.shape}")
    print(f"validation_y shape: {validation_y.shape}")
    print(f"test_x shape: {test_x.shape}")
    print(f"test_y shape: {test_y.shape}")
    print(f"inference shape: {inference.shape}")
    print(f"-----------------------------------")

    train_x = torch.from_numpy(train_x).float()
    train_y = torch.from_numpy(train_y).float()
    validation_x = torch.from_numpy(validation_x).float()
    validation_y = torch.from_numpy(validation_y).float()
    test_x = torch.from_numpy(test_x).float()
    test_y = torch.from_numpy(test_y).float()
    inference = torch.from_numpy(inference).float()

    dataset_train = TensorDataset(train_x, train_y)
    dataset_validation = TensorDataset(validation_x, validation_y)
    dataset_test = TensorDataset(test_x, test_y)

    train_loader = DataLoader(dataset_train, batch_size=batch_size, shuffle=True)
    validation_loader = DataLoader(dataset_validation, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(dataset_test, batch_size=batch_size, shuffle=True)

    return train_loader, validation_loader, test_loader, inference
