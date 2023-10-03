import torch

from functions.test import test
from functions.train import train
from functions.infer import infer
from dataset.dataset import get_data


BATCH_SIZE = 16
EPOCHS = 20
FUNCTIONS = ["train", "test", "infer", "train_test", "train_infer", "test_infer", "full"]


def main():

    """
    Select here the function you want to run, note that testing from a saved model
    might be biased. If you want to test from a saved model, make sure to have the model
    saved in the ./model_saves folder.
    For normal purposes, use the "full" function.
    """
    function = "full"
    if function not in FUNCTIONS:
        print("Invalid function")
        return

    print(f"Running \"{function}\" function")
    train_loader, validation_loader, test_loader, inference = get_data(BATCH_SIZE)

    match function:
        case "train":
            train(train_loader, validation_loader, EPOCHS, device)
        case "test":
            test(test_loader, device)
        case "infer":
            infer(inference, device)
        case "train_test":
            train(train_loader, validation_loader, EPOCHS, device)
            test(test_loader, device)
        case "train_infer":
            train(train_loader, validation_loader, EPOCHS, device)
            infer(inference, device)
        case "test_infer":
            test(test_loader, device)
            infer(inference, device)
        case "full":
            train(train_loader, validation_loader, EPOCHS, device)
            test(test_loader, device)
            infer(inference, device)
        case _:
            print("Invalid function")

    del train_loader, validation_loader, test_loader, inference


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using {device} device")
    main()
