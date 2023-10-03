import torch
import torch.nn as nn

from model import Model


def test(test_loader, device):

    """
    Test the model on the test set
    """
    model = Model.Model().to(device)
    model.load_state_dict(torch.load('./model_saves/best_model'))

    loss_fn = nn.CrossEntropyLoss()

    model.eval()
    running_test_loss = 0.0

    with torch.no_grad():
        for test_index_batch, test_data_batch in enumerate(test_loader):
            test_inputs, test_labels = test_data_batch
            test_inputs = test_inputs.to(device)
            test_labels = test_labels.to(device)
            test_outputs = model(test_inputs)
            test_loss = loss_fn(test_outputs, test_labels)
            running_test_loss += test_loss

    average_test_loss = running_test_loss / (test_index_batch + 1)
    print(f"TESTING PHASE: loss_test_total: {running_test_loss}, loss_test_average: {average_test_loss}")

    del model
    torch.cuda.empty_cache()
