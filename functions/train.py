import torch
import torch.nn as nn

from model import Model


def train(train_loader, validation_loader, epochs, device):

    """
    Train the model for the specified number of epochs.
    """
    model = Model.Model().to(device)

    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0004)

    best_validation_loss = 1000000.0

    for epoch in range(epochs):

        model.train(True)
        running_loss = 0.0
        index_batch = 0

        for index_batch, data_batch in enumerate(train_loader):

            inputs_batch, labels_batch = data_batch
            inputs_batch = inputs_batch.to(device)
            labels_batch = labels_batch.to(device)
            optimizer.zero_grad()
            logit_batch = model(inputs_batch)
            loss = loss_fn(logit_batch, labels_batch)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        average_loss = running_loss / (index_batch + 1)

        model.eval()
        running_validation_loss = 0.0

        with torch.no_grad():
            for validation_index_batch, validation_data_batch in enumerate(validation_loader):
                validation_inputs, validation_labels = validation_data_batch
                validation_inputs = validation_inputs.to(device)
                validation_labels = validation_labels.to(device)
                validation_outputs = model(validation_inputs)
                validation_loss = loss_fn(validation_outputs, validation_labels)
                running_validation_loss += validation_loss

        average_validation_loss = running_validation_loss / (validation_index_batch + 1)
        print(f"EPOCH {epoch}: loss_train_total: {running_loss}, loss_train_average: {average_loss}"
              f", loss_validation_total: {running_validation_loss}, loss_validation_average: {average_validation_loss}")

        if average_validation_loss < best_validation_loss:
            best_validation_loss = average_validation_loss
            model_path = './model_saves/best_model'.format(epoch)
            torch.save(model.state_dict(), model_path)

    del model
    torch.cuda.empty_cache()

