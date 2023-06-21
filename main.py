from utils import *
from Model import Model


def main():

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = Model(device)

    train_loader, val_loader = from_csv_to_loader(model=model, path='train.csv', is_train=True, prints=True)
    test_loader = from_csv_to_loader(model=model, path='test.csv', is_train=False, prints=True)

    for epoch in range(model.NB_EPOCHS):

        loss = None
        val_loss = None
        batch_index = None

        model.train()
        for batch_index, (batch_x, batch_y) in enumerate(train_loader):
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)

            outputs = model(batch_x)
            loss = model.loss(outputs, batch_y)

            model.optimizer.zero_grad()
            loss.backward()
            model.optimizer.step()

        model.eval()
        correct = 0
        total = 0

        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                batch_x, batch_y = batch_x.to(device), batch_y.to(device)

                outputs = model(batch_x)
                val_loss = model.loss(outputs, batch_y)
                predicted = torch.argmax(outputs, dim=1)
                targets = torch.argmax(batch_y, dim=1)
                correct += (predicted == targets).sum().item()
                total += model.BATCH_SIZE

        accuracy = 100 * correct / total

        print(f"Epoch [{epoch + 1}/{model.NB_EPOCHS}],  "
              f"Step [{batch_index + 1}/{len(train_loader)}],  "
              f"Loss: {loss.item():.4f},  "
              f"Val Loss: {val_loss.item():.4f},  "
              f"Val Accuracy: {accuracy:.2f}%")


if __name__ == '__main__':
    main()
