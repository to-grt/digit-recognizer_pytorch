import torch

from model import Model
from functions.usual_functions import plot_ten_predictions, save_results


def infer(inference_data, device):

    """
    Infer the predictions from the inference data
    """
    model = Model.Model().to(device)
    model.load_state_dict(torch.load('./model_saves/best_model'))

    model.eval()

    with torch.no_grad():
        inference_data = inference_data.to(device)
        predictions = model(inference_data)

    inference_data = inference_data.cpu().numpy()
    predictions = predictions.cpu().numpy()

    predictions = predictions.argmax(axis=1)

    plot_ten_predictions(inference_data, predictions)
    save_results(predictions)

    del model
    torch.cuda.empty_cache()
