import os
import numpy as np
import matplotlib.pyplot as plt


def plot_ten_predictions(data, predictions):

    """
    Plot ten random images from the data with their predictions
    """
    random_indices = np.random.randint(0, len(data), 10)
    for image_number, index in enumerate(random_indices):
        plt.subplot(2, 5, image_number + 1)
        plt.imshow(data[index][0], cmap='gray')
        plt.title(f"Prediction: {predictions[index]}")
        plt.axis('off')
    plt.show()


def save_results(predictions):

    """
    create a csv file with the predictions such as
    ImageId,Label
    1,0
    2,0
    3,0
    etc.
    """
    try:
        os.remove('./results.csv')
    except FileNotFoundError:
        pass
    with open('./data/results.csv', 'w') as file:
        file.write('ImageId,Label\n')
        for index, prediction in enumerate(predictions):
            file.write(f"{index + 1},{prediction}\n")
