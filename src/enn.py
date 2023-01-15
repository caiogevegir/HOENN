"""
Trains Embedded Neural Network (ENN) and saves its weights on a .txt file
"""

import numpy
import os.path
import matplotlib.pyplot
import sklearn.metrics
from scipy.ndimage import gaussian_filter
import time
import scipy.stats

from datasets import MNIST, FashionMNIST, CIFAR10

# --------------------------------------------------------------------------------------------------

ROOT_PATH = os.path.dirname(os.path.dirname(__file__))
OUTPUT_PATH = os.path.join(ROOT_PATH, "out")

CONFIDENCE_ARRAY = []

matplotlib.pyplot.rcParams["font.size"] = 16
matplotlib.pyplot.rcParams["font.family"] = "Times New Roman"

# --------------------------------------------------------------------------------------------------

class EmbeddedNN:

    def __init__(self, data_size, num_labels):
        self.__model = numpy.zeros((num_labels, data_size), dtype=int)

    def __attenuate_model_values(self):
        # LOG attenuation
        # self.__model = numpy.log10(self.__model).astype(int)
        # self.__model = numpy.maximum(self.__model, 0)  # Remove underflow and negative results
        # MEAN attenuation + Gaussian Filter + Dropout
        self.__model = (self.__model / 50000).astype(int)
        # self.__model = gaussian_filter(self.__model, sigma=0.3)
        self.__model = numpy.maximum(self.__model, 3) - 3

    def train(self, train_features, train_labels):
        # start_time = time.time()
        for features, label in zip(train_features, train_labels):
            self.__model[label] += features
        self.__attenuate_model_values()
        # print("Time Ellapsed: ", time.time()-start_time)

    @staticmethod
    def __get_confidence(response):
        response.sort()
        # confidence = float(1 - (response[-2] / response[-1]))
        confidence = float(1 - (response[0] / response[1]))
        CONFIDENCE_ARRAY.append(confidence)
        return confidence

    def predict(self, features):
        # Getting Response By Threshold
        # response = numpy.matmul(self.__model, numpy.sign(features))
        # label = int(numpy.argmax(response))
        # Getting Response by Diff
        response = numpy.sum(numpy.abs(self.__model - features), axis=1)
        label = int(numpy.argmin(response))
        confidence = self.__get_confidence(response)
        return label, confidence

    def display_sample(self, label, rows, columns):
        if rows is not None and columns is not None:
            sample = numpy.reshape(self.__model[label], (rows, columns))
        else:
            sample = self.__model[label]
        matplotlib.pyplot.axis("off")
        matplotlib.pyplot.imshow(sample, cmap='gray')
        matplotlib.pyplot.show()

    def dumps(self, path):
        with open(path, "w", encoding="utf-8") as f:
            for label_array in self.__model:
                f.write("{ ")
                for index in label_array:
                    f.write(f"{index}, ")
                f.write("},\n")

# --------------------------------------------------------------------------------------------------

def evaluate(model, validation_features, validation_labels, confidence_threshold):
    """
    Evaluates model's local accuracy and inference rate for a validation set and a threshold.
    """
    predictions = []
    local_right_predictions = 0
    local_predictions = 0
    cloud_predictions = 0
    for features, real_label in zip(validation_features, validation_labels):
        predicted_label, confidence = model.predict(features)
        predictions.append(predicted_label)
        if confidence > confidence_threshold:
            local_predictions += 1
            if predicted_label == real_label:
                local_right_predictions += 1
        else:
            cloud_predictions += 1

    local_accuracy = local_right_predictions / local_predictions
    local_inference_rate = local_predictions / (local_predictions + cloud_predictions)

    return predictions, local_accuracy, local_inference_rate


def plot_confusion_matrix(true_labels, predicted_labels):
    """
    Displays model's confusion matrix.
    """
    confusion_matrix_array = sklearn.metrics.confusion_matrix(true_labels, predicted_labels)
    matplotlib.pyplot.xlabel("Valores Reais")
    matplotlib.pyplot.ylabel("Valores Inferidos")
    for i in range(10):
        for j in range(10):
            matplotlib.pyplot.text(j, i, confusion_matrix_array[i][j], ha="center", va="center", color="w")
    matplotlib.pyplot.imshow(confusion_matrix_array)
    matplotlib.pyplot.colorbar()
    matplotlib.pyplot.show()

def plot_cdf(confidence_threshold):

    count, bins_count = numpy.histogram(CONFIDENCE_ARRAY, bins=1000)
    pdf = count / sum(count)
    cdf = numpy.cumsum(pdf)
    matplotlib.pyplot.plot(bins_count[1:], cdf, label="CDF")
    matplotlib.pyplot.axvline(x=confidence_threshold, color="r", label="Limiar de Confiança")
    matplotlib.pyplot.xlabel("Confiança na Inferência")
    matplotlib.pyplot.ylabel("F(x)")
    matplotlib.pyplot.legend()
    matplotlib.pyplot.show()

# --------------------------------------------------------------------------------------------------

def main():
    # dataset = MNIST(flatten=True)
    dataset = FashionMNIST(flatten=True)
    # dataset = CIFAR10()

    model = EmbeddedNN(data_size=dataset.data_size, num_labels=dataset.num_labels)
    model.train(dataset.train_features, dataset.train_labels)
    model.dumps(path=os.path.join(OUTPUT_PATH, model.__class__.__name__ + ".txt"))

    predictions, local_accuracy, local_inference_rate = \
        evaluate(model, dataset.validation_features, dataset.validation_labels, -1)  # MNIST: 0.0104 - 95% | 0.071 - 90% | 0.0048 - 85%
    print(local_accuracy, local_inference_rate)

    plot_confusion_matrix(predictions, dataset.validation_labels)
    model.display_sample(2, 28, 28)
    # plot_cdf(0.01)

if __name__ == '__main__':
    main()
