"""
Trains Embedded Neural Network (ENN) and saves its weights on a .txt file
"""

import numpy
import os.path
import matplotlib.pyplot
import sklearn.metrics
import time

from datasets import MNIST

# --------------------------------------------------------------------------------------------------

ROOT_PATH = os.path.dirname(os.path.dirname(__file__))
OUTPUT_PATH = os.path.join(ROOT_PATH, "out")

CONFIDENCE_ARRAY = []

matplotlib.pyplot.rcParams["font.size"] = 12
matplotlib.pyplot.rcParams["font.family"] = "Times New Roman"

# --------------------------------------------------------------------------------------------------

class EmbeddedNN:

    def __init__(self, data_size, num_labels):
        self.__model = numpy.zeros((num_labels, data_size), dtype=int)

    def __attenuate_model_values(self):
        # LOG attenuation
        # self.__model = numpy.log10(self.__model).astype(int)
        # self.__model = numpy.maximum(self.__model, 0)  # Remove underflow and negative results
        # MEAN attenuation + Dropout
        self.__model = (self.__model / 50000).astype(int)
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

def plot_cdf():

    count, bins_count = numpy.histogram(CONFIDENCE_ARRAY, bins=1000)
    pdf = count / sum(count)
    cdf = numpy.cumsum(pdf)

    fig, ax = matplotlib.pyplot.subplots(1, 2, figsize=(9,3))
    fig.suptitle("Distribuição Acumulada da Confiança")
    fig.supylabel("F(x)")
    fig.tight_layout()

    ax[0].plot(bins_count[1:], cdf, label="CDF")
    ax[0].grid(linestyle='dotted')
    ax[0].set_yticks([0.0, 0.25, 0.5, 0.75, 1.0])
    ax[0].axvline(x=0.0045, color='red', linestyle='dotted', label="Limiar 75/25")
    ax[0].axhline(y=0.25, color='red', linestyle='dotted')
    ax[0].text(0.03, 0.625, 'Local', color='w', backgroundcolor='red')
    ax[0].text(0.03, 0.125, 'Nuvem', color='w', backgroundcolor='red')
    ax[0].set_xlabel('(a)')
    ax[0].legend(loc='lower right')

    ax[1].plot(bins_count[1:], cdf, label="CDF")
    ax[1].grid(linestyle='dotted')
    ax[1].set_yticks([0.0, 0.25, 0.5, 0.75, 1.0])
    ax[1].axvline(x=0.0105, color='darkred', linestyle='dotted', label="Limiar 50/50")
    ax[1].axhline(y=0.50, color='darkred', linestyle='dotted')
    ax[1].text(0.03, 0.75, 'Local', color='w', backgroundcolor='darkred')
    ax[1].text(0.03, 0.25, 'Nuvem', color='w', backgroundcolor='darkred')
    ax[1].set_xlabel('(b)')
    ax[1].legend(loc='lower right')

    matplotlib.pyplot.savefig('out/validation_cdf.pdf')
    matplotlib.pyplot.show()


# --------------------------------------------------------------------------------------------------

def main():
    dataset = MNIST(flatten=True)

    model = EmbeddedNN(data_size=dataset.data_size, num_labels=dataset.num_labels)
    model.train(dataset.train_features, dataset.train_labels)
    model.dumps(path=os.path.join(OUTPUT_PATH, model.__class__.__name__ + ".txt"))

    predictions, local_accuracy, local_inference_rate = \
        evaluate(model, dataset.validation_features, dataset.validation_labels, -1)  # MNIST: 0.0104 - 95% | 0.071 - 90% | 0.0048 - 85%
    print(local_accuracy, local_inference_rate)

    #plot_confusion_matrix(predictions, dataset.validation_labels)
    #model.display_sample(2, 28, 28)
    plot_cdf()

if __name__ == '__main__':
    main()
