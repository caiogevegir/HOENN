import numpy
import wisardpkg
import time

from skimage.color import rgb2gray
from skimage.filters import threshold_otsu, threshold_niblack, threshold_sauvola
from sklearn.metrics import accuracy_score

from datasets import MNIST

def preprocess(data, num_items, input_size, to_grayscale=False):
    new_data = rgb2gray(data) if to_grayscale else data
    # new_data = threshold_sauvola(new_data, window_size=3)
    return numpy.reshape(numpy.sign(new_data).astype(int), newshape=(num_items, input_size)).tolist()


def main():
    """
    Address Size    Size (kB)       Training Time (s)   Prediction Time (s)    Accuracy (%)
    2                182.134        1.134029            0.00888215              0.7087
    4                216.542        1.048951            0.00171336              0.7432
    7                378.006        1.033938            0.00037850              0.8129
    8                452.346        1.047952            0.00024272              0.8059
    14              1194.134        0.998907            0.00007396              0.8717
    16              1548.848        1.017924            0.00006315              0.8906
    28              4064.334        1.038943            0.00005164              0.9450
    49              6697.142        1.053957            0.00004584              0.9295
    56              6863.010        1.062965            0.00004540              0.8943
    """
    dataset = MNIST(flatten=False)
    model = wisardpkg.Wisard(28)

    # Training
    preprocessed_train_labels = dataset.train_labels.astype(str).tolist()
    start = time.time()
    preprocessed_train_features = preprocess(dataset.train_features, dataset.num_train_items, dataset.data_size)
    model.train(preprocessed_train_features, preprocessed_train_labels)
    finish = time.time()
    print("Training Time: ", finish - start)

    # Evaluating
    preprocessed_validation_labels = dataset.validation_labels.astype(str).tolist()
    start = time.time()
    preprocessed_validation_features = preprocess(dataset.validation_features, dataset.num_validation_items, dataset.data_size)
    predictions = model.classify(preprocessed_validation_features)
    finish = time.time()
    print("Evaluation Time: ", finish - start)
    print("Average Prediction Time: ", (finish-start)/dataset.num_validation_items)
    print("Accuracy: ", accuracy_score(preprocessed_validation_labels, predictions))

    # Export
    model.json(True,"out/WiSARD")


if __name__ == "__main__":
    main()
