"""
Compares inference time using ENN and LeNet5.
"""

import time

from cloud_predict import MODEL_JSON_PATH, MODEL_H5_PATH, LeNet5, get_image_from_buffer
from enn import EmbeddedNN
from datasets import MNIST

# --------------------------------------------------------------------------------------------------

def main():
    dataset = MNIST(flatten=True)
    lenet5 = LeNet5(MODEL_JSON_PATH, MODEL_H5_PATH) # Pre-Trained
    enn = EmbeddedNN(dataset.data_size, dataset.num_labels)
    enn.train(dataset.train_features, dataset.train_labels)

    lenet5_average_time = 0.0
    enn_average_time = 0.0

    print("Benchmark Start...")

    for sample in dataset.test_features:

        start = time.time()
        enn.predict(sample)
        enn_average_time += time.time() - start

        image = get_image_from_buffer(sample.tobytes())
        start = time.time()
        lenet5.predict(image)
        lenet5_average_time += time.time() - start

    lenet5_average_time /= 10000.0
    enn_average_time /= 10000.0

    print("LeNet5 average prediction time:", lenet5_average_time)
    print("ENN average prediction time:", enn_average_time)


if __name__ == "__main__":
    main()
