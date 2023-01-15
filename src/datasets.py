import numpy
import os.path
import pickle
import matplotlib.pyplot


# (FASHION) MNIST ----------------------------------------------------------------------------------

class MNIST:

    def __init__(self, flatten=False):
        self.dataset_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "Dataset", self.__class__.__name__)
        self.train_features_path = os.path.join(self.dataset_path, "train-images-idx3-ubyte")
        self.train_labels_path = os.path.join(self.dataset_path, "train-labels-idx1-ubyte")
        self.test_features_path = os.path.join(self.dataset_path, "t10k-images-idx3-ubyte")
        self.test_labels_path = os.path.join(self.dataset_path, "t10k-labels-idx1-ubyte")

        self.is_flat = flatten
        self.num_train_items = 50000
        self.num_validation_items = 10000
        self.num_test_items = 10000
        self.num_labels = 10
        self.num_rows = 28
        self.num_columns = 28
        self.data_size = self.num_rows * self.num_columns
        self.train_features, self.train_labels, self.test_features, self.test_labels = \
            self.__load_raw_data()

        # Original train data will be split into training and validation
        self.validation_features, self.validation_labels = \
            self.train_features[50000:], self.train_labels[50000:]
        self.train_features, self.train_labels = \
            self.train_features[0:50000], self.train_labels[0:50000]

    def __load_features(self, features_file_path, num_items):
        features = numpy.fromfile(file=features_file_path, offset=16, dtype=numpy.uint8)
        if self.is_flat:
            features = numpy.reshape(a=features, newshape=(num_items, self.data_size))
        else:
            features = numpy.reshape(a=features, newshape=(num_items, self.num_rows, self.num_columns))
        return features

    @staticmethod
    def __load_labels(labels_file_path):
        return numpy.fromfile(file=labels_file_path, offset=8, dtype=numpy.uint8)

    def __load_raw_data(self):
        num_default_train_items = 60000

        train_features = self.__load_features(self.train_features_path, num_default_train_items)
        train_labels = self.__load_labels(self.train_labels_path)
        test_features = self.__load_features(self.test_features_path, self.num_test_items)
        test_labels = self.__load_labels(self.test_labels_path)

        return train_features, train_labels, test_features, test_labels

    def sample(self):
        sample = None
        if self.is_flat:
            sample = numpy.reshape(self.test_features[0], (self.num_rows, self.num_columns))
        else:
            sample = self.test_features[0]
        matplotlib.pyplot.axis("off")
        matplotlib.pyplot.imshow(sample, cmap="gray")
        matplotlib.pyplot.show()


class FashionMNIST(MNIST):

    def __init__(self, flatten=False):
        super().__init__(flatten)

# CIFAR10 ------------------------------------------------------------------------------------------

class CIFAR10:

    def __init__(self) -> None:
        self.dataset_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "Dataset", self.__class__.__name__)
        self.data_size = 3072
        self.num_labels = 10
        self.train_features = numpy.empty(shape=(1,3072), dtype=numpy.uint8)
        self.train_labels = []
        self.validation_features = []
        self.validation_labels = []
        self.test_features = []
        self.test_labels = []

        self.__load_train_dataset()
        self.__load_test_dataset()

    def __load_train_dataset(self):
        for i in range(1,6):
            data_path = os.path.join(self.dataset_path, f"data_batch_{i}")
            with open(data_path, "rb") as dp:
                content = pickle.load(dp, encoding="bytes")
                if i < 5:
                    self.train_features = numpy.concatenate((self.train_features, content[b"data"]))
                    self.train_labels += content[b"labels"]
                else:
                    # Leave the last batch for validation
                    self.validation_features = content[b"data"]
                    self.validation_labels = content[b"labels"]
        self.train_features = self.train_features[1:]
        self.train_features = self.__reshape_rgb_array(self.train_features, 40000)
        self.validation_features = self.__reshape_rgb_array(self.validation_features, 10000)

    def __load_test_dataset(self):
        data_path = os.path.join(self.dataset_path, "test_batch")
        with open(data_path, "rb") as dp:
            content = pickle.load(dp, encoding="bytes")
            self.test_features = content[b"data"]
            self.test_labels = content[b"labels"]
        self.test_features = self.__reshape_rgb_array(self.test_features, 10000)

    @staticmethod
    def __reshape_rgb_array(array, num_images):
        return numpy.reshape(array, newshape=(num_images,3,32,32)).transpose(0,2,3,1)

    def sample(self):
        matplotlib.pyplot.imshow(self.train_features[0])
        matplotlib.pyplot.show()

# --------------------------------------------------------------------------------------------------

def main():
    dataset = MNIST()
    # dataset = FashionMNIST()
    # dataset = CIFAR10()
    dataset.sample()


if __name__ == "__main__":
    main()
