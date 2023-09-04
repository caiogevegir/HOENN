from tensorflow.keras import Sequential, Input
from tensorflow.keras.layers import Conv2D, AveragePooling2D, Flatten, Dense
from tensorflow.keras.utils import to_categorical
from datasets import MNIST
import os.path
import time

REPO_ROOT = os.path.dirname(os.path.dirname(__file__))

OUTPUT_PATH = os.path.join(REPO_ROOT, "LeNet5", "LeNet5-FashionMNIST")

class LeNet5(Sequential):

    def __init__(self, optimizer, loss):
        super(LeNet5, self).__init__(layers=None, name=None)
        self.input_size = (28, 28, 1)
        self.add(Input(shape=self.input_size))
        self.add(Conv2D(filters=6, kernel_size=(3,3), activation="relu"))
        self.add(AveragePooling2D())
        self.add(Conv2D(filters=16, kernel_size=(3,3), activation="relu"))
        self.add(AveragePooling2D())
        self.add(Flatten())
        self.add(Dense(units=120, activation="relu"))
        self.add(Dense(units=84, activation="relu"))
        self.add(Dense(units=10, activation="softmax"))
        self.compile(optimizer=optimizer, loss=loss, metrics=["Accuracy"])

    def train(self, features, labels):
        labels = to_categorical(labels)
        self.fit(x=features, y=labels, batch_size=128, epochs=10)
    
    def test(self, features, labels):
        labels = to_categorical(labels)
        self.evaluate(x=features, y=labels)

    def export(self, path):
        model_json = self.to_json()
        with open(path + ".json", "w+") as json_file:
            json_file.write(model_json)
        self.save_weights(path + ".h5")

def main():
    dataset = MNIST()

    model = LeNet5(optimizer="Adam", loss="categorical_crossentropy")

    train_features = dataset.train_features
    train_labels = dataset.train_labels

    test_features = dataset.validation_features
    test_labels = dataset.validation_labels

    model.summary()
    # start_time = time.time()
    model.train(train_features, train_labels)
    # print("Time Ellapsed: ", time.time() - start_time)
    print("Testing:")
    model.test(test_features, test_labels)
    model.export(OUTPUT_PATH)

if __name__ == "__main__":
    main()
