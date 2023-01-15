"""
Image Prediction using LeNet5 CNN

This script goes on server to receive data from embedded system.
"""

import numpy
import os.path
import socket
from tensorflow import keras

SERVER_HOST = ""
SERVER_PORT = 9000

# LENET5_DATA_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "LeNet5")
# MODEL_JSON_PATH = os.path.join(LENET5_DATA_PATH, "LeNet5-MNIST.json")
# MODEL_H5_PATH = os.path.join(LENET5_DATA_PATH, "LeNet5-MNIST.h5")

MODEL_JSON_PATH = "LeNet5-MNIST.json"
MODEL_H5_PATH = "LeNet5-MNIST.h5"

BUFFER_SIZE = 784
MNIST_IMAGE_SHAPE = (1, 28, 28, 1)

# --------------------------------------------------------------------------------------------------

class LeNet5:

    def __init__(self, structure_path, weights_path):
        self.__model = self.__load_model_from_json(structure_path, weights_path)

    @staticmethod
    def __load_model_from_json(structure_path, weights_path) -> keras.Model:
        with open(structure_path, "r", encoding="utf-8") as j:
            model = keras.models.model_from_json(j.read())
        model.load_weights(weights_path)
        return model

    def predict(self, data) -> bytes:
        return int(numpy.argmax(self.__model.predict(data, verbose=0))).to_bytes(1, "big")

# --------------------------------------------------------------------------------------------------

def get_image_from_buffer(buffer):
    image = numpy.frombuffer(buffer, dtype=numpy.uint8)
    image = numpy.reshape(image, MNIST_IMAGE_SHAPE)
    image = numpy.pad(image, ((0, 0), (2, 2), (2, 2), (0, 0)))  # Pads to 32x32
    return image

# --------------------------------------------------------------------------------------------------

def main():
    print("Loading LeNet5...")
    lenet5 = LeNet5(MODEL_JSON_PATH, MODEL_H5_PATH)
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as server:
        server.bind((SERVER_HOST, SERVER_PORT))
        server.listen()
        print("Listening...")
        while True:
            connection, address = server.accept()
            with connection:
                print(f"(Re)Connected by {address}")
                while True:
                    try:
                        buffer = connection.recv(BUFFER_SIZE)
                        print("Received buffer...")
                        image = get_image_from_buffer(buffer)
                        prediction = lenet5.predict(image)
                        connection.sendall(prediction)
                        print("Sent prediction...")
                    except KeyboardInterrupt:
                        print("Interrupted by user!")
                        break
                    except ConnectionResetError:
                        print("Connection reset by peer. Reconnecting...")
                        break


if __name__ == "__main__":
    main()
