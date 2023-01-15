"""
Generate images to be sent to embedded system.
"""

import os.path
import serial
import time

from datasets import MNIST, FashionMNIST, CIFAR10

# --------------------------------------------------------------------------------------------------

SERIAL_PORT = "COM5"
SERIAL_BAUDRATE = 115200

RESULTS_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "out", "results.csv")
CSV_HEADER = ["Pred", "Real", "Layer", "Time"]

# --------------------------------------------------------------------------------------------------

def predict_on_board(board: serial.Serial, buffer: bytes):
    # Write
    board.write(buffer)
    # Read
    start_time = time.time()
    buffer = board.read()
    time_elapsed = time.time() - start_time
    response = int.from_bytes(buffer, "little")
    # Process
    prediction = response & 0x0F
    layer = "Cloud" if ((response >> 4) == 0x01) else "Local"  # response >> 4 == 0x00
    return prediction, layer, time_elapsed

# --------------------------------------------------------------------------------------------------

def main() -> None:
    dataset = MNIST(flatten=True)
    # dataset = FashionMNIST(flatten=True)
    # dataset = CIFAR10()

    offset = 0
    features = dataset.test_features[offset:]
    labels = dataset.test_labels[offset:]

    with (
        open(RESULTS_PATH, "a", encoding="utf-8") as csv_report,
        serial.Serial(port=SERIAL_PORT, baudrate=SERIAL_BAUDRATE) as board,
    ):
        try:
            for image, real_label in zip(features, labels):
                print("Sending image...")
                buffer = image.tobytes()
                predicted_label, layer, time_elapsed = predict_on_board(board, buffer)
                print(f"{predicted_label},{real_label},{layer},{time_elapsed}\n")
                csv_report.write(f"{predicted_label},{real_label},{layer},{time_elapsed}\n")
        except KeyboardInterrupt:
            board.flush()


if __name__ == "__main__":
    main()
