import sys
import pandas
import sklearn.metrics

CSV_HEADER = ["Pred", "Real", "Layer", "Time"]

# --------------------------------------------------------------------------------------------------

def analyze_results(results_path: str):
    dataframe = pandas.read_csv(results_path, names=CSV_HEADER)
    # Inferences per layer
    print(dataframe["Layer"].value_counts(normalize=True))
    # Accuracy
    prediction = dataframe["Pred"].to_numpy()
    real = dataframe["Real"].to_numpy()
    accuracy = sklearn.metrics.accuracy_score(real, prediction)
    print(f"Accuracy: {accuracy}")
    # Average Inference Time (in ms)
    average_inference_time = dataframe["Time"].mean() * 1000
    print(f"Average Inference Time: {average_inference_time} ms")
    # Confusion Matrix
    confusion_matrix = sklearn.metrics.confusion_matrix(real, prediction)
    print("Confusion Matrix:\n", confusion_matrix)


if __name__ == "__main__":
    analyze_results(sys.argv[1])
