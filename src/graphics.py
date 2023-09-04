import matplotlib.pyplot
import matplotlib.ticker

matplotlib.pyplot.rcParams["font.size"] = 12
matplotlib.pyplot.rcParams["font.family"] = "Times New Roman"

MODEL_LABELS = ["HOENN", "WiSARD-2", "WiSARD-4", "WiSARD-8", "WiSARD-16", "WiSARD-28", "LeNet5"]

MODEL_SPECS = [
  [7.840, 182.134, 216.542, 452.346, 1548.848, 4064.334, 264.184], # Size
  [0.0990, 1.1340, 1.0489, 1.0479, 1.0179, 1.0389, 25.3686], # Training Time (s)
  [0.00004, 0.00888, 0.00171, 0.00024, 0.00006, 0.00005, 0.02560], # Prediction Time (s)
  [0.7681, 0.7087, 0.7432, 0.8059, 0.8906, 0.9450, 0.9849] # Accuracy
]

FIG_SIZE = (8, 3)


def plot_allocated_memory():
  matplotlib.pyplot.figure(figsize=FIG_SIZE)
  matplotlib.pyplot.title("Memória Alocada (kB)")
  matplotlib.pyplot.bar(MODEL_LABELS, MODEL_SPECS[0], color='gray')
  matplotlib.pyplot.yscale('log')
  matplotlib.pyplot.grid(axis='y', which='major', linestyle='dotted')
  matplotlib.pyplot.tight_layout()
  matplotlib.pyplot.savefig('out/allocated_memory.pdf')
  matplotlib.pyplot.show()


def plot_training_time():
  matplotlib.pyplot.figure(figsize=FIG_SIZE)
  matplotlib.pyplot.title("Tempo Total de Treinamento (s)")
  matplotlib.pyplot.bar(MODEL_LABELS, MODEL_SPECS[1], color='lightblue')
  matplotlib.pyplot.yscale('log')
  matplotlib.pyplot.grid(axis='y', which='major', linestyle='dotted')
  matplotlib.pyplot.tight_layout()
  matplotlib.pyplot.savefig('out/training_time.pdf')
  matplotlib.pyplot.show()


def plot_inference_time():
  matplotlib.pyplot.figure(figsize=FIG_SIZE)
  matplotlib.pyplot.title("Tempo Médio de Inferência (s)")
  matplotlib.pyplot.bar(MODEL_LABELS, MODEL_SPECS[2], color='lightblue')
  matplotlib.pyplot.yscale('log')
  matplotlib.pyplot.grid(axis='y', which='major', linestyle='dotted')
  matplotlib.pyplot.tight_layout()
  matplotlib.pyplot.savefig('out/inference_time.pdf')
  matplotlib.pyplot.show()


def plot_validation_accuracy():
  matplotlib.pyplot.figure(figsize=FIG_SIZE)
  matplotlib.pyplot.title("Acurácia de Validação")
  matplotlib.pyplot.bar(MODEL_LABELS, MODEL_SPECS[3], color='lightgreen')
  matplotlib.pyplot.grid(axis='y', which='major', linestyle='dotted')
  matplotlib.pyplot.ylim(0.7, 1.0)
  matplotlib.pyplot.tight_layout()
  matplotlib.pyplot.yticks
  matplotlib.pyplot.savefig('out/validation_accuracy.pdf')
  matplotlib.pyplot.show()


if __name__ == "__main__":
  plot_allocated_memory()
  matplotlib.pyplot.close()
  plot_training_time()
  matplotlib.pyplot.close()
  plot_inference_time()
  matplotlib.pyplot.close()
  plot_validation_accuracy()
  matplotlib.pyplot.close()
