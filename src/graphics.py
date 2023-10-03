import matplotlib.pyplot
import matplotlib.ticker
import numpy
import pandas
import os.path

matplotlib.pyplot.rcParams["font.size"] = 12
matplotlib.pyplot.rcParams["font.family"] = "Times New Roman"

RESULTS_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "out")

MODEL_LABELS = ["HOENN", "WiSARD-2", "WiSARD-4", "WiSARD-8", "WiSARD-16", "WiSARD-28", "LeNet5"]

MODEL_SPECS = [
  [7.840, 182.134, 216.542, 452.346, 1548.848, 4064.334, 264.184], # Size
  [0.0990, 1.1340, 1.0489, 1.0479, 1.0179, 1.0389, 25.3686], # Training Time (s)
  [0.00004, 0.00888, 0.00171, 0.00024, 0.00006, 0.00005, 0.02560], # Prediction Time (s)
  [0.7681, 0.7087, 0.7432, 0.8059, 0.8906, 0.9450, 0.9849] # Accuracy
]

FIG_SIZE = (8, 3)

CSV_HEADER = ["Pred", "Real", "Layer", "Time"]

MNIST_LOCAL_ONLY = pandas.read_csv(os.path.join(RESULTS_PATH, "results_MNIST_Local_Only.csv"), names=CSV_HEADER)
MNIST_USA_CLOUD_ONLY = pandas.read_csv(os.path.join(RESULTS_PATH, "results_MNIST_USA_Cloud_Only.csv"), names=CSV_HEADER)
MNIST_SP_CLOUD_ONLY = pandas.read_csv(os.path.join(RESULTS_PATH, "results_MNIST_SP_Cloud_Only.csv"), names=CSV_HEADER)
MNIST_75_LOCAL_25_SP_CLOUD = pandas.read_csv(os.path.join(RESULTS_PATH, "results_MNIST_75_Local_25_SP_Cloud.csv"), names=CSV_HEADER)
MNIST_50_LOCAL_50_SP_CLOUD = pandas.read_csv(os.path.join(RESULTS_PATH, "results_MNIST_50_Local_50_SP_Cloud.csv"), names=CSV_HEADER)
MNIST_VALIDATION_ACCURACY_INFERENCE = pandas.read_csv(os.path.join(RESULTS_PATH, "accuracy_inference.csv"), names=["Confidence", "Accuracy", "Inferences"])

# --------------------------------------------------------------------------------------------------

def __convert_time_to_numpy(dataframe):
  return dataframe["Time"].apply(lambda x: x * 1000).to_numpy(dtype=int)

def __convert_pred_to_numpy(dataframe):
  return dataframe["Pred"].to_numpy(dtype=int)

def __convert_real_to_numpy(dataframe):
  return dataframe["Real"].to_numpy(dtype=int)

def __convert_layer_to_list(dataframe):
  return dataframe["Layer"].values.tolist()

# --------------------------------------------------------------------------------------------------

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


def plot_local_cloud_boxplot():
  local_only = __convert_time_to_numpy(MNIST_LOCAL_ONLY)
  sp_cloud_only = __convert_time_to_numpy(MNIST_SP_CLOUD_ONLY)
  usa_cloud_only = __convert_time_to_numpy(MNIST_USA_CLOUD_ONLY)

  matplotlib.pyplot.figure(figsize=FIG_SIZE)
  matplotlib.pyplot.title("Tempo de Inferência por Dispositivo (ms)")
  boxplot = matplotlib.pyplot.boxplot([usa_cloud_only, sp_cloud_only, local_only], patch_artist=True, notch=False, showfliers=False, vert=False)
  matplotlib.pyplot.yticks(ticks=[1, 2, 3], labels=["AWS EUA", "AWS SP", "ESP32S"])
  matplotlib.pyplot.xlim(left=40, right=300)
  matplotlib.pyplot.grid(axis='both', which='major', linestyle='dotted')
  boxplot["boxes"][0].set_facecolor("blue")
  boxplot["boxes"][1].set_facecolor("blue")
  boxplot["boxes"][2].set_facecolor("red")
  boxplot["medians"][0].set_color("white")
  boxplot["medians"][1].set_color("white")
  boxplot["medians"][2].set_color("white")
  matplotlib.pyplot.savefig('out/local_cloud_boxplot.pdf')
  matplotlib.pyplot.show()


def plot_accuracy_confidence_rate():
  x_ticks = MNIST_VALIDATION_ACCURACY_INFERENCE["Confidence"].to_numpy(dtype=float)
  y_accuracy = MNIST_VALIDATION_ACCURACY_INFERENCE["Accuracy"].to_numpy(dtype=float)
  y_inferences = MNIST_VALIDATION_ACCURACY_INFERENCE["Inferences"].to_numpy(dtype=float)

  matplotlib.pyplot.figure(figsize=(8,3))
  matplotlib.pyplot.title("Acurácia x Inferências Mantidas")
  matplotlib.pyplot.gca().yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(xmax=1.0))
  matplotlib.pyplot.plot(x_ticks, y_accuracy, color="green", label="Acurácia")
  matplotlib.pyplot.plot(x_ticks, y_inferences, color="blue", label="Inferências")
  matplotlib.pyplot.ylim(bottom=-0.01, top=1.01)
  matplotlib.pyplot.xlim(left=0.0, right=0.135)
  matplotlib.pyplot.legend(loc="center right")
  matplotlib.pyplot.grid(linestyle="dotted")
  matplotlib.pyplot.tight_layout()
  matplotlib.pyplot.xlabel("Limiar de Confiança")
  matplotlib.pyplot.savefig('out/accuracy_confidence_rate.pdf')
  matplotlib.pyplot.show()


def plot_average_inference():

  local_only = __convert_time_to_numpy(MNIST_LOCAL_ONLY)
  sp_cloud_only = __convert_time_to_numpy(MNIST_SP_CLOUD_ONLY)
  local_75_sp_cloud_25 = __convert_time_to_numpy(MNIST_75_LOCAL_25_SP_CLOUD)
  local_50_sp_cloud_50 = __convert_time_to_numpy(MNIST_50_LOCAL_50_SP_CLOUD)

  local_only_mean = numpy.mean(local_only)
  sp_cloud_only_mean = numpy.mean(sp_cloud_only)
  local_75_sp_cloud_25_mean = numpy.mean(local_75_sp_cloud_25)
  local_50_sp_cloud_50_mean = numpy.mean(local_50_sp_cloud_50)
  heights = [sp_cloud_only_mean, local_50_sp_cloud_50_mean, local_75_sp_cloud_25_mean, local_only_mean]

  local_only_error = 1.96 * (numpy.std(local_only)/100.0)
  sp_cloud_only_error = 1.96 * (numpy.std(sp_cloud_only)/100.0)
  local_75_sp_cloud_25_error = 1.96 * (numpy.std(local_75_sp_cloud_25)/100.0)
  local_50_sp_cloud_50_error = 1.96 * (numpy.std(local_50_sp_cloud_50)/100.0)
  errors = [sp_cloud_only_error, local_50_sp_cloud_50_error, local_75_sp_cloud_25_error, local_only_error]

  fig, ax = matplotlib.pyplot.subplots(figsize=FIG_SIZE)
  fig.suptitle("Tempo para Inferência (ms)")
  x_label = ["AWS SP", "50/50", "25/75", "ESP32S"]
  x_pos = numpy.arange(4)
  ax.yaxis.grid(linestyle="dotted")
  ax.set_ylim(bottom=0, top=200)
  rects = ax.bar(x=x_pos, height=heights, yerr=errors, width=0.5, color="lightblue")
  ax.bar_label(rects, fmt="%d")
  ax.set_xticks(x_pos, x_label)
  
  matplotlib.pyplot.savefig('out/average_inference.pdf')
  matplotlib.pyplot.show()


def plot_relative_accuracy_time():

  relative_accuracy = numpy.array([1.0, 0.9597, 0.8951, 0.7615])
  relative_time = numpy.array([1.0, 0.6582, 0.4754, 0.2712])
  x_label = ["AWS SP", "50/50", "25/75", "ESP32S"]
  x_pos = numpy.arange(4)

  matplotlib.pyplot.figure(figsize=FIG_SIZE)
  matplotlib.pyplot.title("Acurácia x Tempo Relativo (%)")
  matplotlib.pyplot.gca().yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(xmax=1.0))
  matplotlib.pyplot.plot(relative_accuracy, color="green", lw=2, marker="o", label="Acurácia")
  matplotlib.pyplot.fill_between([0, 1, 2, 3], relative_accuracy, color="lightgreen")
  matplotlib.pyplot.plot(relative_time, color="blue", lw=2, marker="o", label="Tempo")
  matplotlib.pyplot.fill_between([0, 1, 2, 3], relative_time, color="lightblue")
  matplotlib.pyplot.xticks(x_pos, x_label)
  matplotlib.pyplot.ylabel("Diferença (%)")
  matplotlib.pyplot.grid(linestyle='dotted')
  matplotlib.pyplot.legend()

  matplotlib.pyplot.savefig('out/relative_accuracy_time.pdf')
  matplotlib.pyplot.show()


def plot_inference_map():

  local_rgb = [200, 200, 0]   # Yellow
  cloud_rgb = [0, 0, 127]     # Blue
  both_rgb = [0, 127, 0]      # Green
  none_rgb = [200, 0, 0]      # Red

  local_predictions = __convert_pred_to_numpy(MNIST_LOCAL_ONLY)
  cloud_predictions = __convert_pred_to_numpy(MNIST_SP_CLOUD_ONLY)
  real_values = __convert_real_to_numpy(MNIST_LOCAL_ONLY)

  local_right = 0
  cloud_right = 0
  both_right = 0
  none_right = 0

  map_list = []

  for lpred, cpred, real in zip(local_predictions, cloud_predictions, real_values):
    if ( lpred == real and cpred == real ):
      map_list.append(both_rgb)
      both_right += 1
    elif ( lpred == real ):
      map_list.append(local_rgb)
      local_right += 1
    elif ( cpred == real ):
      map_list.append(cloud_rgb)
      cloud_right += 1
    else:
      map_list.append(none_rgb)
      none_right += 1

  map_array = numpy.reshape(numpy.asarray(map_list), newshape=(100, 100, 3))
  print(local_right, cloud_right, both_right, none_right)

  matplotlib.pyplot.title("Acertos de Inferência")
  matplotlib.pyplot.imshow(map_array)
  matplotlib.pyplot.axis("off")
  yellow_patch = matplotlib.patches.Patch(color='#C8C800', label='Local')
  blue_patch = matplotlib.patches.Patch(color='#00007F', label='Nuvem')
  green_patch = matplotlib.patches.Patch(color="#007F00", label='Ambos')
  red_patch = matplotlib.patches.Patch(color="#C80000", label='Ninguém')
  matplotlib.pyplot.legend(loc="upper center", bbox_to_anchor=(0.5, 0), handles=[yellow_patch, blue_patch, green_patch, red_patch], ncol=4)

  matplotlib.pyplot.savefig('out/inference_map.pdf')
  matplotlib.pyplot.show()


def plot_confidence_assertion():

  # 25/75 - 50/50
  over_confident = numpy.array([0, 0])         # Local Wrong; High Confidence
  right_confident = numpy.array([0, 0])        # Local Right; High Confidence
  under_confident = numpy.array([0, 0])        # Local Right; Low Confidence

  for i in range(10000):
    j=0
    for df in [MNIST_75_LOCAL_25_SP_CLOUD, MNIST_50_LOCAL_50_SP_CLOUD]:
      if ( df.iloc[i]["Pred"] == df.iloc[i]["Real"] ) and ( df.iloc[i]["Layer"] == "Local" ):
        right_confident[j] += 1
      #elif ( df.iloc[i]["Pred"] == df.iloc[i]["Real"] ) and ( df.iloc[i]["Layer"] == "Cloud" ) and ( MNIST_LOCAL_ONLY.iloc[i]["Pred"] != MNIST_LOCAL_ONLY.iloc[i]["Real"] ):
      #  right_confident[j] += 1
      elif ( df.iloc[i]["Pred"] != df.iloc[i]["Real"] ) and ( df.iloc[i]["Layer"] == "Local" ):
        over_confident[j] += 1
      elif ( df.iloc[i]["Layer"] == "Cloud" ) and ( MNIST_LOCAL_ONLY.iloc[i]["Pred"] == MNIST_LOCAL_ONLY.iloc[i]["Real"] ):
        under_confident[j] += 1
      j+=1
  
  print(over_confident, right_confident, under_confident)

  x = numpy.arange(2)
  labels = ["25/75", "50/50"]
  width = 0.1

  fig, ax = matplotlib.pyplot.subplots(figsize=(8,4))

  rects1 = ax.bar(x/2 - width, under_confident, width, label="Inseguro", color="lightblue")
  rects2 = ax.bar(x/2, right_confident, width, label="Seguro", color="green")
  rects3 = ax.bar(x/2 + width, over_confident, width, label="Imprudente", color="brown")

  ax.set_ylabel("Quantidade")
  ax.set_xlabel("Distribuição Nuvem/Local")
  ax.set_title("Confiabilidade Local")
  ax.set_xticks(x/2, labels)
  ax.set_ylim(bottom=0, top=10000)
  ax.yaxis.grid(True, linestyle="dotted")
  ax.legend(loc="upper right")

  ax.bar_label(rects1)
  ax.bar_label(rects2)
  ax.bar_label(rects3)

  fig.tight_layout()

  matplotlib.pyplot.savefig('out/confidence_assertion.pdf')
  matplotlib.pyplot.show()


if __name__ == "__main__":
  #plot_allocated_memory()
  #matplotlib.pyplot.close()
  #plot_training_time()
  #matplotlib.pyplot.close()
  #plot_inference_time()
  #matplotlib.pyplot.close()
  #plot_validation_accuracy()
  #matplotlib.pyplot.close()
  #plot_local_cloud_boxplot()
  #matplotlib.pyplot.close()
  #plot_accuracy_confidence_rate()
  #matplotlib.pyplot.close()
  #plot_average_inference()
  #matplotlib.pyplot.close()
  #plot_relative_accuracy_time()
  #matplotlib.pyplot.close()
  #plot_inference_map()
  #matplotlib.pyplot.close()
  #plot_confidence()
  #matplotlib.pyplot.close()
  #plot_relative_accuracy_time()
  #matplotlib.pyplot.close()
  plot_confidence_assertion()
  matplotlib.pyplot.close()

