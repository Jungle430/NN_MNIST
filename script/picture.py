import re
import matplotlib.pyplot as plt


def extract_data_from_log(log_file):
    loss_values = []
    accuracies = []

    with open(log_file, "r") as file:
        loss_match = None
        accuracy_match = None
        for line in file:

            if loss_match is None:
                loss_match = re.search(r"loss: (\d+\.\d+)", line)
            if accuracy_match is None:
                accuracy_match = re.search(r"right/count: (\d+\.\d+)", line)

            if loss_match is not None and accuracy_match is not None:
                loss_values.append(float(loss_match.group(1)))
                accuracies.append(float(accuracy_match.group(1)))
                loss_match = None
                accuracy_match = None
        return loss_values, accuracies


log_file = "../build/nn.log"
loss_values, accuracies = extract_data_from_log(log_file)

fig, ax1 = plt.subplots()

color = "tab:red"
ax1.set_xlabel("Training Iteration")
ax1.set_ylabel("Loss", color=color)
ax1.plot(range(1, len(loss_values) + 1), loss_values, color=color)
ax1.tick_params(axis="y", labelcolor=color)

ax2 = ax1.twinx()
color = "tab:blue"
ax2.set_ylabel("Accuracy", color=color)
ax2.plot(range(1, len(accuracies) + 1), accuracies, color=color)
ax2.tick_params(axis="y", labelcolor=color)

fig.tight_layout()
plt.title("Loss and Accuracy Over Training Iterations")
plt.show()
