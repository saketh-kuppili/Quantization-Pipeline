import matplotlib.pyplot as plt

def plot_sensitivity_heatmap(results):
    plt.figure(figsize=(10,6))
    plt.barh(list(results.keys()), list(results.values()))
    plt.xlabel("Accuracy")
    plt.title("Layer Sensitivity")
    plt.savefig("outputs/sensitivity.png")
    plt.show()