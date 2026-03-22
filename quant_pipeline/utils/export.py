import csv
import os


def save_results_to_csv(results):
    os.makedirs("outputs", exist_ok=True)

    filepath = "outputs/results.csv"

    with open(filepath, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)

    print(f"Results saved to {filepath}")