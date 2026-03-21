import csv, os

def save_results_to_csv(results):
    os.makedirs("outputs", exist_ok=True)
    with open("outputs/results.csv","w",newline="") as f:
        writer = csv.DictWriter(f, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)