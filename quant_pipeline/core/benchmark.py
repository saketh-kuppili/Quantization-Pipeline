import time
import torch
from quant_pipeline.core.metrics import compute_accuracy


def measure_latency(model, inputs):
    start = time.time()
    with torch.no_grad():
        model(**inputs)
    return (time.time() - start) * 1000


def benchmark(model, tokenizer, texts, labels):
    preds, latencies = [], []

    for text in texts:
        inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)

        latencies.append(measure_latency(model, inputs))

        with torch.no_grad():
            outputs = model(**inputs)

        preds.append(torch.argmax(outputs.logits, dim=1).item())

    return {
        "accuracy": compute_accuracy(preds, labels),
        "avg_latency_ms": sum(latencies) / len(latencies),
        "p99_latency_ms": sorted(latencies)[int(0.99 * len(latencies))],
    }