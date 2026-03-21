from copy import deepcopy

def analyze_sensitivity(model, tokenizer, texts, labels, layer_names, benchmark_fn):
    results = {}

    for layer in layer_names:
        temp_model = deepcopy(model)
        metrics = benchmark_fn(temp_model, tokenizer, texts, labels)
        results[layer] = metrics["accuracy"]

    return results