from quant_pipeline.models.nlp.distilbert import load_model, load_tokenizer
from quant_pipeline.quantization.utils import apply_quantization
from quant_pipeline.data.loaders import load_sst2
from quant_pipeline.core.benchmark import benchmark
from quant_pipeline.utils.memory import get_model_size
from quant_pipeline.utils.export import save_results_to_csv


def run():
    texts, labels = load_sst2()
    tokenizer = load_tokenizer()

    modes = ["fp32", "fp16", "int8_ptq", "int8_qat"]

    results = []

    for mode in modes:
        print(f"\n=== {mode.upper()} ===")

        model = load_model()

        model = apply_quantization(
            model,
            mode,
            tokenizer=tokenizer,
            train_data=(texts, labels),
        )

        metrics = benchmark(model, tokenizer, texts, labels)

        result = {
            "requested_mode": mode,
            "accuracy": metrics["accuracy"],
            "avg_latency_ms": metrics["avg_latency_ms"],
            "p99_latency_ms": metrics["p99_latency_ms"],
            "memory_mb": get_model_size(model),
        }

        results.append(result)

        print(result)

    save_results_to_csv(results)


if __name__ == "__main__":
    run()