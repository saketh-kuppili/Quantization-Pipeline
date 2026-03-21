from quant_pipeline.models.nlp.distilbert import load_model, load_tokenizer
from quant_pipeline.quantization.utils import apply_quantization
from quant_pipeline.data.loaders import load_sst2
from quant_pipeline.core.benchmark import benchmark
from quant_pipeline.utils.memory import get_model_size
from quant_pipeline.utils.export import save_results_to_csv

def run():
    texts, labels = load_sst2()
    tokenizer = load_tokenizer()

    modes = ["fp32","fp16","int8_ptq","int8_qat"]
    results = []

    for mode in modes:
        model = load_model()
        model = apply_quantization(model, mode, tokenizer, (texts,labels))

        metrics = benchmark(model, tokenizer, texts, labels)

        results.append({
            "mode":mode,
            **metrics,
            "memory_mb":get_model_size(model)
        })

    save_results_to_csv(results)

if __name__=="__main__":
    run()