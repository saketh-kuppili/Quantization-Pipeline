from quant_pipeline.models.nlp.distilbert import load_model, load_tokenizer
from quant_pipeline.data.loaders import load_sst2
from quant_pipeline.analysis.sensitivity import analyze_sensitivity
from quant_pipeline.analysis.visualization import plot_sensitivity_heatmap
from quant_pipeline.core.benchmark import benchmark

def run():
    model = load_model()
    tokenizer = load_tokenizer()
    texts, labels = load_sst2(sample_size=100)

    layers = [n for n,_ in model.named_modules()]
    results = analyze_sensitivity(model, tokenizer, texts, labels, layers, benchmark)

    plot_sensitivity_heatmap(results)

if __name__=="__main__":
    run()