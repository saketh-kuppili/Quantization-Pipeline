from quant_pipeline.models.nlp.distilbert import load_model
from quant_pipeline.quantization.utils import apply_quantization

def test_quant():
    model = load_model()
    q = apply_quantization(model,"int8_ptq")
    assert q is not None