from quant_pipeline.core.pipeline import Pipeline

def test_pipeline():
    pipe = Pipeline()
    assert pipe.predict("good") is not None