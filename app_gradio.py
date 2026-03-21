import gradio as gr
from quant_pipeline.core.pipeline import Pipeline

def predict(text, precision):
    pipe = Pipeline(precision=precision)
    res = pipe.predict(text)
    return f"{res['label']} ({res['confidence']:.2f})"

gr.Interface(
    fn=predict,
    inputs=["text", gr.Dropdown(["fp32","fp16","int8_ptq","int8_qat"])],
    outputs="text",
    title="Quantized DistilBERT"
).launch()