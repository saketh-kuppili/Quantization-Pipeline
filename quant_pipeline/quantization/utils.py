import torch

from quant_pipeline.quantization.qat import (
    prepare_model_for_qat,
    convert_qat_model,
)
from quant_pipeline.quantization.qat_trainer import train_qat



#  CROSS-PLATFORM BACKEND SETUP
def set_quantization_backend():
    supported = torch.backends.quantized.supported_engines

    if "fbgemm" in supported:
        torch.backends.quantized.engine = "fbgemm"
    elif "qnnpack" in supported:
        torch.backends.quantized.engine = "qnnpack"
    else:
        print(" No quantization backend available:", supported)


# Call once
set_quantization_backend()



#  SAFE TEST FUNCTION

def test_model_forward(model):
    """
    Runs a small dummy forward pass to ensure model works after quantization.
    """
    try:
        dummy_input = {
            "input_ids": torch.randint(0, 100, (1, 10)),
            "attention_mask": torch.ones(1, 10),
        }

        with torch.no_grad():
            model(**dummy_input)

        return True

    except Exception as e:
        print(" Quantized model failed during forward pass:", e)
        return False



#  MAIN QUANTIZATION FUNCTION

def apply_quantization(model, mode, tokenizer=None, train_data=None):
    if mode == "fp32":
        return model

    elif mode == "fp16":
        return model.half()


    #  INT8 POST-TRAINING QUANTIZATION
    elif mode == "int8_ptq":
        try:
            model_q = torch.quantization.quantize_dynamic(
                model,
                {torch.nn.Linear},
                dtype=torch.qint8,
            )

            if test_model_forward(model_q):
                print(" INT8 PTQ applied successfully")
                return model_q
            else:
                raise RuntimeError("Forward pass failed after PTQ")

        except Exception as e:
            print("\n INT8 PTQ not supported on this system.")
            print("Fallback → FP16")
            print("Reason:", e, "\n")

            return model.half()


    #  INT8 QAT (SAFE VERSION)
    elif mode == "int8_qat":
        try:
            if tokenizer is None or train_data is None:
                raise ValueError("QAT requires tokenizer and training data")

            texts, labels = train_data

            model_qat = prepare_model_for_qat(model)

            model_qat = train_qat(
                model_qat,
                tokenizer,
                texts,
                labels,
                epochs=1,
            )

            model_qat = convert_qat_model(model_qat)

            if test_model_forward(model_qat):
                print(" INT8 QAT applied successfully")
                return model_qat
            else:
                raise RuntimeError("Forward pass failed after QAT")

        except Exception as e:
            print("\n INT8 QAT not supported on this system.")
            print("Fallback → FP16")
            print("Reason:", e, "\n")

            return model.half()

    else:
        raise ValueError(f"Invalid mode: {mode}")