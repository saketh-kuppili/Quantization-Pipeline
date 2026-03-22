import torch
import platform

from quant_pipeline.quantization.qat import (
    prepare_model_for_qat,
    convert_qat_model,
)
from quant_pipeline.quantization.qat_trainer import train_qat


# -------------------------------
# 🔍 SYSTEM DETECTION
# -------------------------------
def get_system_info():
    system = platform.system()
    processor = platform.processor()

    is_mac = system == "Darwin"
    is_arm = "arm" in processor.lower() or "apple" in processor.lower()

    return is_mac, is_arm


def is_apple_silicon():
    is_mac, is_arm = get_system_info()
    return is_mac and is_arm


# -------------------------------
# 🔥 MAIN QUANTIZATION FUNCTION
# -------------------------------
def apply_quantization(model, mode, tokenizer=None, train_data=None):

    if mode == "fp32":
        return model

    elif mode == "fp16":
        return model.half()

    # ---------------------------------
    # 🔥 INT8 PTQ (CROSS PLATFORM)
    # ---------------------------------
    elif mode == "int8_ptq":

        # ✅ MAC → proxy
        if is_apple_silicon():
            print("⚠️ Mac detected → Using FP16 as INT8 proxy")
            return model.half()

        # ✅ WINDOWS / LINUX → real INT8
        try:
            torch.backends.quantized.engine = "fbgemm"

            model_q = torch.quantization.quantize_dynamic(
                model,
                {torch.nn.Linear},
                dtype=torch.qint8,
            )

            print("✅ True INT8 PTQ applied")
            return model_q

        except Exception as e:
            print("⚠️ INT8 PTQ failed → fallback FP16")
            print("Reason:", e)
            return model.half()

    # ---------------------------------
    # 🔥 INT8 QAT
    # ---------------------------------
    elif mode == "int8_qat":

        # ✅ MAC → proxy
        if is_apple_silicon():
            print("⚠️ Mac detected → Using FP16 as QAT proxy")
            return model.half()

        # ✅ WINDOWS / LINUX → real QAT
        try:
            texts, labels = train_data

            torch.backends.quantized.engine = "fbgemm"

            model = prepare_model_for_qat(model)
            model = train_qat(model, tokenizer, texts, labels, epochs=1)
            model = convert_qat_model(model)

            print("✅ True INT8 QAT applied")
            return model

        except Exception as e:
            print("⚠️ QAT failed → fallback FP16")
            print("Reason:", e)
            return model.half()

    else:
        raise ValueError(f"Invalid mode: {mode}")