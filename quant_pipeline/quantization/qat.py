import torch
from torch.quantization import get_default_qat_qconfig, prepare_qat, convert


def prepare_model_for_qat(model):
    model.train()

    #  disable global quantization
    model.qconfig = None

    qconfig = get_default_qat_qconfig("qnnpack")

    #  ONLY apply to Linear layers
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Linear):
            module.qconfig = qconfig

    model = prepare_qat(model)

    return model


def convert_qat_model(model):
    model.eval()
    return convert(model)