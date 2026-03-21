def get_model_size(model):
    size = sum(p.numel()*p.element_size() for p in model.parameters())
    return size / (1024**2)