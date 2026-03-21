from datasets import load_dataset

def load_sst2(split="validation", sample_size=200):
    dataset = load_dataset("glue", "sst2", split=split)
    dataset = dataset.shuffle(seed=42).select(range(sample_size))
    return dataset["sentence"], dataset["label"]