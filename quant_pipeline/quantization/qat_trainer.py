import torch
from torch.optim import AdamW
from tqdm import tqdm


def train_qat(model, tokenizer, texts, labels, epochs=1):
    optimizer = AdamW(model.parameters(), lr=5e-5)

    for _ in range(epochs):
        loop = tqdm(zip(texts, labels), total=len(texts))
        for text, label in loop:
            inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)

            outputs = model(**inputs, labels=torch.tensor([label]))
            loss = outputs.loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    return model