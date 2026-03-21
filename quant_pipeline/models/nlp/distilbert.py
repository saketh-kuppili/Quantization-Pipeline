from transformers import AutoModelForSequenceClassification, AutoTokenizer

MODEL_NAME = "distilbert-base-uncased-finetuned-sst-2-english"

def load_model():
    return AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)

def load_tokenizer():
    return AutoTokenizer.from_pretrained(MODEL_NAME)