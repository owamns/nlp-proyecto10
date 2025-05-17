from datasets import load_dataset
from transformers import BertTokenizer
import torch

def load_cnn_dailymail():
    """Carga el dataset CNN/DailyMail completo."""
    dataset = load_dataset("cnn_dailymail", "3.0.0")
    train_data = dataset["train"]
    valid_data = dataset["validation"]
    return train_data, valid_data

def preprocess_article(article, tokenizer, max_length=512):
    """Tokeniza un artículo y devuelve input_ids y attention_mask."""
    tokens = tokenizer(article, truncation=True, max_length=max_length, return_tensors="pt", padding=False)
    return tokens["input_ids"].squeeze(), tokens["attention_mask"].squeeze()

def preprocess_data(data, tokenizer):
    """Preprocesa el dataset, tokenizando artículos y resúmenes."""
    processed = []
    for example in data:
        article = example["article"]
        summary = example["highlights"]
        
        article_ids, article_mask = preprocess_article(article, tokenizer)
        summary_tokens = tokenizer(summary, truncation=True, max_length=128, return_tensors="pt")
        
        processed.append({
            "article_ids": article_ids,
            "article_mask": article_mask,
            "summary_ids": summary_tokens["input_ids"].squeeze(),
            "summary_mask": summary_tokens["attention_mask"].squeeze()
        })
    return processed

if __name__ == "__main__":
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    train_data, valid_data = load_cnn_dailymail()
    processed_train = preprocess_data(train_data.select(range(100)), tokenizer)
    print(f"Procesados {len(processed_train)} ejemplos de entrenamiento.")
