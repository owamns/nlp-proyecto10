from datasets import load_dataset
from transformers import BertTokenizer
import torch

def load_cnn_dailymail():
    """Carga el dataset CNN/DailyMail completo."""
    dataset = load_dataset("cnn_dailymail", "3.0.0")
    train_data = dataset["train"]
    valid_data = dataset["validation"]
    return train_data, valid_data

def dynamic_chunking(input_ids, chunk_size=512, max_chunks=None):
    """Fragmenta dinámicamente un documento."""
    total_length = len(input_ids)
    if total_length <= chunk_size:
        return [input_ids]
    
    if max_chunks and total_length > chunk_size * max_chunks:
        chunk_size = total_length // max_chunks + 1
    
    return [input_ids[i:i + chunk_size] for i in range(0, total_length, chunk_size)]

def preprocess_article(article, tokenizer, chunk_size=512, max_chunks=5):
    """Tokeniza y fragmenta un artículo."""
    tokens = tokenizer.tokenize(article)
    input_ids = tokenizer.convert_tokens_to_ids(tokens)
    chunks = dynamic_chunking(input_ids, chunk_size, max_chunks)
    
    chunked_data = []
    for chunk in chunks:
        chunk_ids = torch.tensor(chunk)
        chunk_mask = torch.ones_like(chunk_ids)
        chunked_data.append({"input_ids": chunk_ids, "attention_mask": chunk_mask})
    return chunked_data

def generate_schema_target(article, tokenizer, max_length=50):
    """Genera un esquema simulado (primeras frases del artículo)."""
    sentences = article.split(". ")[:3]  # Tomar primeras 3 oraciones
    schema_text = ". ".join(sentences)
    tokens = tokenizer(schema_text, truncation=True, max_length=max_length, return_tensors="pt")
    return tokens["input_ids"].squeeze(), tokens["attention_mask"].squeeze()

def preprocess_data(data, tokenizer, chunk_size=512, max_chunks=5):
    """Preprocesa el dataset, incluyendo targets para esquema y secciones."""
    processed = []
    for example in data:
        article = example["article"]
        summary = example["highlights"]
        
        chunked_article = preprocess_article(article, tokenizer, chunk_size, max_chunks)
        schema_ids, schema_mask = generate_schema_target(article, tokenizer)
        summary_tokens = tokenizer(summary, truncation=True, max_length=128, return_tensors="pt")
        
        processed.append({
            "chunks": chunked_article,
            "schema_ids": schema_ids,
            "schema_mask": schema_mask,
            "summary_ids": summary_tokens["input_ids"].squeeze(),
            "summary_mask": summary_tokens["attention_mask"].squeeze()
        })
    return processed

if __name__ == "__main__":
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    train_data, valid_data = load_cnn_dailymail()
    processed_train = preprocess_data(train_data.select(range(100)), tokenizer)
    print(f"Procesados {len(processed_train)} ejemplos de entrenamiento.")
