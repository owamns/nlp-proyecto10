from datasets import load_dataset
from transformers import BertTokenizer
import torch

def load_cnn_dailymail():
    """Carga el dataset CNN/DailyMail completo."""
    train_data = load_dataset("cnn_dailymail", "3.0.0", split="train[:1%]")
    valid_data = load_dataset("cnn_dailymail", "3.0.0", split="validation[:1%]")
    return train_data, valid_data

def dynamic_chunking(input_ids, chunk_size=512, max_chunks=None):
    """Fragmenta dinámicamente un documento, respetando el límite de 512 tokens de BERT."""
    total_length = len(input_ids)
    max_chunk_size = 512  # Límite estricto de BERT

    # Si el documento es menor o igual al chunk_size, devolver como un solo fragmento
    if total_length <= chunk_size:
        return [input_ids]

    # Calcular el número de fragmentos necesarios
    if max_chunks and total_length > chunk_size * max_chunks:
        # Ajustar el número de fragmentos para respetar max_chunk_size
        num_chunks = (total_length + max_chunk_size - 1) // max_chunk_size
        if max_chunks and num_chunks > max_chunks:
            num_chunks = max_chunks
        chunk_size = min(max_chunk_size, (total_length + num_chunks - 1) // num_chunks)
    else:
        chunk_size = min(chunk_size, max_chunk_size)

    # Dividir en fragmentos
    chunks = [input_ids[i:i + chunk_size] for i in range(0, total_length, chunk_size)]
    
    # Asegurar que ningún fragmento supere max_chunk_size
    final_chunks = []
    for chunk in chunks:
        while len(chunk) > max_chunk_size:
            final_chunks.append(chunk[:max_chunk_size])
            chunk = chunk[max_chunk_size:]
        if len(chunk) > 0:
            final_chunks.append(chunk)
    
    # Limitar a max_chunks si es necesario
    if max_chunks and len(final_chunks) > max_chunks:
        final_chunks = final_chunks[:max_chunks]
    
    return final_chunks

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
    sentences = article.split(". ")[:3]
    schema_text = ". ".join(sentences)
    tokens = tokenizer(schema_text, truncation=True, max_length=max_length, return_tensors="pt")
    return tokens["input_ids"].squeeze(), tokens["attention_mask"].squeeze()

def preprocess_data(data, tokenizer, chunk_size=512, max_chunks=5):
    """Preprocesa el dataset, incluyendo targets para esquema y secciones."""
    if isinstance(data, dict):
        keys = list(data.keys())
        length = len(data[keys[0]])
        data = [{k: data[k][i] for k in keys} for i in range(length)]
    
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
