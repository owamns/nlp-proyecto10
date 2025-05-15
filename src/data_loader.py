from datasets import load_dataset
import numpy as np


def load_and_chunk_data(chunk_size=512):
    dataset = load_dataset("cnn_dailymail", "3.0.0", split="train[:1%]")
    documents = dataset["article"]
    chunked_docs = []

    for doc in documents:
        tokens = doc.split()
        chunks = [tokens[i:i + chunk_size] for i in range(0, len(tokens), chunk_size)]
        chunked_docs.append([" ".join(chunk) for chunk in chunks])
    return chunked_docs, dataset["highlights"]


if __name__ == "__main__":
    chunks, summaries = load_and_chunk_data()
    print(f"Documento 1, chunk 1: {chunks[0][0][:50]}...")
    print(f"Resumen 1: {summaries[0]}")