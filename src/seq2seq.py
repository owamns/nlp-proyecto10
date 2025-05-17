import torch
import torch.nn as nn
from encoder import HierarchicalEncoder
from decoder import HierarchicalDecoder
from datasets import load_dataset

class HierarchicalSeq2Seq(nn.Module):
    def __init__(self, vocab_size=30522, hidden_size=128, num_layers=1, num_heads=4, chunk_size=128):
        super(HierarchicalSeq2Seq, self).__init__()
        self.encoder = HierarchicalEncoder(chunk_size=chunk_size, hidden_size=hidden_size, 
                                        num_heads=num_heads, num_layers=num_layers)
        self.decoder = HierarchicalDecoder(vocab_size=vocab_size, hidden_size=hidden_size, 
                                        num_layers=num_layers, num_heads=num_heads)
        self.tokenizer = self.encoder.tokenizer

    def forward(self, document, max_chunks=None, device='cpu'):
        H, local_reps = self.encoder(document, max_chunks=max_chunks, device=device)
        schema = self.decoder.generate_schema(H)
        section_titles = ["Introducción", "Metodología", "Conclusión"]
        sections = []
        for title in section_titles:
            section, attn_weights = self.decoder.generate_section(title, H, local_reps)
            sections.append((title, section))
        penalty = self.decoder.compute_coverage_penalty(attn_weights)
        return schema, sections, penalty

def load_and_chunk_data(chunk_size=128):
    dataset = load_dataset("cnn_dailymail", "3.0.0", split="train[:1%]")
    documents = dataset["article"]
    chunked_docs = []
    for doc in documents:
        tokens = doc.split()
        chunks = [tokens[i:i + chunk_size] for i in range(0, len(tokens), chunk_size)]
        chunked_docs.append([" ".join(chunk) for chunk in chunks])
    return chunked_docs, dataset["highlights"]
