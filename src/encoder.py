import torch
import torch.nn as nn
from transformers import BertModel, BertTokenizer

class LocalEncoder(nn.Module):
    def __init__(self, model_name='prajjwal1/bert-tiny', hidden_size=128):
        super(LocalEncoder, self).__init__()
        self.bert = BertModel.from_pretrained(model_name)
        self.bert.gradient_checkpointing = True  # Activar checkpointing
        self.hidden_size = hidden_size

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        h_i = outputs.last_hidden_state[:, 0, :]  # [batch_size, hidden_size]
        return h_i

class GlobalEncoder(nn.Module):
    def __init__(self, hidden_size=128, num_heads=4, num_layers=1):
        super(GlobalEncoder, self).__init__()
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=hidden_size, nhead=num_heads)
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=num_layers)

    def forward(self, local_reps):
        H = self.transformer_encoder(local_reps)
        return H

class HierarchicalEncoder(nn.Module):
    def __init__(self, chunk_size=128, hidden_size=128, num_heads=4, num_layers=1):
        super(HierarchicalEncoder, self).__init__()
        self.local_encoder = LocalEncoder(hidden_size=hidden_size)
        self.global_encoder = GlobalEncoder(hidden_size=hidden_size, num_heads=num_heads, num_layers=num_layers)
        self.chunk_size = chunk_size
        self.tokenizer = BertTokenizer.from_pretrained('prajjwal1/bert-tiny')
        self.hidden_size = hidden_size

    def dynamic_chunking(self, input_ids, max_chunks=None):
        """Ajusta dinámicamente el tamaño de los fragmentos según la longitud del documento."""
        total_length = len(input_ids)
        if total_length <= self.chunk_size:
            return [input_ids]
        
        chunks = [input_ids[i:i + self.chunk_size] for i in range(0, total_length, self.chunk_size)]
        if max_chunks and len(chunks) > max_chunks:
            chunks = chunks[:max_chunks]
        return chunks

    def forward(self, document, max_chunks=None, device='cpu'):
        tokens = self.tokenizer.tokenize(document)
        input_ids = self.tokenizer.convert_tokens_to_ids(tokens)
        chunks = self.dynamic_chunking(input_ids, max_chunks)
        
        local_reps = []
        for chunk in chunks:
            input_ids_chunk = torch.tensor([chunk], dtype=torch.long).to(device)
            attention_mask = torch.ones_like(input_ids_chunk).to(device)
            h_i = self.local_encoder(input_ids_chunk, attention_mask)
            local_reps.append(h_i)
        
        local_reps = torch.stack(local_reps, dim=0)
        global_reps = self.global_encoder(local_reps)
        
        return global_reps, local_reps
