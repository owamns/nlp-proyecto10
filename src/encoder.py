import torch
import torch.nn as nn
from transformers import BertModel, BertTokenizer

class LocalEncoder(nn.Module):
    def __init__(self, model_name='bert-base-uncased', hidden_size=768):
        super(LocalEncoder, self).__init__()
        self.bert = BertModel.from_pretrained(model_name)
        self.hidden_size = hidden_size

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        # Usamos la representación del token [CLS] para cada fragmento
        h_i = outputs.last_hidden_state[:, 0, :]  # [batch_size, hidden_size]
        return h_i

class GlobalEncoder(nn.Module):
    def __init__(self, hidden_size=768, num_heads=8, num_layers=2):
        super(GlobalEncoder, self).__init__()
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=hidden_size, nhead=num_heads)
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=num_layers)

    def forward(self, local_reps):
        # local_reps: [num_chunks, batch_size, hidden_size]
        H = self.transformer_encoder(local_reps)
        return H  # [num_chunks, batch_size, hidden_size]

class HierarchicalEncoder(nn.Module):
    def __init__(self, chunk_size=512, hidden_size=768, num_heads=8, num_layers=2):
        super(HierarchicalEncoder, self).__init__()
        self.local_encoder = LocalEncoder(hidden_size=hidden_size)
        self.global_encoder = GlobalEncoder(hidden_size=hidden_size, num_heads=num_heads, num_layers=num_layers)
        self.chunk_size = chunk_size
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.hidden_size = hidden_size

    def dynamic_chunking(self, input_ids, max_chunks=None):
        """Ajusta dinámicamente el tamaño de los fragmentos según la longitud del documento."""
        total_length = len(input_ids)
        if total_length <= self.chunk_size:
            return [input_ids]
        
        # Si se especifica max_chunks, ajustamos el chunk_size
        if max_chunks and total_length > self.chunk_size * max_chunks:
            self.chunk_size = total_length // max_chunks + 1
        
        chunks = [input_ids[i:i + self.chunk_size] for i in range(0, total_length, self.chunk_size)]
        return chunks

    def forward(self, document, max_chunks=None):
        """
        Procesa un documento completo y devuelve representaciones jerárquicas.
        
        Args:
            document (str): Texto del documento completo.
            max_chunks (int, optional): Número máximo de fragmentos para controlar memoria.
        
        Returns:
            torch.Tensor: Representaciones globales [num_chunks, batch_size, hidden_size]
        """
        # Tokenizar el documento
        tokens = self.tokenizer.tokenize(document)
        input_ids = self.tokenizer.convert_tokens_to_ids(tokens)
        
        # Aplicar chunking dinámico
        chunks = self.dynamic_chunking(input_ids, max_chunks)
        
        # Procesar cada fragmento con el encoder local
        local_reps = []
        for chunk in chunks:
            input_ids_chunk = torch.tensor([chunk], dtype=torch.long)
            attention_mask = torch.ones_like(input_ids_chunk)
            h_i = self.local_encoder(input_ids_chunk, attention_mask)
            local_reps.append(h_i)
        
        # Apilar las representaciones locales
        local_reps = torch.stack(local_reps, dim=0)  # [num_chunks, batch_size, hidden_size]
        
        # Pasar por el encoder global
        global_reps = self.global_encoder(local_reps)
        
        return global_reps
