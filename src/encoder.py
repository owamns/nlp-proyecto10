from transformers import BertTokenizer, BertModel
import torch
import torch.nn as nn

class HierarchicalEncoder(nn.Module):
 def __init__(self, local_model_name="bert-base-uncased", global_layers=2):
     super().__init__()
     self.tokenizer = BertTokenizer.from_pretrained(local_model_name)
     self.local_encoder = BertModel.from_pretrained(local_model_name)
     self.global_encoder = nn.TransformerEncoder(
         nn.TransformerEncoderLayer(d_model=768, nhead=4), num_layers=global_layers
     )

 def forward(self, chunks):
     local_reps = []
     for chunk in chunks:
         inputs = self.tokenizer(chunk, return_tensors="pt", padding=True, truncation=True)
         outputs = self.local_encoder(**inputs).last_hidden_state[:, 0, :]
         local_reps.append(outputs)
     local_reps = torch.stack(local_reps).squeeze(1)
     global_reps = self.global_encoder(local_reps)
     return global_reps

if __name__ == "__main__":
 encoder = HierarchicalEncoder()
 sample_chunks = ["Este es un chunk de prueba.", "Otro chunk para probar."]
 reps = encoder(sample_chunks)
 print(f"Dimensiones de salida: {reps.shape}")