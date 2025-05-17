import torch
import torch.nn as nn
from transformers import BertTokenizer

class SchemaDecoder(nn.Module):
    def __init__(self, vocab_size, hidden_size, num_layers, num_heads):
        super(SchemaDecoder, self).__init__()
        self.embedding = nn.Embedding(vocab_size, hidden_size)
        self.pos_encoder = nn.Embedding(512, hidden_size)
        decoder_layer = nn.TransformerDecoderLayer(d_model=hidden_size, nhead=num_heads)
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)
        self.fc_out = nn.Linear(hidden_size, vocab_size)

    def forward(self, tgt, memory):
        tgt_pos = torch.arange(0, tgt.size(1)).unsqueeze(0).to(tgt.device)
        tgt_emb = self.embedding(tgt) + self.pos_encoder(tgt_pos)
        tgt_emb = tgt_emb.transpose(0, 1)
        output = self.transformer_decoder(tgt_emb, memory)
        logits = self.fc_out(output)
        return logits

class CustomDecoderLayer(nn.TransformerDecoderLayer):
    def __init__(self, d_model, nhead, dim_feedforward=512, dropout=0.1, activation="relu"):
        super(CustomDecoderLayer, self).__init__(d_model, nhead, dim_feedforward, dropout, activation)
        self.attention_weights = None

    def forward(self, tgt, memory, tgt_mask=None, memory_mask=None, 
                tgt_key_padding_mask=None, memory_key_padding_mask=None, 
                tgt_is_causal=None, memory_is_causal=None):
        tgt2, _ = self.self_attn(tgt, tgt, tgt, attn_mask=tgt_mask, 
                                key_padding_mask=tgt_key_padding_mask, 
                                is_causal=tgt_is_causal)
        tgt = tgt + self.dropout1(tgt2)
        tgt = self.norm1(tgt)
        
        tgt2, self.attention_weights = self.multihead_attn(tgt, memory, memory, 
                                                         attn_mask=memory_mask, 
                                                         key_padding_mask=memory_key_padding_mask, 
                                                         is_causal=memory_is_causal,
                                                         need_weights=True)
        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)
        
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout3(tgt2)
        tgt = self.norm3(tgt)
        return tgt

class SectionDecoder(nn.Module):
    def __init__(self, vocab_size, hidden_size, num_layers, num_heads):
        super(SectionDecoder, self).__init__()
        self.embedding = nn.Embedding(vocab_size, hidden_size)
        self.pos_encoder = nn.Embedding(512, hidden_size)
        decoder_layer = CustomDecoderLayer(d_model=hidden_size, nhead=num_heads, dim_feedforward=512)
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)
        self.fc_out = nn.Linear(hidden_size, vocab_size)
        self.memory_projection = nn.Linear(2 * hidden_size, hidden_size)

    def forward(self, tgt, memory, tgt_mask=None, memory_key_padding_mask=None):
        memory_proj = self.memory_projection(memory)
        tgt_pos = torch.arange(0, tgt.size(1)).unsqueeze(0).to(tgt.device)
        tgt_emb = self.embedding(tgt) + self.pos_encoder(tgt_pos)
        output = self.transformer_decoder(tgt_emb.transpose(0, 1), memory_proj.transpose(0, 1), 
                                        tgt_mask=tgt_mask, memory_key_padding_mask=memory_key_padding_mask)
        output = output.transpose(0, 1)
        attn_weights = [layer.attention_weights.transpose(0, 1) for layer in self.transformer_decoder.layers]
        return output, attn_weights

class HierarchicalDecoder(nn.Module):
    def __init__(self, vocab_size, hidden_size, num_layers, num_heads):
        super(HierarchicalDecoder, self).__init__()
        self.schema_decoder = SchemaDecoder(vocab_size, hidden_size, num_layers, num_heads)
        self.section_decoder = SectionDecoder(vocab_size, hidden_size, num_layers, num_heads)
        self.tokenizer = BertTokenizer.from_pretrained('prajjwal1/bert-tiny')
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size

    def generate_schema(self, H, max_length=50):
        batch_size = H.size(1)
        tgt = torch.tensor([[self.tokenizer.cls_token_id]] * batch_size, dtype=torch.long).to(H.device)
        for _ in range(max_length):
            logits = self.schema_decoder(tgt, H)
            next_token = logits[-1, :, :].argmax(dim=-1).unsqueeze(0)
            tgt = torch.cat([tgt, next_token], dim=0)
            if next_token[0, 0].item() == self.tokenizer.sep_token_id:
                break
        schema = self.tokenizer.decode(tgt[:, 0].tolist())
        return schema

    def generate_section(self, title, H, local_reps, max_length=200):
        batch_size = H.size(1)
        memory = torch.cat([H, local_reps], dim=-1)
        memory = memory.transpose(0, 1)
        title_tokens = self.tokenizer.encode(f"[CLS] {title} [SEP]", add_special_tokens=False)
        tgt = torch.tensor([title_tokens] * batch_size, dtype=torch.long).to(H.device)
        
        seq_len = tgt.size(1)
        tgt_mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1).bool().to(H.device)
        
        outputs = []
        for _ in range(max_length):
            output, attn_weights = self.section_decoder(tgt, memory, tgt_mask=tgt_mask)
            logits = self.section_decoder.fc_out(output)
            next_token = logits[:, -1, :].argmax(dim=-1).unsqueeze(1)
            outputs.append(next_token)
            tgt = torch.cat([tgt, next_token], dim=1)
            if next_token[0, 0].item() == self.tokenizer.sep_token_id:
                break
            seq_len += 1
            tgt_mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1).bool().to(H.device)
        
        section = self.tokenizer.decode(torch.cat(outputs, dim=1)[0].tolist())
        return section, attn_weights

    def compute_coverage_penalty(self, attn_weights):
        total_attention = torch.stack([w.sum(dim=-2) for w in attn_weights], dim=0).sum(dim=0)
        coverage = torch.min(total_attention, torch.ones_like(total_attention))
        penalty = coverage.sum()
        return penalty
