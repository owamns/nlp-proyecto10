import torch
import torch.nn as nn

class CombinedLoss(nn.Module):
    def __init__(self, lambda_penalty=0.1):
        super(CombinedLoss, self).__init__()
        self.cross_entropy = nn.CrossEntropyLoss(ignore_index=0)
        self.lambda_penalty = lambda_penalty
    
    def forward(self, logits, target, attn_weights=None):
        """Calcula pérdida combinada: entropía cruzada + coverage penalty."""
        batch_size, seq_len, vocab_size = logits.shape
        logits = logits.view(-1, vocab_size)
        target = target.view(-1)
        ce_loss = self.cross_entropy(logits, target)
        
        penalty = 0
        if attn_weights is not None:
            total_attention = torch.stack([w.sum(dim=-2) for w in attn_weights], dim=0).sum(dim=0)
            coverage = torch.min(total_attention, torch.ones_like(total_attention))
            penalty = coverage.sum()
        
        return ce_loss + self.lambda_penalty * penalty
