import unittest
import torch
from loss import CombinedLoss

class TestLoss(unittest.TestCase):
    def setUp(self):
        self.loss_fn = CombinedLoss(lambda_penalty=0.1)
        self.logits = torch.randn(2, 10, 30522)
        self.target = torch.randint(0, 30522, (2, 10))
        self.attn_weights = [torch.randn(2, 10, 5) for _ in range(2)]
    
    def test_loss_computation(self):
        loss = self.loss_fn(self.logits, self.target, self.attn_weights)
        self.assertIsInstance(loss, torch.Tensor)
        self.assertGreater(loss.item(), 0)

if __name__ == "__main__":
    unittest.main()
