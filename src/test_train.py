import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
import torch
import unittest
from train import train_model, save_checkpoint
from seq2seq import HierarchicalSeq2Seq
from dataset import get_dataloader
from data_preprocessing import load_cnn_dailymail, preprocess_data
from transformers import BertTokenizer

class TestTraining(unittest.TestCase):
    def setUp(self):
        self.tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
        self.model = HierarchicalSeq2Seq(vocab_size=30522, hidden_size=768, num_layers=1, num_heads=4, chunk_size=256)
        self.train_data, _ = load_cnn_dailymail()
        self.processed_data = preprocess_data(self.train_data[:2], self.tokenizer, chunk_size=256)
        self.train_loader = get_dataloader(self.processed_data, batch_size=1, num_workers=0)
        self.valid_loader = self.train_loader
        torch.cuda.empty_cache()

    def test_training_loop(self):
        train_model(self.model, self.train_loader, self.valid_loader, num_epochs=1, device="cpu", lr=1e-4)
        self.assertTrue(os.path.exists("checkpoints/checkpoint_epoch_1.pt"))
    
    def test_checkpoint_saving(self):
        optimizer = torch.optim.Adam(self.model.parameters())
        save_checkpoint(self.model, optimizer, 1, 0.5)
        self.assertTrue(os.path.exists("checkpoints/checkpoint_epoch_1.pt"))

if __name__ == "__main__":
    unittest.main()
