import unittest
from dataset import CNNDailyMailDataset, get_dataloader
from data_preprocessing import load_cnn_dailymail, preprocess_data
from transformers import BertTokenizer

class TestDataset(unittest.TestCase):
    def setUp(self):
        self.tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
        self.train_data, _ = load_cnn_dailymail()
        self.processed_data = preprocess_data(self.train_data[:10], self.tokenizer)
    
    def test_dataset_length(self):
        dataset = CNNDailyMailDataset(self.processed_data)
        self.assertEqual(len(dataset), 10)
    
    def test_dataloader_batch(self):
        dataloader = get_dataloader(self.processed_data, batch_size=2, num_workers=0)
        batch = next(iter(dataloader))
        self.assertIn("chunks", batch)
        self.assertIn("schema_ids", batch)
        self.assertIn("summary_ids", batch)

if __name__ == "__main__":
    unittest.main()
