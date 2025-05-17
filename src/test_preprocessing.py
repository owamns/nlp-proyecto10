import unittest
from data_loader import load_cnn_dailymail, preprocess_data, preprocess_article, generate_schema_target
from transformers import BertTokenizer

class TestPreprocessing(unittest.TestCase):
    def setUp(self):
        self.tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
        self.train_data, self.valid_data = load_cnn_dailymail()
        self.sample_article = self.train_data[0]["article"]
        self.sample_summary = self.train_data[0]["highlights"]
    
    def test_load_dataset(self):
        self.assertGreater(len(self.train_data), 0)
        self.assertGreater(len(self.valid_data), 0)
    
    def test_preprocess_article(self):
        chunks = preprocess_article(self.sample_article, self.tokenizer, chunk_size=512, max_chunks=5)
        self.assertTrue(all("input_ids" in chunk and "attention_mask" in chunk for chunk in chunks))
    
    def test_generate_schema(self):
        schema_ids, schema_mask = generate_schema_target(self.sample_article, self.tokenizer)
        self.assertEqual(schema_ids.shape, schema_mask.shape)
    
    def test_preprocess_data(self):
        processed = preprocess_data(self.train_data[:10], self.tokenizer)
        self.assertEqual(len(processed), 10)
        self.assertTrue(all("chunks" in item and "schema_ids" in item and "summary_ids" in item for item in processed))

if __name__ == "__main__":
    unittest.main()
