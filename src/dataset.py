import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence

class CNNDailyMailDataset(Dataset):
    def __init__(self, processed_data):
        self.data = processed_data
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        return {
            "chunks": item["chunks"],
            "schema_ids": item["schema_ids"],
            "schema_mask": item["schema_mask"],
            "summary_ids": item["summary_ids"],
            "summary_mask": item["summary_mask"]
        }

def collate_fn(batch):
    """Aplica padding dinámico a los batches."""
    chunked_inputs = []
    schema_ids = []
    schema_masks = []
    summary_ids = []
    summary_masks = []
    
    max_chunks = max(len(item["chunks"]) for item in batch)
    
    for item in batch:
        chunks = item["chunks"]
        padded_chunks = chunks + [{"input_ids": torch.tensor([]), "attention_mask": torch.tensor([])}] * (max_chunks - len(chunks))
        chunked_inputs.append(padded_chunks)
        
        schema_ids.append(item["schema_ids"])
        schema_masks.append(item["schema_mask"])
        summary_ids.append(item["summary_ids"])
        summary_masks.append(item["summary_mask"])
    
    # Aplicar padding a schema y summary
    schema_ids = pad_sequence(schema_ids, batch_first=True, padding_value=0)
    schema_masks = pad_sequence(schema_masks, batch_first=True, padding_value=0)
    summary_ids = pad_sequence(summary_ids, batch_first=True, padding_value=0)
    summary_masks = pad_sequence(summary_masks, batch_first=True, padding_value=0)
    
    # Estructurar chunks para el encoder
    chunked_data = []
    for i in range(max_chunks):
        chunk_batch = {
            "input_ids": pad_sequence([item[i]["input_ids"] for item in chunked_inputs if len(item[i]["input_ids"]) > 0], batch_first=True, padding_value=0),
            "attention_mask": pad_sequence([item[i]["attention_mask"] for item in chunked_inputs if len(item[i]["attention_mask"]) > 0], batch_first=True, padding_value=0)
        }
        chunked_data.append(chunk_batch)
    
    return {
        "chunks": chunked_data,
        "schema_ids": schema_ids,
        "schema_mask": schema_masks,
        "summary_ids": summary_ids,
        "summary_mask": summary_masks
    }

def get_dataloader(processed_data, batch_size=4):
    """Crea un DataLoader con padding."""
    dataset = CNNDailyMailDataset(processed_data)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
