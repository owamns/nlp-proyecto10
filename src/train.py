import torch
from torch.optim import Adam
from tqdm import tqdm
from seq2seq import HierarchicalSeq2Seq
from dataset import get_dataloader
from loss import CombinedLoss
import os
import logging

def setup_logging():
    """Configura el registro de entrenamiento."""
    logging.basicConfig(
        filename="training.log",
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s"
    )

def save_checkpoint(model, optimizer, epoch, loss, path="checkpoints"):
    """Guarda un checkpoint del modelo."""
    os.makedirs(path, exist_ok=True)
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "epoch": epoch,
        "loss": loss
    }
    torch.save(checkpoint, os.path.join(path, f"checkpoint_epoch_{epoch}.pt"))

def train_model(model, train_loader, valid_loader, num_epochs=10, device="cuda", lr=1e-4):
    """Entrena el modelo Seq2Seq."""
    setup_logging()
    model = model.to(device)
    optimizer = Adam(model.parameters(), lr=lr)
    loss_fn = CombinedLoss(lambda_penalty=0.1)
    
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}"):
            chunks = batch["chunks"]
            schema_ids = batch["schema_ids"].to(device)
            summary_ids = batch["summary_ids"].to(device)
            
            optimizer.zero_grad()
            
            chunk_outputs = []
            for chunk in chunks:
                input_ids = chunk["input_ids"].to(device)
                attention_mask = chunk["attention_mask"].to(device)
                if input_ids.numel() > 0:
                    chunk_output = model.encoder.local_encoder(input_ids, attention_mask)
                    chunk_outputs.append(chunk_output)
            
            local_reps = torch.stack(chunk_outputs, dim=0)
            H = model.encoder.global_encoder(local_reps)
            
            schema_logits = model.decoder.schema_decoder(schema_ids[:-1], H)
            schema_loss = loss_fn(schema_logits.transpose(0, 1), schema_ids[1:])
            
            section_logits, attn_weights = model.decoder.section_decoder(summary_ids[:-1], torch.cat([H, local_reps], dim=-1).transpose(0, 1))
            section_loss = loss_fn(section_logits, summary_ids[1:], attn_weights)
            
            loss = schema_loss + section_loss
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        
        avg_loss = total_loss / len(train_loader)
        logging.info(f"Epoch {epoch+1}, Loss: {avg_loss}")
        print(f"Epoch {epoch+1}, Loss: {avg_loss}")
        save_checkpoint(model, optimizer, epoch + 1, avg_loss)
