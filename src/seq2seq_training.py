import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast, GradScaler
from transformers import BertTokenizer
from data_preprocessing import load_cnn_dailymail, preprocess_data
from dataset import get_dataloader
from seq2seq import HierarchicalSeq2Seq
from loss import CombinedLoss
from tqdm import tqdm
import os

# Configurar manejo de memoria de PyTorch
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

# Configuraciones
device = torch.device("cuda")
batch_size = 1  # Reducido para minimizar uso de memoria
num_epochs = 10
learning_rate = 1e-4
chunk_size = 256  # Reducido de 512
max_chunks = 3    # Reducido de 5
checkpoint_dir = "./checkpoints"
os.makedirs(checkpoint_dir, exist_ok=True)
accum_steps = 4   # Acumulación de gradientes para simular batch_size=4

# Cargar y preprocesar datos
print("Cargando y preprocesando datos...")
train_data, valid_data = load_cnn_dailymail()
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
processed_train = preprocess_data(train_data, tokenizer, chunk_size, max_chunks)
processed_valid = preprocess_data(valid_data, tokenizer, chunk_size, max_chunks)

# Crear dataloaders
train_loader = get_dataloader(processed_train, batch_size=batch_size, num_workers=2)
valid_loader = get_dataloader(processed_valid, batch_size=batch_size, num_workers=2)

# Instanciar modelo
model = HierarchicalSeq2Seq(
    vocab_size=tokenizer.vocab_size,
    hidden_size=768,
    num_layers=2,
    num_heads=8,
    chunk_size=chunk_size
).to(device)

# Optimizador, función de pérdida y escalador para precisión mixta
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
loss_fn = CombinedLoss(lambda_penalty=0.1).to(device)
scaler = GradScaler()

# Máscara autoregresiva
def generate_square_subsequent_mask(sz):
    mask = torch.triu(torch.ones(sz, sz), diagonal=1).bool()
    return mask.to(device)

# Función de entrenamiento
def train_epoch(model, dataloader, optimizer, loss_fn, accum_steps):
    model.train()
    total_loss = 0
    optimizer.zero_grad()
    for i, batch in enumerate(tqdm(dataloader, desc="Entrenando")):
        chunks = [{k: v.to(device) for k, v in chunk.items()} for chunk in batch["chunks"]]
        schema_ids = batch["schema_ids"].to(device)
        summary_ids = batch["summary_ids"].to(device)

        # Precisión mixta
        with autocast():
            # Encoder
            local_reps = []
            batch_size = schema_ids.size(0)
            for chunk in chunks:
                input_ids = chunk["input_ids"]
                attention_mask = chunk["attention_mask"]
                if input_ids.size(0) == batch_size and input_ids.numel() > 0:
                    h_i = model.encoder.local_encoder(input_ids, attention_mask)
                    local_reps.append(h_i)
            if not local_reps:
                continue
            local_reps = torch.stack(local_reps, dim=0)
            H = model.encoder.global_encoder(local_reps)

            # Schema Decoder
            tgt_schema = schema_ids[:, :-1]
            target_schema = schema_ids[:, 1:]
            logits_schema = model.decoder.schema_decoder(tgt_schema, H)
            logits_schema = logits_schema.transpose(0, 1)
            loss_schema = loss_fn(logits_schema, target_schema)

            # Section Decoder
            memory = torch.cat([H, local_reps], dim=-1).transpose(0, 1)
            tgt_section = summary_ids[:, :-1]
            target_section = summary_ids[:, 1:]
            tgt_mask = generate_square_subsequent_mask(tgt_section.size(1))
            output, attn_weights = model.decoder.section_decoder(tgt_section, memory, tgt_mask=tgt_mask)
            logits_section = model.decoder.section_decoder.fc_out(output)
            loss_section = loss_fn(logits_section, target_section, attn_weights)

            # Pérdida total
            loss = (loss_schema + loss_section) / accum_steps

        # Acumulación de gradientes
        scaler.scale(loss).backward()
        if (i + 1) % accum_steps == 0 or (i + 1) == len(dataloader):
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

        total_loss += loss.item() * accum_steps
        torch.cuda.empty_cache()  # Liberar caché

    return total_loss / len(dataloader)

# Función de validación
def validate_epoch(model, dataloader, loss_fn):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Validando"):
            chunks = [{k: v.to(device) for k, v in chunk.items()} for chunk in batch["chunks"]]
            schema_ids = batch["schema_ids"].to(device)
            summary_ids = batch["summary_ids"].to(device)

            with autocast():
                local_reps = []
                batch_size = schema_ids.size(0)
                for chunk in chunks:
                    input_ids = chunk["input_ids"]
                    attention_mask = chunk["attention_mask"]
                    if input_ids.size(0) == batch_size and input_ids.numel() > 0:
                        h_i = model.encoder.local_encoder(input_ids, attention_mask)
                        local_reps.append(h_i)
                if not local_reps:
                    continue
                local_reps = torch.stack(local_reps, dim=0)
                H = model.encoder.global_encoder(local_reps)

                # Schema Decoder
                tgt_schema = schema_ids[:, :-1]
                target_schema = schema_ids[:, 1:]
                logits_schema = model.decoder.schema_decoder(tgt_schema, H)
                logits_schema = logits_schema.transpose(0, 1)
                loss_schema = loss_fn(logits_schema, target_schema)

                # Section Decoder
                memory = torch.cat([H, local_reps], dim=-1).transpose(0, 1)
                tgt_section = summary_ids[:, :-1]
                target_section = summary_ids[:, 1:]
                tgt_mask = generate_square_subsequent_mask(tgt_section.size(1))
                output, attn_weights = model.decoder.section_decoder(tgt_section, memory, tgt_mask=tgt_mask)
                logits_section = model.decoder.section_decoder.fc_out(output)
                loss_section = loss_fn(logits_section, target_section, attn_weights)

                loss = loss_schema + loss_section
            total_loss += loss.item()
            torch.cuda.empty_cache()
    return total_loss / len(dataloader)

# Bucle principal de entrenamiento
for epoch in range(num_epochs):
    print(f"\nEpoca {epoch + 1}/{num_epochs}")
    train_loss = train_epoch(model, train_loader, optimizer, loss_fn, accum_steps)
    print(f"Pérdida de entrenamiento: {train_loss:.4f}")

    val_loss = validate_epoch(model, valid_loader, loss_fn)
    print(f"Pérdida de validación: {val_loss:.4f}")

    # Guardar checkpoint
    torch.save(model.state_dict(), os.path.join(checkpoint_dir, f"model_epoch_{epoch + 1}.pth"))

print("Entrenamiento completado!")
torch.save(model.state_dict(), os.path.join(checkpoint_dir, "hierarchical_seq2seq_final.pth"))
