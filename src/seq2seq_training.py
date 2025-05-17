import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from transformers import BertTokenizer
from data_preprocessing import load_cnn_dailymail, preprocess_data
from dataset import get_dataloader
from seq2seq import HierarchicalSeq2Seq
from loss import CombinedLoss
from tqdm import tqdm
import os
import logging
from torch.amp import autocast, GradScaler

# Configurar logging para depuración
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Configuraciones
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
batch_size = 1
num_epochs = 10
learning_rate = 1e-4
chunk_size = 128
max_chunks = 2
accum_steps = 1
checkpoint_dir = os.path.abspath("./checkpoints")  # Usar path absoluto
os.makedirs(checkpoint_dir, exist_ok=True)

# Verificar que el directorio es escribible
try:
    test_file = os.path.join(checkpoint_dir, "test.txt")
    with open(test_file, 'w') as f:
        f.write("test")
    os.remove(test_file)
    logger.info(f"Directorio {checkpoint_dir} es escribible")
except Exception as e:
    logger.error(f"Error: No se puede escribir en {checkpoint_dir}: {str(e)}")
    raise

# Configurar PyTorch para evitar fragmentación
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

# Cargar y preprocesar datos
logger.info("Cargando y preprocesando datos...")
train_data, valid_data = load_cnn_dailymail()
# train_data = train_data.select(range(100))  # Comentar para usar dataset completo
tokenizer = BertTokenizer.from_pretrained('prajjwal1/bert-tiny')
processed_train = preprocess_data(train_data, tokenizer, chunk_size, max_chunks)
processed_valid = preprocess_data(valid_data, tokenizer, chunk_size, max_chunks)

# Crear dataloaders
train_loader = get_dataloader(processed_train, batch_size=batch_size, num_workers=0)
valid_loader = get_dataloader(processed_valid, batch_size=batch_size, num_workers=0)

# Instanciar modelo
model = HierarchicalSeq2Seq(
    vocab_size=tokenizer.vocab_size,
    hidden_size=128,
    num_layers=1,
    num_heads=4,
    chunk_size=chunk_size
).to(device)

# Optimizador, pérdida y escalador
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
loss_fn = CombinedLoss(lambda_penalty=0.1).to(device)
scaler = GradScaler(device='cuda')

# Máscara autoregresiva
def generate_square_subsequent_mask(sz):
    mask = torch.triu(torch.ones(sz, sz), diagonal=1).bool()
    return mask.to(device)

# Función para guardar checkpoint
def save_checkpoint(model, path):
    try:
        torch.save(model.state_dict(), path)
        logger.info(f"Checkpoint guardado exitosamente en: {path}")
    except Exception as e:
        logger.error(f"Error al guardar checkpoint en {path}: {str(e)}")
        raise

# Función de entrenamiento
def train_epoch(model, dataloader, optimizer, loss_fn, accum_steps):
    model.train()
    total_loss = 0
    optimizer.zero_grad()
    for i, batch in enumerate(tqdm(dataloader, desc="Entrenando")):
        chunks = [{k: v.to(device) for k, v in chunk.items()} for chunk in batch["chunks"]]
        schema_ids = batch["schema_ids"].to(device)
        summary_ids = batch["summary_ids"].to(device)

        with autocast(device_type='cuda'):
            local_reps = []
            for chunk in chunks:
                input_ids = chunk["input_ids"]
                attention_mask = chunk["attention_mask"]
                if input_ids.numel() > 0:
                    h_i = model.encoder.local_encoder(input_ids, attention_mask)
                    local_reps.append(h_i)
            local_reps = torch.stack(local_reps, dim=0)
            H = model.encoder.global_encoder(local_reps)

            tgt_schema = schema_ids[:, :-1]
            target_schema = schema_ids[:, 1:]
            logits_schema = model.decoder.schema_decoder(tgt_schema, H)
            logits_schema = logits_schema.transpose(0, 1)
            loss_schema = loss_fn(logits_schema, target_schema)

            memory = torch.cat([H, local_reps], dim=-1).transpose(0, 1)
            tgt_section = summary_ids[:, :-1]
            target_section = summary_ids[:, 1:]
            tgt_mask = generate_square_subsequent_mask(tgt_section.size(1))
            output, attn_weights = model.decoder.section_decoder(tgt_section, memory, tgt_mask=tgt_mask)
            logits_section = model.decoder.section_decoder.fc_out(output)
            loss_section = loss_fn(logits_section, target_section, attn_weights)

            loss = loss_schema + loss_section

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad()
        torch.cuda.empty_cache()

        total_loss += loss.item()
        del chunks, schema_ids, summary_ids, local_reps, H, memory, logits_schema, logits_section, output, attn_weights, loss

    return total_loss / len(dataloader)

# Bucle principal de entrenamiento
for epoch in range(num_epochs):
    logger.info(f"\nEpoca {epoch + 1}/{num_epochs}")
    train_loss = train_epoch(model, train_loader, optimizer, loss_fn, accum_steps)
    logger.info(f"Pérdida de entrenamiento: {train_loss:.4f}")

    # Guardar checkpoint por época
    checkpoint_path = os.path.join(checkpoint_dir, f"model_epoch_{epoch + 1}.pth")
    save_checkpoint(model, checkpoint_path)

logger.info("Entrenamiento completado!")
# Guardar modelo final
final_checkpoint_path = os.path.join(checkpoint_dir, "hierarchical_seq2seq_final.pth")
save_checkpoint(model, final_checkpoint_path)
