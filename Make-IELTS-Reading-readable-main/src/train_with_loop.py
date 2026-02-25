import os
import sys
import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW
from transformers import get_scheduler, DataCollatorForSeq2Seq
from tqdm.auto import tqdm
import wandb  

from dataset import IELTSTranslationDataset
from model import load_model
from utils import compute_metrics

# Cấu hình encoding UTF-8 cho console
try:
    if sys.stdout.encoding != 'utf-8':
        sys.stdout.reconfigure(encoding='utf-8')
        sys.stderr.reconfigure(encoding='utf-8')
except (AttributeError, TypeError):
    # Jupyter notebook hoặc môi trường không hỗ trợ reconfigure
    pass

# Cấu hình một số tham số của model
MODEL_CHECKPOINT = "Helsinki-NLP/opus-mt-en-vi"
MAX_LENGTH = 128
BATCH_SIZE = 8
GRADIENT_ACCUMULATION_STEPS = 4 
LEARNING_RATE = 2e-5
NUM_EPOCHS = 3
SAMPLE_SIZE = 50000 

def train_manual_with_wandb():
    # Thiết lập device
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    print(f"Device: {device}")

    wandb.init(
        project="IELTS_Translator_EnVi",
        name="manual_run_base",
        config={
            "model": MODEL_CHECKPOINT,
            "batch_size": BATCH_SIZE,
            "learning_rate": LEARNING_RATE,
            "epochs": NUM_EPOCHS,
            "max_length": MAX_LENGTH
        }
    )

    # --- RESUME FROM CHECKPOINT ---
    start_epoch = 0
    checkpoint_dir = None
    existing_checkpoints = [
        d for d in os.listdir(".")
        if d.startswith("checkpoint_epoch_") and os.path.isdir(d)
    ]
    if existing_checkpoints:
        checkpoint_dir = max(existing_checkpoints, key=lambda x: int(x.split("_")[-1]))
        start_epoch = int(checkpoint_dir.split("_")[-1]) + 1
        print(f"Resuming from {checkpoint_dir}, starting at epoch {start_epoch}")

    # Load Data & Model
    dataset_handler = IELTSTranslationDataset(MODEL_CHECKPOINT, MAX_LENGTH, SAMPLE_SIZE)
    tokenized_datasets, tokenizer = dataset_handler.load_data()
    model_to_load = checkpoint_dir if checkpoint_dir else MODEL_CHECKPOINT
    model = load_model(model_to_load, device)

    # DataLoader
    data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)
    train_dataloader = DataLoader(tokenized_datasets["train"], shuffle=True, batch_size=BATCH_SIZE, collate_fn=data_collator)
    eval_dataloader = DataLoader(tokenized_datasets["validation"], batch_size=BATCH_SIZE, collate_fn=data_collator)

    # Optimizer & Scheduler
    optimizer = AdamW(model.parameters(), lr=LEARNING_RATE)
    num_update_steps_per_epoch = len(train_dataloader) // GRADIENT_ACCUMULATION_STEPS
    max_train_steps = NUM_EPOCHS * num_update_steps_per_epoch
    
    lr_scheduler = get_scheduler(
        "linear", optimizer=optimizer, num_warmup_steps=0, num_training_steps=max_train_steps
    )

    # Restore optimizer & scheduler states nếu resume
    if checkpoint_dir:
        opt_path = os.path.join(checkpoint_dir, "optimizer.pt")
        sched_path = os.path.join(checkpoint_dir, "scheduler.pt")
        if os.path.exists(opt_path):
            optimizer.load_state_dict(torch.load(opt_path, map_location=device))
            print("Loaded optimizer state")
        if os.path.exists(sched_path):
            lr_scheduler.load_state_dict(torch.load(sched_path))
            print("Loaded scheduler state")

    # --- TRAINING LOOP ---
    print("Bắt đầu Training...")
    global_step = start_epoch * num_update_steps_per_epoch

    for epoch in range(start_epoch, NUM_EPOCHS):
        model.train()
        total_train_loss = 0
        progress_bar = tqdm(range(num_update_steps_per_epoch), desc=f"Epoch {epoch+1}")
        
        for step, batch in enumerate(train_dataloader):
            batch = {k: v.to(device) for k, v in batch.items()}
            
            outputs = model(**batch)
            loss = outputs.loss / GRADIENT_ACCUMULATION_STEPS
            loss.backward()
            
            if (step + 1) % GRADIENT_ACCUMULATION_STEPS == 0:
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()
                progress_bar.update(1)
                global_step += 1
                
                # Log Train Loss theo từng step
                # wandb.log nhận vào một dictionary
                wandb.log({
                    "train_loss": loss.item() * GRADIENT_ACCUMULATION_STEPS, 
                    "learning_rate": optimizer.param_groups[0]['lr'],
                    "epoch": epoch
                })

        # --- EVALUATION LOOP ---
        print("Evaluating...")
        model.eval()
        all_preds = []
        all_labels = []
        
        wandb_table = wandb.Table(columns=["Source (En)", "Target (Vi)", "Prediction (Vi)"])
        
        examples_to_log = [] 

        for i, batch in enumerate(tqdm(eval_dataloader, desc="Evaluating")):
            batch = {k: v.to(device) for k, v in batch.items()}
            with torch.no_grad():
                generated_tokens = model.generate(
                    batch["input_ids"], 
                    attention_mask=batch["attention_mask"],
                    max_length=128,
                    num_beams=4
                )
            
            decoded_preds = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)
            decoded_labels = tokenizer.batch_decode(
                torch.where(batch["labels"] != -100, batch["labels"], tokenizer.pad_token_id), 
                skip_special_tokens=True
            )
            decoded_inputs = tokenizer.batch_decode(batch["input_ids"], skip_special_tokens=True)

            all_preds.extend(decoded_preds)
            all_labels.extend(decoded_labels)
            
            # 5 mẫu đầu tiên của batch đầu tiên vào bảng wandb
            if i == 0:
                for src, trg, pred in zip(decoded_inputs[:5], decoded_labels[:5], decoded_preds[:5]):
                    wandb_table.add_data(src, trg, pred)

        # Tính metrics (all_preds/all_labels đã là decoded strings)
        metrics = compute_metrics((all_preds, all_labels), tokenizer, already_decoded=True)
        print(f"Epoch {epoch+1} | BLEU: {metrics['bleu']:.2f} | chrF: {metrics['chrf']:.2f} | TER: {metrics['ter']:.2f}")

        wandb.log({
            "eval/bleu": metrics['bleu'],
            "eval/chrf": metrics['chrf'],
            "eval/ter": metrics['ter'],
            "eval/gen_len": metrics['gen_len'],
            "examples": wandb_table
        })
        
        # Save checkpoint
        model.save_pretrained(f"checkpoint_epoch_{epoch}")
        tokenizer.save_pretrained(f"checkpoint_epoch_{epoch}")
        torch.save(optimizer.state_dict(), f"checkpoint_epoch_{epoch}/optimizer.pt")
        torch.save(lr_scheduler.state_dict(), f"checkpoint_epoch_{epoch}/scheduler.pt")
        print(f"Checkpoint saved: checkpoint_epoch_{epoch}")

    wandb.finish()

if __name__ == "__main__":
    train_manual_with_wandb()