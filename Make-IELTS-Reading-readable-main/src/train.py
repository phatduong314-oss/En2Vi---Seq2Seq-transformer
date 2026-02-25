import os
import sys
import torch
from transformers import Seq2SeqTrainingArguments, Seq2SeqTrainer, DataCollatorForSeq2Seq
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

MODEL_CHECKPOINT = "Helsinki-NLP/opus-mt-en-vi"
MAX_LENGTH = 128
NUM_BEAMS = 4 
BATCH_SIZE = 8 
GRADIENT_ACCUMULATION_STEPS = 4 
LEARNING_RATE = 2e-5
NUM_EPOCHS = 3
SAMPLE_SIZE = 50000 # train với 50_000 câu (dataset 4 triệu câu)

def train():
    # Chuẩn bị device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Đang sử dụng thiết bị: {device}")

    # Load data
    dataset_handler = IELTSTranslationDataset(MODEL_CHECKPOINT, MAX_LENGTH, SAMPLE_SIZE)
    tokenized_datasets, tokenizer = dataset_handler.load_data()

    # Load Model
    model = load_model(MODEL_CHECKPOINT, device)

    # Data Collator gom data thành 1 batch, tự động padding
    data_collator = DataCollatorForSeq2Seq(tokenizer, model=model, label_pad_token_id = -100)

    # Cấu hình tham số Training 
    args = Seq2SeqTrainingArguments(
        output_dir="./models/ielts-translator-v1",
        eval_strategy="epoch", # Đánh giá sau mỗi epoch
        save_strategy="epoch",       # Lưu model sau mỗi epoch 
        learning_rate=LEARNING_RATE,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        weight_decay=0.01,
        save_total_limit=2,          # Chỉ giữ lại 2 checkpoint 
        num_train_epochs=NUM_EPOCHS,
        predict_with_generate=True,  # Để model sinh văn bản khi đánh giá thay vì chỉ tính loss
        fp16=torch.cuda.is_available(), # Dùng mixed precision nếu có GPU 
        gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
        
        # Cấu hình Beam Search cho quá trình Validation
        generation_config=model.generation_config, 
    )
    
    # Dùng Beam Search khi generate 
    args.generation_max_length = MAX_LENGTH
    args.generation_num_beams = NUM_BEAMS 

    # Khởi tạo Trainer
    trainer = Seq2SeqTrainer(
        model=model,
        args=args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["validation"],
        data_collator=data_collator,
        compute_metrics=lambda p: compute_metrics(p, tokenizer),
    )
    
    trainer.train()

    trainer.save_model("./models/final_model")
    tokenizer.save_pretrained("my_ielts_model")
    print("Model đã được lưu tại ./models/final_model")

if __name__ == "__main__":
    train()