import os
import sys
from datasets import load_dataset, DatasetDict
from transformers import AutoTokenizer

# Cấu hình encoding UTF-8 cho console
try:
    if sys.stdout.encoding != 'utf-8':
        sys.stdout.reconfigure(encoding='utf-8')
        sys.stderr.reconfigure(encoding='utf-8')
except (AttributeError, TypeError):
    # Jupyter notebook hoặc môi trường không hỗ trợ reconfigure
    pass

class IELTSTranslationDataset:
    def __init__(self, model_checkpoint, max_length=128, sample_size=None):
        self.model_checkpoint = model_checkpoint
        self.max_length = max_length
        self.sample_size = sample_size
        self.tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
        
        self.source_lang = "en" 
        self.target_lang = "vi"

    def preprocess_function(self, examples):

        raw_inputs = examples[self.source_lang]
        raw_targets = examples[self.target_lang]

        # Chuyển câu thành list và bỏ các dòng trống
        inputs = [str(x) if x is not None else "" for x in raw_inputs]
        targets = [str(x) if x is not None else "" for x in raw_targets]

        
        # tokenize input tiếng anh
        model_inputs = self.tokenizer(
            inputs, 
            max_length=self.max_length, 
            truncation=True
            #padding="max_length"
        )

        # tokenize ouput tiếng việt
        labels = self.tokenizer(
            text_target=targets,
            max_length=self.max_length, 
            truncation=True
            #padding="max_length"
        )
        ## model_inputs đang chứa list các list độ dài không đều nhau.
        # DataCollatorForSeq2Seq sẽ tự động padding chúng thành hình chữ nhật khi tạo batch. Tuy nhiên, ta cần giúp DataCollator biết đâu là labels.
        model_inputs["labels"] = labels["input_ids"]
        return model_inputs
    
    def load_data(self):
        raw_dataset = load_dataset("hiimbach/mtet", split="train")

        # Chọn ra sample_size câu để train (50_000)
        raw_dataset = raw_dataset.shuffle(seed=2016).select(range(self.sample_size))

        # Vì dữ liệu load về chỉ lấy tập train (Có thể tham khảo notebook để xem cấu trúc của dữ liệu gốc) nên cần chia lại thành tập train, validation và test
        # Lấy tỉ lệ 80/10/10
        train_test_split = raw_dataset.train_test_split(test_size=0.2, seed=2016)
        test_valid_split = train_test_split['test'].train_test_split(test_size=0.5, seed=2016)

        dataset = DatasetDict({
            'train': train_test_split['train'],
            'validation': test_valid_split['train'],
            'test': test_valid_split['test']
        })

        # Sau khi chia dữ liệu xong ta bắt đầu tokenize dữ liệu và dùng hàm map để ánh xạ cho cả tập dữ liệu
        tokenized_datasets = dataset.map(
            self.preprocess_function, 
            batched=True,
            remove_columns=dataset["train"].column_names 
        )
        
        return tokenized_datasets, self.tokenizer

if __name__ == "__main__":
    # Kiểm tra nhanh một chút
    dataset_handler = IELTSTranslationDataset("Helsinki-NLP/opus-mt-en-vi", sample_size=100)
    data, tokenizer = dataset_handler.load_data()
    print(data['train'][0])






