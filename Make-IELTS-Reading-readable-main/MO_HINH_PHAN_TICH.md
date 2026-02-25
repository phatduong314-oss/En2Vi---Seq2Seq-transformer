# PHÂN TÍCH MÔ HÌNH HELSINKI-NLP/OPUS-MT-EN-VI

## 1. TỔNG QUAN KIẾN TRÚC

**Loại mô hình**: MarianMT (Transformer Seq2Seq)
- Kiến trúc: Encoder-Decoder (Transformer)
- Nhiệm vụ: Dịch máy English → Vietnamese
- Base model: Helsinki-NLP/opus-mt-en-vi

---

## 2. CẤU HÌNH CHI TIẾT

### 2.1 ENCODER (Mã hóa văn bản tiếng Anh)
```
Số layers: 6
Hidden size (d_model): 512
Số attention heads: 8
FFN dimension: 2048
Dropout: 0.1
Attention dropout: 0.0
```

### 2.2 DECODER (Giải mã sang tiếng Việt)
```
Số layers: 6
Hidden size (d_model): 512
Số attention heads: 8
FFN dimension: 2048
Dropout: 0.1
Attention dropout: 0.0
```

### 2.3 VOCABULARY
```
Vocab size: 53,685 tokens
Max position embeddings: 512 positions
Activation function: swish
```

---

## 3. LUỒNG DỮ LIỆU QUA MÔ HÌNH (CHÚ Ý SIZE)

### **BƯỚC 1: TOKENIZATION (Input)**
```python
Input text: "The Titanic struck an iceberg"
          ↓
Tokenizer.encode()
          ↓
input_ids: [1234, 5678, 912, 3456, 2]  # Mỗi từ → số
          ↓
Shape: (batch_size, seq_length)
Ví dụ: (8, 128)  # 8 câu, mỗi câu max 128 tokens
```

### **BƯỚC 2: EMBEDDING LAYER**
```python
Input: (batch_size, seq_length) = (8, 128)
          ↓
Embedding lookup trong vocab_size=53,685
          ↓
Output: (batch_size, seq_length, d_model)
Shape: (8, 128, 512)  # 512 chiều vector cho mỗi token
```

### **BƯỚC 3: ENCODER (6 layers)**

#### Layer 1/6 - Multi-Head Self-Attention
```python
Input: (8, 128, 512)
          ↓
Split thành 8 heads: 512 / 8 = 64 dimensions/head
          ↓
Mỗi head:
  Q (Query):  (8, 128, 64)
  K (Key):    (8, 128, 64)
  V (Value):  (8, 128, 64)
          ↓
Attention Score = softmax(Q @ K^T / √64)
Shape: (8, 8, 128, 128)  # Attention matrix
          ↓
Output = Attention @ V
Per head: (8, 128, 64)
          ↓
Concat 8 heads → (8, 128, 512)
          ↓
Linear projection → (8, 128, 512)
```

#### Feed-Forward Network (FFN)
```python
Input: (8, 128, 512)
          ↓
Linear 1: (8, 128, 512) → (8, 128, 2048)  # Mở rộng
          ↓
Swish Activation
          ↓
Dropout (p=0.1)
          ↓
Linear 2: (8, 128, 2048) → (8, 128, 512)  # Thu hẹp
          ↓
Residual + LayerNorm
          ↓
Output: (8, 128, 512)
```

**Lặp lại 6 lần** → Encoder final output: **(8, 128, 512)**

---

### **BƯỚC 4: DECODER (6 layers)**

#### Layer 1/6 - Masked Self-Attention (Decoder tự chú ý vào output đã sinh)
```python
Input: (8, target_len, 512)  # target_len = độ dài câu tiếng Việt
          ↓
Masked Multi-Head Attention (8 heads)
  - Chỉ nhìn tokens phía trước (causal mask)
  - Tránh nhìn tương lai khi dịch
          ↓
Output: (8, target_len, 512)
```

#### Cross-Attention (Decoder nhìn vào Encoder)
```python
Query (từ Decoder): (8, target_len, 512)
Key (từ Encoder):   (8, 128, 512)
Value (từ Encoder): (8, 128, 512)
          ↓
Split thành 8 heads: 512/8 = 64 per head
          ↓
Attention Score = softmax(Q @ K^T / √64)
Shape: (8, 8, target_len, 128)  
# target_len tokens Việt nhìn vào 128 tokens Anh
          ↓
Output = Attention @ V
          ↓
Concat heads → (8, target_len, 512)
```

#### Feed-Forward Network
```python
(8, target_len, 512) → (8, target_len, 2048) → (8, target_len, 512)
```

**Lặp lại 6 lần** → Decoder final output: **(8, target_len, 512)**

---

### **BƯỚC 5: OUTPUT PROJECTION**
```python
Decoder output: (8, target_len, 512)
          ↓
Linear (Language Model Head):
  (8, target_len, 512) → (8, target_len, 53685)
          ↓
Softmax over vocab_size
          ↓
Probabilities: (8, target_len, 53685)
# Mỗi vị trí có xác suất cho 53,685 từ tiếng Việt
          ↓
Argmax → Predicted token IDs
Shape: (8, target_len)
```

### **BƯỚC 6: DETOKENIZATION**
```python
Token IDs: [45, 234, 789, 12, 3]
          ↓
Tokenizer.decode()
          ↓
"Tàu Titanic đâm vào tảng băng"
```

---

## 4. CÁC ĐIỂM ĐẶC BIỆT CẦN LƯU Ý

### 4.1 **ATTENTION MECHANISM**

#### Self-Attention (Encoder)
- **Vai trò**: Mỗi từ tiếng Anh nhìn toàn bộ câu để hiểu ngữ cảnh
- **Attention Matrix**: (seq_len, seq_len) = (128, 128)
- **Không bị mask**: Nhìn cả trước và sau
- **Ví dụ**: "struck" sẽ chú ý đến "Titanic" và "iceberg" để hiểu đúng nghĩa

#### Masked Self-Attention (Decoder)
- **Vai trò**: Tự hồi quy, chỉ nhìn tokens đã sinh
- **Causal mask**: Vị trí i chỉ nhìn được vị trí ≤ i
- **Ngăn "nhìn tương lai"** khi training

#### Cross-Attention (Encoder-Decoder)
- **Vai trò**: Decoder nhìn vào Encoder để "đọc" câu gốc
- **Attention weights**: Cho biết từ Việt nào tương ứng với từ Anh nào
- **Dynamic**: Mỗi từ Việt chú ý khác nhau vào câu Anh

### 4.2 **KỸ THUẬT ATTENTION**

```python
# Scaled Dot-Product Attention
Attention(Q, K, V) = softmax(Q @ K^T / √d_k) @ V

Trong đó:
- d_k = 64 (dimension per head)
- √d_k = 8 để scale, tránh gradient vanishing
- Softmax → tổng weights = 1
```

### 4.3 **MULTI-HEAD ATTENTION**

```
8 heads × 64 dims = 512 total dims

Head 1: Học cú pháp (syntax)
Head 2: Học ngữ nghĩa (semantics)
Head 3: Học vị trí (position)
Head 4: Học từ vựng domain
...

→ Mỗi head học khía cạnh khác nhau của ngôn ngữ
```

### 4.4 **POSITION EMBEDDINGS**

```python
Max position: 512 tokens
→ Câu dài hơn 512 tokens sẽ bị truncate

# Trong code:
max_length = 128  # Giới hạn để tiết kiệm bộ nhớ
```

### 4.5 **FEED-FORWARD NETWORK**

```python
512 → 2048 → 512

Tại sao mở rộng 
- Tăng capacity học
- 2048 neurons học features phức tạp
- Swish activation: x * sigmoid(x) - smooth hơn ReLU
```

---

## 5. KỸ THUẬT TRAINING (Theo code)

### 5.1 **Hyperparameters**
```python
MAX_LENGTH = 128              # Giới hạn độ dài câu
BATCH_SIZE = 8                # Số câu/batch
GRADIENT_ACCUMULATION = 4     # Tích lũy 4 batches → effective batch = 32
LEARNING_RATE = 2e-5          # Learning rate nhỏ
NUM_EPOCHS = 3
SAMPLE_SIZE = 50,000          # Train trên subset
```

### 5.2 **Data Processing**
```python
# Tokenize input (English)
inputs = tokenizer(
    text, 
    max_length=128, 
    truncation=True,     # Cắt nếu > 128
    padding="max_length" # Pad đến 128
)
→ Shape: (batch, 128)

# Tokenize target (Vietnamese)
labels = tokenizer(
    text_target=targets,
    max_length=128,
    truncation=True,
    padding="max_length"
)
→ Shape: (batch, 128)
```

### 5.3 **Inference (Beam Search)**
```python
num_beams = 4               # Duy trì 4 hypotheses tốt nhất
early_stopping = True       # Dừng khi tìm được EOS
no_repeat_ngram_size = 2    # Tránh lặp 2-gram
length_penalty = 1.0        # Không ưu tiên câu dài/ngắn

# Beam Search flow:
Step 1: Sinh 4 từ có xác suất cao nhất
Step 2: Từ mỗi từ, sinh tiếp 4 từ → 16 sequences
Step 3: Giữ 4 sequences có tổng log-prob cao nhất
...
→ Output: Câu có likelihood cao nhất
```

---

## 6. KÍCH THƯỚC BỘ NHỚ

### 6.1 **Model Parameters**
```
Tổng tham số: ~74 million params
- Embeddings: 53,685 × 512 = ~27M
- Encoder (6 layers): ~18M
- Decoder (6 layers): ~18M
- LM Head: 512 × 53,685 = ~27M

GPU Memory (inference):
- Model: ~300MB (float32) hoặc ~150MB (float16)
- Activations: Phụ thuộc batch_size và seq_length
```

### 6.2 **Activation Memory (Forward Pass)**
```python
Với batch_size=8, seq_len=128:

Encoder:
- Embeddings: 8 × 128 × 512 × 4 bytes = 2.1 MB
- Attention: 8 × 8 × 128 × 128 × 4 bytes = 4.2 MB (per layer) × 6 = 25 MB
- FFN: 8 × 128 × 2048 × 4 bytes = 8.4 MB (per layer) × 6 = 50 MB

Decoder: Tương tự
→ Total activations: ~200-300 MB
→ Training cần gradient → × 2 = ~600 MB
```

---

## 7. SO SÁNH VỚI CÁC MÔ HÌNH KHÁC

| Đặc điểm | Helsinki-NLP | BERT-base | GPT-2 | T5-small |
|----------|--------------|-----------|-------|----------|
| **Kiến trúc** | Encoder-Decoder | Encoder-only | Decoder-only | Encoder-Decoder |
| **Layers** | 6 + 6 | 12 | 12 | 6 + 6 |
| **Hidden size** | 512 | 768 | 768 | 512 |
| **Attention heads** | 8 | 12 | 12 | 8 |
| **Parameters** | 74M | 110M | 124M | 60M |
| **Max length** | 512 | 512 | 1024 | 512 |
| **Vocab size** | 53K | 30K | 50K | 32K |

**Ưu điểm Helsinki-NLP/opus-mt**:
- ✅ Nhỏ gọn, nhanh
- ✅ Chuyên biệt cho translation
- ✅ Pre-trained sẵn trên EN-VI corpus lớn

**Nhược điểm**:
- ❌ Max length = 512 (không dịch được văn bản quá dài)
- ❌ Vocabulary size nhỏ hơn → có thể không biết từ hiếm

---

## 8. TÓM TẮT CHO IELTS READING

Khi dịch bài IELTS Reading dài (1000+ words):

```python
# ❌ KHÔNG THỂ:
Input toàn bộ 1000 words vào 1 lần (vượt max_length=512)

# ✅ GIẢI PHÁP:
1. Chia nhỏ thành từng đoạn < 128 tokens
2. Dịch từng đoạn
3. Ghép lại

# Hoặc tăng max_length:
MAX_LENGTH = 256  # Tăng lên nhưng tốn GPU memory hơn
```

**Lưu ý quan trọng**:
- Attention complexity: O(n²) với n = sequence length
- Length 256 → Attention matrix gấp 4 lần so với 128
- GPU memory tăng đáng kể

---

## 9. DEBUGGING SIZE

Khi gặp lỗi shape mismatch, check:

```python
print(f"Input IDs shape: {input_ids.shape}")
# Expected: (batch_size, max_length)

print(f"Attention mask shape: {attention_mask.shape}")
# Expected: (batch_size, max_length)

print(f"Labels shape: {labels.shape}")
# Expected: (batch_size, max_length)

print(f"Model output shape: {outputs.logits.shape}")
# Expected: (batch_size, max_length, vocab_size)
```

---

**File này tóm tắt toàn bộ luồng xử lý từ text → tokens → embeddings → attention → output → text, với kích thước cụ thể ở mỗi bước!**
