# En2Vi - Seq2Seq transformer

Fine-tune model MarianMT (`Helsinki-NLP/opus-mt-en-vi`) trên 50.000 cặp câu song ngữ Anh-Việt để dịch bài đọc IELTS sang tiếng Việt.

---

## Mục lục

- [Tại sao lại có project này?](#tại-sao-lại-có-project-này)
- [Dữ liệu](#dữ-liệu)
- [Kiến trúc model: MarianMT](#kiến-trúc-model-marianmt)
  - [Nhìn chung](#nhìn-chung)
  - [Embedding](#embedding)
  - [Encoder](#encoder)
  - [Attention](#attention)
  - [Multi-Head Attention](#multi-head-attention)
  - [Feed-Forward Network](#feed-forward-network)
  - [Decoder](#decoder)
- [Tổng hợp kích thước](#tổng-hợp-kích-thước)
- [Training](#training)
- [Metrics](#metrics)
- [Cấu trúc project](#cấu-trúc-project)
- [Cách chạy](#cách-chạy)

---

## Tại sao lại có project này?

Ai thi IELTS cũng biết cảm giác mở đề Reading ra, thấy bài 800 từ về "the sinking of the Titanic" hay "coral reef ecosystems", chỉ muốn hiểu nhanh nội dung trước khi đi vào câu hỏi.

Project này dịch bài đọc IELTS từ tiếng Anh sang tiếng Việt, nhưng không dùng Google Translate mà tự fine-tune model. Mục đích chính là để hiểu rõ model dịch nó hoạt động ra sao từ bên trong.

---

## Dữ liệu

### Nguồn gốc

Dataset lấy từ [hiimbach/mtet](https://huggingface.co/datasets/hiimbach/mtet) trên Hugging Face, khoảng 4 triệu cặp câu song ngữ Anh-Việt, tổng hợp từ nhiều nguồn dịch thuật công khai.

### Cách xử lý

4 triệu câu thì GPU không kham nổi, mà cũng không cần cho mục đích học. Nên mình chỉ lấy 50.000 cặp câu (shuffle ngẫu nhiên, seed=2016), rồi chia:

- Train: 40.000 câu (80%)
- Validation: 5.000 câu (10%)
- Test: 5.000 câu (10%)

### Mỗi mẫu dữ liệu trông thế nào?

Mỗi mẫu là một cặp câu song ngữ, ví dụ:

| Trường | Nội dung |
|--------|----------|
| `en` (tiếng Anh) | "The Titanic struck an iceberg on its maiden voyage." |
| `vi` (tiếng Việt) | "Tàu Titanic đã va phải một tảng băng trôi trong chuyến đi đầu tiên." |

Sau khi tokenize, mỗi mẫu sẽ có 3 trường:

```
input_ids:      [1234, 5678, 912, 3456, 0]      ← câu tiếng Anh, dạng số
attention_mask: [1, 1, 1, 1, 1]                  ← "phần nào là chữ thật, phần nào là padding"
labels:         [8765, 4321, 567, 890, 0]        ← câu tiếng Việt (target), dạng số
```

### Phân bố độ dài câu

Đa số câu trong dataset có độ dài dưới 128 tokens. Đồ thị phân bố trông đại khái như này:

```
Số câu
  │
  │  ██
  │  ██ ██
  │  ██ ██ ██
  │  ██ ██ ██ ██
  │  ██ ██ ██ ██ ██
  │  ██ ██ ██ ██ ██ ██
  │  ██ ██ ██ ██ ██ ██ ██ ██
  └───────────────────────────── Số tokens
     10  20  40  60  80 100 120
```

Nên `max_length=128` là đủ cho hầu hết câu. Câu nào dài hơn thì bị cắt (truncation).

### Pipeline xử lý dữ liệu

```
┌─────────────────────┐
│  Dataset thô (text) │  "The Titanic struck an iceberg"
│  hiimbach/mtet      │  "Tàu Titanic đã va phải..."
└────────┬────────────┘
         │
         ▼
┌─────────────────────┐
│  Shuffle + Select   │  Lấy 50.000 cặp câu ngẫu nhiên
│  (seed=2016)        │
└────────┬────────────┘
         │
         ▼
┌─────────────────────┐
│  Train/Val/Test     │  80% / 10% / 10%
│  Split              │
└────────┬────────────┘
         │
         ▼
┌─────────────────────┐
│  Tokenize           │  Chữ → SentencePiece tokens → Số (IDs)
│  (max_length=128)   │  Truncate câu dài, KHÔNG padding ở đây
└────────┬────────────┘
         │
         ▼
┌─────────────────────┐
│  DataCollator       │  Gom thành batch, tự động padding
│  (dynamic padding)  │  mỗi batch có cùng chiều dài
└─────────────────────┘
```

---

## Kiến trúc model: MarianMT

### Nhìn chung

MarianMT là model dịch máy dùng kiến trúc Transformer (họ hàng với ChatGPT, nhưng nhỏ hơn nhiều). Gồm hai phần:

```
                    ┌──────────────────────────┐
  "The Titanic      │                          │     "Tàu Titanic
   struck an   ───▶ │   ENCODER    │  DECODER  │ ──▶  đã va phải
   iceberg"         │  (đọc hiểu)  │ (viết ra) │      một tảng
                    │              │           │      băng trôi"
                    └──────────────────────────┘
                          ▲              ▲
                    6 layers         6 layers
                    8 heads          8 heads
                    d_model=512      d_model=512
```

Các con số chính:

| Thông số | Giá trị | Ý nghĩa |
|----------|---------|---------|
| `vocab_size` | 53,685 | Tổng số từ model biết (cả Anh lẫn Việt) |
| `d_model` | 512 | Mỗi từ biểu diễn bằng vector 512 chiều |
| `encoder_layers` | 6 | 6 lớp xử lý bên Encoder |
| `decoder_layers` | 6 | 6 lớp xử lý bên Decoder |
| `attention_heads` | 8 | Mỗi lớp có 8 attention heads |
| `ffn_dim` | 2048 | Kích thước lớp feed-forward |
| `max_positions` | 512 | Câu dài tối đa 512 tokens |
| Tổng params | 154.6M | 154.6 triệu tham số |

### Embedding

Máy tính không đọc được chữ, nó chỉ hiểu số. Nên bước đầu tiên là biến mỗi từ thành một vector (mảng số).

```
"Titanic" ──▶ Token ID: 1234 ──▶ Embedding lookup ──▶ [0.23, -0.45, ..., 0.67]
                                                         └──── 512 số ────┘
```

Nhưng chưa đủ. Từ "Titanic" ở đầu câu và cuối câu nên có biểu diễn khác nhau vì ngữ cảnh khác. Nên model cộng thêm Positional Embedding để mã hóa vị trí của từ trong câu:

$$PE_{(pos, 2i)} = \sin\left(\frac{pos}{10000^{2i/512}}\right)$$

$$PE_{(pos, 2i+1)} = \cos\left(\frac{pos}{10000^{2i/512}}\right)$$

Trong đó:
- $pos$: vị trí từ trong câu (0, 1, 2, ...)
- $i$: chiều thứ mấy trong vector (0 → 255)

Dùng sin/cos vì chúng tạo ra pattern lặp lại ở các tần số khác nhau, giúp model phân biệt khoảng cách giữa các từ, kể cả với những câu dài hơn lúc train.

```
Final Embedding = Token Embedding × √d_model  +  Positional Embedding
                = Token Embedding × √512      +  PE
```

Nhân với $\sqrt{512} \approx 22.6$ để token embedding không bị lấn át bởi positional encoding.

Shape: `(batch_size, seq_len)` → `(batch_size, seq_len, 512)`. Ví dụ 8 câu, mỗi câu 128 từ thì ra `(8, 128, 512)`.

### Encoder

Encoder gồm 6 layers xếp chồng nhau. Mỗi layer làm 2 việc:
1. Self-Attention: mỗi từ nhìn toàn bộ câu để hiểu ngữ cảnh
2. Feed-Forward: xử lý thêm thông tin sau đó

```
Input (8, 128, 512)
    │
    ▼
┌─────────────────────────┐
│  Encoder Layer 1        │
│  ┌───────────────────┐  │
│  │  Self-Attention    │  │  ← Mỗi từ nhìn mọi từ khác
│  │  + LayerNorm       │  │
│  │  + Residual        │  │
│  ├───────────────────┤  │
│  │  Feed-Forward      │  │  ← 512 → 2048 → 512
│  │  + LayerNorm       │  │
│  │  + Residual        │  │
│  └───────────────────┘  │
└────────┬────────────────┘
         │ (8, 128, 512)   ← Shape KHÔNG đổi qua mỗi layer
         ▼
┌─────────────────────────┐
│  Encoder Layer 2        │
│  ... (giống hệt)       │
└────────┬────────────────┘
         │
         ▼
   ... (×6 layers) ...
         │
         ▼
    Output (8, 128, 512)   ← "Hiểu biết" về câu tiếng Anh
```

### Attention

Lấy ví dụ câu:

> *"The bank by the river was covered in grass."*

"bank" ở đây là bờ sông hay ngân hàng? Người đọc biết là bờ sông vì thấy "river" và "grass" xung quanh. Attention hoạt động y vậy: cho mỗi từ nhìn vào mọi từ khác trong câu để xác định ngữ cảnh.

Mỗi từ sẽ tạo ra 3 vector:

- Query (Q): tôi đang tìm thông tin gì?
- Key (K): tôi có thông tin gì để cung cấp?
- Value (V): nội dung thực sự của tôi

Công thức Attention:

$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right) V$$

Đi qua từng bước:

**Bước 1** - tính điểm liên quan giữa các từ:

$$\text{scores} = Q \times K^T$$

Nhân Q của từ này với K của mọi từ khác. Số càng lớn thì hai từ càng liên quan.

```
             The    bank    river   grass
  bank   [  0.1    0.3     0.8     0.6  ]
               ↑               ↑
          ít liên quan    rất liên quan
```

**Bước 2** - chia cho $\sqrt{d_k}$ (scaling):

$$\text{scaled\_scores} = \frac{scores}{\sqrt{64}} = \frac{scores}{8}$$

Khi $d_k$ lớn, tích vô hướng cũng lớn theo, softmax sẽ ra giá trị rất gần 0 hoặc 1, gradient gần 0, model không học được gì. Chia cho $\sqrt{d_k}$ để giữ giá trị ổn định.

**Bước 3** - softmax biến thành xác suất:

$$\text{weights} = \text{softmax}(\text{scaled\_scores})$$

Mỗi hàng tổng bằng 1.0, thể hiện phân bố sự chú ý:

```
              The    bank    river   grass
  bank   [  0.05    0.15    0.45    0.35  ]
                              ↑       ↑
                         45% chú ý  35% chú ý
```

**Bước 4** - lấy thông tin theo trọng số:

$$\text{output} = \text{weights} \times V$$

0.45 ở cột "river" nghĩa là lấy 45% nội dung (Value) của từ "river" trộn vào biểu diễn mới của "bank". Sau bước này, "bank" mang theo ngữ cảnh sông nước nên model hiểu đó là bờ sông, không phải ngân hàng.

### Multi-Head Attention

Một head attention chỉ nhìn được một kiểu quan hệ. Ngôn ngữ thì phức tạp hơn thế, ví dụ:
- Head 1 có thể học quan hệ cú pháp (chủ-vị)
- Head 2 học quan hệ ngữ nghĩa (từ đồng nghĩa)
- Head 3 học quan hệ vị trí (từ gần nhau)

Nên thay vì 1 attention cỡ 512, model tách thành 8 heads, mỗi head cỡ 64:

$$d_k = d_v = \frac{d_{model}}{h} = \frac{512}{8} = 64$$

```
Input (batch=8, seq=128, dim=512)
        │
        ├── q_proj(x) ──▶ Q (8, 128, 512) ──▶ reshape ──▶ (8, 8, 128, 64)
        ├── k_proj(x) ──▶ K (8, 128, 512) ──▶ reshape ──▶ (8, 8, 128, 64)
        └── v_proj(x) ──▶ V (8, 128, 512) ──▶ reshape ──▶ (8, 8, 128, 64)
                                                              │
                                              8 heads chạy song song
                                                              │
                                              ┌───────────────┤
                                              ▼               ▼
                                          Head 1           Head 8
                                        (8,128,64)       (8,128,64)
                                              │               │
                                              └───────┬───────┘
                                                      ▼
                                              Concat (8, 128, 512)
                                                      │
                                                      ▼
                                              out_proj(x) ──▶ (8, 128, 512)
```

Công thức chính thức:

$$\text{MultiHead}(Q, K, V) = \text{Concat}(\text{head}_1, ..., \text{head}_8) \cdot W^O$$

$$\text{head}_i = \text{Attention}(QW_i^Q, KW_i^K, VW_i^V)$$

Trong đó $W_i^Q, W_i^K, W_i^V \in \mathbb{R}^{512 \times 64}$ và $W^O \in \mathbb{R}^{512 \times 512}$.

Tham số Attention mỗi layer:

| Projection | Shape | Params |
|-----------|-------|--------|
| `q_proj` | 512 × 512 + 512 | 262,656 |
| `k_proj` | 512 × 512 + 512 | 262,656 |
| `v_proj` | 512 × 512 + 512 | 262,656 |
| `out_proj` | 512 × 512 + 512 | 262,656 |
| Tổng | | 1,050,624 |

### Feed-Forward Network

Sau Attention, model biết từ nào liên quan đến từ nào. FFN làm thêm một bước biến đổi phi tuyến để trích xuất features phức tạp hơn:

$$\text{FFN}(x) = W_2 \cdot \text{SiLU}(W_1 \cdot x + b_1) + b_2$$

SiLU (hay Swish) là hàm kích hoạt:

$$\text{SiLU}(x) = x \cdot \sigma(x) = \frac{x}{1 + e^{-x}}$$

```
Input (8, 128, 512)
    │
    ▼
 fc1: Linear(512 → 2048)      ← Mở rộng 4 lần: "nhìn rộng hơn"
    │
    ▼
 SiLU activation               ← Phi tuyến
    │
    ▼
 fc2: Linear(2048 → 512)      ← Nén lại về kích thước gốc
    │
    ▼
Output (8, 128, 512)
```

Mở rộng lên 2048 rồi nén về 512 giống kiểu model cần không gian rộng hơn để xử lý thông tin, rồi tổng hợp lại. Trong không gian lớn hơn, nó biểu diễn được những quan hệ phức tạp mà 512 chiều không đủ.

Tham số FFN mỗi layer:

| Layer | Shape | Params |
|-------|-------|--------|
| `fc1` | 512 × 2048 + 2048 | 1,050,624 |
| `fc2` | 2048 × 512 + 512 | 1,049,088 |
| Tổng | | 2,099,712 |

### Residual Connection & Layer Norm

Một chi tiết quan trọng: residual connections (kết nối tắt). Thay vì `output = SubLayer(x)`, model làm:

```
output = x + SubLayer(LayerNorm(x))
```

Với 6 layers chồng lên nhau, gradient phải đi qua rất nhiều phép biến đổi. Residual connection cho gradient một đường tắt để chảy ngược, giúp model hội tụ nhanh và ổn định hơn.

### Decoder

Decoder giống Encoder nhưng phức tạp hơn. Mỗi layer có 3 sub-layers thay vì 2:

```
┌────────────────────────────────────┐
│  Decoder Layer                     │
│  ┌──────────────────────────────┐  │
│  │ 1. Masked Self-Attention     │  │  ← Từ chỉ nhìn được những từ
│  │    + LayerNorm + Residual    │  │     đã sinh TRƯỚC nó
│  ├──────────────────────────────┤  │
│  │ 2. Cross-Attention           │  │  ← Nhìn sang output Encoder
│  │    Q từ Decoder              │  │     (biểu diễn câu tiếng Anh)
│  │    K, V từ Encoder           │  │
│  │    + LayerNorm + Residual    │  │
│  ├──────────────────────────────┤  │
│  │ 3. Feed-Forward              │  │  ← Giống Encoder
│  │    + LayerNorm + Residual    │  │
│  └──────────────────────────────┘  │
└────────────────────────────────────┘
```

Masked Self-Attention: khi sinh từ thứ 3 ("va"), model không được nhìn trước từ thứ 4 ("phải"). Mask che đi tương lai:

```
             Tàu   Titanic   đã    va    phải
Tàu       [  ✓       ✗       ✗     ✗      ✗  ]
Titanic   [  ✓       ✓       ✗     ✗      ✗  ]
đã        [  ✓       ✓       ✓     ✗      ✗  ]
va        [  ✓       ✓       ✓     ✓      ✗  ]
phải      [  ✓       ✓       ✓     ✓      ✓  ]

✓ = được nhìn, ✗ = bị che (set = -∞ trước softmax)
```

Cross-Attention là cầu nối giữa hai ngôn ngữ. Khi Decoder đang sinh từ "bờ" trong tiếng Việt, nó gửi Query sang Encoder để tìm từ tiếng Anh nào liên quan nhất. Trong trường hợp này "river" và "bank" sẽ có score cao nhất.

---

## Tổng hợp kích thước

### Luồng dữ liệu end-to-end

```
"The Titanic struck an iceberg"
         │
         ▼
┌──────────────────────┐
│  Tokenizer           │  Shape: (batch=8, seq=128)
│  (SentencePiece)     │  Loại: integer IDs
└────────┬─────────────┘
         │
         ▼
┌──────────────────────┐
│  Embedding           │  (8, 128) → (8, 128, 512)
│  + Positional        │  53,685 × 512 = 27.5M params
└────────┬─────────────┘
         │
         ▼
┌──────────────────────┐
│  Encoder ×6          │  (8, 128, 512) → (8, 128, 512)
│  Self-Attn + FFN     │  Mỗi layer: 3.15M params
└────────┬─────────────┘
         │
         ├──────────────────── K, V gửi sang Decoder
         │
         ▼
┌──────────────────────┐
│  Decoder ×6          │  (8, target_len, 512) → (8, target_len, 512)
│  Self + Cross + FFN  │  Mỗi layer: 4.2M params
└────────┬─────────────┘
         │
         ▼
┌──────────────────────┐
│  LM Head (Linear)    │  (8, target_len, 512) → (8, target_len, 53685)
│  + Softmax           │  Chọn từ có xác suất cao nhất
└────────┬─────────────┘
         │
         ▼
"Tàu Titanic đã va phải một tảng băng trôi"
```

### Đếm tham số

| Thành phần | Params | Ghi chú |
|-----------|--------|---------|
| Shared Embedding | 27,486,720 | 53,685 × 512 (dùng chung Enc/Dec/LM Head) |
| Positional Embedding | 262,144 | 512 × 512 (cố định, sin/cos) |
| Encoder × 6 layers | 18,914,304 | 6 × 3,152,384 |
| Decoder × 6 layers | 25,224,192 | 6 × 4,204,032 |
| LM Head | 27,486,720 | Tied với Embedding (không thêm param) |
| Tổng | 154,609,664 | ~154.6 triệu |

Về tied embeddings: Embedding đầu vào và Linear cuối cùng (LM Head) dùng chung trọng số. Tiết kiệm ~27M params, và cũng hợp lý vì ánh xạ chữ→vector và vector→chữ về bản chất là quá trình nghịch đảo nhau.

---

## Training

### Hyperparameters

| Tham số | Giá trị | Tại sao? |
|---------|---------|----------|
| Learning rate | 2e-5 | Fine-tune nên lr nhỏ, tránh phá model gốc |
| Batch size | 8 | GPU memory hạn chế |
| Gradient accumulation | 4 steps | Effective batch = 8 × 4 = 32 |
| Epochs | 3 | Đủ converge cho fine-tune |
| Max length | 128 tokens | Đủ cho hầu hết câu IELTS |
| Optimizer | AdamW | Standard cho Transformer |
| Weight decay | 0.01 | Regularization nhẹ |
| FP16 | Có (nếu GPU) | Tiết kiệm memory, tăng tốc ~2x |
| Beam search | 4 beams | Khi eval, sinh text chất lượng hơn greedy |

### Pipeline training

```
┌─────────────────────────────────────────────────────┐
│                    TRAINING LOOP                     │
│                                                     │
│  for epoch in 1..3:                                 │
│    for batch in train_loader:                       │
│      ┌─────────────────────────┐                    │
│      │ Forward pass            │                    │
│      │ input_ids → Encoder     │                    │
│      │ → Decoder → logits     │                    │
│      │ → CrossEntropyLoss     │                    │
│      └──────────┬──────────────┘                    │
│                 │                                    │
│      ┌──────────▼──────────────┐                    │
│      │ Backward pass           │                    │
│      │ loss.backward()         │                    │
│      │ (tích lũy 4 steps)     │                    │
│      └──────────┬──────────────┘                    │
│                 │                                    │
│      ┌──────────▼──────────────┐                    │
│      │ Optimizer step          │                    │
│      │ AdamW + LR scheduler   │                    │
│      └─────────────────────────┘                    │
│                                                     │
│    ── Evaluation trên Validation set ──            │
│    Beam search → decode → tính BLEU, chrF, TER     │
└─────────────────────────────────────────────────────┘
```

---

## Metrics

3 metrics dùng để đánh giá:

| Metric | Ý nghĩa | Cao/Thấp tốt? |
|--------|---------|---------------|
| BLEU | So sánh n-gram giữa bản dịch và đáp án | Cao = tốt (0-100) |
| chrF | So sánh ở mức ký tự (phù hợp cho tiếng Việt có dấu) | Cao = tốt (0-100) |
| TER | Số lần edit để biến bản dịch thành đáp án | Thấp = tốt |

---

## Cấu trúc project

```
Make-IELTS-Reading-readable-main/
├── src/
│   ├── model.py          # Load model MarianMT, in summary
│   ├── dataset.py        # Load & tokenize dataset
│   ├── train.py          # Training bằng HuggingFace Trainer
│   ├── train_with_loop.py # Training bằng custom loop + WandB logging
│   └── utils.py          # Compute metrics (BLEU, chrF, TER)
├── notebooks/
│   ├── dataset.ipynb     # Khám phá dữ liệu
│   └── test_model.ipynb  # Test model đã train
├── models/
│   └── final_model/      # Model đã fine-tune xong
├── requirements.txt
└── README.md
```

---

## Cách chạy

### 1. Cài đặt

```bash
pip install -r requirements.txt
```

### 2. Train model

```bash
# Cách 1: Dùng HuggingFace Trainer (đơn giản)
python src/train.py

# Cách 2: Custom training loop + WandB tracking
python src/train_with_loop.py
```

### 3. Xem cấu trúc model

```bash
pip install torchinfo
python src/model.py
```

---

Project này không nhằm tạo ra bản dịch hoàn hảo hơn Google Translate. Mục đích là hiểu rõ mọi thứ bên trong một model dịch thuật: embedding, attention, beam search. Nếu đọc đến đây mà hiểu thêm được chút gì thì coi như đạt mục tiêu rồi.
