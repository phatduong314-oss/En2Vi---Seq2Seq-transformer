import sys
from transformers import AutoConfig, AutoModelForSeq2SeqLM
import torch
import torchinfo

# Cấu hình encoding UTF-8 cho console
try:
    if sys.stdout.encoding != 'utf-8':
        sys.stdout.reconfigure(encoding='utf-8')
        sys.stderr.reconfigure(encoding='utf-8')
except (AttributeError, TypeError):
    # Jupyter notebook hoặc môi trường không hỗ trợ reconfigure
    pass

def load_model(model_checkpoint, device):
    
    # Load Config trước để xem tham số 
    config = AutoConfig.from_pretrained(model_checkpoint)
    
    # Load Model 
    model = AutoModelForSeq2SeqLM.from_pretrained(model_checkpoint)
    
    # Chuyển model sang GPU nếu có
    model.to(device)
        
    return model

if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = load_model("Helsinki-NLP/opus-mt-en-vi", device)

    torchinfo.summary(model, depth = 5)
    print(model.config)
    

"""
=====================================================================================
Layer (type:depth-idx)                                       Param #
=====================================================================================
MarianMTModel                                                --
├─MarianModel: 1-1                                           --
│    └─Embedding: 2-1                                        27,486,720
│    └─MarianEncoder: 2-2                                    --
│    │    └─Embedding: 3-1                                   27,486,720
│    │    └─MarianSinusoidalPositionalEmbedding: 3-2         262,144
│    │    └─ModuleList: 3-3                                  --
│    │    │    └─MarianEncoderLayer: 4-1                     3,152,384
│    │    │    └─MarianEncoderLayer: 4-2                     3,152,384
│    │    │    └─MarianEncoderLayer: 4-3                     3,152,384
│    │    │    └─MarianEncoderLayer: 4-4                     3,152,384
│    │    │    └─MarianEncoderLayer: 4-5                     3,152,384
│    │    │    └─MarianEncoderLayer: 4-6                     3,152,384
│    └─MarianDecoder: 2-3                                    --
│    │    └─Embedding: 3-4                                   27,486,720
│    │    └─MarianSinusoidalPositionalEmbedding: 3-5         262,144
│    │    └─ModuleList: 3-6                                  --
│    │    │    └─MarianDecoderLayer: 4-7                     4,204,032
│    │    │    └─MarianDecoderLayer: 4-8                     4,204,032
│    │    │    └─MarianDecoderLayer: 4-9                     4,204,032
│    │    │    └─MarianDecoderLayer: 4-10                    4,204,032
│    │    │    └─MarianDecoderLayer: 4-11                    4,204,032
│    │    │    └─MarianDecoderLayer: 4-12                    4,204,032
├─Linear: 1-2                                                27,486,720
=====================================================================================
Total params: 154,609,664
Trainable params: 154,609,664
Non-trainable params: 0
=====================================================================================
MarianConfig {
  "activation_dropout": 0.0,
  "activation_function": "swish",
  "add_bias_logits": false,
  "add_final_layer_norm": false,
  "architectures": [
    "MarianMTModel"
  ],
  "attention_dropout": 0.0,
  "bos_token_id": 0,
  "classif_dropout": 0.0,
  "classifier_dropout": 0.0,
  "d_model": 512,
  "decoder_attention_heads": 8,
  "decoder_ffn_dim": 2048,
  "decoder_layerdrop": 0.0,
  "decoder_layers": 6,
  "decoder_start_token_id": 53684,
  "decoder_vocab_size": 53685,
  "dropout": 0.1,
  "dtype": "float32",
  "encoder_attention_heads": 8,
  "encoder_ffn_dim": 2048,
  "encoder_layerdrop": 0.0,
  "encoder_layers": 6,
  "eos_token_id": 0,
  "extra_pos_embeddings": 53685,
  "id2label": {
    "0": "LABEL_0",
    "1": "LABEL_1",
    "2": "LABEL_2"
  },
  "init_std": 0.02,
  "is_decoder": false,
  "is_encoder_decoder": true,
  "label2id": {
    "LABEL_0": 0,
    "LABEL_1": 1,
    "LABEL_2": 2
  },
  "max_position_embeddings": 512,
  "model_type": "marian",
  "normalize_before": false,
  "normalize_embedding": false,
  "num_hidden_layers": 6,
  "pad_token_id": 53684,
  "scale_embedding": true,
  "share_encoder_decoder_embeddings": true,
  "static_position_embeddings": true,
  "tie_word_embeddings": true,
  "transformers_version": "5.1.0",
  "use_cache": true,
  "vocab_size": 53685
}



=====================================================================================
Layer (type:depth-idx)                                       Param #
=====================================================================================
MarianMTModel                                                --
├─MarianModel: 1-1                                           --
│    └─Embedding: 2-1                                        27,486,720
│    └─MarianEncoder: 2-2                                    --
│    │    └─Embedding: 3-1                                   27,486,720
│    │    └─MarianSinusoidalPositionalEmbedding: 3-2         262,144
│    │    └─ModuleList: 3-3                                  --
│    │    │    └─MarianEncoderLayer: 4-1                     --
│    │    │    │    └─MarianAttention: 5-1                   1,050,624
│    │    │    │    └─LayerNorm: 5-2                         1,024
│    │    │    │    └─SiLU: 5-3                              --
│    │    │    │    └─Linear: 5-4                            1,050,624
│    │    │    │    └─Linear: 5-5                            1,049,088
│    │    │    │    └─LayerNorm: 5-6                         1,024
│    │    │    └─MarianEncoderLayer: 4-2                     --
│    │    │    │    └─MarianAttention: 5-7                   1,050,624
│    │    │    │    └─LayerNorm: 5-8                         1,024
│    │    │    │    └─SiLU: 5-9                              --
│    │    │    │    └─Linear: 5-10                           1,050,624
│    │    │    │    └─Linear: 5-11                           1,049,088
│    │    │    │    └─LayerNorm: 5-12                        1,024
│    │    │    └─MarianEncoderLayer: 4-3                     --
│    │    │    │    └─MarianAttention: 5-13                  1,050,624
│    │    │    │    └─LayerNorm: 5-14                        1,024
│    │    │    │    └─SiLU: 5-15                             --
│    │    │    │    └─Linear: 5-16                           1,050,624
│    │    │    │    └─Linear: 5-17                           1,049,088
│    │    │    │    └─LayerNorm: 5-18                        1,024
│    │    │    └─MarianEncoderLayer: 4-4                     --
│    │    │    │    └─MarianAttention: 5-19                  1,050,624
│    │    │    │    └─LayerNorm: 5-20                        1,024
│    │    │    │    └─SiLU: 5-21                             --
│    │    │    │    └─Linear: 5-22                           1,050,624
│    │    │    │    └─Linear: 5-23                           1,049,088
│    │    │    │    └─LayerNorm: 5-24                        1,024
│    │    │    └─MarianEncoderLayer: 4-5                     --
│    │    │    │    └─MarianAttention: 5-25                  1,050,624
│    │    │    │    └─LayerNorm: 5-26                        1,024
│    │    │    │    └─SiLU: 5-27                             --
│    │    │    │    └─Linear: 5-28                           1,050,624
│    │    │    │    └─Linear: 5-29                           1,049,088
│    │    │    │    └─LayerNorm: 5-30                        1,024
│    │    │    └─MarianEncoderLayer: 4-6                     --
│    │    │    │    └─MarianAttention: 5-31                  1,050,624
│    │    │    │    └─LayerNorm: 5-32                        1,024
│    │    │    │    └─SiLU: 5-33                             --
│    │    │    │    └─Linear: 5-34                           1,050,624
│    │    │    │    └─Linear: 5-35                           1,049,088
│    │    │    │    └─LayerNorm: 5-36                        1,024
│    └─MarianDecoder: 2-3                                    --
│    │    └─Embedding: 3-4                                   27,486,720
│    │    └─MarianSinusoidalPositionalEmbedding: 3-5         262,144
│    │    └─ModuleList: 3-6                                  --
│    │    │    └─MarianDecoderLayer: 4-7                     --
│    │    │    │    └─MarianAttention: 5-37                  1,050,624
│    │    │    │    └─SiLU: 5-38                             --
│    │    │    │    └─LayerNorm: 5-39                        1,024
│    │    │    │    └─MarianAttention: 5-40                  1,050,624
│    │    │    │    └─LayerNorm: 5-41                        1,024
│    │    │    │    └─Linear: 5-42                           1,050,624
│    │    │    │    └─Linear: 5-43                           1,049,088
│    │    │    │    └─LayerNorm: 5-44                        1,024
│    │    │    └─MarianDecoderLayer: 4-8                     --
│    │    │    │    └─MarianAttention: 5-45                  1,050,624
│    │    │    │    └─SiLU: 5-46                             --
│    │    │    │    └─LayerNorm: 5-47                        1,024
│    │    │    │    └─MarianAttention: 5-48                  1,050,624
│    │    │    │    └─LayerNorm: 5-49                        1,024
│    │    │    │    └─Linear: 5-50                           1,050,624
│    │    │    │    └─Linear: 5-51                           1,049,088
│    │    │    │    └─LayerNorm: 5-52                        1,024
│    │    │    └─MarianDecoderLayer: 4-9                     --
│    │    │    │    └─MarianAttention: 5-53                  1,050,624
│    │    │    │    └─SiLU: 5-54                             --
│    │    │    │    └─LayerNorm: 5-55                        1,024
│    │    │    │    └─MarianAttention: 5-56                  1,050,624
│    │    │    │    └─LayerNorm: 5-57                        1,024
│    │    │    │    └─Linear: 5-58                           1,050,624
│    │    │    │    └─Linear: 5-59                           1,049,088
│    │    │    │    └─LayerNorm: 5-60                        1,024
│    │    │    └─MarianDecoderLayer: 4-10                    --
│    │    │    │    └─MarianAttention: 5-61                  1,050,624
│    │    │    │    └─SiLU: 5-62                             --
│    │    │    │    └─LayerNorm: 5-63                        1,024
│    │    │    │    └─MarianAttention: 5-64                  1,050,624
│    │    │    │    └─LayerNorm: 5-65                        1,024
│    │    │    │    └─Linear: 5-66                           1,050,624
│    │    │    │    └─Linear: 5-67                           1,049,088
│    │    │    │    └─LayerNorm: 5-68                        1,024
│    │    │    └─MarianDecoderLayer: 4-11                    --
│    │    │    │    └─MarianAttention: 5-69                  1,050,624
│    │    │    │    └─SiLU: 5-70                             --
│    │    │    │    └─LayerNorm: 5-71                        1,024
│    │    │    │    └─MarianAttention: 5-72                  1,050,624
│    │    │    │    └─LayerNorm: 5-73                        1,024
│    │    │    │    └─Linear: 5-74                           1,050,624
│    │    │    │    └─Linear: 5-75                           1,049,088
│    │    │    │    └─LayerNorm: 5-76                        1,024
│    │    │    └─MarianDecoderLayer: 4-12                    --
│    │    │    │    └─MarianAttention: 5-77                  1,050,624
│    │    │    │    └─SiLU: 5-78                             --
│    │    │    │    └─LayerNorm: 5-79                        1,024
│    │    │    │    └─MarianAttention: 5-80                  1,050,624
│    │    │    │    └─LayerNorm: 5-81                        1,024
│    │    │    │    └─Linear: 5-82                           1,050,624
│    │    │    │    └─Linear: 5-83                           1,049,088
│    │    │    │    └─LayerNorm: 5-84                        1,024
├─Linear: 1-2                                                27,486,720
=====================================================================================
Total params: 154,609,664
Trainable params: 154,609,664
Non-trainable params: 0
=====================================================================================
"""

