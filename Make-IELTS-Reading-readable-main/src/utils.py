import numpy as np
import evaluate

metric_bleu = evaluate.load("sacrebleu")
metric_chrf = evaluate.load("chrf")
metric_ter = evaluate.load("ter")

def compute_metrics(eval_preds, tokenizer, already_decoded=False):
    preds, labels = eval_preds

    if already_decoded:
        # all_preds/all_labels đã là list of strings (dùng trong manual training loop)
        decoded_preds = preds
        decoded_labels = labels
    else:
        if isinstance(preds, tuple):
            preds = preds[0]

        # Inference
        decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)

        # thay padding token bằng -100
        labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
        decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

    # Clean 
    decoded_preds = [pred.strip() for pred in decoded_preds]
    decoded_labels = [label.strip() for label in decoded_labels]

    # Chuẩn bị dữ liệu cho Metrics
    # Các metrics dịch thuật thường yêu cầu references dạng list of list vì một câu tiếng anh dịch sang tiếng việt có thể có nhiều cách viết (chung label)
    # Ví dụ: references = [['Tôi đi học', 'Tôi tới trường'], ['Xin chào', 'Chào bạn']]
    # Ở đây ta chỉ có 1 nhãn đúng duy nhất nên wrap nó lại.
    formatted_refs = [[l] for l in decoded_labels]
    
    # BLEU Score
    result_bleu = metric_bleu.compute(predictions=decoded_preds, references=formatted_refs)
    
    # chrF Score (character F-score)
    result_chrf = metric_chrf.compute(predictions=decoded_preds, references=formatted_refs)
    
    # TER (Translation Edit Rate) 
    result_ter = metric_ter.compute(predictions=decoded_preds, references=formatted_refs)

    return {
        "bleu": result_bleu["score"],
        "chrf": result_chrf["score"],
        "ter": result_ter["score"],     
        "gen_len": np.mean([len(t) for t in decoded_preds])
    }