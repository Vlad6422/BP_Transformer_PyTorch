from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from rouge_score import rouge_scorer
import torch

def calculate_bleu(iterator, model, tokenizer, sos_token, eos_token, pad_token_id,device):
    model.eval()
    total_bleu = 0
    count = 0
    smooth = SmoothingFunction().method4

    with torch.no_grad():
        for i, (src, trg) in enumerate(iterator):
            src, trg = src.to(device), trg.to(device)

            # Generate predictions
            output = model(src, trg[:, :-1])
            output_words = output.argmax(dim=-1).cpu().numpy()
            
            trg_words = trg[:, 1:].cpu().numpy()

            for j in range(len(output_words)):
                predicted_tokens = output_words[j].tolist()
                reference_tokens = trg_words[j].tolist()

                if eos_token in predicted_tokens:
                    predicted_tokens = predicted_tokens[:predicted_tokens.index(eos_token)]

                if eos_token in reference_tokens:
                    reference_tokens = reference_tokens[:reference_tokens.index(eos_token)]

                predicted_tokens = [t for t in predicted_tokens if t not in [pad_token_id, eos_token, sos_token]]
                reference_tokens = [t for t in reference_tokens if t not in [pad_token_id, eos_token, sos_token]]

                predicted_sentence = tokenizer.decode(predicted_tokens, skip_special_tokens=True)
                actual_sentence = tokenizer.decode(reference_tokens, skip_special_tokens=True)

                if reference_tokens:
                    reference_list = [actual_sentence.split()]
                    predicted_list = predicted_sentence.split()
                    bleu_score = sentence_bleu(reference_list, predicted_list, smoothing_function=smooth)
                    total_bleu += bleu_score
                    count += 1

    avg_bleu = total_bleu / count if count > 0 else 0
    print(f"Average BLEU score: {avg_bleu:.4f}")
    return avg_bleu * 100


def calculate_rouge(iterator, model, tokenizer, sos_token, eos_token, pad_token_id,device):
    model.eval()
    total_rouge = 0
    count = 0

    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)

    with torch.no_grad():
        for i, (src, trg) in enumerate(iterator):
            src, trg = src.to(device), trg.to(device)


            output = model(src, trg[:, :-1])
            output_words = output.argmax(dim=-1).cpu().numpy()

            trg_words = trg[:, 1:].cpu().numpy()

            for j in range(len(output_words)):
                predicted_tokens = output_words[j].tolist()
                reference_tokens = trg_words[j].tolist()

                if eos_token in predicted_tokens:
                    predicted_tokens = predicted_tokens[:predicted_tokens.index(eos_token)]

                if eos_token in reference_tokens:
                    reference_tokens = reference_tokens[:reference_tokens.index(eos_token)]


                predicted_tokens = [t for t in predicted_tokens if t not in [pad_token_id, eos_token, sos_token]]
                reference_tokens = [t for t in reference_tokens if t not in [pad_token_id, eos_token, sos_token]]


                predicted_sentence = tokenizer.decode(predicted_tokens, skip_special_tokens=True)
                actual_sentence = tokenizer.decode(reference_tokens, skip_special_tokens=True)


                if reference_tokens:
                    scores = scorer.score(actual_sentence, predicted_sentence)
                    total_rouge += scores['rouge1'].fmeasure
                    count += 1


    avg_rouge = total_rouge / count if count > 0 else 0
    print(f"Average ROUGE-1 score: {avg_rouge:.4f}")
    return avg_rouge * 100